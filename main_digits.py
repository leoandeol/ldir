import torch
import argparse
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import shutil
import torch.nn as nn

from nfnets.agc import AGC

from model import Model_linear, Model
from data import get_dataset, InfiniteSampler
from utils import predict, score_dis, score_clf
from knn import get_adaptive_radius_vat

from loss import VATLoss, VMTLoss, EntMinLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Invariant Networks")
    parser.add_argument(
        "--train_datasets", type=str, default="mnist,svhn", help="Train datasets"
    )
    parser.add_argument(
        "--known_labels",
        type=str,
        default="3000,3000",
        help="Number of known labels for each domain",
    )
    parser.add_argument("--test_datasets", type=str, default="", help="Test datasets")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--backbone", type=str, default="vat", help="Backbone")
    parser.add_argument(
        "--coeff_disc", type=float, default=0.1, help="Discriminator Coefficient"
    )
    parser.add_argument("--optim", type=str, default="adam", help="Optimizer choice")
    parser.add_argument(
        "--disc_type",
        help="Discriminator, possible values = ours, marginal, joint, none",
        type=str,
        default="ours",
    )
    parser.add_argument("--vmt", dest="vmt", action="store_true")
    parser.add_argument("--no-vmt", dest="vmt", action="store_false")
    parser.set_defaults(vmt=False)
    parser.add_argument("--vat", dest="vat", action="store_true")
    parser.add_argument("--no-vat", dest="vat", action="store_false")
    parser.set_defaults(vat=True)
    parser.add_argument("--entmin", dest="entmin", action="store_true")
    parser.add_argument("--no-entmin", dest="entmin", action="store_false")
    parser.set_defaults(entmin=True)

    args = parser.parse_args()
    disc_type = args.disc_type
    optim = args.optim
    n_epoch = args.n_epochs
    batch_size = args.batch_size
    use_vmt = args.vmt
    use_vat = args.vat
    use_entmin = args.entmin
    exp_folder = "exps"
    coeff_dicriminator = args.coeff_disc
    train_domains = args.train_datasets.split(",")
    n_train_sets = len(train_domains)
    assert n_train_sets == 2
    test_domains = args.test_datasets.split(",")
    if len(args.test_datasets) == 0:
        test_domains = []
    test_domains = train_domains + test_domains
    known_labels = list([int(x) for x in args.known_labels.split(",")])
    assert len(train_domains) == len(known_labels)
    model_type = "ours"
    backbone = args.backbone
    instance_norm = False
    spectral_norm = True
    xi = 10
    comments = ""
    run_name = "digits"
    config = {
        "model": model_type,
        "backbone": backbone,
        "train_domains": {
            domain: known for domain, known in zip(train_domains, known_labels)
        },
        "test_domains": test_domains,
        "n_epoch": n_epoch,
        "batch_size": batch_size,
        "coeff_dicriminator": coeff_dicriminator,
        "comments": comments,
        "instance_norm": instance_norm,
        "spectral_norm": spectral_norm,
        "disc_type": disc_type,
        "vmt": use_vmt,
        "vat": use_vat,
        "entmin": use_entmin,
        "optimizer": optim,
        "xi":xi,
    }
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    datasets_train = [
        get_dataset(domain, test=False, known=known)
        for domain, known in zip(train_domains, known_labels)
    ]

    epsilons = []
    print("Computing adaptive radiuses")
    for i, dataset in enumerate(datasets_train):
        X = dataset.dataset.data
        eps = torch.ones((len(X),)) * 2.5
        #        print(eps.shape)
        epsilons.append(eps)
        dataset.epsilons = eps

    datasets_test = [
        get_dataset(domain, test=True, known=-1) for domain in test_domains
    ]

    alpha_betas = list([k / len(d) for k, d in zip(known_labels, datasets_train)])

    dataloaders_train_semi = [
        torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        for dataset in datasets_train
    ]

    dataloaders_train_sup = [
        torch.utils.data.DataLoader(
            dataset.get_sup_dataset(),
            batch_size=batch_size // 4 if batch_size >= 128 else batch_size,
            drop_last=True,
            sampler=InfiniteSampler(known),
        )
        for dataset, known in zip(datasets_train, known_labels)
    ]

    dataloaders_test = [
        torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        for dataset in datasets_test
    ]

    lens = [len(dataset) for dataset in datasets_train]
    n_domains = len(train_domains)

    classif_loss_criterion = torch.nn.NLLLoss()  # CrossEntropyLoss()
    # last case for mnist_recs
    if (
        "mnist" in train_domains
        or "svhn" in train_domains
        or "mnist" in train_domains[0]
    ):
        net = Model(
            mode="digits",
            instance_norm=instance_norm,
            spectral_norm=spectral_norm,
            backbone=backbone,
            disc_type=disc_type,
        )
    else:
        raise ValueError("Wrong dataset?")
    multi_gpu = False
    if False:  # torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)
        multi_gpu = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    net_ = net.module if multi_gpu else net
    if optim.lower() == "nfnet":
        print("Using NFNet's AGC")
        optim_emb = torch.optim.SGD(net_.embedder.parameters(), 1e-3)
        optim_emb = AGC(
            net_.embedder.parameters(),
            optim_emb,
            model=net_.embedder,
            ignore_agc=["head.fc"],
        )
        optim_clf = torch.optim.SGD(net_.classifier.parameters(), lr=1e-3, momentum=0.9)
        optim_clf = AGC(net_.classifier.parameters(), optim_clf)
        optim_disc = torch.optim.Adam(net_.discriminator.parameters(), lr=1e-4)
        # optim_disc = AGC(
        #   net.discriminator.parameters(), optim, model=net.discriminator)
        # )
    elif optim.lower() == "adam":
        print("Using Adam")
        optim_emb = torch.optim.Adam(net_.embedder.parameters(), lr=1e-4)
        optim_clf = torch.optim.Adam(net_.classifier.parameters(), lr=1e-4)
        optim_disc = torch.optim.Adam(net_.discriminator.parameters(), lr=1e-4)
    elif optim.lower() == "sgd":
        print("Using SGD")
        optim_emb = torch.optim.SGD(
            net_.embedder.parameters(), lr=1e-3, nesterov=True, momentum=0.9
        )
        optim_clf = torch.optim.SGD(
            net_.classifier.parameters(), lr=1e-3, nesterov=True, momentum=0.9
        )
        optim_disc = torch.optim.SGD(
            net_.discriminator.parameters(), lr=1e-3, nesterov=True, momentum=0.9
        )

    val_sum_errs = []
    val_max_errs = []

    if use_vmt:
        vmt_loss = VMTLoss(net_)
    if use_vat:
        vat_loss_criterion = [
            VATLoss(net_, eps=epsilons[0], xi=xi),
            VATLoss(net_, eps=epsilons[1], xi=xi),
        ]
    if use_entmin:
        entmin_loss_criterion = EntMinLoss()

    net_.train()
    alpha = 1
            
    for epoch in range(n_epoch):
        data_iters_semi = [iter(dataloader) for dataloader in dataloaders_train_semi]
        data_iters_sup = [iter(dataloader) for dataloader in dataloaders_train_sup]
        i = 0
        n_iters = len(data_iters_semi)

        len_dataloader = np.min([len(x) for x in dataloaders_train_semi])
        while i < len_dataloader:
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            net.zero_grad()
            optim_emb.zero_grad()
            optim_clf.zero_grad()
            optim_disc.zero_grad()

            i += 1

            all_classif_losses = []
            all_domain_outputs = []
            all_class_outputs = []
            all_mixed_y = []
            all_x_semi = []
            all_vat_losses = []
            all_entmin_losses = []
            all_y_est_semi = []
            all_y_mixed_semi = []

            for j in range(n_iters):
                #######################
                # Classification
                #######################

                data_semi = data_iters_semi[j].next()
                data_sup = data_iters_sup[j].next()

                x_semi, y_semi, eps_semi = data_semi
                x_sup, y_sup, _ = data_sup
                
                y_est_sup = net.forward_log_classifier(x_sup)

                classif_loss = classif_loss_criterion(y_est_sup, torch.argmax(y_sup, 1))
                all_classif_losses.append(classif_loss)

                #################################
                # Domain Critic and Regularizers
                #################################

                embeddings_semi, y_est_semi, disc_semi, y_mixed_semi = net(
                    x_semi, y_semi, alpha
                )

                all_domain_outputs.append(disc_semi)
                all_x_semi.append(x_semi)
                all_y_mixed_semi.append(y_mixed_semi)
                all_y_est_semi.append(y_est_semi)

                # Virtual Adversarial training
                if use_vat:
                    loss_vat = vat_loss_criterion[j](x_semi, eps_semi)
                    all_vat_losses.append(loss_vat)

                # Conditional Entropy
                if use_entmin:
                    loss_entmin = entmin_loss_criterion(y_est_semi)
                    all_entmin_losses.append(loss_entmin)

            #####################
            # Loss and Backprop
            #####################
            alpha_betas = [0, 0]
            loss_classif = torch.sum(
                torch.stack(
                    [
                        loss  # * (1 - weight)
                        for loss, weight in zip(all_classif_losses, alpha_betas)
                    ]
                )
            )

            if disc_type == "ours":
                loss_disc = (all_y_mixed_semi[0] * all_domain_outputs[0]).mean(
                    dim=0
                ) - (all_y_mixed_semi[1] * all_domain_outputs[1]).mean(dim=0)
                loss_disc = loss_disc.mean()
            elif disc_type == "joint" or disc_type == "marginal":
                loss_disc = all_domain_outputs[0].mean() - all_domain_outputs[1].mean()
            elif disc_type.lower() == "none":
                loss_disc = torch.zeros(1, requires_grad=True).cuda().mean()

            loss_vat = 0
            loss_vmt = 0
            loss_entmin = 0
            if use_vat:
                loss_vat = torch.sum(torch.stack(all_vat_losses))
            if use_entmin:
                loss_entmin = torch.sum(torch.stack(all_entmin_losses))

            # Virtual Mixup Training
            if use_vmt:
                loss_vmt += vmt_loss(all_x_semi[0], all_y_est_semi[0])
                loss_vmt += vmt_loss(all_x_semi[1], all_y_est_semi[1])

            loss = (
                loss_classif
                + coeff_dicriminator * loss_disc
                + loss_vat
                + 0.2 * loss_vmt
                + loss_entmin
            )

            loss.backward()
            optim_clf.step()
            optim_disc.step()
            optim_emb.step()

        net.eval()
        test_domains_scores = {
            domain: score_clf(net, dataloader)
            for domain, dataloader in zip(test_domains, dataloaders_test)
        }
        dataloader1t = list(dataloaders_test)[0]
        dataloader2t = list(dataloaders_test)[1]
        net.train()
        score1 = test_domains_scores[test_domains[0]]
        score2 = test_domains_scores[test_domains[1]]
        sum_error = (1 - score1) + (1 - score2)
        max_error = max(1 - score1, 1 - score2)
        val_sum_errs.append(sum_error)
        val_max_errs.append(max_error)

        # do checkpointing
        torch.save(net.state_dict(), f"{exp_folder}/{run_name}_{epoch}.pth")
        if val_sum_errs[-1] == np.max(val_sum_errs):
            shutil.copy(
                f"{exp_folder}/{run_name}_{epoch}.pth",
                f"{exp_folder}/{run_name}_best_net.pth",
            )
