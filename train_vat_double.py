import torch
import argparse
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize
import wandb
from sklearn.svm import SVC
import shutil
import torch.nn as nn

from nfnets.agc import AGC

from model import JDDAN_linear, JDDAN, VAT
from data import get_dataset, InfiniteSampler
from utils import predict, score_dis, score_clf, calc_gradient_penalty, TensorArray  #
from knn import get_adaptive_radius_vat

# python3 main.py --train_datasets mnist,svhn --known_labels 3000,3000 --test_datasets usps --batch_size 256 --n_epochs 50 --backbone nf_regnet_b0
#
from loss import VATLoss, VMTLoss, EntMinLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Invariant Networks")
    parser.add_argument(
        "--train_datasets", type=str, default="mnist,svhn", help="Train datasets"
    )
    parser.add_argument(
        "--known_labels",
        type=str,
        default="500,500",
        help="Number of known labels for each domain",
    )
    parser.add_argument("--test_datasets", type=str, default="", help="Test datasets")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone")
    parser.add_argument(
        "--coeff_disc", type=float, default=0.1, help="Discriminator Coefficient"
    )
    parser.add_argument("--optim", type=str, default="adam", help="Optimizer choice")
    parser.add_argument(
        "--disc_type",
        help="Discriminator, possible values = conditional, marginal, joint, none",
        type=str,
        default="conditional",
    )
    parser.add_argument("--vmt", dest="vmt", action="store_true")
    parser.add_argument("--no-vmt", dest="vmt", action="store_false")
    parser.set_defaults(vmt=True)
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
    assert n_train_sets == 2, "code only supports two until further update"
    test_domains = args.test_datasets.split(",")
    if len(args.test_datasets) == 0:
        test_domains = []
    test_domains = train_domains + test_domains
    known_labels = list([int(x) for x in args.known_labels.split(",")])
    assert len(train_domains) == len(known_labels)
    model_type = "jddan"
    backbone = args.backbone
    instance_norm = False
    spectral_norm = True
    comments = "VAT-augs-lessbatch-stability-VATmodel"
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
    }
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    domain = "mnist-svhn"

    dataset1 = get_dataset("mnist", test=False, known=3000)
    dataset2 = get_dataset("svhn", test=False, known=3000)

    dataset1.epsilons = torch.ones((len(dataset1.dataset.data))) * 2.5
    dataset2.epsilons = torch.ones((len(dataset2.dataset.data))) * 2.5

    dataset1_test = get_dataset("mnist", test=True, known=-1)
    dataset2_test = get_dataset("svhn", test=True, known=-1)

    dataloader1_train_semi = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=True, drop_last=True
    )

    dataloader1_train_sup = torch.utils.data.DataLoader(
        dataset1.get_sup_dataset(),
        batch_size=batch_size // 4,
        drop_last=True,
        sampler=InfiniteSampler(3000),
    )

    dataloader1_test = torch.utils.data.DataLoader(
        dataset1_test, batch_size=batch_size, shuffle=False, drop_last=False
    )

    dataloader2_train_semi = torch.utils.data.DataLoader(
        dataset2, batch_size=batch_size, shuffle=True, drop_last=True
    )

    dataloader2_train_sup = torch.utils.data.DataLoader(
        dataset2.get_sup_dataset(),
        batch_size=batch_size // 4,
        drop_last=True,
        sampler=InfiniteSampler(3000),
    )

    dataloader2_test = torch.utils.data.DataLoader(
        dataset2_test, batch_size=batch_size, shuffle=False, drop_last=False
    )

    classif_loss_criterion = torch.nn.NLLLoss()  # CrossEntropyLoss()
    # last case for mnist_recs
    if (
        "mnist" in train_domains
        or "svhn" in train_domains
        or "mnist" in train_domains[0]
    ):
        net = JDDAN(
            mode="digits",
            instance_norm=instance_norm,
            spectral_norm=spectral_norm,
            backbone=backbone,
            disc_type=disc_type,
        )
    elif (
        "art" in train_domains
        or "sketch" in train_domains
        or "cartoon" in train_domains
        or "photo" in train_domains
    ):
        net = JDDAN(
            mode="pacs",
            instance_norm=instance_norm,
            spectral_norm=spectral_norm,
            backbone=backbone,
            disc_type=disc_type,
        )
    else:
        raise ValueError("Wrong dataset?")

    #    net = VAT()

    multi_gpu = False
    if False:  # torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = nn.DataParallel(net)
        multi_gpu = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    wandb.init(project="jddan", config=config)
    # wandb.watch(net)

    # optim = torch.optim.Adam(
    #    net.parameters(),
    #    lr=0.0001,
    #    # betas=(0.5, 0.999),
    # )
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
        vat_loss_criterion = VATLoss(net_, eps="yolo", xi=10)
    if use_entmin:
        entmin_loss_criterion = EntMinLoss()

    net_.train()
    for epoch in range(n_epoch):
        data1_iters_semi = iter(dataloader1_train_semi)
        data1_iters_sup = iter(dataloader1_train_sup)
        data2_iters_semi = iter(dataloader2_train_semi)
        data2_iters_sup = iter(dataloader2_train_sup)
        i = 0

        len_dataloader = len(dataloader1_train_semi)
        while i < len_dataloader:
            # p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 1  # 2.0 / (1.0 + np.exp(-10 * p)) - 1

            # clamp parameters to a cube
            # for p in emb.parameters():
            #        p.data.clamp_(-0.01, 0.01)

            # emb.zero_grad()
            # dis.zero_grad()
            # clf.zero_grad()
            i += 1
            net.zero_grad()
            optim_emb.zero_grad()
            optim_clf.zero_grad()
            optim_disc.zero_grad()

            #######################
            # Classification
            #######################

            data_semi = data1_iters_semi.next()
            data_sup = data1_iters_sup.next()

            x_semi, y_semi, eps_semi = data_semi
            x_sup, y_sup, _ = data_sup

            y_est_sup = net.forward_log_classifier(x_sup)

            classif_loss = classif_loss_criterion(y_est_sup, torch.argmax(y_sup, 1))

            embeddings_semi, y_est_semi, disc_semi, y_mixed_semi = net(
                x_semi, y_semi, alpha
            )

            # Virtual Adversarial training
            if use_vat:
                loss_vat = vat_loss_criterion(x_semi, eps_semi)

            # Conditional Entropy
            if use_entmin:
                loss_entmin = entmin_loss_criterion(y_est_semi)

            #######################
            # Classification BIS
            #######################

            data_semi = data2_iters_semi.next()
            data_sup = data2_iters_sup.next()

            x_semi, y_semi, eps_semi = data_semi
            x_sup, y_sup, _ = data_sup

            y_est_sup = net.forward_log_classifier(x_sup)

            classif_loss += classif_loss_criterion(y_est_sup, torch.argmax(y_sup, 1))

            embeddings_semi, y_est_semi, disc_semi, y_mixed_semi = net(
                x_semi, y_semi, alpha
            )

            # Virtual Adversarial training
            if use_vat:
                loss_vat += vat_loss_criterion(x_semi, eps_semi)

            # Conditional Entropy
            if use_entmin:
                loss_entmin += entmin_loss_criterion(y_est_semi)

            #####################
            # Loss and Backprop
            #####################

            loss = (
                classif_loss
                # + coeff_dicriminator * loss_disc
                + loss_vat
                + loss_entmin
            )

            loss.backward()
            optim_clf.step()
            optim_disc.step()
            optim_emb.step()
            # [[x.detach_() for x in f.array] for f in featurequeues]
            loss_classif = classif_loss
            wandb.log(
                {
                    "epoch": epoch,
                    "iter": i,
                    "err_clf": loss_classif.item()
                    if not isinstance(loss_classif, float)
                    else loss_classif,
                    "loss_vat": loss_vat,
                }
            )

        net.eval()
        # train_domains_scores = {
        #    domain: score_clf(net, dataloader)
        #    for domain, dataloader in zip(train_domains, dataloaders_train)
        # }
        test_domains_scores = {
            "mnist": score_clf(net, dataloader1_test),
            "svhn": score_clf(net, dataloader2_test),
        }
        # scoreDis = score_dis(net, dataloader1t, dataloader2t)
        net.train()

        wandb.log(
            {
                "scores_test": test_domains_scores,
            }
        )

        # do checkpointing
        torch.save(net.state_dict(), f"{exp_folder}/net_epoch_{epoch}.pth")

        # shutil.copy(
        #    f"{exp_folder}/net_epoch_{np.argmin(val_sum_errs)}.pth",
        #    f"{exp_folder}/{wandb.run.name}_best_net.pth",
        #    #)

        # do checkpointing

        # if val_sum_errs[-1] == np.max(val_sum_errs):
        #     shutil.copy(
        #         f"{exp_folder}/net_epoch_{np.argmin(val_sum_errs)}.pth",
        #         f"{exp_folder}/{wandb.run.name}_best_net.pth",
        #     )
