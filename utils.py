import torch
import numpy as np
import umap
from torch import autograd
from tqdm import tqdm
from ot import emd2, sinkhorn2
from scipy.spatial.distance import cdist

def predict(model, dataloader):
    prediction_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            pred = model(batch[0])
            prediction_list.append(pred.cpu())
    return torch.cat(prediction_list, 0).numpy()


def score_clf(model, dataloader, domain=None):
    prediction_list = []
    labels = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x, y, _ = batch
            pred = model(x, y, 0)[1]
            prediction_list.append(pred.cpu())
            labels.append(y.cpu())

    score = np.mean(
        np.argmax(torch.cat(prediction_list, 0).numpy(), 1)
        == np.argmax(torch.cat(labels, 0).numpy(), 1)
    )
    return score

def score_dis(model, dataloader1, dataloader2):
    prediction_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader1):
            x, y, _ = batch
            pred = model(x, y, 0)[2]
            prediction_list.append(pred.cpu() <= 0.5)
        score1 = torch.cat(prediction_list, 0)
        prediction_list = []
        for i, batch in enumerate(dataloader2):
            x, y, _ = batch
            pred = model(x, y, 0)[2]
            prediction_list.append(pred.cpu() > 0.5)
        score2 = torch.cat(prediction_list, 0)
    return torch.true_divide(
        torch.sum(torch.cat((score1, score2), 0)), (score1.shape[0] + score2.shape[0])
    )


class TensorArray:
    def __init__(self, batch_size, number_classes, *args):
        self.array = [np.random.rand(1, number_classes, *args) for _ in range(batch_size)]
        self.array = list(
            [
                torch.Tensor(x / np.sum(x, axis=1, keepdims=True)).cuda()
                for x in self.array
            ]
        )
        self.batch_size = batch_size
        self.number_classes = number_classes

    def __call__(self):
        return torch.cat(self.array)

    def append(self, element):
        length = element.shape[0]
        if length > 0:
            self.array = self.array[length:]
            # make all except the new tensors detached from the graph
            #self.array = list([x.detach() for x in self.array])
            # make a list of tensors and add them to the list
            list_elements = list(torch.chunk(element, length, dim=0))
            self.array = self.array + list_elements

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_feature_matrix(net, dataloader1, dataloader2=None, with_labels=False):
    d1_feats = []
    d2_feats = []

    if not with_labels:
        with torch.no_grad():
            for data in tqdm(dataloader1):
                x, _, _ = data
                feat = net.embedder(x)
                d1_feats.append(feat)
            d1_feats = torch.cat(d1_feats).detach().cpu().numpy()
            if dataloader2 is not None:
                for data in tqdm(dataloader2):
                    x, _, _ = data
                    feat = net.embedder(x)
                    d2_feats.append(feat)
                d2_feats = torch.cat(d2_feats).detach().cpu().numpy()
                return d1_feats, d2_feats
            else:
                return d1_feats
    else:
        with torch.no_grad():
            for data in tqdm(dataloader1):
                x, y, _ = data
                feat, _, _, ymix = net(x, y, 1)
                d1_feats.append(torch.hstack((feat, ymix)))
            d1_feats = torch.cat(d1_feats).detach().cpu().numpy()
            if dataloader2 is not None:
                for data in tqdm(dataloader2):
                    x, y, _ = data
                    feat, _, _, ymix = net(x, y, 1)
                    d2_feats.append(torch.hstack((feat, ymix)))
                d2_feats = torch.cat(d2_feats).detach().cpu().numpy()
                return d1_feats, d2_feats
            else:
                return d1_feats


def wasserstein_distance(feats1, feats2, sinkhorn=False):
    C_feats = cdist(feats1, feats2)
    if sinkhorn:
        out_feats = sinkhorn2([], [], C_feats, log=True, processes=36)
    else:
        out_feats = emd2([], [], C_feats, log=True, processes=36)
    return out_feats[0]

def get_umap_features(feats1, feats2):
    all_feats = np.concatenate([feats1, feats2])
    stop = len(feats1)
    ubeds = umap.UMAP(verbose=True).fit_transform(all_feats)
    ubeds1 = ubeds[:stop]
    ubeds2 = ubeds[stop:]
    return ubeds1, ubeds2
