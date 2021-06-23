import os
import time
import torch
import numpy as np
import nmslib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class NMSlibTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

    def __init__(self, n_neighbors=5, metric='euclidean', method='sw-graph',
                 n_jobs=1):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        self.n_samples_fit_ = X.shape[0]

        # see more metric in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual
        space = {
            'euclidean': 'l2',
            'cosine': 'cosinesimil',
            'l1': 'l1',
            'l2': 'l2',
        }[self.metric]

        self.nmslib_ = nmslib.init(method=self.method, space=space)
        self.nmslib_.addDataPointBatch(X)
        self.nmslib_.createIndex()
        return self

    def transform(self, X):
        n_samples_transform = X.shape[0]

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        results = self.nmslib_.knnQueryBatch(X, k=n_neighbors,
                                             num_threads=self.n_jobs)
        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        indptr = np.arange(0, n_samples_transform * n_neighbors + 1,
                           n_neighbors)
        kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
                                       indptr), shape=(n_samples_transform,
                                                       self.n_samples_fit_))
        return kneighbors_graph



def get_adaptive_radius_vat(X, dataset, device):
    start = time.time()
    alpha = 0.25
    K = 10 # Number of neighbors
    n = X.shape[0]
    X = X.reshape((n, -1))
    if n > 5*10**5:
        nms = NMSlibTransformer(n_neighbors = K) # Compute approximated knn graph
        Knn_graph = nms.fit_transform(X)
        # define adaptie radius for VAT
        Knn_dist = Knn_graph.data.reshape(n,K+1)
        R = alpha*Knn_dist[:,K]
        R = R.reshape(X.shape[0],1) 
        del Knn_graph, Knn_dist
        end = time.time()
        print(f"{end-start} seconds by Approximated KNN.")
    else:
        knncachestr = "cache/%s-k%d" % (dataset, K+1)
        if dataset != "unknown" and os.path.exists(knncachestr + ".npy"):
            print("Loaded cached kNN from %s" % knncachestr + ".npy")
            distances = np.load(knncachestr + ".npy")
            indices = np.load(knncachestr + "idx.npy")
        else:
            nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='brute').fit(X)
            distances, indices = nbrs.kneighbors(X)
            np.save(knncachestr+".npy", distances)
            np.save(knncachestr+"idx.npy", indices)
        K_vat = 10
        R = alpha*distances[:,K_vat]
        R = R.reshape(n,1)
        end = time.time()
        print(f"{end-start} seconds by Brute-Force KNN.")
    R = torch.tensor(R.ravel().astype('f')).to(device)
    #X = torch.tensor(X.astype('f')).to(dev) # this unlabeled dataset (set of feature vectors) is input of IMSAT
    return R
