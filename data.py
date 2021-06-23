import os
from sklearn.decomposition import PCA
import scipy.io
from pandas import DataFrame
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, Subset
from torch.utils.data.sampler import Sampler
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
import h5py
from randaugment import RandAugment


def get_dataset(dataset_name, test=False, known=None, **kwargs):
    if known is None:
        raise ValueError("The number of known labels must be specified")

    if dataset_name == "mnist":
        if test:
            return JDDAtasetWrapper("mnist_test", known=known)
        else:
            return JDDAtasetWrapper("mnist", known=known)
    elif dataset_name == "svhn":
        if test:
            return JDDAtasetWrapper("svhn_test", known=known)
        else:
            return JDDAtasetWrapper("svhn", known=known)
    elif dataset_name == "mnistm":
        if test:
            return JDDAtasetWrapper("mnistm_test", known=known)
        else:
            return JDDAtasetWrapper("mnistm", known=known)
    elif dataset_name == "usps":
        if test:
            return JDDAtasetWrapper("usps_test", known=known)
        else:
            return JDDAtasetWrapper("usps", known=known)
    elif dataset_name in ["amazon", "caltech", "dslr", "webcam"]:
        if test is False:
            return AmazonCaltechDataset(dataset_name, known)
    elif dataset_name in ["photo", "art", "cartoon", "sketch"]:
        if test:
            return JDDAtasetWrapper(f"{dataset_name}_test", known=known)
        else:
            return JDDAtasetWrapper(f"{dataset_name}", known=known)
    elif "mnist_rec" in dataset_name:
        splits = dataset_name.split("-")
        d = int(splits[-2])
        size= int(splits[-1])
        if test:
            return JDDAtasetWrapper("mnist_rec", known=known, test=True, d=d, size=size, **kwargs)
        else:
            return JDDAtasetWrapper("mnist_rec", known=known, test=False, d=d, size=size, **kwargs)

classes = {"C15": 0, "CCAT": 1, "E21": 2, "ECAT": 3, "GCAT": 4, "M11": 5}
# classes = { "C15": 0,
#           "CCAT" : 1,
#           "E21" : 1,
#          "ECAT" : 1,
#          "GCAT" : 1,
#          "M11" : 1
#            }

class InfiniteSampler(Sampler):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        while True:
            order = np.random.permutation(self.num_samples)
            for i in range(self.num_samples):
                yield order[i]

    def __len__(self):
        return None


class MNISTM(datasets.VisionDataset):
    """MNIST-M Dataset."""

    resources = [
        (
            "https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz",
            "191ed53db9933bd85cc9700558847391",
        ),
        (
            "https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz",
            "e11cb4d7fff76d7ec588b1134907db59",
        ),
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        print(os.path.join(self.processed_folder, data_file))

        self.data, self.targets = torch.load(
            os.path.join(self.processed_folder, data_file)
        )

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST-M data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition("/")[2]
            download_and_extract_archive(
                url,
                download_root=self.raw_folder,
                extract_root=self.processed_folder,
                filename=filename,
                md5=md5,
            )

        print("Done!")

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class PACS(datasets.VisionDataset):

    # resources = [
    #     (
    #         "https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz",
    #         "191ed53db9933bd85cc9700558847391",
    #     ),
    #     (
    #         "https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz",
    #         "e11cb4d7fff76d7ec588b1134907db59",
    #     ),
    # ]

    NUM_CLASSES = 7  # 7 classes for each domain: 'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'
    DOMAIN_NAMES = ["photo", "art", "cartoon", "sketch"]
    CLASS_NAMES = ["Dog", "Elephant", "Giraffe", "Guitar", "Horse", "House", "Person"]
    SPLITS = ["train", "test"]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(
        self,
        root,
        domain,
        split="train",
        transform=None,
        target_transform=None,
        download=False,
    ):
        """Init PACS dataset."""
        super(PACS, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if download:
            self.download()

        # if not self._check_exists():
        #    raise RuntimeError(
        #        "Dataset not found." + " You can use download=True to download it"
        #    )

        if domain not in self.DOMAIN_NAMES:
            raise ValueError(f"Domain '{domain}' is not one of {self.DOMAIN_NAMES}")
        elif domain == "art":
            domain_path = "art_painting"
        else:
            domain_path = domain
        self.domain = domain

        if split not in self.SPLITS:
            raise ValueError(f"Split must be one of {self.SPLITS} but got '{split}'")
        self.split = split

        self.file_path = os.path.join(
            self.processed_folder, f"{domain_path}_{split}.hdf5"
        )
        with h5py.File(self.file_path, "r") as f:
            self.images = np.array(f["images"])
            self.labels = np.array(f["labels"])

        if split == "train":
            val_path = os.path.join(self.processed_folder, f"{domain_path}_val.hdf5")
            with h5py.File(val_path, "r") as f:
                self.images = np.concatenate((self.images, np.array(f["images"])))
                self.labels = np.concatenate((self.labels, np.array(f["labels"])))

        # default dtype is int16, leads to bug, should use uint8
        self.images = self.images.astype(np.uint8)
        # shift the labels to start from 0
        self.labels -= np.min(self.labels)

        assert len(self.images) == len(self.labels)

        self.file_num_train = len(self.labels)

        self.data = self.images
        self.targets = self.labels

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def download(self):
        pass

    def extra_repr(self):
        return f"Split: {self.split}"


def load_file(filename, sub=None):
    X = dict()
    y = dict()
    print("\n\nLoading", filename)
    with open(filename, "r") as f:
        line = f.readline()
        cnt = 1
        while line:
            split = line.replace(" \n", "").split(" ")
            if len(split) <= 1:
                continue
            cat = split[0]
            X[cnt] = dict()
            y[cnt] = cat
            for pair in split[1:]:
                key, value = pair.split(":")
                X[cnt][int(key)] = float(value)
            line = f.readline()
            cnt += 1
    print("Loaded ...\n Some Statistics :")
    assert len(y) == len(X)
    print("Number of entries", len(y))

    print("Number of classes : ", len(np.unique(y.values())))
    print("Mean number of features : ", sum([len(x) for x in X.values()]) / len(X))
    if sub is not None:
        # print(cnt)
        idxs = list(range(1, cnt))
        np.random.shuffle(idxs)
        idxs = idxs[:sub]
        X = {key: val for key, val in X.items() if key in idxs}
        y = {key: val for key, val in y.items() if key in idxs}
    return X, y


def densify(X, y):
    print("Transforming the data to a dense matrix")
    yy = dict()
    for key, val in y.items():
        yy[key] = classes[val]
    yy = {123: yy}
    yy = DataFrame.from_dict(yy, orient="columns", dtype=None).values.reshape((-1, 1))
    XX = DataFrame.from_dict(X, orient="columns", dtype=None).fillna(0).values.T
    # NORMAIZE : MAY ALTER DATA ?
    mins = np.amin(XX, axis=0).reshape((1, -1))
    XX = XX - (np.ones((XX.shape[0], 1)) @ mins)
    print("Size of the new matrix : ", XX.shape)
    return XX, yy


def reduce_dims(X, energy=0.60):  # Energy kept in HFA experiments
    """
    Transforms the sparse representations into dense representations and applies PCA on X
    """
    print("Computing the PCA")
    pca = PCA(n_components=energy, svd_solver="auto")
    X = pca.fit_transform(X)
    print("Size of the reduced matrix : ", X.shape)
    # NORMAIZE : MAY ALTER DATA ?
    # mins = np.amin(X, axis=0).reshape((1,-1))
    # X = X - (np.ones((X.shape[0],1)) @ mins)
    return X


def load_data(src, sub, energy=0.60):
    X, y = load_file(
        "datasets/MultiLingualReuters/" + src + "/Index_" + src + "-" + src, sub=sub
    )
    X, y = densify(X, y)
    X = reduce_dims(X, energy)
    return X, y.ravel()


def export_data(path, X, y):
    with open("datasets/MultiLingualReuters/" + path + "/X.dat", "wb") as f:
        pickle.dump(X, f)
    with open("datasets/MultiLingualReuters/" + path + "/y.dat", "wb") as f:
        pickle.dump(y, f)


def import_data(path, sub=None):
    with open("datasets/MultiLingualReuters/" + path + "/X.dat", "rb") as f:
        X = pickle.load(f)
    with open("datasets/MultiLingualReuters/" + path + "/y.dat", "rb") as f:
        y = pickle.load(f)
    if sub is not None:
        return X[sub, :], y[sub]
    return X, y


# TO prepare HDA data
# if __name__ == "__main__":
#    srcs = ["FR","SP","IT","GR","EN"]
#    energies = [1230,807,1041,1417,1131]
#    for src,e in zip(srcs,energies):
#        X,y = load_data(src,None, e)
#        export_data(src,X,y)


def import_caltech_office():
    dic = dict()
    return dic


def import_amazon_review():
    dic = dict()
    v = DictVectorizer(sparse=True)
    domains = ["books", "dvd", "electronics", "kitchen"]
    label_transform = {"negative": 0, "positive": 1}
    filenames = ["negative", "positive", "unlabeled"]
    ext = ".review"
    base = "/home/leoandeol/sjdot/datasets/processed_acl/"
    total = []
    # First run to fit the vectorizer
    for domain in domains:
        dic[domain] = dict()
        for filename in filenames:
            with open(base + domain + "/" + filename + ext, "r", encoding="utf-8") as f:
                text = f.read()
                lines = text.split("\n")
                sentences = list(
                    [
                        {
                            word.split(":")[0]: word.split(":")[1]
                            for word in line.split(" ")[1:-1]
                        }
                        for line in lines[:-1]
                    ]
                )
                labels = list(
                    [
                        label_transform[line.split(" ")[-1].split(":")[1]]
                        for line in lines[:-1]
                    ]
                )
                total.append(sentences)
                dic[domain][filename] = sentences, np.array(labels)
    total = list([i for j in total for i in j])
    v.fit(total)
    # print(len(v.vocabulary_))
    del total
    for domain in domains:
        for filename in filenames:
            #            with open(base+domain+"/"+filename+ext, "r", encoding='utf-8') as f:
            #                text = f.read()
            #                lines = text.split("\n")
            #                sentences = list([   {   word.split(":")[0]:word.split(":")[1]  for word in line.split(" ")[1:-1]  }   for line in lines[:-1] ])
            #                labels = list([  label_transform[line.split(" ")[-1].split(":")[1]] for line in lines[:-1]])
            sentences, labels = dic[domain][filename]
            dic[domain][filename] = v.transform(sentences), labels
    return dic


# class JDDAtaset(Dataset):
#     def __init__(self, domain1, domain2):
#         X1, y1 = domain1
#         X2, y2 = domain2

#         n_classes = len(np.unique(y1))
#         assert n_classes == len(np.unique(y2))
#         eye = torch.eye(n_classes)
#         y1_one_hot = eye[y1]
#         y2_one_hot = eye[y2]

#         self.task_labels = torch.cat((y1_one_hot, y2_one_hot), 0).cuda()
#         self.data = torch.cat((torch.Tensor(X1), torch.Tensor(X2)), 0).cuda()

#         self.domain_labels = torch.cat(
#             (torch.zeros(len(y1)), torch.ones(len(y2))), 0
#         ).cuda()

#     def __len__(self):
#         return len(self.domain_labels)

#     def __getitem__(self, idx):
#         return (self.data[idx], self.task_labels[idx]), self.domain_labels[idx]


class RectangleTransformation(torch.nn.Module):
    """
    Args:

    """

    def __init__(self, d, size=1, rec=None, total_size=16):
        super().__init__()
        assert size >= 1 and size <= 15
        # (x1, x2, y1, y2)
        if rec is not None:
            self.rec = rec
            
        if size == "medium":
            if d == 1:
                self.rec = (8, 18, 4, 22)
            elif d == 2:
                self.rec = (4, 22, 10, 20)
        elif size == "small":
            if d == 1:
                self.rec = (12, 18, 6, 18)
            elif d == 2:
                self.rec = (6, 18, 12, 18)
        elif size == "big":
            if d == 1:
                self.rec = (0, 18, 0, 32)
            elif d == 2:
                self.rec = (0, 32, 8, 22)
        else:
            half_size_odd = size // 2
            half_size_even = (size+1)//2
            
            if d==1:
                self.rec = (total_size-size, total_size+size, total_size-half_size_odd, total_size+half_size_even)
            elif d==2:
                self.rec = (total_size-half_size_odd, total_size+half_size_even, total_size-size, total_size+size)
            #print(self.rec)
                
                
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor:
        """
        assert len(img.shape) == 3
        base_out = torch.rand(img.shape[1:3])[None,:,:]
        base_out = np.repeat(base_out, 3, axis=0)
        out = base_out
        #print(base_out.shape)
        out[:, self.rec[0] : self.rec[1], self.rec[2] : self.rec[3]] = img[
            :, self.rec[0] : self.rec[1], self.rec[2] : self.rec[3]
        ]
        return out

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class JDDAtaset(Dataset):
    def __init__(self, domain, known):
        X1, y = domain

        n_classes = len(np.unique(y))
        y_onehot = torch.FloatTensor(len(y), n_classes)

        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, torch.LongTensor(y), 1)

        self.true_labels = y_onehot.cuda()
        self.data = torch.Tensor(X1).cuda()

        self.seen_labels = torch.ones_like(self.true_labels)
        self.seen_labels = self.seen_labels * float("inf")
        idx = np.random.choice(
            list(range(len(self.true_labels))), size=known, replace=False
        )

        self.seen_labels[idx] = self.true_labels[idx]

    def __len__(self):
        return len(self.true_labels)

    def __getitem__(self, idx):
        return self.data[idx], self.seen_labels[idx], self.true_labels[idx]


class AmazonCaltechDataset(JDDAtaset):
    def __init__(self, domain_name, known):
        domain = loadmat("datasets/officecaltech/decaf6/" + domain_name + "_decaf.mat")
        domain["feas"] = normalize(domain["feas"])
        domain["labels"] -= 1
        super().__init__((domain["feas"], domain["labels"]), known=known)


class JDDAtasetWrapper(Dataset):
    def __init__(self, dataset, known, test=False, epsilons=None, **kwargs):
        self.epsilons = epsilons
        if isinstance(dataset, str):
            if dataset == "mnist":
                self.n_classes = 10
                dataset = datasets.MNIST(
                    "datasets/",
                    train=True,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.Resize(32),
                            transforms.Grayscale(3),
                            transforms.ToTensor(),
                            transforms.Lambda(
                                lambda x: (x + 0.1)
                                * torch.Tensor(
                                    [
                                        [
                                            [
                                                np.random.uniform(),
                                                np.random.uniform(),
                                                np.random.uniform(),
                                            ]
                                        ]
                                    ]
                                ).reshape((-1, 1, 1))
                            ),
                            transforms.ToPILImage(),
                            transforms.ColorJitter(0.2, 0.2, 0.2, 0.3),
                            transforms.RandomAffine(degrees=0,translate=(0.07, 0.07)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
            elif dataset == "mnist_test":
                self.n_classes = 10
                dataset = datasets.MNIST(
                    "datasets/",
                    train=False,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.Resize(32),
                            transforms.Grayscale(3),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
            elif dataset == "mnist_rec":
                self.n_classes = 10
                dataset = datasets.MNIST(
                    "datasets/",
                    train=True if not test else False,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.Resize(32),
                            transforms.Grayscale(3),
                            # transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                            transforms.ToTensor(),
                            RectangleTransformation(**kwargs),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
            elif dataset == "svhn":
                self.n_classes = 10
                dataset = datasets.SVHN(
                    "datasets/",
                    split="train",
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.ColorJitter(0.2, 0.1, 0.3, 0.5),
                            #transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                            transforms.RandomAffine(degrees=0,translate=(0.07, 0.07)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]
                    ),
                )
            elif dataset == "svhn_test":
                self.n_classes = 10
                dataset = datasets.SVHN(
                    "datasets/",
                    split="test",
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]
                    ),
                )
            elif dataset == "mnistm":
                self.n_classes = 10
                dataset = MNISTM(
                    "datasets/",
                    train=True,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.Resize(32),
                            transforms.ColorJitter(0.2, 0.1, 0.3, 0.5),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
            elif dataset == "mnistm_test":
                self.n_classes = 10
                dataset = MNISTM(
                    "datasets/",
                    train=False,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.Resize(32),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
            elif dataset == "usps":
                self.n_classes = 10
                dataset = datasets.USPS(
                    "datasets/",
                    train=True,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.Resize(32),
                            transforms.Grayscale(3),
                            transforms.ToTensor(),
                            transforms.Lambda(
                                lambda x: (x + 0.1)
                                * torch.Tensor(
                                    [
                                        [
                                            [
                                                np.random.uniform(),
                                                np.random.uniform(),
                                                np.random.uniform(),
                                            ]
                                        ]
                                    ]
                                ).reshape((-1, 1, 1))
                            ),
                            transforms.ToPILImage(),
                            transforms.ColorJitter(0.2, 0.2, 0.2, 0.3),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
            elif dataset == "usps_test":
                self.n_classes = 10
                dataset = datasets.USPS(
                    "datasets/",
                    train=False,
                    download=True,
                    transform=transforms.Compose(
                        [
                            transforms.Resize(32),
                            transforms.Grayscale(3),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                )
            elif dataset in ["photo", "art", "cartoon", "sketch"]:
                if "photo" in dataset:
                    mean = [0.43963096, 0.4831598,  0.5085434 ]
                    std = [0.24552175, 0.23584251, 0.24376714]
                elif "sketch" in dataset:
                    mean = [0.9566002, 0.9566002, 0.9566002]
                    std = [0.18300444, 0.18300444, 0.18300444]
                elif "cartoon" in dataset:
                    mean = [0.73600024, 0.7829197,  0.8078039 ]
                    std = [0.29773897, 0.25352597, 0.23099977]
                elif "art" in dataset:
                    mean = [0.45786348, 0.5085214,  0.555027  ]
                    std = [0.21990362, 0.21782503, 0.228203  ]
                self.n_classes = 7
                dataset = PACS(
                    "datasets/",
                    domain=dataset,
                    split="train",
                    transform=transforms.Compose(
                        [
                            RandAugment(n=2, m=5),  # from NFNET
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                            transforms.ToTensor(),
                            # transforms.RandomCrop(224),
                            transforms.Normalize(
                                mean=mean, std=std
                            ),
                        ]
                    ),
                )
            elif dataset in [
                x + "_test" for x in ["photo", "art", "cartoon", "sketch"]
            ]:
                if "photo" in dataset:
                    mean = [0.43963096, 0.4831598,  0.5085434 ]
                    std = [0.24552175, 0.23584251, 0.24376714]
                elif "sketch" in dataset:
                    mean = [0.9566002, 0.9566002, 0.9566002]
                    std = [0.18300444, 0.18300444, 0.18300444]
                elif "cartoon" in dataset:
                    mean = [0.73600024, 0.7829197,  0.8078039 ]
                    std = [0.29773897, 0.25352597, 0.23099977]
                elif "art" in dataset:
                    mean = [0.45786348, 0.5085214,  0.555027  ]
                    std = [0.21990362, 0.21782503, 0.228203  ]
                self.n_classes = 7
                dataset = PACS(
                    "datasets/",
                    domain=dataset[0:-5],
                    split="test",
                    transform=transforms.Compose(
                        [
                            # transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=mean, std=std
                            ),
                        ]
                    ),
                )
            else:
                raise ValueError(f"Wrong dataset name, got {dataset}")
        else:
            print("N_classes undefined currently if dataset is torch dataset")

        self.dataset = dataset
        self._len = len(dataset)
        self._cuda = torch.cuda.is_available()
        if known >= 1:
            idx = np.random.choice(list(range(self._len)), size=known, replace=False)
            self.seen_labels_idx = idx
        else:
            self.seen_labels_idx = None
        if self.epsilons is None:
            self.epsilons = torch.zeros(self._len)

    def __len__(self):
        return self._len

    def get_sup_dataset(self):
        return Subset(self, self.seen_labels_idx)
        
    def __getitem__(self, idx):
        X, y = self.dataset.__getitem__(idx)
        X = torch.FloatTensor(X)
        y = torch.LongTensor([y])


        eps = self.epsilons[idx]
        true = torch.nn.functional.one_hot(y, self.n_classes)[0].float()

        if self.seen_labels_idx is not None:
            is_not_seen = np.logical_not(np.isin(idx, self.seen_labels_idx))
            seen = (
                true.clone()
                if idx in self.seen_labels_idx
                else torch.ones_like(true) * float("inf")
            ).float()
        else:
            seen = true
        if self._cuda:
            #return X.cuda(), seen.cuda(), true.cuda(), eps.cuda()
            return X.cuda(), seen.cuda(), eps.cuda()
        else:
            #return X, seen, true, eps
            return X, seen, eps
