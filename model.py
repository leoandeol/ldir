import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

from torch.autograd import Function
from torchvision.models import resnet18, resnet50, vgg11

import timm


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def max_singular_value(weight, u, Ip):
    assert Ip >= 1

    _u = u
    for _ in range(Ip):
        _v = F.normalize(torch.mm(_u, weight), p=2, dim=1).detach()
        _u = F.normalize(torch.mm(_v, weight.transpose(0, 1)), p=2, dim=1).detach()
    sigma = torch.sum(F.linear(_u, weight.transpose(0, 1)) * _v)
    return sigma, _u, _v


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long).cuda()
    return x[tuple(indices)]


class VAT(nn.Module):
    def __init__(self, top_bn=False):
        super(VAT, self).__init__()
        self.top_bn = top_bn
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2, 1),
            nn.Dropout2d(),
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 1, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, 1, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedder = self.main
        self.classifier = nn.Linear(128, 10)
        self.discriminator = nn.Linear(1, 1)
        self.bn = nn.BatchNorm1d(10)

    # def forward(self, input):
    #    return self.main(input).view(input.size()[0], -1)

    def forward(self, input, *args):
        output = self.main(input)
        output = self.classifier(output.view(input.size()[0], -1))
        if self.top_bn:
            output = self.bn(output)
        return output  # None, output, None, None

    # def forward_classifier(self, input):
    #     return F.softmax(self.forward(input)[1], dim=-1)

    # def forward_log_classifier(self, input):
    #     return F.log_softmax(self.forward(input)[1], dim=-1)


class SNLinear(nn.Linear):
    def __init__(
        self, in_features, out_features, bias=True, init_u=None, use_gamma=False
    ):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.Ip = 1
        self.register_buffer(
            "u", init_u if init_u is not None else torch.randn(1, out_features)
        )
        self.gamma = nn.Parameter(torch.zeros(1)) if use_gamma else None

        self.Ip_grad = 8
        self.r = 10
        self.register_buffer(
            "u0", init_u if init_u is not None else torch.randn(1, out_features)
        )
        self.register_buffer(
            "u1", init_u if init_u is not None else torch.randn(1, out_features)
        )

    @property
    def W_bar(self):
        sigma, u, _ = max_singular_value(self.weight, self.u, self.Ip)
        self.u[:] = u
        return self.weight / sigma

    def forward(self, x):
        if self.gamma is not None:
            return torch.exp(self.gamma) * F.linear(x, self.W_bar, self.bias)
        else:
            return F.linear(x, self.W_bar, self.bias)


class Model_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 10),
            nn.Softmax(dim=-1),
        )
        self.discriminator_1 = nn.Sequential(
            nn.Linear(512 + 10, 256),
            nn.ReLU(True),
        )
        self.discriminator_2 = nn.Sequential(
            nn.Linear(256 + 10, 128),
            nn.ReLU(True),
        )
        self.discriminator_3 = nn.Sequential(
            nn.Linear(128 + 10, 1),
        )

    def forward(self, input_data, input_labels, alpha, domain=None):
        # Embeddings
        embeddings = self.embedder(input_data)

        # Classifier
        output_labels = self.classifier(embeddings)

        # Domain Discriminator
        # TODO : have nan or inf for unknown labels and replace them with classifier output
        labels = input_labels.clone()
        idx = torch.isfinite(labels.sum(1))
        labels[~idx] = output_labels[~idx]
        assert (torch.isfinite(labels) == True).all()

        reversed_embeddings = ReverseLayerF.apply(embeddings, alpha)
        disc_1 = self.discriminator_1(torch.cat((reversed_embeddings, labels), 1))
        disc_2 = self.discriminator_2(torch.cat((disc_1, labels), 1))
        disc_output = self.discriminator_3(torch.cat((disc_2, labels), 1))

        return embeddings, output_labels, disc_output, labels

    def forward_discriminator(self, embeddings, labels):
        disc_1 = self.discriminator_1(torch.cat((embeddings, labels), 1))
        disc_2 = self.discriminator_2(torch.cat((disc_1, labels), 1))
        disc_output = self.discriminator_3(torch.cat((disc_2, labels), 1))
        return disc_output


class Model(nn.Module):
    def __init__(
        self,
        mode,  # digits or pacs
        instance_norm=False,
        spectral_norm=False,
        backbone="resnet18",
        disc_type="ours",
        n_features = None
    ):
        super().__init__()
        if n_features is None:
            n_features = 512 if mode == "pacs" else 256
        self.n_features = n_features
        if backbone == "vgg11":
            self.embedder = vgg11(pretrained=True)
            self.embedder.classifier[-1] = nn.Linear(4096, self.n_features)
        elif backbone == "vat":
            self.embedder = VAT()
            self.embedder.classifier = nn.Linear(128, self.n_features)
        elif backbone == "resnet50":
            self.embedder = resnet50(pretrained=True)
            self.embedder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.embedder.fc = nn.Linear(2048, self.n_features)
        elif backbone == "resnet18":
            self.embedder = resnet18(pretrained=True)
            self.embedder.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.embedder.fc = nn.Linear(512, self.n_features)
        elif backbone == "nf_resnet26":
            self.embedder = timm.create_model("nf_resnet26", pretrained=True)
            self.embedder.head.fc = nn.Linear(2048, self.n_features)
        elif backbone == "nf_regnet_b0":
            self.embedder = timm.create_model(
                "nf_regnet_b0", pretrained=True, num_classes=self.n_features
            )
            # self.embedder.head.fc = nn.Linear(960, 512)
        elif backbone == "nf_regnet_b1":
            self.embedder = timm.create_model(
                "nf_regnet_b1", pretrained=True, num_classes=self.n_features
            )
            # self.embedder.head.fc = nn.Linear(960, 512)
        elif backbone == "nfnet":
            self.embedder = timm.create_model("dm_nfnet_f0", pretrained=False)
            # todo: changer conv ?
            self.embedder.head.fc = nn.Linear(3072, self.n_features)
        self.n_classes = 7 if mode == "pacs" else 10
        if mode == "digits":
            self.classifier = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, self.n_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, self.n_classes),
            )
        self.disc_type = disc_type
        if disc_type.lower() == "ours":
            self.discriminator = nn.Sequential(
                SNLinear(n_features + self.n_classes, 256)
                if spectral_norm
                else nn.Linear(n_features + self.n_classes, 256),
                nn.ReLU(True),
                SNLinear(256, 256) if spectral_norm else nn.Linear(256, 256),
                nn.ReLU(True),
                SNLinear(256, self.n_classes)
                if spectral_norm
                else nn.Linear(256, self.n_classes),
            )
        elif disc_type.lower() == "joint":
            self.discriminator = nn.Sequential(
                SNLinear(n_features + self.n_classes, 256)
                if spectral_norm
                else nn.Linear(n_features + self.n_classes, 256),
                nn.ReLU(True),
                SNLinear(256, 256) if spectral_norm else nn.Linear(256, 256),
                nn.ReLU(True),
                SNLinear(256, 1) if spectral_norm else nn.Linear(256, 1),
            )
        elif disc_type.lower() == "marginal":
            self.discriminator = nn.Sequential(
                SNLinear(n_features, 256)
                if spectral_norm
                else nn.Linear(n_features, 256),
                nn.ReLU(True),
                SNLinear(256, 256) if spectral_norm else nn.Linear(256, 256),
                nn.ReLU(True),
                SNLinear(256, 1) if spectral_norm else nn.Linear(256, 1),
            )
        elif disc_type.lower() == "none":
            self.discriminator = nn.Sequential(
                SNLinear(n_features + self.n_classes, 1)
                if spectral_norm
                else nn.Linear(n_features + self.n_classes, 1),
            )
        self.instance_norm = instance_norm
        self.instance_norm_input = nn.InstanceNorm2d(3)
        self.instance_norm_embd = (
            lambda x: x / x.shape[1]
        )  # torch.norm(x, dim=1, keepdim=True)
        self.spectral_norm = spectral_norm

    def forward(self, input_data, input_labels, alpha, domain=None):
        # Embeddings
        if self.instance_norm:
            input_data = self.instance_norm_input(input_data)
        embeddings = self.embedder(input_data)
        embeddings_clf = embeddings
        embeddings_disc = embeddings
        if self.instance_norm:
            embeddings_disc = self.instance_norm_embd(embeddings_disc)

        # Classifier
        # if domain == 1# :
        #     output_labels = self.classifier_1(embeddings)
        # elif domain == 2:
        #     output_labels = self.classifier_2(embeddings)
        # else:
        #     output_labels_1 = self.classifier_1(embeddings)
        #     output_labels_2 = self.classifier_2(embeddings)
        #     output_labels = output_labels_1 * output_labels_2
        #     output_lab
        #    els = output_labels / torch.norm(output_labels, dim=1, keepdim=True)
        output_labels = self.classifier(embeddings_clf)

        # Domain Discriminator
        labels = input_labels.clone()
        idx = torch.isfinite(labels.sum(1))
        labels[~idx] = F.softmax(output_labels[~idx], dim=-1)

        # print(torch.norm(embeddings_disc, dim=-1))
        # assert False
        assert (torch.isfinite(labels) == True).all()

        reversed_embeddings = ReverseLayerF.apply(embeddings_disc, alpha)
        if self.disc_type == "marginal":
            disc_output = self.discriminator(reversed_embeddings)
        else:
            disc_output = self.discriminator(
                torch.cat((reversed_embeddings, labels), 1)
            )
        return embeddings, output_labels, disc_output, labels

    def forward_discriminator(self, embeddings, labels):
        if self.disc_type == "marginal":
            disc_output = self.discriminator(embeddings)
        else:
            disc_output = self.discriminator(torch.cat((embeddings, labels), 1))
        return disc_output

    def forward_classifier(self, input_data):
        # Embeddings
        # if self.instance_norm:
        #    input_data = self.instance_norm_input(input_data)
        embeddings = self.embedder(input_data)
        # embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        # print(torch.norm(embeddings,dim=1))
        # Classifier
        output_labels = self.classifier(embeddings)
        output_labels = F.softmax(output_labels, dim=-1)
        return output_labels

    def forward_log_classifier(self, input_data):
        # Embeddings
        # if self.instance_norm:
        #    input_data = self.instance_norm_input(input_data)
        embeddings = self.embedder(input_data)
        # embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
        # print(torch.norm(embeddings,dim=1))
        # Classifier
        output_labels = self.classifier(embeddings)
        output_labels = F.log_softmax(output_labels, dim=-1)
        return output_labels
