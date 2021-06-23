import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

#def kl(p, q):
#    loss = torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8))) / p.shape[0]
#    return loss

def conditional_entropy(soft):
    loss = torch.sum(-soft*torch.log(soft + 1e-8)) / soft.shape[0]
    return loss

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.linalg.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

class VMTLoss(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, x1, f_x1, x2=None, f_x2=None):
        if x2 is None and f_x2 is None:
            x2 = x1
            f_x2 = f_x1
        shuffle_idx = torch.randperm(x2.size()[0])
        x2 = x2[shuffle_idx]
        f_x2 = f_x2[shuffle_idx]
        bs = len(x1)
        shape_narrowed = list(x1.shape)
        shape_narrowed = [shape_narrowed[0]] + [
            1 for i in range(len(shape_narrowed) - 1)
        ]
        alphas = torch.rand(bs, device=x1.device).unsqueeze(-1)
        x_mixed = x1 * alphas.view(*shape_narrowed) + x2 * (
            1 - alphas.view(*shape_narrowed)
        )
        y_mixed = F.log_softmax(f_x1 * alphas + f_x2 * (1 - alphas), dim=-1)
        f_mixed = self.net.forward_log_classifier(x_mixed)
        return self.loss(y_mixed, f_mixed)


class EntMinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_x):
        soft_f_x = F.softmax(f_x, dim=-1)
        log_soft_f_x = F.log_softmax(f_x, dim=-1)
        ent = -torch.sum(soft_f_x*log_soft_f_x)/f_x.shape[0]
        #ent = conditional_entropy(soft_f_x)
        #ent = f_x * torch.log(f_x + 1e-8)
        #ent = ent.sum(dim=1)
        #ent = -1 * ent.mean()

        return ent

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

class VATLoss(nn.Module):
    def __init__(self, net, xi=1e-6, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.net = net
        self.loss = nn.KLDivLoss(reduction="batchmean")#, log_target=True)

    def forward(self, x, eps):
        with _disable_tracking_bn_stats(self.net):
            with torch.no_grad():
                pred = self.net.forward_classifier(x)
            # prepare random unit tensor
            d = torch.randn(x.shape).to(x.device)
            d = _l2_normalize(d)
            
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                logp_hat = self.net.forward_log_classifier(x + self.xi * d)
                #p_hat = self.net.forward_classifier(x + self.xi * d)
                # logp_hat = torch.log(pred_hat)  # , dim=1)
                adv_distance = self.loss(logp_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                self.net.zero_grad()

            #print(x.shape, d.shape, eps.shape)
            if len(x.shape)==4:
                eps = eps[:, None, None, None]
            elif len(x.shape)==2:
                eps = eps[:, None]
            # calc LDS
            r_adv = d * eps
            logp_hat = self.net.forward_log_classifier(x + r_adv)
            #p_hat = self.net.forward_classifier(x + r_adv)
            # logp_hat = torch.log(pred_hat)  # , dim=1)
            lds = self.loss(logp_hat, pred)
        return lds
