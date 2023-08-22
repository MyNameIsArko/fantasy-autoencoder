import math

import torch
import os
import torch.nn.functional as F


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)


def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "./saves/" + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))


def reduce_with_choice(mu1, mu2, var1, var2, choice=None):
    term1 = torch.pow(mu1-mu2, 2)
    term2 = torch.div(term1, var1+var2)
    term3 = torch.mul(term2, -0.5)
    term4 = torch.exp(term3)
    term5 = torch.sqrt(var1+var2)
    res = torch.div(term4, term5)
    return torch.mean(res) if choice == 'mean' else torch.sum(res)


def gaussian_distance(mu_a, logvar_a, mu_b, logvar_b, reduce='mean'):
    var_a = torch.exp(logvar_a)
    var_b = torch.exp(logvar_b)

    mu_a1 = mu_a.view(mu_a.size(0), 1, -1)
    mu_a2 = mu_a.view(1, mu_a.size(0), -1)
    var_a1 = var_a.view(var_a.size(0), 1, -1)
    var_a2 = var_a.view(1, var_a.size(0), -1)

    mu_b1 = mu_b.view(mu_b.size(0), 1, -1)
    mu_b2 = mu_b.view(1, mu_b.size(0), -1)
    var_b1 = var_b.view(var_b.size(0), 1, -1)
    var_b2 = var_b.view(1, var_b.size(0), -1)

    if reduce == 'mean':
        vaa = reduce_with_choice(mu_a1, mu_a2, var_a1, var_a2, choice='mean')
        vab = reduce_with_choice(mu_a1, mu_b2, var_a1, var_b2, choice='mean')
        vbb = reduce_with_choice(mu_b1, mu_b2, var_b1, var_b2, choice='mean')

    else:
        vaa = reduce_with_choice(mu_a1, mu_a2, var_a1, var_a2, choice='sum')
        vab = reduce_with_choice(mu_a1, mu_b2, var_a1, var_b2, choice='sum')
        vbb = reduce_with_choice(mu_b1, mu_b2, var_b1, var_b2, choice='sum')

    loss = vaa + vbb - torch.mul(vab, 2.0)

    return loss


def get_coef(iter_, epoch_iter, epoch, mode='linear'):
    total = epoch_iter * epoch
    scaled_iter_ = iter_ * 5 / total
    if mode == 'tanh':
        scaled_iter_ = math.tanh(scaled_iter_)
    return scaled_iter_ if scaled_iter_ <= 1 else 1
