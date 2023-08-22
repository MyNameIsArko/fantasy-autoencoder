"""
Train Soft-Intro VAE for image datasets
Author: Tal Daniel
"""

# standard
import os
import random
import time

import numpy as np
# imports
# torch and friends
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import FantasySet
from ema import EMA
from model import SoftIntroVAE
from utils import *


def train(z_dim=128, lr_e=2e-4, lr_d=2e-4, batch_size=128, num_workers=4,
                         start_epoch=0, start_iter=0, exit_on_negative_diff=False,
                         num_epochs=250, save_interval=50, recon_loss_type="mse",
                         beta_kl=1.0, beta_rec=1.0, beta_neg=1.0, test_iter=100, seed=-1, pretrained=None,
                         device=torch.device("cpu"), gamma_r=1e-8):
    """
    :param z_dim: latent dimensions
    :param lr_e: learning rate for encoder
    :param lr_d: learning rate for decoder
    :param batch_size: batch size
    :param num_workers: num workers for the loading the data
    :param start_epoch: epoch to start from
    :param exit_on_negative_diff: stop run if mean kl diff between fake and real is negative after 50 epochs
    :param num_epochs: total number of epochs to run
    :param save_interval: epochs between checkpoint saving
    :param recon_loss_type: type of reconstruction loss ('mse', 'l1', 'bce')
    :param beta_kl: beta coefficient for the kl divergence
    :param beta_rec: beta coefficient for the reconstruction loss
    :param beta_neg: beta coefficient for the kl divergence in the expELBO function
    :param test_iter: iterations between sample image saving
    :param seed: seed
    :param pretrained: path to pretrained model, to continue training
    :param device: device to run calculation on - torch.device('cuda:x') or torch.device('cpu')
    :param num_row: number of images in a row gor the sample image saving
    :param gamma_r: coefficient for the reconstruction loss for fake data in the decoder
    :return:
    """
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    # Get Dataset
    channels = [64, 128, 256, 512, 512, 512]
    image_size = 256
    ch = 3
    train_set = FantasySet()

    # Build Model
    model = SoftIntroVAE(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size).to(device)
    if pretrained is not None:
        load_model(model, pretrained, device)

    # Build Optimizer
    optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
    optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)

    # Build Learning Rate Scheduler
    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(250, 500), gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(250, 500), gamma=0.1)

    # Build Exponential Moving Average
    ema = EMA(model.decoder, 0.999)
    ema.register()

    # Normalize
    scale = 1 / (ch * image_size ** 2)

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True)

    writer = SummaryWriter(f'runs/fantasy_soft_intro_betas_{beta_kl}_{beta_neg}_{beta_rec}')

    cur_iter = start_iter

    for epoch in range(start_epoch, num_epochs):
        # save models
        if (epoch % save_interval == 0 and epoch > 0) or epoch == num_epochs - 1:
            prefix = f'fantasy_soft_intro_betas_{beta_kl}_{beta_neg}_{beta_rec}_'
            save_checkpoint(model, epoch, cur_iter, prefix)

        model.train()

        diff_kls = []
        batch_kls_real = []
        batch_kls_fake = []
        batch_kls_rec = []
        batch_rec_errs = []
        batch_exp_elbo_f = []
        batch_exp_elbo_r = []

        pbar = tqdm(iterable=train_data_loader)
        total_iter = len(pbar)

        for batch in pbar:
            c = get_coef(cur_iter, epoch_iter=total_iter, epoch=num_epochs, mode='tanh')
            writer.add_scalar('c', c, cur_iter)

            # c = 1.0
            b_size = batch.size(0)
            noise_batch = torch.randn(size=(b_size, z_dim)).to(device)

            real_batch = batch.to(device)

            # =========== Update E ================
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = False

            fake = model.sample(noise_batch)

            real_mu, real_logvar = model.encode(real_batch)
            z = reparameterize(real_mu, real_logvar)
            rec = model.decoder(z)

            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")

            lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")

            rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
            fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())

            kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
            kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")

            loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction='none')
            while len(loss_rec_rec_e.shape) > 1:
                loss_rec_rec_e = loss_rec_rec_e.sum(-1)
            loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction='none')
            while len(loss_rec_fake_e.shape) > 1:
                loss_rec_fake_e = loss_rec_fake_e.sum(-1)

            rec_mu = nn.Tanh()(rec_mu)
            rec_logvar = nn.Tanh()(rec_logvar)
            fake_mu = nn.Tanh()(fake_mu)
            fake_logvar = nn.Tanh()(fake_logvar)

            dis = gaussian_distance(rec_mu, rec_logvar, fake_mu, fake_logvar, reduce='none')

            expelbo_rec = (-2 * scale * (beta_rec * loss_rec_rec_e + beta_neg * (c * kl_rec + (1-c) * dis))).exp().mean()
            expelbo_fake = (-2 * scale * (beta_rec * loss_rec_fake_e + beta_neg * (c * kl_fake + (1-c) * dis))).exp().mean()

            lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
            lossE_real = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl)

            lossE = lossE_real + lossE_fake
            optimizer_e.zero_grad()
            lossE.backward()
            optimizer_e.step()

            # ========= Update D ==================
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = True

            fake = model.sample(noise_batch)
            rec = model.decoder(z.detach())
            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")

            rec_mu, rec_logvar = model.encode(rec)
            z_rec = reparameterize(rec_mu, rec_logvar)

            fake_mu, fake_logvar = model.encode(fake)
            z_fake = reparameterize(fake_mu, fake_logvar)

            rec_rec = model.decode(z_rec.detach())
            rec_fake = model.decode(z_fake.detach())

            loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type,
                                                    reduction="mean")
            loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type,
                                                     reduction="mean")

            rec_mu = nn.Tanh()(rec_mu)
            rec_logvar = nn.Tanh()(rec_logvar)
            fake_mu = nn.Tanh()(fake_mu)
            fake_logvar = nn.Tanh()(fake_logvar)

            lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
            lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")
            dis = gaussian_distance(rec_mu, rec_logvar, fake_mu, fake_logvar, reduce='mean')

            lossD = scale * (loss_rec * beta_rec + (
                    c*(lossD_rec_kl + lossD_fake_kl) + (1-c)*dis) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (
                                     loss_rec_rec + loss_fake_rec))

            optimizer_d.zero_grad()
            lossD.backward()
            optimizer_d.step()

            ema.update()

            if torch.isnan(lossD) or torch.isnan(lossE):
                raise SystemError

            dif_kl = (-lossE_real_kl + lossD_fake_kl).cpu().item()
            pbar.set_description_str('epoch #{}'.format(epoch))
            pbar.set_postfix(r_loss=loss_rec.cpu().item(), kl=lossE_real_kl.cpu().item(),
                             diff_kl=dif_kl, expelbo_f=expelbo_fake.cpu().item())

            diff_kls.append(dif_kl)
            batch_kls_real.append(lossE_real_kl.cpu().item())
            batch_kls_fake.append(lossD_fake_kl.cpu().item())
            batch_kls_rec.append(lossD_rec_kl.cpu().item())
            batch_rec_errs.append(loss_rec.cpu().item())
            batch_exp_elbo_f.append(expelbo_fake.cpu().item())
            batch_exp_elbo_r.append(expelbo_rec.cpu().item())

            if cur_iter % test_iter == 0:
                _, _, _, rec_det = model(real_batch, deterministic=True)
                ema.apply_shadow()
                fake = model.sample(noise_batch)
                img_grid = vutils.make_grid(torch.cat([real_batch, rec_det, fake], dim=0).cpu(), nrow=real_batch.size(0))
                writer.add_image('figures', img_grid, cur_iter)
                ema.restore()

            cur_iter += 1
        e_scheduler.step()
        d_scheduler.step()
        pbar.close()
        if exit_on_negative_diff and epoch > 50 and np.mean(diff_kls) < -1.0:
            print(
                f'the kl difference [{np.mean(diff_kls):.3f}] between fake and real is negative (no sampling improvement)')
            print("try to lower beta_neg hyperparameter")
            print("exiting...")
            raise SystemError("Negative KL Difference")

        # Epoch summary
        writer.add_scalar('kl', np.mean(batch_kls_real), epoch)
        writer.add_scalar('kl_fake', np.mean(batch_kls_fake), epoch)
        writer.add_scalar('kl_rec', np.mean(batch_kls_rec), epoch)
        writer.add_scalar('rec', np.mean(batch_rec_errs), epoch)
        writer.add_scalar('exp_elbo_f', np.mean(batch_exp_elbo_f), epoch)
        writer.add_scalar('exp_elbo_r', np.mean(batch_exp_elbo_r), epoch)
        writer.add_scalar('diff_kl', np.mean(diff_kls), epoch)


if __name__ == '__main__':
    """
    Recommended hyper-parameters:
    - CIFAR10: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
    - SVHN: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
    - MNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
    - FashionMNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
    - Monsters: beta_kl: 0.2, beta_rec: 0.2, beta_neg: 256, z_dim: 128, batch_size: 16
    - CelebA-HQ: beta_kl: 1.0, beta_rec: 0.5, beta_neg: 1024, z_dim: 256, batch_size: 8
    """
    beta_kl = 1.0
    beta_rec = 0.5
    beta_neg = 8.0
    pretrained = 'saves/fantasy_soft_intro_betas_1.0_4.0_0.5_model_epoch_1550_iter_21700.pth'
    # pretrained = None
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print("betas: ", beta_kl, beta_neg, beta_rec)
    try:
        train(z_dim=256, batch_size=6, num_workers=12, num_epochs=2000,
                             beta_kl=beta_kl, beta_neg=beta_neg, beta_rec=beta_rec, exit_on_negative_diff=False,
                             device=device, save_interval=50, start_epoch=1550, start_iter=21700, lr_e=2e-4, lr_d=2e-4,
                             pretrained=pretrained, test_iter=100)
    except SystemError:
        print("Error, probably loss is NaN, try again...")
