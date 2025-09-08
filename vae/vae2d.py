import torch
import torch.nn as nn

import copy
import math

class VAE2D(nn.Module):
    def __init__(self, hidden_size=100):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 4))
        self.decoder = copy.deepcopy(self.encoder)

    # alternatively, could use logprob method of torch.distributions multivariate normal
    @staticmethod
    def multivariate_normal_log_prob(x, means, covariance):
        part1 = (x[:, 0] - means[:, 0])**2 * covariance[:, 0]**-1
        part2 = (x[:, 1] - means[:, 1])**2 * covariance[:, 1]**-1
        log_det_cov = torch.log(covariance[:, 0]) + torch.log(covariance[:, 1])
        return -(math.log(2 * math.pi) + 0.5 * (log_det_cov + part1 + part2))

    @staticmethod
    def compute_kl_term(q_means, q_covariances):
        return 0.5 * (-q_covariances.log().sum(1) - 2 + q_covariances.sum(1) + (q_means**2).sum(1))

    def loss(self, x):  # -ELBO, (N, 2)
        # z ~ q(z | x)
        q_params = self.encoder(x)  # (N, 4)
        q_means, q_covariances = q_params[:, :2], q_params[:, 2:].exp()  # (N, 2)
        z = torch.randn_like(x) * q_covariances.sqrt() + q_means
        # log_prob(p(x | z))
        p_params = self.decoder(z)
        p_means, p_covariance = p_params[:, :2], p_params[:, 2:].exp()  # (N, 2)

        p_log_probs = self.multivariate_normal_log_prob(x, p_means, p_covariance)
        kl_term = self.compute_kl_term(q_means, q_covariances)
        neg_elbo = - (p_log_probs - kl_term)
        return neg_elbo.mean(), -p_log_probs.mean(), kl_term.mean()

    @torch.no_grad()
    def generate(self, n_samples, with_noise=True):
        x = torch.randn(n_samples, 2)
        p_params = self.decoder(x)
        p_means, p_covariance = p_params[:, :2], p_params[:, 2:].exp()  # (N, 2)
        if with_noise:
            p_means += torch.randn(n_samples, 2) * p_covariance.sqrt()
        return p_means

    def test_loss(model, test_loader):
        neg_elbo, recon_loss, kl_term = 0, 0, 0
        for test_labels in test_loader:
            neg_elbo_, recon_loss_, kl_term_ = model.loss(test_labels)
            neg_elbo += neg_elbo_.item()
            recon_loss += recon_loss_.item()
            kl_term += kl_term_.item()
        return neg_elbo/len(test_loader), recon_loss/len(test_loader), kl_term/len(test_loader)

    def train_one_epoch(model, optimizer, loader, history):
        for labels in loader:
            neg_elbo, recon_loss, kl_term = model.loss(labels)
            neg_elbo.backward()
            optimizer.step()
            optimizer.zero_grad()
            history.append([neg_elbo.item(), recon_loss.item(), kl_term.item()])
