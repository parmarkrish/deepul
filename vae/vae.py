import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 8 x 8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 4 x 4
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(4 * 4 * 256, 2 * latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 128),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def compute_kl_term(self, q_means, q_covariances):
        return 0.5 * (-q_covariances.log().sum(1) - self.latent_dim + q_covariances.sum(1) + (q_means**2).sum(1))

    def loss(self, x):  # N x 3 x 32 x 32
        batch_size = x.shape[0]

        q_params = self.encoder(x)
        q_means, q_covariances = torch.chunk(q_params, 2, dim=1)
        q_covariances = q_covariances.exp()

        z = torch.randn(batch_size, self.latent_dim) * q_covariances.sqrt() + q_means  # N x latent_dim
        p_means = self.decoder(z)  # N x 3 x 32 x 32

        recon_loss = torch.sum((x - p_means)**2, dim=(1, 2, 3))
        # recon_loss = F.mse_loss(x, p_means)
        kl_term = self.compute_kl_term(q_means, q_covariances)
        neg_elbo = recon_loss + kl_term
        return neg_elbo.mean(), recon_loss.mean(), kl_term.mean()

    @torch.no_grad()
    def generate(self, n_samples, with_noise=True):
        z = torch.randn(n_samples, self.latent_dim)
        p_means = self.decoder(z)
        if with_noise:
            p_means += torch.randn(n_samples, 3, 32, 32)
        return p_means

    @torch.no_grad()
    def reconstruct(self, x):
        q_params = self.encoder(x)
        q_means, q_covariances = torch.chunk(q_params, 2, dim=1)
        q_covariances = q_covariances.exp()

        # z = torch.randn(batch_size, self.latent_dim, device='cuda') * q_covariances.sqrt() + q_means  # N x latent_dim
        p_means = self.decoder(q_means)  # N x 3 x 32 x 32

        return torch.stack((x, p_means), dim=1).view(-1, 3, 32, 32)  # (N, 2, 3, 32, 32) -> (N*2, 3, 32, 32) (interleave x and p_means)

    def test_loss(model, test_loader):
        neg_elbo, recon_loss, kl_term = 0, 0, 0
        for img in test_loader:
            neg_elbo_, recon_loss_, kl_term_ = model.loss(img)
            neg_elbo += neg_elbo_.item()
            recon_loss += recon_loss_.item()
            kl_term += kl_term_.item()
        return neg_elbo/len(test_loader), recon_loss/len(test_loader), kl_term/len(test_loader)

    def train_one_epoch(model, optimizer, loader, history):
        for img in loader:
            neg_elbo, recon_loss, kl_term = model.loss(img)
            neg_elbo.backward()
            optimizer.step()
            optimizer.zero_grad()
            history.append([neg_elbo.item(), recon_loss.item(), kl_term.item()])