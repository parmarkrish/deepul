import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae import VAE

class HVAE(VAE):
    def __init__(self):
        super().__init__()
        self.latent_dim = 12 * 2 * 2
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + 12, 32, 3, padding=1),  # (32, 32, 32)
            nn.LayerNorm([32, 32, 32]),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (64, 16, 16)
            nn.LayerNorm([64, 16, 16]),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # (64, 8, 8)
            nn.LayerNorm([64, 8, 8]),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # (64, 4, 4)
            nn.LayerNorm([64, 4, 4]),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # (64, 2, 2)
            nn.LayerNorm([64, 2, 2]),
            nn.ReLU(),

            nn.Conv2d(64, 12*2, 3, padding=1),  # (12*2, 2, 2)
        )
        self.p_z2_given_z1 = nn.Sequential(
            nn.Flatten(start_dim=1),

            nn.Linear(12*2*2, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 12*2*2),
            nn.Unflatten(dim=1, unflattened_size=(12, 2, 2))
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),  # (64, 2, 2)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # (64, 4, 4)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # (64, 8, 8)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # (64, 16, 16)
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # (32, 32, 32)
            nn.ReLU(),

            nn.Conv2d(32, 3, 3, padding=1)  # (3, 32, 32)
        )
    
    def encode_to_z1(self, x):
        x_ = torch.cat([x, torch.zeros(x.shape[0], 12, 32, 32)], dim=1)
        z1_params = self.encoder(x_)  # (B, 12*2, 2, 2)
        z1_means, z1_log_stds = torch.chunk(z1_params, 2, dim=1)  # (B, 12, 2, 2), (B, 12, 2, 2)
        return z1_means, z1_log_stds
    
    def encode_to_z2(self, x, z1):
        z1_upsample = F.interpolate(z1, size=(32, 32))  # (B, 12, 32, 32)
        x_z1 = torch.cat([x, z1_upsample], dim=1)
        z2_params = self.encoder(x_z1)
        z2_residual_means, z2_residual_log_stds = torch.chunk(z2_params, 2, dim=1)  # (B, 12, 2, 2), (B, 12, 2, 2)
        return z2_residual_means, z2_residual_log_stds

    
    def encode(self, x, with_noise=True):
        z1_means, z1_log_stds = self.encode_to_z1(x)
        z1 = z1_means
        if with_noise:  # pytorch doesn't allow inplace updates on views
            z1 = z1 + torch.randn_like(z1) * z1_log_stds.exp()

        z2_residual_means, z2_residual_log_stds = self.encode_to_z2(x, z1)
        z2_means = self.p_z2_given_z1(z1)
        z2 = z2_means + z2_residual_means

        if with_noise:
            z2 = z2 + torch.randn_like(z2) * z2_residual_log_stds.exp()

        return z2, z1, z1_means, z1_log_stds, z2_residual_means, z2_residual_log_stds

    def loss(self, x):  # (B, 3, 32, 32)
        z2, z1, z1_means, z1_log_stds, z2_residual_means, z2_residual_log_stds = self.encode(x, with_noise=True)
        x_recon = self.decoder(z2)

        recon_loss = torch.sum((x - x_recon)**2, dim=(1, 2, 3)) # (B, )

        # kl_z1 = 0.5 * (-z1_log_stds - 1 + z1_log_stds.exp() + z1_means**2) 
        kl_z1 = -z1_log_stds + 0.5 * (torch.exp(2*z1_log_stds) + z1_means**2 - 1)
        # print('different kl')
        kl_z1 = kl_z1.sum(dim=(1, 2, 3))  # (B, )

        # kl_z2 = 0.5 * (-z2_residual_log_stds - 1 + z2_residual_log_stds.exp() + z2_residual_means**2) 
        kl_z2 = -z2_residual_log_stds - 0.5 + 0.5 * (torch.exp(2*z2_residual_log_stds) + z2_residual_means**2)
        kl_z2 = kl_z2.sum(dim=(1, 2, 3))  # (B, )

        kl_loss = kl_z1 + kl_z2
        neg_elbo = recon_loss + kl_loss

        return neg_elbo.mean(), recon_loss.mean(), kl_loss.mean()
    
    @torch.no_grad()
    def generate(self, n_samples, with_noise=False):
        z1 = torch.randn(n_samples, 12, 2, 2)
        z2_means = self.p_z2_given_z1(z1)
        z2 = z2_means + torch.randn_like(z2_means)
        x = self.decoder(z2)
        if with_noise:
            x += torch.randn(n_samples, 3, 32, 32)
        return x
    
    @torch.no_grad()
    def reconstruct(self, x):
        z2 = self.encode(x, with_noise=True)[0]
        x_recon = self.decoder(z2)  # (B, 3, 32, 32)
        return torch.stack((x, x_recon), dim=1).view(-1, 3, 32, 32)  # (N, 2, 3, 32, 32) -> (N*2, 3, 32, 32) (interleave x and p_means)
    
    @torch.no_grad()
    def create_interpolations(self, test_data, n_samples=10, n_interp_states=10):
        x = torch.as_tensor(test_data[:n_samples+1])
        latents = self.encode(x, with_noise=True)[0]  # (B, 12, 2, 2)
        alphas = torch.linspace(0, 1, n_interp_states)
        return self.decoder(
            torch.stack([
                start_latent*(1-alpha) + end_latent*alpha
                for start_latent, end_latent in zip(latents, latents[1:])
                for alpha in alphas
            ], dim=0)
        )


if __name__ == '__main__':
    torch.manual_seed(1)
    model = HVAE()
    # encoder test
    # x = torch.rand(1, 3, 32, 32)
    # z2, z1, z1_means, z1_log_stds, z2_residual_means, z2_residual_log_stds = model.encode(x)
    # print(f'{z1[0][0][0][0]=}')
    # print(f'{z1_means[0][0][0][0]=}')
    # decoder test
    z = torch.rand(1, 12, 2, 2)
    y = model.p_z2_given_z1(z)
    print(f'{y.shape=}')


    # test loss
    # x = torch.rand(2, 3, 32, 32)
    # neg_elbo, recon_loss, kl_loss = model.loss(x)
    # print(f'{neg_elbo=}')
    # print(f'{recon_loss=}')
    # print(f'{kl_loss=}')

    # test interpolate
    # test_data = torch.randn(12, 3, 32, 32)
    # interps = model.create_interpolations(test_data)
    # print(f'{interps.shape=}')