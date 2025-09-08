import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.block(x)


class VQVAE(nn.Module):
    def __init__(self, k=128, beta=0.25):
        super().__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, 2, 1),  # (16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),  # (8, 8)
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),  # (16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 4, 2, 1),  # (32, 32)
        )

        self.codebook = nn.Embedding(k, 256)
        # initalize uniform randomly between [-1/k, 1/k]
        self.codebook.weight.data = torch.rand(k, 256) * (2/k) - (1/k)

    def encode(self, x, idx=False):
        latents = self.encoder(x)
        return self._quantize_latents(latents)[idx]

    def _quantize_latents(self, latents):  # (B, 256, 8, 8)
        e = self.codebook.weight  # (128, 256)
        k, emb_dim = e.shape

        dot = F.conv2d(latents, e.view(k, emb_dim, 1, 1))  # (B, 128, 8, 8)
        latents2 = (latents**2).sum(dim=1, keepdims=True)  # (B, 1, 8, 8)
        e2 = (e**2).sum(dim=1).view(k, 1, 1)  # (128, 1, 1)

        # ||z - e||^2 = z^2 + e^2 - 2*z*e
        dist = latents2 + e2 - 2 * dot  # (B, 128, 8, 8)
        quantized_latents_idx = dist.min(dim=1).indices  # (B, 8, 8)
        quantized_latents = self.codebook(quantized_latents_idx).permute(0, 3, 1, 2)
        return quantized_latents, quantized_latents_idx

    def loss(self, x):
        latents = self.encoder(x)  # -> (N, 256, 8, 8)
        quantized_latents = self._quantize_latents(latents)[0]

        # straight-though estimator in pytorch
        quantized_latents_st = (quantized_latents - latents).detach() + latents

        reconstruction_loss = F.mse_loss(x, self.decoder(quantized_latents_st))
        vq_loss = F.mse_loss(latents.detach(), quantized_latents)
        commitment_loss = self.beta * F.mse_loss(latents, quantized_latents.detach())

        return reconstruction_loss + vq_loss + commitment_loss

    @torch.no_grad()
    def generate(self, N, prior_model, with_noise=False):
        latent_idx = prior_model.generate(N).reshape(-1, 8, 8)
        latents = self.codebook(latent_idx).permute(0, 3, 1, 2)
        out = self.decoder(latents)
        if with_noise:
            out += torch.randn(N, 3, 32, 32, device='cuda')
        return out

    @torch.no_grad()
    def reconstruct(self, x):
        quantized_latents = self.encode(x)
        out = self.decoder(quantized_latents)  # N x 3 x 32 x 32
        return torch.stack((x, out), dim=1).reshape(-1, 3, 32, 32)  # (N, 2, 3, 32, 32) -> (N*2, 3, 32, 32) (interleave x and out)


if __name__ == '__main__':
    model = VQVAE()
    x = torch.rand(1, 256, 8, 8)
    ql1 = model._quantize_latents(x)[1]
