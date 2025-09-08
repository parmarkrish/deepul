import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.functional as F
import math
import itertools

from gan import ResBlock, ResnetBlockDown, GlobalSumPooling
from vae import VQVAE
from deepul.hw3_utils.lpips import LPIPS
from diffusion import Patchify, unpatchify
from autoregressive.transformer import Block


def patchify(x: torch.tensor, patch_size=8):  # (B, C, H, W) -> (B * (H/P) * (H/P), C, P, P)
    _, C, _, _ = x.shape
    y = F.unfold(x, kernel_size=patch_size, stride=patch_size)  # -> (B, C*P*P, L)
    y = y.permute(0, 2, 1)  # -> (B, L, C*P*P)
    y = y.reshape(-1, C, patch_size, patch_size)
    return y

class VQGAN(VQVAE):
    class Discriminator(nn.Module):
        def __init__(self, n_filters=128, patch_size=8):
            super().__init__()
            self.patch_size = patch_size
            self.net = nn.Sequential(
                ResnetBlockDown(3, n_filters=n_filters),
                ResnetBlockDown(n_filters, n_filters=n_filters),
                ResBlock(n_filters, n_filters=n_filters),
                ResBlock(n_filters, n_filters=n_filters),
                nn.ReLU()
            )
            self.linear = nn.Linear(n_filters, 1)

        def forward(self, x):  # (B, 3, 32, 32)
            x = patchify(x, patch_size=self.patch_size)  # -> (B*4*4, 3, 8, 8)
            x = self.net(x)
            x = torch.sum(x, dim=[2, 3])  # -> (B*4*4, 128)
            x = self.linear(x).view(-1)  # (B, 1) -> (B, )
            return x

    def __init__(self, k=128, beta=0.25, n_filters=128, patch_size=8, perceptual_loss_weight=0.5, gan_loss_weight=0.1):
        super().__init__(k, beta)
        self.decoder.append(nn.Tanh())  # VQGAN requires decoder output to be [-1, 1]
        self.perceptual_loss_weight = perceptual_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.discriminator = self.Discriminator(n_filters=n_filters, patch_size=patch_size)
        self.perceptual_loss = LPIPS()

    def disc_loss(self, x, x_recon):
        logits_real = self.discriminator(x)  # (B,)
        # detach to stop the flow of gradients to the rest of the model
        logits_fake = self.discriminator(x_recon.detach())  # (B,)
        # log(1 - sigmoid(x)) = -x + logsigmoid(x)
        disc_loss = -(F.logsigmoid(logits_real) + (-logits_fake + F.logsigmoid(logits_fake))).mean()
        return disc_loss
    
    def ae_losses(self, x, x_recon, latents, quant_latents):
        l2_loss = F.mse_loss(x, x_recon)  # trains encoder & decoder
        codebook_loss = F.mse_loss(latents.detach(), quant_latents)  # trains codebook
        commitment_loss = self.beta * F.mse_loss(latents, quant_latents.detach())  # trains encoder
        vq_loss = l2_loss + codebook_loss + commitment_loss

        perceptual_loss = self.perceptual_loss(x, x_recon).mean()
        gan_loss = -F.logsigmoid(self.discriminator(x_recon)).mean()  # non saturating formulation

        return l2_loss, vq_loss, perceptual_loss, gan_loss
    
    def train_one_step(self, x, disc_opt, ae_opt):
        latents = self.encoder(x)  # -> (N, 256, 8, 8)
        quant_latents = self._quantize_latents(latents)[0]
        # straight-though estimator in pytorch
        quant_latents_st = (quant_latents - latents).detach() + latents
        x_recon = self.decoder(quant_latents_st)

        # discriminator training
        disc_loss = self.disc_loss(x, x_recon)
        disc_loss.backward()
        disc_opt.step()
        disc_opt.zero_grad()

        # ae losses
        l2_loss, vq_loss, perceptual_loss, gan_loss = self.ae_losses(x, x_recon, latents, quant_latents)

        ae_loss = (
            vq_loss + 
            self.perceptual_loss_weight * perceptual_loss + 
            self.gan_loss_weight * gan_loss
        )

        # ae training
        ae_loss.backward()
        ae_opt.step()
        self.zero_grad()  # clear out discriminator grads as well

        return disc_loss.item(), perceptual_loss.item(), l2_loss.item()


# differences from Transformer in Autoregressive: removed token embedding, non-causal and removed head
class Transformer(nn.Module):
    def __init__(self, max_seq_len, n_layers=2, d_model=128, n_heads=4, d_mlp=2048, dropout=0.1, act=nn.GELU()):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(max_seq_len, d_model) * 0.1)

        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads, d_mlp, dropout, act, is_causal=False) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):  # (B, T, D)
        x = x + self.pos_enc
        x = self.blocks(x)
        out = self.ln(x)
        return out  # (B, T, D)


class ViTVQGAN(VQGAN):
    class Encoder(nn.Module):
        def __init__(self, in_channels=3, out_channels=256, patch_size=4, img_size=32, n_layer=4, n_heads=8):
            super().__init__()
            self.patch_size = patch_size
            self.latent_size = img_size // patch_size
            self.patchify_linear_proj = Patchify(in_channels=in_channels, out_channels=out_channels, patch_size=patch_size)
            self.transformer = Transformer(max_seq_len=(img_size // patch_size)**2, n_layers=n_layer, d_model=out_channels, n_heads=n_heads)

        def forward(self, x):  # (B, 3, 32, 32)
            x = self.patchify_linear_proj(x)  # -> (B, 64, 256)
            out = self.transformer(x)  # -> (B, 64, 256)
            B, T, D = out.shape
            # -> (B, 64, 256) -> (B, 256, 64) -> (B, 256, 8, 8)
            out = out.permute(0, 2, 1).reshape(B, D, self.latent_size, self.latent_size)
            return out
    
    class Decoder(nn.Module):
        def __init__(self, in_channels=256, patch_size=4, img_size=32, n_layer=4, n_heads=8):
            super().__init__()
            self.img_size=img_size
            self.patch_size = patch_size
            out_channels = 3 * patch_size * patch_size
            self.transformer = Transformer(max_seq_len=(img_size // patch_size)**2, n_layers=n_layer, d_model=in_channels, n_heads=n_heads)
            self.linear = nn.Linear(in_channels, out_channels)

        def forward(self, x):  # (B, 256, 8, 8)
            B, D, _, _ = x.shape
            x = x.view(B, D, -1).permute(0, 2, 1)  # -> (B, 64, 256)
            x = self.transformer(x)  # -> (B, 64, 256)
            x = F.tanh(self.linear(x))  # -> (B, 64, 48)
            out = unpatchify(x, output_size=self.img_size, patch_size=self.patch_size)
            return out
        
    def __init__(self, k=128, beta=0.25, n_filters=128, patch_size=8, perceptual_loss_weight=0.5, gan_loss_weight=0.1, l1_loss_weight=0.1):
        super().__init__(k, beta, n_filters, patch_size, perceptual_loss_weight, gan_loss_weight)
        self.l1_loss_weight = l1_loss_weight
        self.encoder = self.Encoder()
        self.decoder = self.Decoder()
        self.discriminator = nn.Sequential(
            ResnetBlockDown(3, n_filters=n_filters, act=nn.LeakyReLU()),
            ResnetBlockDown(n_filters, n_filters=n_filters, act=nn.LeakyReLU()),
            ResBlock(n_filters, n_filters=n_filters, act=nn.LeakyReLU()),
            ResBlock(n_filters, n_filters=n_filters, act=nn.LeakyReLU()),
            nn.LeakyReLU(),
            GlobalSumPooling(keep_dims=False),
            nn.Linear(n_filters, 1)
        )

        # apply spectral norm to weight matrices
        for m in self.discriminator.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                spectral_norm(m)
    
    # hacky way to add l1 loss without modifying VQGAN code
    def ae_losses(self, x, x_recon, latents, quant_latents):
        l2_loss, vq_loss, perceptual_loss, gan_loss = super().ae_losses(x, x_recon, latents, quant_latents)
        vq_loss += self.l1_loss_weight * F.l1_loss(x, x_recon)  # trains encoder & decoder
        return l2_loss, vq_loss, perceptual_loss, gan_loss


if __name__ == '__main__':
    import torch.optim as optim

    torch.manual_seed(1)
    model = ViTVQGAN(n_filters=3)

    discriminator_opt = optim.Adam(
        model.discriminator.parameters(),
        lr=1e-3, 
        betas=(0.5, 0.9)
    )

    ae_opt = optim.Adam(
        itertools.chain(
            model.encoder.parameters(),
            model.decoder.parameters(),
            model.codebook.parameters()
        ),
        lr=1e-3,
        betas=(0.5, 0.9)
    )

    x = torch.rand(1, 3, 32, 32)
    disc_loss, perceptual_loss, l2_loss = model.train_one_step(x, discriminator_opt, ae_opt)
    print(f'{disc_loss=}')
    print(f'{perceptual_loss=}')
    print(f'{l2_loss=}')







