import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaGAN(nn.Module):
    def __init__(self, H=128, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(1, H), act,
            nn.Linear(H, H), act,
            nn.Linear(H, H), act,
            nn.Linear(H, 1), nn.Tanh()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(1, H), act,
            nn.Linear(H, H), act,
            nn.Linear(H, H), act,
            nn.Linear(H, 1), nn.Sigmoid()
        )
    def generate(self, n_samples):
        return self.generator(torch.randn(n_samples, 1))

    def discriminator_loss(self, x):
        D, G = self.discriminator, self.generator
        gz = self.generate(x.shape[0])
        return - D(x).log().mean() - (1 - D(gz)).log().mean()

    def generator_loss(self, x):
        D, G = self.discriminator, self.generator
        gz = self.generate(x.shape[0])
        return (1 - D(gz)).log().mean()


    def train_one_step(self, x, d_opt, g_opt, K):
        # discriminator update
        for _ in range(K):
            d_loss = self.discriminator_loss(x)
            d_loss.backward()
            d_opt.step()
            self.zero_grad()

        # generator update
        g_loss = self.generator_loss(x)
        g_loss.backward()
        g_opt.step()
        self.zero_grad()
        return d_loss.item()


class NonSaturatingGAN(VanillaGAN):
    def generator_loss(self, x):
        D, G = self.discriminator, self.generator
        gz = self.generate(x.shape[0])
        return -D(gz).log().mean()