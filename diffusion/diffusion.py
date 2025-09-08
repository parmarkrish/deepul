import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContinuousDiffusion(nn.Module):
    def __init__(self, model, shape, clip_bounds=None):
        super().__init__()
        self.model = model
        self.shape = shape
        self.clip_bounds = clip_bounds

    def forward(self, *args):
        eps_hat = self.model.forward(*args)
        return eps_hat

    def get_alpha_t_sigma_t(self, t):
        t = t.view(*t.shape, *([1] * len(self.shape)))
        alpha_t, sigma_t = torch.cos(math.pi / 2 * t), torch.sin(math.pi / 2 * t)
        return alpha_t, sigma_t

    def loss(self, x, y=None):
        batch_size = x.shape[0]
        eps = torch.randn_like(x)
        t = torch.rand(batch_size)
        alpha_t, sigma_t = self.get_alpha_t_sigma_t(t)

        xt = alpha_t * x + sigma_t * eps
        if y is None:
            eps_hat = self.forward(xt, t)
        else:
            eps_hat = self.forward(xt, t, y)

        loss = F.mse_loss(eps, eps_hat)
        return loss

    def ddpm_update(self, xt, eps_hat, t, tm1):
        alpha_t, sigma_t = self.get_alpha_t_sigma_t(t)
        alpha_tm1, sigma_tm1 = self.get_alpha_t_sigma_t(tm1)

        N = sigma_tm1 / sigma_t * torch.sqrt(1 - alpha_t**2 / alpha_tm1**2)
        eps = torch.randn_like(xt)

        x0_hat = (xt - sigma_t * eps_hat) / alpha_t

        if self.clip_bounds is not None:
            x0_hat = x0_hat.clip(*self.clip_bounds)

        xtm1 = (
            alpha_tm1 * (x0_hat)
            + torch.sqrt((sigma_tm1**2 - N**2).clamp(0)) * eps_hat
            + N * eps
        )
        return xtm1

    @torch.no_grad()
    def generate(self, num_samples, num_steps, class_idxs=None, cfg_scale=1):
        self.eval()
        ts = torch.linspace(1 - 1e-4, 1e-4, num_steps + 1)
        x = torch.randn(num_samples, *self.shape)
        for i in range(num_steps):
            t = ts[i].expand(num_samples)
            tm1 = ts[i + 1].expand(num_samples)
            if class_idxs is None:
                eps_hat = self(x, t)
            else:
                if cfg_scale == 1:
                    eps_hat = self(x, t, class_idxs)
                else:
                    # set up batch such that the first half is conditional and second half is unconditional
                    x_repeat = x.repeat(2, 1, 1, 1)
                    t_repeat = t.repeat(2)
                    class_idxs_repeat = torch.cat(
                        (class_idxs, torch.full((num_samples,), self.model.num_classes))
                    )
                    eps_hat_cond, eps_hat_uncond = self(x_repeat, t_repeat, class_idxs_repeat).chunk(2)
                    eps_hat = (1 - cfg_scale) * eps_hat_uncond + cfg_scale * eps_hat_cond

            x = self.ddpm_update(x, eps_hat, t, tm1)
        return x
