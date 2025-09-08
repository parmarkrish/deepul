import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.parametrizations import weight_norm
from torch.distributions import MultivariateNormal
from functools import partial


class ResnetBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_filters, n_filters, 1))
        self.conv2 = weight_norm(nn.Conv2d(n_filters, n_filters, 3, padding=1))
        self.conv3 = weight_norm(nn.Conv2d(n_filters, n_filters, 1))
    def forward(self, x):
        h = self.conv1(x)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.conv3(h)
        h = F.relu(h)
        return h + x


class SimpleResnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=128, n_blocks=8):
        super().__init__()
        self.inital_conv = weight_norm(nn.Conv2d(in_channels, n_filters, 3, padding=1))
        self.resnet_blocks = nn.Sequential(*[ResnetBlock(n_filters) for _ in range(n_blocks)])
        self.out_conv = weight_norm(nn.Conv2d(n_filters, out_channels, 3, padding=1))
    def forward(self, x):
        x = self.inital_conv(x)
        x = self.resnet_blocks(x)
        x = F.relu(x)
        x = self.out_conv(x)
        return x


class AffineCoupling(nn.Module):
    def __init__(self, mask, in_channels, n_filters=128, n_blocks=8):
        super().__init__()
        self.register_buffer('mask', mask)
        self.scale = nn.Parameter(torch.rand(1) * 0.1)
        self.scale_shift = nn.Parameter(torch.rand(1) * 0.1)
        self.simple_resnest = SimpleResnet(in_channels, in_channels*2, n_filters=n_filters, n_blocks=n_blocks)

    def forward(self, x):
        x_ = x * self.mask
        out = self.simple_resnest(x_)  # (B, C, H, W) -> (B, 2C, H, W)
        log_s, t = torch.chunk(out, 2, dim=1)  # -> (B, C, H, W)
        log_scale = self.scale * F.tanh(log_s) + self.scale_shift

        t = t * (1 - self.mask)
        log_scale = log_scale * (1 - self.mask)

        self.log_det_jac_per_dim = log_scale.mean()  # save as attribute (per dim log abs det jacobian)
        # print(f'in affine coupling {self.log_det_jac_per_dim=}')

        return x * torch.exp(log_scale) + t

    @torch.no_grad()
    def reverse(self, y):
        y_ = y * self.mask
        out = self.simple_resnest(y_)  # (B, C, H, W) -> (B, 2C, H, W)
        log_s, t = torch.chunk(out, 2, dim=1)  # -> (B, C, H, W)
        log_scale = self.scale * F.tanh(log_s) + self.scale_shift

        t = t * (1 - self.mask)
        log_scale = log_scale * (1 - self.mask)

        return (y - t) * torch.exp(-log_scale)


def create_checkerboard_mask(H, W, alt=False):
    y_pattern, x_pattern = torch.meshgrid(torch.arange(W) + alt, torch.arange(H))
    return (y_pattern + x_pattern) % 2


class AffineCouplingWithCheckerboard(AffineCoupling):
    def __init__(self, feature_map_size, in_channels, n_filters=128, n_blocks=8, alt=False):
        super().__init__(
            create_checkerboard_mask(*feature_map_size, alt=alt),
            in_channels,
            n_filters=n_filters,
            n_blocks=n_blocks
        )


def create_channelwise_mask(n_channels, alt=False):
    x = torch.ones(n_channels, dtype=torch.bool)
    x[:n_channels//2] = 0
    if alt:
        x = ~x
    return x.view(1, n_channels, 1, 1).float()


class AffineCouplingWithChannel(AffineCoupling):
    def __init__(self, in_channels, n_filters=128, n_blocks=8, alt=False):
        super().__init__(
            create_channelwise_mask(in_channels, alt=alt),
            in_channels,
            n_filters=n_filters,
            n_blocks=n_blocks
        )


def squeeze(x):
    N, C, H, W = x.shape
    return F.unfold(x, 2, stride=2).view(N, C*4, H//2, W//2)


def unsqueeze(x):
    N, C, H, W = x.shape
    x = x.view(N, C, H*W)
    return F.fold(x, output_size=(H*2, W*2), kernel_size=2, stride=2)


class ActNorm(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.scale = nn.Parameter(torch.empty(1, n_channels, 1, 1))
        self.bias = nn.Parameter(torch.empty(1, n_channels, 1, 1))
        self.is_initialized = False

    def forward(self, x):
        assert x.ndim == 4, "Only implemented for 4D tensor (N, C, H, W)"
        _, _, H, W = x.shape

        if not self.is_initialized:  # use first batch to initialize scale and bias
            self.bias.data = -x.detach().mean(dim=(0, 2, 3), keepdim=True)
            self.scale.data = 1. / (x.detach() + self.bias.data).std(dim=(0, 2, 3), keepdim=True)
            self.is_initialized = True

        self.log_det_jac_per_dim = self.scale.abs().log().mean()
        # print(f'in act norm{self.log_det_jac_per_dim=}')
        return self.scale * (x + self.bias)

    def reverse(self, y):
        return y / self.scale - self.bias


class RealNVP(nn.Module):
    def __init__(self, image_shape: tuple[int, int, int], n_filters):
        super().__init__()
        self.image_shape = image_shape
        n_channels = image_shape[0]
        self.n_dims = np.prod(image_shape)
        # not sure how to set automatically set device of distribution without passing it in
        self.distribution = MultivariateNormal(torch.zeros(self.n_dims, device='cuda'), torch.eye(self.n_dims, device='cuda'))
        # self.distribution = MultivariateNormal(torch.zeros(self.n_dims), torch.eye(self.n_dims))

        self.stage1 = self._create_stage('checkerboard', 4, in_channels=n_channels, n_filters=n_filters)
        self.stage2 = self._create_stage('channel', 3, in_channels=n_channels*4, n_filters=n_filters)
        self.stage3 = self._create_stage('checkerboard', 3, in_channels=n_channels, n_filters=n_filters)

    def _create_stage(self, type, n_reps, in_channels, n_filters=128, n_blocks=8):
        assert type in ('checkerboard', 'channel'), "Invalid type. Choose from either 'checkerboard' or 'channel'"
        AffineCouplingType = partial(AffineCouplingWithCheckerboard, self.image_shape[1:]) if type == 'checkerboard' else AffineCouplingWithChannel
        layers = nn.ModuleList()
        # for i in range(n_reps):
            # layers.append(AffineCouplingType(in_channels, n_filters=n_filters, n_blocks=n_blocks, alt=(i % 2)))
        for i in range(n_reps):
            layers.extend([
                AffineCouplingType(in_channels, n_filters=n_filters, n_blocks=n_blocks, alt=(i % 2)),
                ActNorm(in_channels)
            ])
        return layers

    def forward(self, x):
        for layer in self.stage1:
            x = layer(x)
        x = squeeze(x)
        for layer in self.stage2:
            x = layer(x)
        x = unsqueeze(x)
        for layer in self.stage3:
            x = layer(x)
        return x

    def loss(self, x):
        out = self.forward(x)
        # modules() method recursively gets all modules (i.e sub-modules as well), so we have to filter using isinstance
        log_det_jac_per_dim = sum([m.log_det_jac_per_dim for m in self.modules() if isinstance(m, (AffineCoupling, ActNorm))])
        log_prob_z_per_dim = self.distribution.log_prob(out.flatten(start_dim=1)).mean() / self.n_dims
        loss = -(log_prob_z_per_dim + log_det_jac_per_dim)
        return loss

    @torch.no_grad()
    def reverse(self, y):  # used for testing, could be folded into samples
        for layer in reversed(self.stage3):
            y = layer.reverse(y)
        y = squeeze(y)
        for layer in reversed(self.stage2):
            y = layer.reverse(y)
        y = unsqueeze(y)
        for layer in reversed(self.stage1):
            y = layer.reverse(y)
        return y

    @torch.no_grad()
    def samples(self, n_samples, device='cuda'):
        y = torch.randn(n_samples, *self.image_shape, device=device)
        return self.reverse(y)
