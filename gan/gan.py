import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


# Spatial Upsampling with Nearest Neighbors
class UpsampleConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.depth_to_space = DepthToSpace(block_size=2)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        x = self.depth_to_space(x)
        x = self.conv2d(x)
        return x


# Spatial Downsampling with Spatial Mean Pooling
class DownsampleConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.space_to_depth = SpaceToDepth(block_size=2)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        x = self.space_to_depth(x)
        x = sum(x.chunk(4, dim=1)) / 4.0
        x = self.conv2d(x)
        return x


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256, act=nn.ReLU()):
        super().__init__()
        self.n_filters = n_filters
        self.act = act
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.conv = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)
        self.upsample_conv2d_residual = UpsampleConv2d(n_filters, n_filters, kernel_size, padding=1)
        self.upsample_conv2d_shortcut = UpsampleConv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)
    def forward(self, x):
        _x = x
        _x = self.act(self.bn1(_x))
        _x = self.conv(_x)
        _x = self.act(self.bn2(_x))
        residual = self.upsample_conv2d_residual(_x)
        shortcut = self.upsample_conv2d_shortcut(x)
        return residual + shortcut


class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256, act=nn.ReLU()):
        super().__init__()
        self.n_filters = n_filters
        self.act = act
        self.conv = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)
        self.downsample_conv2d_residual = DownsampleConv2d(n_filters, n_filters, kernel_size, padding=1)
        self.downsample_conv2d_shortcut = DownsampleConv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)
    def forward(self, x):
        _x = x
        _x = self.act(x)
        _x = self.act(self.conv(_x))
        residual = self.downsample_conv2d_residual(_x)
        shortcut = self.downsample_conv2d_shortcut(x)
        return residual + shortcut


class ResBlock(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256, act=nn.ReLU()):
        super().__init__()
        assert in_dim == n_filters
        self.act = act
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, n_filters, kernel_size, padding=1)

        self.bn2 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size, padding=1)
    def forward(self, x):
        _x = x
        _x = self.act(self.bn1(_x))
        _x = self.conv1(_x)

        _x = self.act(self.bn2(_x))
        _x = self.conv2(_x)
        return _x + x


class GlobalSumPooling(nn.Module):
    def __init__(self, keep_dims=True):
        super().__init__()
        self.keep_dims = keep_dims
    def forward(self, x):
        return torch.sum(x, dim=(-1, -2), keepdims=self.keep_dims)


class GAN(nn.Module):
    def __init__(self, n_filters=128):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(128, 4*4*256),
            nn.Unflatten(1, (256, 4, 4)),
            ResnetBlockUp(in_dim=256, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            ResnetBlockDown(in_dim=3, n_filters=n_filters),
            ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
            ResBlock(in_dim=n_filters, n_filters=n_filters),
            ResBlock(in_dim=n_filters, n_filters=n_filters),
            nn.ReLU(),
            GlobalSumPooling(keep_dims=False),
            nn.Linear(n_filters, 1),
        )

    def generate(self, n_samples):
        z = torch.randn(n_samples, 128)
        return self.generator(z)