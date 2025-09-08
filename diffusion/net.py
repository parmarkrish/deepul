import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from autoregressive.transformer import MultiHeadSelfAttention


class MLPConcat(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, 2)
        )

    def forward(self, x, t):
        return self.mlp(torch.cat((x, t.unsqueeze(1)), dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.projection = nn.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.temb_linear = nn.Linear(temb_channels, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

    def forward(self, x, temb):
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)

        temb = self.temb_linear(temb)
        h += temb[:, :, None, None]  # h is (B, D, H, W), temb is (B, D, 1, 1)

        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h)

        x = self.projection(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, hidden_dims, blocks_per_dim):
        super().__init__()
        self.hidden_dims = hidden_dims
        temb_channels = hidden_dims[0] * 4

        self.emb_block = nn.Sequential(
            nn.Linear(hidden_dims[0], temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels)
        )

        self.down_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)])
        prev_ch = hidden_dims[0]
        down_block_chs = [prev_ch]

        for i, hidden_dim in enumerate(hidden_dims):
            for _ in range(blocks_per_dim):
                self.down_blocks.append(ResidualBlock(
                    prev_ch, hidden_dim, temb_channels))
                prev_ch = hidden_dim
                down_block_chs.append(prev_ch)
            if i != len(hidden_dims) - 1:
                self.down_blocks.append(Downsample(prev_ch))
                down_block_chs.append(prev_ch)

        self.mid_blocks = nn.ModuleList(ResidualBlock(prev_ch, prev_ch, temb_channels) for _ in range(blocks_per_dim))

        self.up_blocks = nn.ModuleList()
        for i, hidden_dim in list(enumerate(hidden_dims))[::-1]:
            for j in range(blocks_per_dim + 1):
                dch = down_block_chs.pop()
                self.up_blocks.append(ResidualBlock(
                    prev_ch + dch, hidden_dim, temb_channels))
                prev_ch = hidden_dim
                if i and j == blocks_per_dim:  # upsample except final level
                    self.up_blocks.append(Upsample(prev_ch))

        self.head = nn.Sequential(
            nn.GroupNorm(8, prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, in_channels, 3, padding=1)
        )

    def forward(self, x, t):
        emb = timestep_embedding(t, self.hidden_dims[0])
        emb = self.emb_block(emb)
        hs = []
        h = x
        for down_block in self.down_blocks:
            h = down_block(h, emb) if isinstance(down_block, ResidualBlock) else down_block(h)
            hs.append(h)

        for mid_block in self.mid_blocks:
            h = mid_block(h, emb)

        for up_block in self.up_blocks:
            h = up_block(torch.cat((h, hs.pop()), dim=1), emb) if isinstance(up_block, ResidualBlock) else up_block(h)

        out = self.head(h)
        return out


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.cond_mlp = nn.Linear(hidden_size, 6 * hidden_size)
        nn.init.zeros_(self.cond_mlp.weight)
        nn.init.zeros_(self.cond_mlp.bias)
        self.layer_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.msa = MultiHeadSelfAttention(hidden_size, num_heads)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size),
                                 nn.SiLU(), nn.Linear(4 * hidden_size, hidden_size))

    def forward(self, x, c):  # x: (B, L, D), c: (B, D)
        c = self.cond_mlp(F.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(c, 6, dim=1)

        h = self.layer_norm1(x)
        h = modulate(h, shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.msa(h)

        h = self.layer_norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.cond_mlp = nn.Linear(hidden_size, 2 * hidden_size)
        nn.init.zeros_(self.cond_mlp.weight)
        nn.init.zeros_(self.cond_mlp.bias)
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.out_mlp = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x, c):  # x: (B, L, D), c: (B, D)
        c = self.cond_mlp(F.silu(c))
        shift, scale = torch.chunk(c, 2, dim=1)

        x = self.layer_norm(x)
        x = modulate(x, shift, scale)
        x = self.out_mlp(x)
        return x


class Patchify(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.patchify = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(in_channels * patch_size * patch_size, out_channels)

    def forward(self, x):
        x = self.patchify(x).transpose(-1, -2)  # (B, C, H, W) -> (B, C*P*P, H//P*W//P) -> (B, H//P*W//P, C*P*P)
        x = self.proj(x)  # -> (B, H//P*W//P, D)
        return x


class DiT(nn.Module):
    def __init__(self, input_shape, patch_size, hidden_size, num_heads, num_layers, num_classes, dropout_prob):
        super().__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

        self.patchify_lin_proj = Patchify(input_shape[0], hidden_size, patch_size)
        self.class_embedding = nn.Embedding(num_classes + 1, hidden_size)
        # use ModuleList instead of Sequential since we have multiple inputs
        self.DiT_blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.final_layer = DiTFinalLayer(hidden_size, patch_size, input_shape[0])

    def forward(self, x, t, y):  # x: (B, C, H, W), y: (B,) t: (B,)
        # (B, C, H, W) -> (B, D, (H // P * W // P)) -> (B, (H // P * W // P), D)
        x = self.patchify_lin_proj(x)

        pos_emb = torch.tensor(
            get_2d_sincos_pos_embed(self.hidden_size, grid_size=self.input_shape[1] // self.patch_size),
            dtype=torch.float32
        )

        x += pos_emb

        t_emb = timestep_embedding(t, self.hidden_size)
        if self.training:
            y = dropout_classes(y, self.dropout_prob, self.num_classes)
        y = self.class_embedding(y)
        c = t_emb + y

        for block in self.DiT_blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = unpatchify(x, output_size=self.input_shape[1:], patch_size=self.patch_size)
        return x
