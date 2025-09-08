import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import math


def timestep_embedding(timesteps, dim, scale_factor=1000, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half) / half)
    args = scale_factor * timesteps[:, None] * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding  # (B, D)


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def dropout_classes(y, dropout_prob, num_classes):  # y: (b,)
    dropout_mask = torch.rand_like(y, dtype=torch.float32) < dropout_prob
    y_dropped_out = torch.where(dropout_mask, num_classes, y)
    return y_dropped_out


def unpatchify(x, output_size, patch_size):
    return F.fold(
        x.transpose(-1, -2),  # (B, H//P*W//P, C*P*P) -> (B, C*P*P, H//P*W//P)
        output_size,
        kernel_size=patch_size,
        stride=patch_size
    )  # -> (B, C, H, W)


if __name__ == "__main__":
    y = torch.tensor([0, 1, 2, 4, 5], dtype=torch.int64)
    dropout_prob = 0.1
    print(dropout_classes(y, dropout_prob, 6))
