import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, padding=0):
        assert mask_type == 'A' or mask_type == 'B', "Mask type must be 'A' or 'B'"
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        self.mask = torch.zeros(kernel_size, kernel_size)
        mid_index = kernel_size // 2
        self.mask[:mid_index] = 1
        self.mask[mid_index, :mid_index + (mask_type == 'B')] = 1 # go one more left if mask_type == 'B'
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNNSimple(nn.Module):
  def __init__(self, image_shape, num_channels):
    super().__init__()
    self.image_shape = image_shape
    self.layers = nn.ModuleList([MaskedConv2d(1, num_channels, 7, mask_type='A', padding='same'), nn.ReLU()])
    for _ in range(5):
        self.layers.extend([MaskedConv2d(num_channels, num_channels, 7, mask_type='B', padding='same'), nn.ReLU()])

    self.layers.extend([MaskedConv2d(num_channels, num_channels, 1, mask_type='B'), nn.ReLU()])
    self.layers.append(MaskedConv2d(num_channels, 1, 1, mask_type='B')) # layer layer with no relu

  def forward(self, x):
    for layer in self.layers:
        x = layer(x)
    return x

  def loss(self, x):
    logits = self.forward(x)
    return F.binary_cross_entropy_with_logits(logits, x)

  @torch.no_grad()
  def samples(self, N):
    num_dims = self.image_shape[0] * self.image_shape[1]
    x = torch.zeros(N, 1, *(self.image_shape)) # (N, 1, H, W)
    for i in range(num_dims):
        logits = self.forward(x).flatten(1)[:, i] # (N, H*W)[:, i] -> (N,)
        probs = torch.sigmoid(logits)
        next = torch.bernoulli(probs)
        x.flatten(1)[:, i] = next
    return x


class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Conv2d(2*num_channels, num_channels, 1), nn.ReLU(),
            MaskedConv2d(num_channels, num_channels, 7, mask_type='B', padding='same'), nn.ReLU(),
            nn.Conv2d(num_channels, 2*num_channels, 1), nn.ReLU()
        ])

    def forward(self, x):
        return self.layers(x) + x


class PixelCNN(nn.Module):
    def __init__(self, image_shape, num_channels, num_layers):
        super().__init__()
        self.image_shape_channels_first = image_shape[::-1] # assumes that H == W
        # 1x1 conv to make increase channels to 2*num_channels
        layers = [MaskedConv2d(3, 2*num_channels, 7, mask_type='A', padding='same'), nn.ReLU()]
        layers += [ResBlock(num_channels) for _ in range(num_layers-1)]
        layers.append(nn.Conv2d(2*num_channels, 3*4, 1)) # logits layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def loss(self, x):
        logits = self.net(x)
        B, _, H, W = logits.shape # (B, 12, H, W)

        logits = logits.view(B * 3, 4, H, W)
        y = x.view(B * 3, H, W).long()
        return F.cross_entropy(logits, y)

    @torch.no_grad()
    def samples(self, N):
        x = torch.zeros((N, *self.image_shape_channels_first))
        _, C, H, W = x.shape
        for i in range(H * W):
            logits = self.net(x) # (N, 12, H, W)
            logits = logits.view(N*3, 4, H*W)
            logits_i = logits[:, :, i] # (N*3, 4)
            probs = F.softmax(logits_i, dim=1)
            next = torch.multinomial(probs, 1) # (N*3, 1)
            x.view(N, C, H*W)[:, :, i] = next.view(N, 3) # slot in next dim
        return x
