import torch
import torch.nn as nn
import torch.nn.functional as F


class Made2d(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d1_weights = nn.Parameter(torch.rand(d))
        self.d2_weights = nn.Parameter(torch.rand(d, d))
    def loss(self, x):
        # x (32, 2)
        d1, d2 = x[:,0], x[:,1]
        d1_loss = F.cross_entropy(self.d1_weights.expand(x.shape[0], -1), d1)
        d2_loss = F.cross_entropy(self.d2_weights[d1], d2)
        loss = (d1_loss + d2_loss) / 2
        return loss

    @torch.no_grad()
    def prob(self):
        return F.softmax(self.d2_weights, dim=1) * \
            F.softmax(self.d1_weights, dim=0).unsqueeze(1)


class MADE(nn.Module):
    def __init__(self, image_shape, hidden_dims):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.image_shape = image_shape
        self.input_size = self.image_shape[0] * self.image_shape[1]
        dims = [self.input_size] + hidden_dims + [self.input_size] # add input and output dims
        self.layers = nn.ModuleList([nn.Linear(h1, h2) for h1, h2 in zip(dims, dims[1:])])
        self.masks = self._construct_masks()

    def _construct_masks(self):
        connections = [torch.arange(1, self.input_size+1)]
        connections += [
            torch.randint(min(connections[-1]), self.input_size, (hidden_dim,))
            for hidden_dim in self.hidden_dims
        ]
        masks = [
            c2.unsqueeze(1) >= c1 # use broacasting semantics to create mask
            for c1, c2 in zip(connections, connections[1:])
        ]
        masks += [connections[0].unsqueeze(1) > connections[-1]] # add last layer mask
        return masks

    def forward(self, x):
        x = x.flatten(1).float() # (B, H*W)
        for i, (layer, mask) in enumerate(zip(self.layers, self.masks)):
            layer.weight.data = layer.weight * mask
            x = layer(x)
        if i != len(self.layers)-1:
            x = F.relu(x)
        out = x.unflatten(1, (*self.image_shape, 1)) # (B, H, W, 1)
        return out

    def loss(self, x):
        logits = self.forward(x)
        return F.binary_cross_entropy_with_logits(logits, x.float())

    @torch.no_grad()
    def samples(self, N):
        samples = torch.zeros(N, *self.image_shape, 1)
        for i in range(self.input_size):
            logits = self.forward(samples)
            probs = torch.sigmoid(logits.flatten(1)) # (B, H*W)
            next = torch.bernoulli(probs)
            samples.flatten(1)[:, i] = next[:, i]
        return samples
