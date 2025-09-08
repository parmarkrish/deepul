import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.uniform import Uniform
from torch.autograd import grad

class MixtureOfLogistics2D(nn.Module):
    def __init__(self, n_logistics, h_dim):
        super().__init__()
        self.n_logistics = n_logistics

        self.weights1 = nn.Parameter(torch.rand(n_logistics))
        self.means1 = nn.Parameter(torch.rand(n_logistics))
        self.log_scales1 = nn.Parameter(torch.rand(n_logistics))

        self.mlp = nn.Sequential(*[nn.Linear(1, h_dim), nn.ReLU(), nn.Linear(h_dim, n_logistics * 3)])

    def forward(self, x):  # (N, 2)
        # seperate into column vectors
        x1 = x[:,0].unsqueeze(1)
        x2 = x[:,1].unsqueeze(1)

        z1s = torch.sigmoid((x1 - self.means1) / torch.exp(self.log_scales1))  # (N, n_logistics)

        params2 = self.mlp(x1)
        weights2, means2, log_scales2 = torch.chunk(params2, 3, dim=1)  # (N, n_logistics) * 3

        z2s = torch.sigmoid((x2 - means2) / torch.exp(log_scales2))  # (N, n_logistics)

        z1 = z1s @ F.softmax(self.weights1, dim=0)
        z2 = torch.sum(z2s * F.softmax(weights2, dim=1), dim=1)
        z = torch.stack((z1, z2), dim=1)
        return z

class AutoregressiveFlow2D(nn.Module):
    def __init__(self, n_logistics, h_dim, distribution=Uniform(torch.tensor(0.), torch.tensor(1.)), epsilon=1e-4):
        super().__init__()
        self.MoL = MixtureOfLogistics2D(n_logistics, h_dim)
        self.dist = distribution
        self.epsilon = epsilon

    def forward(self, x):
        x.requires_grad = True
        return self.MoL(x)

    def log_prob(self, x):
        z = self(x)

        # floating point arithmetic can make values slightly above/below 1/0
        # which is problematic since Uniform(0) can only accept values from (0, 1)
        if isinstance(self.distribution, Uniform):
            z[z >= self.dist.high] -= self.epsilon
            z[z <= self.dist.low] += self.epsilon

        z1, z2 = z[:, 0], z[:, 1]

        dx1 = grad(z1, x, grad_outputs=torch.ones_like(z1), create_graph=True)[0][:, 0]  # grad returns tuple of tensors
        dx2 = grad(z2, x, grad_outputs=torch.ones_like(z2), create_graph=True)[0][:, 1]

        log_probs1 = self.dist.log_prob(z1) + dx1.abs().log()
        log_probs2 = self.dist.log_prob(z2) + dx2.abs().log()

        return log_probs1 + log_probs2

    def loss(self, x):
        return -torch.mean(self.log_prob(x))
