import torch
import torch.nn as nn
import torch.nn.functional as F


class Histogram(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.logits = nn.Parameter(torch.rand(d))
        
    def loss(self, labels):
        logits_expanded = self.logits.expand(len(labels), -1)
        return F.cross_entropy(logits_expanded, labels)

    @torch.no_grad()
    def prob(self):
      return F.softmax(self.logits, dim=0)


class DiscretizedMixtureOfLogistics(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.weights = nn.Parameter(torch.rand(4))
        self.means = nn.Parameter(torch.rand(4) * d)
        self.log_scales = nn.Parameter(torch.rand(4))

    def _prob(self, x):
        scales = torch.exp(self.log_scales)

        plus_mix = torch.sigmoid((x.unsqueeze(1) + 0.5 - self.means) / scales)
        minus_mix = torch.sigmoid((x.unsqueeze(1) - 0.5 - self.means) / scales)
        mix = plus_mix - minus_mix # (b, 4)

        zero_case_mix = torch.sigmoid((0.5 - self.means) / scales)
        end_case_mix = 1 - torch.sigmoid((self.d-1 - 0.5 - self.means) / scales)

        mix[x == 0] = zero_case_mix
        mix[x == self.d-1] = end_case_mix

        probs = mix @ F.softmax(self.weights, dim=0) # average the logistics
        return probs

    def loss(self, x):
      return torch.mean(-torch.log(self._prob(x)))

    @torch.no_grad()
    def prob(self):
        x = torch.arange(0, self.d).float()
        return self._prob(x)
