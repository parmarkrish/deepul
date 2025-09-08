import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import math


class CosineScheduler:
    def __init__(self, warmup_iters, lr_decay_iters, base_lr, min_lr):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.iter = 0

    def step(self):
        if self.iter < self.warmup_iters:
            lr = self.base_lr * self.iter / self.warmup_iters
        else:
            iter = self.iter - self.warmup_iters
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * iter / self.lr_decay_iters)
            )
        self.iter += 1
        return lr


@torch.no_grad()
def test(model, test_loader, dtype=torch.float32):
    model.eval()
    test_loss = 0
    for x in test_loader:
        if not isinstance(x, list):
            x = [x]
        with torch.autocast(device_type=torch.get_default_device().type, dtype=dtype):
            test_loss += model.loss(*x).item()
    return test_loss / len(test_loader)


def train_one_epoch(
    model, optimizer, loader, dtype=torch.float32, lr_scheduler=None, clip=None
):
    assert dtype in (torch.float32, torch.bfloat16), "only supports float32 or bfloat16"
    losses = []
    model.train()
    for x in loader:
        with torch.autocast(device_type=torch.get_default_device().type, dtype=dtype):
            if not isinstance(x, list):
                x = [x]
            loss = model.loss(*x)
            loss.backward()

        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)

        if lr_scheduler is not None:
            lr = lr_scheduler.step()
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses


def train(
    model,
    optimizer,
    train_loader,
    test_loader,
    num_epochs=10,
    dtype=torch.float32,
    lr_scheduler=None,
    clip=None,
):
    train_losses, test_losses = [], []

    initial_loss = test(model, test_loader, dtype)
    test_losses.append(initial_loss)
    print(f"initial loss: {initial_loss}")

    for epoch_num in range(1, num_epochs + 1):
        epoch_losses = train_one_epoch(
            model, optimizer, train_loader, dtype, lr_scheduler, clip
        )
        train_losses.extend(epoch_losses)

        test_loss = test(model, test_loader, dtype)
        test_losses.append(test_loss)

        print(f"epoch {epoch_num}: {test_loss}")

    return train_losses, test_losses
