"""behavior-cloning: the PyTorch float64 twin of trainer 'bc' (kind B parity, design §6.3).

`npm run atlas -- parity behavior-cloning` runs the TS trainer in fine detail and writes its initial weights and every
batch it trained on (x [n, 2]: each dataset cell's column and row scaled to [-1, 1]; y [n]: the move made there, 0 up,
1 right, 2 down, 3 left) to site/.atlas/behavior-cloning/train_fixture.bc.<optimizer>.json. TS is the source of truth
for the dataset (the demonstrations, and for DAgger the cells its rounds added: they are played with the TS policy and
the TS grid, which reference.py checks). This twin trains on the same batches in float64 and must reach the same
losses, step-1 gradients and final weights: plain SGD for 50 steps within 1e-10 relative, Adam within 1e-7.

The loss, as trainer.ts: cross-entropy of the four move scores against the demonstrated move, averaged over the dataset.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    """TS MLP('net', [2, hidden, 4], 'tanh'): net.0.weight, net.0.bias, net.2.weight, net.2.bias."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build(cfg: dict) -> nn.Module:
    return Policy(int(cfg.get("hidden", 16))).double()


def loss(model: Policy, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    return F.cross_entropy(model(batch["x"]), batch["y"].reshape(-1).long()), {}
