"""cart-pole: the PyTorch float64 twin of trainer 'reinforce' (kind B parity, design §6.2).

`npm run atlas -- parity cart-pole` runs the TS trainer in fine detail and writes its initial weights and every round it
played (states x [N, 4], left [N] with 1 = pushed left, normalized advantages adv [N], ...) to
site/.atlas/cart-pole/train_fixture.reinforce.<optimizer>.json. TS is the source of truth for the episodes (they are
played with the TS policy and the TS simulator; the simulator itself is checked by reference.py). This twin trains on
the same rounds in float64 and must reach the same losses, step-1 gradients and final weights: plain SGD for 50 steps
within 1e-10 relative, Adam within 1e-7.

The loss, as trainer.ts: mean over every step of the round of A * -log pi(action | state), where
-log pi = relu(z) - z*y + log(1 + exp(-|z|)) on the score z (y = 1 for left): binary cross-entropy with logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Policy(nn.Module):
    """TS MLP('net', [4, hidden, 1], 'tanh', { outAct: 'sigmoid' }): net.0.weight, net.0.bias, net.2.weight, net.2.bias."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1), nn.Sigmoid())

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net[2](self.net[1](self.net[0](x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build(cfg: dict) -> nn.Module:
    return Policy(int(cfg.get("hidden", 16))).double()


def loss(model: Policy, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    z = model.logits(batch["x"])
    y = batch["left"].reshape(-1, 1)
    nll = torch.relu(z) - z * y + torch.log(torch.exp(-torch.abs(z)) + 1)
    a = batch["adv"].reshape(-1, 1)
    return torch.mean(nll * a), {}
