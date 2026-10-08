"""flow-2d: the PyTorch float64 twin of trainer 'fit' (kind B parity, design §4.11).

The velocity network v(x, y, t): Sequential(Linear(3, 64), SiLU, Linear(64, 64), SiLU, Linear(64, 64), SiLU,
Linear(64, 2)), the TS MLP('net', [3, 64, 64, 64, 2], 'silu'), with the same state_dict names (net.0.weight, ...).
Loss: the mean squared error between v(x_t, t) and the target x1 - x0, averaged over every number of the batch, as
the TS mse(). The batches come from the TS trainer (TS is the source of truth for init and synthetic data):

    npm run atlas -- parity flow-2d [--steps 20]     # SGD <= 1e-10 and Adam <= 1e-7 relative

pipeline/flow-2d/reference.py imports build() from here for the Euler check.
"""
from __future__ import annotations

import torch
import torch.nn as nn

SIZES = [3, 64, 64, 64, 2]


class Velocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(SIZES) - 1):
            layers.append(nn.Linear(SIZES[i], SIZES[i + 1]))
            if i + 2 < len(SIZES):
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build(cfg: dict) -> nn.Module:
    """The TS model: MLP('net', [3, 64, 64, 64, 2], 'silu')."""
    return Velocity().double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'inp': [B, 3] (x_t, y_t, t), 'target': [B, 2] (x1 - x0)} as float64 tensors."""
    return torch.mean((model(batch["inp"]) - batch["target"]) ** 2), {}
