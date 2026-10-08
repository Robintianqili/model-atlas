"""hello-xor: the PyTorch float64 twin of trainer 'xor' (kind B parity, design §4.11).

`npm run atlas -- parity hello-xor` runs the TS trainer in fine detail, writes its init params and batches to
site/.atlas/hello-xor/train_fixture.xor.<optimizer>.json, and runs `python -m atlas_kit.train_reference` with this twin.
Losses, step-1 gradients and final params must agree: SGD 1e-10 relative and Adam 1e-7 relative, over 50 steps.
The state_dict names equal the TS Module.named() names: MLP('net', [2, 2, 1], 'tanh') -> net.0.*, net.2.*.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build(cfg: dict) -> nn.Module:
    """The TS model: MLP('net', [2, hidden, 1], 'tanh') = Linear, Tanh, Linear (no activation after the last layer)."""
    return Net(int(cfg.get("hidden", 2))).double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'x': [4, 2], 'y': [4, 1]} as float64 tensors (written by the TS trainer). Mean squared error, as mse()."""
    yhat = model(batch["x"])
    return torch.mean((yhat - batch["y"]) ** 2), {}
