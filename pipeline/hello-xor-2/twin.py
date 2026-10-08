"""hello-xor-2: the PyTorch float64 twin of trainer 'xorSigmoid' (kind B parity).

`npm run atlas -- parity hello-xor-2` runs the TS trainer in fine detail, writes its init params and batches to
site/.atlas/hello-xor-2/train_fixture.xorSigmoid.<optimizer>.json, and runs `python -m atlas_kit.train_reference` with
this twin. Losses, step-1 gradients and final params must agree: SGD 1e-10 and Adam 1e-7 relative, over 50 steps.
The state_dict names equal the TS Module.named() names: SigmoidMLP wraps MLP('net', [2, 2, 1], 'tanh') -> net.0.*, net.2.*.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def build(cfg: dict) -> nn.Module:
    """The TS model: Linear, Tanh, Linear, then a sigmoid (no parameters in the sigmoid)."""
    return Net(int(cfg.get("hidden", 2))).double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'x': [4, 2], 'y': [4, 1]} float64. Binary cross-entropy from the logits, as the TS bceWithLogits()."""
    return F.binary_cross_entropy_with_logits(model.logits(batch["x"]), batch["y"]), {}
