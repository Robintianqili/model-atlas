"""mlp-playground: the PyTorch float64 twin of trainer 'fit' (kind B parity, design §4.11).

`npm run atlas -- parity mlp-playground` runs the TS trainer in fine detail on the page's start data (two moons, 200
points), writes its init params and batches to site/.atlas/mlp-playground/train_fixture.fit.<optimizer>.json, then runs
`python -m atlas_kit.train_reference` with this twin: losses, step-1 gradients and final params must agree over 50 steps
(plain SGD <= 1e-10 relative, Adam <= 1e-7 relative).

The network is site/src/lib/explainers/mlp-playground/trainer.ts: MLP('net', [2, *hidden, 2], act), i.e.
nn.Sequential(Linear(2, h1), Tanh(), Linear(h1, h2), Tanh(), Linear(h2, 2)) with state_dict names net.0.*, net.2.*,
net.4.*; the loss is the mean cross-entropy of the two class scores over the batch (all the points: batch 0).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, hidden: list[int], act: str):
        super().__init__()
        sizes = [2, *hidden, 2]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i + 2 < len(sizes):
                layers.append(nn.ReLU() if act == "relu" else nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build(cfg: dict) -> nn.Module:
    """The TS model for this run's options: hidden sizes (default [4, 4]) and activation (default tanh)."""
    return Net([int(h) for h in cfg.get("hidden", [4, 4])], str(cfg.get("act", "tanh"))).double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'x': [B, 2], 'y': [B]} float64 (labels arrive as floats). Mean cross-entropy, as the TS crossEntropy()."""
    return F.cross_entropy(model(batch["x"]), batch["y"].long()), {}
