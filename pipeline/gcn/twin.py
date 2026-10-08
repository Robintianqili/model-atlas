"""gcn: the PyTorch float64 twin of trainer 'gcn' (kind B parity, design §4.11).

The TS trainer (site/src/lib/explainers/gcn/trainer.ts) writes the tensors it trained on into the fixture (Â of the
learner's graph, the one-hot X, the factions and the labelled members), so this twin needs no graph code. It is
model.py's GCN with the same state_dict names (gc1.weight, gc1.bias, gc2.*, head.*): H1 = tanh(Â (X W1ᵀ) + b1),
H2 = tanh(Â (H1 W2ᵀ) + b2), Z = H2 Whᵀ + bh, cross-entropy over the labelled members only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GCNConv  # noqa: E402


class Twin(nn.Module):
    def __init__(self):
        super().__init__()
        self.gc1 = GCNConv(34, 4)
        self.gc2 = GCNConv(4, 2)
        self.head = nn.Linear(2, 2)

    def forward(self, ahat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.gc1(x, ahat))
        return self.head(torch.tanh(self.gc2(h, ahat)))


def build(cfg: dict) -> nn.Module:
    return Twin().double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'ahat': [n, n], 'x': [n, n], 'y': [n], 'labelled': [k]} float64 (indices arrive as floats)."""
    z = model(batch["ahat"], batch["x"])
    idx = batch["labelled"].long().reshape(-1)
    if idx.numel() == 0:
        return model.head.weight.sum() * 0, {}
    return F.cross_entropy(z[idx], batch["y"].long()[idx]), {}
