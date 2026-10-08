"""matrix-factorization: the PyTorch float64 twin of trainer 'fit' (kind B parity, design §4.11).

`npm run atlas -- parity matrix-factorization` runs the TS trainer in fine detail on the page's start input (24 made-up
people plus your sample profile), writes its init params and batches (one known rating per step: person u, movie i,
rating r) to site/.atlas/matrix-factorization/train_fixture.fit.sgd.json, then runs `python -m atlas_kit.train_reference`
with this twin: losses, step-1 gradients and final params must agree over 50 steps (plain SGD, <= 1e-10 relative).

The model is site/src/lib/explainers/matrix-factorization/trainer.ts: two tables of two numbers each, U = user.weight
[25, 2] (one row per person, the learner last) and V = item.weight [20, 2] (one row per movie). The predicted rating is
the dot product U[u] . V[i]; the loss is the mean squared error over the batch (one rating).
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_USERS, N_ITEMS, RANK = 25, 20, 2


class MF(nn.Module):
    def __init__(self, users: int = N_USERS, items: int = N_ITEMS, rank: int = RANK):
        super().__init__()
        self.user = nn.Embedding(users, rank)
        self.item = nn.Embedding(items, rank)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.user(u) * self.item(i)).sum(-1)


def build(cfg: dict) -> nn.Module:
    """The TS model (sizes fixed by the synthetic world); the init comes from the TS fixture."""
    return MF().double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'u': [B], 'i': [B], 'r': [B]} float64 (indices arrive as floats). Mean squared error, as the TS mse()."""
    pred = model(batch["u"].long(), batch["i"].long())
    return torch.mean((pred - batch["r"]) ** 2), {}
