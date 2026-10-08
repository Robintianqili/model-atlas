"""two-tower: a two-tower retrieval model in PyTorch (design brief §5, Recommendation).

    item tower  8 numbers (genre as 6 switches, year, length; world.ts itemFeatures)
                -> linear 8 -> 16 -> ReLU -> linear 16 -> 8: the movie's vector
    user tower  15 numbers (liked genres, genre shares / mean year / mean length of the picked movies, "picked any";
                world.ts userFeatures) -> linear 15 -> 16 -> ReLU -> linear 16 -> 8: the viewer's vector
    score       the dot product of the two vectors. For retrieval the item tower runs once per movie, offline: its
                outputs form the item table [36, 8], and a viewer's scores are one matrix-vector product with it.

The features come from site/src/lib/explainers/two-tower/world.ts (TS is the source of truth): train.py and export.py
read them from the TS logs tape, never recompute them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_USER, N_ITEM, HIDDEN, DIM = 15, 8, 16, 8


class Tower(nn.Module):
    def __init__(self, n_in: int, hidden: int = HIDDEN, dim: int = DIM):
        super().__init__()
        self.fc1 = nn.Linear(n_in, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.item = Tower(N_ITEM)
        self.user = Tower(N_USER)

    def scores(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """[B, 15] viewers and [M, 8] movies -> [B, M] dot products"""
        return self.user(users) @ self.item(items).t()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
