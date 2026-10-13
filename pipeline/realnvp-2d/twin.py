"""realnvp-2d: the PyTorch float64 twin of trainer 'fit' (kind B parity, design §4.11).

The flow: 8 affine coupling layers. Layer k keeps coordinate k % 2 (x for even k) and moves the other one,
v' = v * exp(s) + t, with s = tanh(out[:, 0]) and t = out[:, 1] from the small network c<k> =
Sequential(Linear(1, 24), Tanh, Linear(24, 24), Tanh, Linear(24, 2)) of the kept coordinate: the TS
MLP('c<k>', [1, 24, 24, 2], 'tanh'), with the same state_dict names (c0.0.weight, ..., c7.4.bias).
Loss: the mean negative log-likelihood of the batch, 0.5 * |z|^2 + log(2 pi) - sum_k s_k, z = f(x).
The initial weights and the batches come from the TS trainer (TS is the source of truth; its last layers start at 0):

    npm run atlas -- parity realnvp-2d     # SGD <= 1e-10 and Adam <= 1e-7 relative over 50 steps

pipeline/realnvp-2d/reference.py imports build() from here for the process checks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

LAYERS, HIDDEN = 8, 24


def mlp() -> nn.Sequential:
    return nn.Sequential(nn.Linear(1, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 2))


class Flow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for k in range(LAYERS):
            setattr(self, f"c{k}", mlp())

    def couple(self, k: int, h: torch.Tensor, inverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer k on h [n, 2] (or its inverse): the new points and s [n]."""
        c = k % 2
        u, v = h[:, c:c + 1], h[:, 1 - c:2 - c]
        o = getattr(self, f"c{k}")(u)
        s, t = torch.tanh(o[:, 0:1]), o[:, 1:2]
        v2 = (v - t) * torch.exp(-s) if inverse else v * torch.exp(s) + t
        return (torch.cat([u, v2], 1) if c == 0 else torch.cat([v2, u], 1)), s[:, 0]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ld = torch.zeros(x.shape[0], dtype=x.dtype)
        for k in range(LAYERS):
            x, s = self.couple(k, x)
            ld = ld + s
        return x, ld


def build(cfg: dict) -> nn.Module:
    """The TS model: Flow() of trainer.ts."""
    return Flow().double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'x': [B, 2]} data points as float64."""
    z, ld = model(batch["x"])
    return torch.mean(0.5 * (z ** 2).sum(1) + math.log(2 * math.pi) - ld), {}
