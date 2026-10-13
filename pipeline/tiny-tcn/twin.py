"""tiny-tcn: the PyTorch float64 twin of trainer 'tcn' (kind B parity, design §4.11).

`npm run atlas -- parity tiny-tcn` writes the TS init params and batches (the changes of the training part and the
targets: TS is the source of truth for data and initial weights) and runs `python -m atlas_kit.train_reference` with
this twin. The network is explainers/tiny-tcn/model.ts: four causal convolutions with kernel 2 and dilations 1, 2, 4, 8
(zeros padded on the left only, F.pad), ReLU, residual blocks from layer 2 on, and a 1x1 head. State_dict names equal the
TS names: l1.weight [8, 1, 2], l1.bias [8], l2..l4.weight [8, 8, 2], head.weight [1, 8, 1], head.bias [1].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DILATIONS = (1, 2, 4, 8)


class TCN(nn.Module):
    def __init__(self, ch: int = 8):
        super().__init__()
        self.l1 = nn.Conv1d(1, ch, 2, dilation=1)
        self.l2 = nn.Conv1d(ch, ch, 2, dilation=2)
        self.l3 = nn.Conv1d(ch, ch, 2, dilation=4)
        self.l4 = nn.Conv1d(ch, ch, 2, dilation=8)
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [N, 1, T] changes -> [N, 1, T] predicted next changes"""
        h = x
        for i, layer in enumerate((self.l1, self.l2, self.l3, self.l4)):
            a = F.relu(layer(F.pad(h, (DILATIONS[i], 0))))
            h = a if i == 0 else h + a
        return self.head(h)


def build(cfg: dict) -> nn.Module:
    return TCN().double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'x': [1, 1, T] changes, 'y': [1, 1, n] targets c_{t+1} of t = T - 1 - n .. T - 2}. Mean squared error."""
    x, y = batch["x"], batch["y"]
    T, n = x.shape[2], y.shape[2]
    out = model(x)[:, :, T - 1 - n:T - 1]
    return torch.mean((out - y) ** 2), {}
