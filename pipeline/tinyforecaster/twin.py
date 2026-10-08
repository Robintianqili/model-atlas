"""tinyforecaster: the PyTorch float64 twin of trainer 'lstm' (kind B parity, design §4.11).

`npm run atlas -- parity tinyforecaster` writes the TS init params and batches (windows of the series: TS is the source
of truth for data and initial weights) and runs `python -m atlas_kit.train_reference` with this twin. The network is
explainers/tinyforecaster/model.ts: the window minus its last value goes through an LSTM cell one value at a time; a
linear head turns the last h into the predicted change; the forecast is the last value plus that change. State_dict
names equal the TS names: lstm.weight_ih [64, 1], lstm.weight_hh [64, 16], lstm.bias_ih, lstm.bias_hh, head.weight
[1, 16], head.bias.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Forecaster(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.lstm = nn.LSTMCell(1, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T] -> [B, 1]"""
        last = x[:, -1:]
        r = x - last
        h = torch.zeros(x.shape[0], self.lstm.hidden_size, dtype=x.dtype)
        c = torch.zeros_like(h)
        for t in range(x.shape[1]):
            h, c = self.lstm(r[:, t:t + 1], (h, c))
        return last + self.head(h)


def build(cfg: dict) -> nn.Module:
    return Forecaster(int(cfg.get("hidden", 16))).double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'x': [B, 24], 'y': [B, 1]} float64. Mean squared error, as the TS mse()."""
    return torch.mean((model(batch["x"]) - batch["y"]) ** 2), {}
