"""ecg-autoencoder: the PyTorch model (kind A): a small dense autoencoder for one heartbeat of 140 numbers.

    encoder   140 heights -> 32 (weighted sums + ReLU) -> CODE numbers (weighted sums, no activation): the code
    decoder   CODE -> 32 (weighted sums + ReLU) -> 140 heights (weighted sums, no activation): the rebuilt beat

Trained (train.py) on synthetic normal beats only (beats.ts in the package; the TS process `data` wrote them) to rebuild
its input: the loss is the mean squared error over the 140 points (mse()). The anomaly score of a beat is its root:
the typical size of the error, sqrt(mean over the 140 points of (x - x_rebuilt)^2) (score()), in the beat's own
units. Submodule names become layer ids and tensor names in spec.json.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N, H, CODE = 140, 32, 4   # CODE: the validation AUC screen (train.py --code 2, 4, 8, 16) chose 4
KINDS = ["normal", "noP", "wideQRS", "stUp", "noisy"]


class AE(nn.Module):
    def __init__(self, n: int = N, h: int = H, code: int = CODE):
        super().__init__()
        self.enc1 = nn.Linear(n, h)
        self.code = nn.Linear(h, code)
        self.dec1 = nn.Linear(code, h)
        self.out = nn.Linear(h, n)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.code(F.relu(self.enc1(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(F.relu(self.dec1(z)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def mse(x: torch.Tensor, xr: torch.Tensor) -> torch.Tensor:
    """The training loss of each beat: the mean of the 140 squared errors."""
    return (x - xr).pow(2).mean(-1)


def score(x: torch.Tensor, xr: torch.Tensor) -> torch.Tensor:
    """The anomaly score of each beat: the root of the mean squared error (the typical size of the error)."""
    return mse(x, xr).sqrt()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = AE()
    print(m, n_params(m), "params")
