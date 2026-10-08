"""adversarial-tinycnn: TinyCNN (the shipped weights) and the two attacks, in PyTorch.

The same arithmetic as site/src/lib/explainers/adversarial-tinycnn/net.ts and process.ts:
  loss  L = cross-entropy of the logits with y = the model's own top class on the clean drawing
  FGSM  x' = clip(x + eps * sign(dL/dx), 0, 1)
  PGD   x_0 = x;  x_k = clip(x_{k-1} + alpha * sign(dL/dx at x_{k-1}), max(x - eps, 0), min(x + eps, 1)),  k = 1..K
        (no random start, so a run is deterministic)
Used by reference.py (the kit's process parity, float64 on the CPU) and stats.py (the measured facts).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NAMES = ["conv1_1", "conv1_2", "conv2_1", "conv2_2"]


class TinyCNN(nn.Module):
    """TinyCNN's TinyVGG (pipeline/tinycnn/common.py), written out so this file stands alone."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 8, 3)
        self.conv1_2 = nn.Conv2d(8, 8, 3)
        self.conv2_1 = nn.Conv2d(8, 16, 3)
        self.conv2_2 = nn.Conv2d(16, 16, 3)
        self.dense = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(F.relu(self.conv1_2(x)), 2)
        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(F.relu(self.conv2_2(x)), 2)
        return self.dense(x.flatten(1))


def build(weights: dict[str, np.ndarray], dtype=torch.float64, device="cpu") -> TinyCNN:
    m = TinyCNN()
    m.load_state_dict({k: torch.tensor(np.asarray(v, dtype=np.float64)).reshape(m.state_dict()[k].shape) for k, v in weights.items()})
    return m.to(device=device, dtype=dtype).eval()


def static_weights(site: str | Path | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """The weights the page loads: static/models/tinycnn/{spec.json, weights.bin} (float32), and the spec."""
    site = Path(site or os.environ.get("ATLAS_SITE") or Path(__file__).resolve().parents[2] / "site")
    d = site / "static" / "models" / "tinycnn"
    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    buf = np.frombuffer((d / "weights.bin").read_bytes(), dtype="<f4")
    return {t["name"]: buf[t["offset"]:t["offset"] + t["length"]].reshape(t["shape"]).copy() for t in spec["tensors"]}, spec


def input_grad(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(logits, per-drawing loss, dL/dx) for x [N, 1, 28, 28]; each drawing's own loss (sum over the batch)."""
    x = x.detach().clone().requires_grad_(True)
    z = m(x)
    loss = F.cross_entropy(z, y, reduction="none")
    (g,) = torch.autograd.grad(loss.sum(), x)
    return z.detach(), loss.detach(), g


def fgsm(x: torch.Tensor, g: torch.Tensor, eps) -> torch.Tensor:
    return torch.clamp(x + eps * torch.sign(g), 0, 1)


def pgd_step(x0: torch.Tensor, xk: torch.Tensor, g: torch.Tensor, eps, alpha) -> torch.Tensor:
    lo = torch.clamp(x0 - eps, min=0)
    hi = torch.clamp(x0 + eps, max=1)
    return torch.minimum(hi, torch.maximum(lo, xk + alpha * torch.sign(g)))


def pgd(m: nn.Module, x: torch.Tensor, y: torch.Tensor, eps, steps: int, alpha) -> torch.Tensor:
    xk = x
    for _ in range(steps):
        _, _, g = input_grad(m, xk, y)
        xk = pgd_step(x, xk, g, eps, alpha)
    return xk
