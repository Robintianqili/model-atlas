"""pinn: the PyTorch float64 twin of trainers 'plain' and 'pinn' (kind B parity, design §4.11).

`npm run atlas -- parity pinn` writes the TS init params and batches (the points and the 200 physics times: TS is the
source of truth for data) and runs `python -m atlas_kit.train_reference` with this twin, for every train source.
State_dict names equal the TS names: MLP('net', [1, 32, 32, 1], 'tanh') -> net.0.*, net.2.*, net.4.*.

loss = data + lam * physics (if physics) + start * u'(0)^2 / k (if start > 0), as site/src/lib/explainers/pinn/trainer.ts:
  data    = mean (u(t_i) - u_i)^2 over the points
  physics = mean ((u'' + c u' + k u) / k)^2 over the 200 times, u' and u'' by torch.autograd.grad(create_graph=True)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def build(cfg: dict) -> nn.Module:
    return Net(int(cfg.get("hidden", 32))).double()


def derivs(model: nn.Module, t: torch.Tensor):
    """u, u' and u'' at the times t ([n, 1]), autograd twice with create_graph (the TS derivs())."""
    t = t.detach().clone().requires_grad_(True)
    u = model(t)
    du = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, t, torch.ones_like(du), create_graph=True)[0]
    return u, du, d2u


def residual(u, du, d2u, c: float, k: float) -> torch.Tensor:
    return (d2u + c * du + k * u) * (1 / k)


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'to': [n, 1], 'uo': [n, 1], 'tc': [200, 1]} float64."""
    to, uo = batch["to"], batch["uo"]
    data = torch.mean((model(to) - uo) ** 2) if to.numel() else 0 * model.net[0].weight.sum()
    parts = {"data": data}
    total = data
    c, k = float(cfg.get("c", 3.6)), float(cfg.get("k", 144.0))
    if cfg.get("physics", True):
        u, du, d2u = derivs(model, batch["tc"])
        parts["physics"] = torch.mean(residual(u, du, d2u, c, k) ** 2)
        total = total + parts["physics"] * float(cfg.get("lam", 1.0))
    start = float(cfg.get("start", 1.0))
    if start > 0:
        t0 = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
        u0 = model(t0)
        du0 = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
        parts["start"] = torch.sum(du0 ** 2) * (1 / k)
        total = total + parts["start"] * start
    return total, parts
