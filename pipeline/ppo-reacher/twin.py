"""ppo-reacher: the PyTorch float64 twin of trainer 'ppo' (kind B parity, design §6.2).

`npm run atlas -- parity ppo-reacher` runs the TS trainer in fine detail and writes its initial weights and every
minibatch it trained on (obs [n, 6], act [n, 2], logpOld [n], adv [n], ret [n], eps [1], ...) to
site/.atlas/ppo-reacher/train_fixture.ppo.<optimizer>.json. TS is the source of truth for the episodes (played with the
TS policy and the TS arm; the arm itself is checked by reference.py). This twin trains on the same minibatches in float64
and must reach the same losses, step-1 gradients and final weights: plain SGD for 50 steps within 1e-10 relative, Adam
within 1e-7.

The loss, as trainer.ts writes it:
    -mean(min(r * A, clip(r, 1 - eps, 1 + eps) * A)) + vcoef * mean((V(s) - R)^2),   r = exp(log pi(a|s) - logpOld)
log pi of the diagonal Gaussian: sum over the two torques of -0.5 * ((a - mu) / sigma)^2 - log sigma - 0.5 * log(2 pi).
clip and min use masks taken from the forward values (inside [1 - eps, 1 + eps]; which term the min picks, ties to r * A),
as the TS loss does, so the gradients agree at the kinks too.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

LOG2PI = math.log(2 * math.pi)


class ActorCritic(nn.Module):
    """TS ActorCritic: MLP('pi', [6, hidden, 2], 'tanh'), logstd [2], MLP('v', [6, hidden, 1], 'tanh')."""

    def __init__(self, hidden: int, logstd0: float):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.logstd = nn.Parameter(torch.full((2,), float(logstd0)))
        self.v = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, 1))


def build(cfg: dict) -> nn.Module:
    return ActorCritic(int(cfg.get("hidden", 32)), float(cfg.get("logStd0", -0.5))).double()


def loss(model: ActorCritic, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    x, act, A, R = batch["obs"], batch["act"], batch["adv"], batch["ret"]
    eps = float(batch["eps"].reshape(-1)[0])
    n = A.shape[0]
    mu = model.pi(x)
    z = (act - mu) / torch.exp(model.logstd)
    logp = torch.sum(z * z * -0.5 - model.logstd - 0.5 * LOG2PI, dim=1)
    ratio = torch.exp(logp - batch["logpOld"])
    r = ratio.detach()
    lo, hi = 1 - eps, 1 + eps
    inside = ((r >= lo) & (r <= hi)).double()
    fixed = torch.where(r < lo, torch.full_like(r, lo), torch.where(r > hi, torch.full_like(r, hi), torch.zeros_like(r)))
    c = torch.where(inside > 0, r, fixed)
    pick = (r * A <= c * A).double()
    surr1 = ratio * A
    surr2 = (ratio * inside + fixed) * A
    obj = surr1 * pick + surr2 * (1 - pick)
    pg = -torch.mean(obj)
    V = model.v(x).reshape(n)
    vf = torch.mean((V - R) * (V - R))
    return pg + vf * float(cfg.get("vcoef", 0.5)), {}
