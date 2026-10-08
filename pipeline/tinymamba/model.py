"""tinymamba: a tiny selective state-space model (Mamba-style), written out in plain PyTorch.

Per block, for the block input x [T, d]:
    z     = LayerNorm(x)                        [T, d]   each row rescaled to mean 0 and spread 1 (pre-norm, as Mamba)
    u     = silu(W_in z + b_in)                 [T, D]   the D channels the scan works on
    dt    = W_dt u + b_dt                       [T, D]
    delta = softplus(dt)                        [T, D]   one step size per channel: how much to forget and to write
    B     = W_B u                               [T, N]   how strongly the token is written into each of the N slots
    C     = W_C u                               [T, N]   how strongly each slot is read
    A     = -exp(A_log)                         [D, N]   learned decay rates, always negative
    h_t   = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t     (per channel d and slot n; h before the first token = 0)
    y_t   = sum_n C_t[n] * h_t[d, n] + Dskip[d] * u_t[d]
    out   = x + W_out y + b_out                 (a residual connection)
After the blocks a last LayerNorm and a linear head give the scores of the 10 tokens at every position.

Differences from Mamba (Gu and Dao, 2023): no causal convolution before the scan, no gate branch, a full-rank delta
projection, LayerNorm instead of RMSNorm. The scan inside each block is the ssm_scan op (ops_ssm_scan.py, ops/scan.ts).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {"vocab": 10, "d": 16, "D": 16, "N": 4, "layers": 2}


class Block(nn.Module):
    def __init__(self, d: int, D: int, N: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(d, D)
        self.dt = nn.Linear(D, D)
        self.B = nn.Linear(D, N, bias=False)
        self.C = nn.Linear(D, N, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, N + 1, dtype=torch.float32)).repeat(D, 1))
        self.Dskip = nn.Parameter(torch.ones(D))
        self.out = nn.Linear(D, d)
        with torch.no_grad():  # Mamba's dt init: softplus(b_dt) spread log-uniformly over [1e-3, 1e-1]
            dt = torch.exp(torch.rand(D) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            self.dt.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def parts(self, x: torch.Tensor):
        u = F.silu(self.inp(self.norm(x)))
        return u, self.dt(u), self.B(u), self.C(u)

    def scan(self, u, dt, Bm, Cm, keep: bool = False):
        """Sequential scan over the time axis (-2). Returns y [..., T, D] (and h [..., T, D, N] when keep)."""
        delta = F.softplus(dt)
        A = -torch.exp(self.A_log)
        h = torch.zeros(*u.shape[:-2], u.shape[-1], A.shape[1], dtype=u.dtype, device=u.device)
        ys, hs = [], []
        for t in range(u.shape[-2]):
            dl = delta[..., t, :, None]
            h = torch.exp(dl * A) * h + dl * Bm[..., t, None, :] * u[..., t, :, None]
            ys.append((h * Cm[..., t, None, :]).sum(-1) + self.Dskip * u[..., t, :])
            if keep:
                hs.append(h)
        y = torch.stack(ys, -2)
        return (y, torch.stack(hs, -3)) if keep else y

    def forward(self, x):
        return x + self.out(self.scan(*self.parts(x)))


class TinyMamba(nn.Module):
    def __init__(self, vocab: int = 10, d: int = 16, D: int = 16, N: int = 4, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, D, N) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))
