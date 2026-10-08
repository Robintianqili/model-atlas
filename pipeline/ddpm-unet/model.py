"""ddpm-unet: the PyTorch model (kind A): a tiny DDPM noise predictor, a small U-Net with a timestep embedding.

    schedule   T = 100 steps; beta_t rises linearly (DDPM's linear schedule, 1e-4 .. 0.02 for 1,000 steps, scaled by
               1000 / T as in Nichol & Dhariwal 2021); alpha_t = 1 - beta_t; abar_t = alpha_1 * ... * alpha_t
    pixels     the drawing's 0 (paper) .. 1 (ink) are stretched to -1 .. 1 (2 x - 1), centred like the noise; a sample
               is drawn back as (x + 1) / 2
    noising    x_t = sqrt(abar_t) * (2 x_0 - 1) + sqrt(1 - abar_t) * eps  (q_sample: the forward process in one jump)
    network    eps_hat = UNet(x_t, t): the noise it thinks is in x_t
    sampling   x_{t-1} = (x_t - beta_t / sqrt(1 - abar_t) * eps_hat) / sqrt(alpha_t) + sigma_t * z,  sigma_t = sqrt(beta_t)

Pixels: a 28x28 drawing, white strokes on black, value = pixel / 255 (TinyCNN's scaling; the page uses the same
preprocessStrokes): 0 is paper, 1 is ink. The U-Net (channels C1, C2, C3 at 28, 14 and 7 pixels):

    temb   t -> sin/cos of t at D/2 frequencies (time_embed) -> Linear(D, E) -> SiLU      one vector for the whole net
    inc    conv 1 -> C1 (28x28), SiLU                                                     skip A
    down   maxpool -> conv C1 -> C2 (14x14) + time bias, SiLU                               skip B
    mid    maxpool -> conv C2 -> C3 (7x7) + time bias, SiLU -> conv C3 -> C2, SiLU
    up     upsample -> concat skip B -> conv 2*C2 -> C1 (14x14) + time bias, SiLU
           upsample -> concat skip A -> conv 2*C1 -> C1 (28x28), SiLU -> conv C1 -> 1: eps_hat
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["star", "umbrella", "house"]
ZH = {"star": "星星", "umbrella": "雨伞", "house": "房子", "sun": "太阳", "apple": "苹果", "tree": "树", "fish": "鱼",
      "clock": "时钟", "car": "汽车", "bicycle": "自行车", "moon": "月亮", "donut": "甜甜圈", "cup": "杯子"}
T = 100
CH = (16, 32, 64)
D, E = 16, 32


def schedule(t_steps: int = T) -> dict:
    """float64 arrays indexed by t - 1 (t = 1..T): beta, alpha, abar (the page computes the same in TS, ddpm.ts)."""
    scale = 1000 / t_steps
    beta = np.linspace(1e-4 * scale, 0.02 * scale, t_steps, dtype=np.float64)
    alpha = 1.0 - beta
    return {"beta": beta, "alpha": alpha, "abar": np.cumprod(alpha)}


def time_embed(t: torch.Tensor, d: int = D) -> torch.Tensor:
    """[B] timesteps -> [B, d]: sin(t * f_i) then cos(t * f_i), f_i = 10000^(-i / (d/2)) (float64, then float32)."""
    half = d // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64, device=t.device) / half)
    a = t.to(torch.float64)[:, None] * f[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1).float()


def add_time(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] + [B, C]: the same number added to every pixel of a channel (add_channels)."""
    return h + v[:, :, None, None]


class UNet(nn.Module):
    def __init__(self, ch=CH, d: int = D, e: int = E):
        super().__init__()
        c1, c2, c3 = ch
        self.d = d
        self.tmlp = nn.Linear(d, e)
        self.inc = nn.Conv2d(1, c1, 3, padding=1)
        self.down = nn.Conv2d(c1, c2, 3, padding=1)
        self.down_t = nn.Linear(e, c2)
        self.mid1 = nn.Conv2d(c2, c3, 3, padding=1)
        self.mid1_t = nn.Linear(e, c3)
        self.mid2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.up2 = nn.Conv2d(2 * c2, c1, 3, padding=1)
        self.up2_t = nn.Linear(e, c1)
        self.up1 = nn.Conv2d(2 * c1, c1, 3, padding=1)
        self.out = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = F.silu(self.tmlp(time_embed(t, self.d)))
        a = F.silu(self.inc(xt))
        b = F.silu(add_time(self.down(F.max_pool2d(a, 2)), self.down_t(te)))
        h = F.silu(add_time(self.mid1(F.max_pool2d(b, 2)), self.mid1_t(te)))
        h = F.silu(self.mid2(h))
        h = torch.cat([F.interpolate(h, scale_factor=2, mode="nearest"), b], dim=1)
        h = F.silu(add_time(self.up2(h), self.up2_t(te)))
        h = torch.cat([F.interpolate(h, scale_factor=2, mode="nearest"), a], dim=1)
        h = F.silu(self.up1(h))
        return self.out(h)


class Noised(nn.Module):
    """What the page traces: (x_0, eps, t) -> x_t -> eps_hat (export.py's module for AK103)."""

    def __init__(self, unet: UNet, t_steps: int = T):
        super().__init__()
        self.unet = unet
        self.register_buffer("abar", torch.tensor(schedule(t_steps)["abar"], dtype=torch.float32))

    def forward(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ab = self.abar[t.long() - 1][:, None, None, None]
        xt = torch.sqrt(ab) * (2 * x0 - 1) + torch.sqrt(1 - ab) * eps
        return self.unet(xt, t)


@torch.no_grad()
def sample(unet: UNet, x: torch.Tensor, z: list, steps: list[int], t_steps: int = T) -> torch.Tensor:
    """Ancestral DDPM sampling on the timesteps `steps` (descending, ending at 1), respaced as in Nichol & Dhariwal 2021:
    abar' of the kept steps, beta'_i = 1 - abar_i / abar_prev. x: [B, 1, 28, 28] noise; z[i]: the noise of step i."""
    ab = schedule(t_steps)["abar"]
    for i, t in enumerate(steps):
        prev = ab[steps[i + 1] - 1] if i + 1 < len(steps) else 1.0
        a_t = ab[t - 1] / prev
        b_t = 1 - a_t
        eps = unet(x, torch.full((x.shape[0],), t, dtype=torch.float32))
        x = (x - (b_t / math.sqrt(1 - ab[t - 1])) * eps) / math.sqrt(a_t)
        if i + 1 < len(steps):
            x = x + math.sqrt(b_t) * z[i]
    return x


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def macs(ch=CH) -> int:
    """multiply-adds of the convolutions of one U-Net forward (the time MLP adds under 0.1%)."""
    c1, c2, c3 = ch
    return 9 * (784 * (c1 + 2 * c1 * c1 + c1) + 196 * (c1 * c2 + 2 * c2 * c1) + 49 * (c2 * c3 + c3 * c2))


if __name__ == "__main__":
    m = UNet()
    print(m, n_params(m), "params", macs() / 1e6, "M MACs")
