"""vae-2d: the PyTorch model (kind A): a small variational autoencoder (VAE) with a 2-D latent space.

    encoder   784 pixels -> 256 -> 64 (weighted sums + ReLU) -> mu [2] and logvar [2] (two separate weighted sums)
    sample    z = mu + exp(0.5 * logvar) * eps, with eps drawn from N(0, 1)   (the reparameterization trick)
    decoder   z [2] -> 64 -> 256 (weighted sums + ReLU) -> 784 scores -> 28x28 -> sigmoid (ink 0..1)

Input: a 28x28 drawing, white strokes on black, value = pixel / 255 (TinyCNN's scaling; the page uses the same
preprocessStrokes). Submodule names become layer ids and tensor names in spec.json (export.py).
The loss (train.py): binary cross-entropy summed over the 784 pixels + KL(N(mu, sigma^2) || N(0, 1)) summed over the 2
latent numbers, averaged over the batch (the standard VAE loss, beta = 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Five visually distinct Quick, Draw! classes (screen.py measured how well a 2-D VAE separates candidate sets).
CLASSES = ["star", "ladder", "umbrella", "eyeglasses", "t-shirt"]
ZH = {"star": "星星", "donut": "甜甜圈", "ladder": "梯子", "lightning": "闪电", "t-shirt": "T 恤",
      "house": "房子", "sun": "太阳", "fish": "鱼", "envelope": "信封", "moon": "月亮", "cup": "杯子",
      "umbrella": "雨伞", "key": "钥匙", "eyeglasses": "眼镜", "apple": "苹果", "cloud": "云"}
H1, H2, ZDIM = 256, 64, 2


class VAE(nn.Module):
    def __init__(self, h1: int = H1, h2: int = H2, zdim: int = ZDIM):
        super().__init__()
        self.enc1 = nn.Linear(784, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.mu = nn.Linear(h2, zdim)
        self.logvar = nn.Linear(h2, zdim)
        self.dec1 = nn.Linear(zdim, h2)
        self.dec2 = nn.Linear(h2, h1)
        self.dec3 = nn.Linear(h1, 784)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc1(x.flatten(1)))
        h = F.relu(self.enc2(h))
        return self.mu(h), self.logvar(h)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return self.dec3(h).view(-1, 1, 28, 28)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * eps
        return self.decode(z)


def losses(logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-drawing reconstruction loss (BCE summed over 784 pixels) and KL (summed over the latent numbers)."""
    rec = F.binary_cross_entropy_with_logits(logits.flatten(1), x.flatten(1), reduction="none").sum(1)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
    return rec, kl


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = VAE()
    print(m, n_params(m), "params")
