"""vq-vae: the PyTorch model (kind A): a tiny VQ-VAE (vector-quantized autoencoder) on 28x28 drawings.

    encoder   a 7x7 kernel with stride 7 cuts the drawing into a 4x4 grid of patches: 1 -> 64 channels (ReLU), then a
              1x1 kernel -> 8 numbers per patch, z_e [8, 4, 4]. Each patch is encoded on its own (no overlap).
    quantize  each patch's 8 numbers snap to the nearest of K = 32 codebook vectors (squared distance, ties to the
              smaller index): z_q [8, 4, 4], the code indices [4, 4] (the drawing as 16 small integers).
    decoder   a 3x3 kernel (padding 1) reads each code with its 8 neighbours: 8 -> 64 (ReLU) -> 64 (1x1, ReLU) -> 49
              (1x1) scores per cell; pixel_shuffle(7) lays the 49 scores of each cell out as its 7x7 block of the
              28x28 picture; a sigmoid turns the scores into ink between 0 and 1.

Training (train.py) passes the decoder's gradient straight through the lookup: z_q_st = z_e + (z_q - z_e).detach().
Loss = BCE averaged over the 784 pixels + BETA * commitment ||z_e - sg[z_q]||^2 averaged over the 16 x 8 numbers, mean
over the batch. The codebook is not moved by the gradient: after every batch each code moves 1% of the way towards the
average z_e of the patches that snapped to it (an exponential moving average, EMA; van den Oord et al., 2017, appendix;
Razavi et al., 2019).
Input: pixel / 255 of a Quick, Draw! bitmap (TinyCNN's scaling; the page uses the same preprocessStrokes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["star", "ladder", "umbrella", "eyeglasses", "t-shirt"]   # vae-2d's five classes
ZH = {"star": "星星", "ladder": "梯子", "umbrella": "雨伞", "eyeglasses": "眼镜", "t-shirt": "T 恤", "house": "房子", "fish": "鱼"}
P, G = 7, 4          # patch side, grid side (28 = 4 x 7)
K, D, H = 32, 8, 64  # codes, numbers per code, hidden channels
BETA = 0.25          # commitment weight
DECAY = 0.99         # EMA decay of the codebook


class VQVAE(nn.Module):
    def __init__(self, k: int = K, d: int = D, h: int = H):
        super().__init__()
        self.enc1 = nn.Conv2d(1, h, P, stride=P)
        self.enc2 = nn.Conv2d(h, d, 1)
        self.codebook = nn.Parameter(torch.zeros(k, d))
        self.dec1 = nn.Conv2d(d, h, 3, padding=1)
        self.dec2 = nn.Conv2d(h, h, 1)
        self.dec3 = nn.Conv2d(h, P * P, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc2(F.relu(self.enc1(x)))

    def quantize(self, ze: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z_q [B, D, G, G], indices [B, G, G], squared distances [B, K, G, G] (explicit differences, as the TS op)."""
        dist = ((ze[:, None] - self.codebook[None, :, :, None, None]) ** 2).sum(2)
        idx = dist.argmin(1)
        zq = self.codebook[idx].permute(0, 3, 1, 2)
        return zq, idx, dist

    def decode_logits(self, zq: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(zq))
        h = F.relu(self.dec2(h))
        return F.pixel_shuffle(self.dec3(h), P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zq, _, _ = self.quantize(self.encode(x))
        return torch.sigmoid(self.decode_logits(zq))


def losses(m: VQVAE, x: torch.Tensor):
    """Per-drawing reconstruction BCE (mean over the 784 pixels) and commitment term (mean over 16 x 8 numbers), with the
    straight-through estimator; also z_e and the indices (for the EMA codebook update)."""
    ze = m.encode(x)
    zq, idx, _ = m.quantize(ze)
    logits = m.decode_logits(ze + (zq - ze).detach())
    rec = F.binary_cross_entropy_with_logits(logits.flatten(1), x.flatten(1), reduction="none").mean(1)
    commit = ((ze - zq.detach()) ** 2).flatten(1).mean(1)
    return rec, commit, ze, idx


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = VQVAE()
    print(m, n_params(m), "params")
