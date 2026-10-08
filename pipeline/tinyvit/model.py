"""tinyvit: a tiny Vision Transformer for 28x28 Quick, Draw! drawings (kind A).

The drawing (white strokes on black, value = pixel / 255, exactly TinyCNN's input) is cut into 16 patches of 7x7
pixels. Each patch becomes 48 numbers (a convolution whose stride equals its kernel), a learned [CLS] row is put in
front, a learned position row is added to each of the 17 rows, then 4 pre-norm Transformer blocks (4 heads of 12,
MLP 48 -> 192 -> 48 with exact GELU) mix the rows. The [CLS] row after a last LayerNorm gives the 10 class scores.

Module names are the layer ids export.py writes (patch, cls, pos, blocks[i] -> "B{i+1}.*", ln, head), so the
Code tab and the weight names line up with this file.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["apple", "bicycle", "car", "clock", "fish", "house", "star", "sun", "tree", "umbrella"]  # TinyCNN's 10
ZH = {"apple": "苹果", "bicycle": "自行车", "car": "汽车", "clock": "时钟", "fish": "鱼", "house": "房子",
      "star": "星星", "sun": "太阳", "tree": "树", "umbrella": "雨伞", "ladder": "梯子", "envelope": "信封"}
HELD_OUT = ["ladder", "envelope"]   # TinyCNN's held-out classes: presets the model was never trained on

D, HEADS, BLOCKS, MLP, PATCH = 48, 4, 4, 192, 7
GRID = 28 // PATCH            # 4 x 4 patches
T = GRID * GRID + 1           # 16 patches + [CLS]


class Block(nn.Module):
    """Pre-norm Transformer block: x + attn(ln1(x)), then x + mlp(ln2(x))."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.q = nn.Linear(D, D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, MLP)
        self.fc2 = nn.Linear(MLP, D)

    def attn_weights(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, D] (already normalized) -> attention weights [B, H, T, T]."""
        B, n, _ = h.shape
        q = self.q(h).view(B, n, HEADS, D // HEADS).transpose(1, 2)
        k = self.k(h).view(B, n, HEADS, D // HEADS).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) * (D // HEADS) ** -0.5
        return s.softmax(-1)

    def forward(self, x: torch.Tensor, keep: list | None = None) -> torch.Tensor:
        B, n, _ = x.shape
        h = self.ln1(x)
        w = self.attn_weights(h)
        if keep is not None:
            keep.append(w)
        v = self.v(h).view(B, n, HEADS, D // HEADS).transpose(1, 2)
        a = (w @ v).transpose(1, 2).reshape(B, n, D)
        x = x + self.proj(a)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyViT(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        self.patch = nn.Conv2d(1, D, PATCH, stride=PATCH)        # 49 pixels -> 48 numbers per patch
        self.cls = nn.Parameter(torch.zeros(D))                  # the learned [CLS] row
        self.pos = nn.Parameter(torch.zeros(T, D))               # one learned row per position (row 0: [CLS])
        self.blocks = nn.ModuleList([Block() for _ in range(BLOCKS)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, 1, 28, 28] -> the 17 rows that enter block 1, [B, 17, 48]."""
        p = self.patch(x).flatten(2).transpose(1, 2)              # [B, 16, 48], patch t = row-major (y, x)
        cls = self.cls.expand(x.shape[0], 1, D)
        return torch.cat([cls, p], 1) + self.pos

    def forward(self, x: torch.Tensor, keep: list | None = None) -> torch.Tensor:
        h = self.tokens(x)
        for b in self.blocks:
            h = b(h, keep)
        return self.head(self.ln(h)[:, 0])


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = TinyViT()
    print(m)
    print(n_params(m), "params")
    print(m(torch.zeros(2, 1, 28, 28)).shape)
