"""contrastive-learning: TinyCLIP in PyTorch (design §7): an image tower and a text tower that meet in one 32-number space.

    image tower  [1, 28, 28] -> conv 3x3 (8) -> ReLU -> max-pool 2 -> conv 3x3 (16) -> ReLU -> max-pool 2 -> flatten (400)
                 -> linear 400 -> 32 -> divide by its length (l2norm)
    text tower   word ids [L] (0 pads) -> embedding (24 x 32) -> mean of the word rows (pads skipped) -> linear 32 -> 32
                 -> divide by its length
    score        s * (image vector . caption vector), s = exp(logit_scale) learned (CLIP's temperature, capped at 100)

The word list and the scenes come from site/src/lib/explainers/contrastive-learning/scenes.ts (the TS side is the source of truth:
it renders the training drawings; VOCAB below must equal its VOCAB, which export.py writes into the spec and a claim
test compares).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = ['<pad>', 'a', 'the', 'small', 'big', 'large', 'circle', 'square', 'triangle', 'cross', 'star', 'on', 'at', 'in',
         'left', 'right', 'top', 'bottom', 'middle', 'center', 'above', 'below', 'of', 'and']
SHAPES = ['circle', 'square', 'triangle', 'cross', 'star']
PLACES = ['left', 'right', 'top', 'bottom', 'middle']
MAX_SCALE = 100.0


def tokenize(text: str) -> list[int]:
    """Known words -> ids, unknown words skipped (the same rule as scenes.ts tokenize)."""
    import re
    return [VOCAB.index(w) for w in re.findall(r"[a-z]+", text.lower()) if w in VOCAB[1:]][:8]


def mean_pool(e: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """[..., L, E] word rows and [..., L] ids -> [..., E]: the average of the rows whose id is not 0 (0 when none)."""
    m = (ids != 0).to(e.dtype).unsqueeze(-1)
    return (e * m).sum(-2) / m.sum(-2).clamp(min=1)


class TinyCLIP(nn.Module):
    def __init__(self, vocab: int = len(VOCAB), width: int = 32, dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.img_proj = nn.Linear(16 * 5 * 5, dim)
        self.emb = nn.Embedding(vocab, width, padding_idx=0)
        self.txt_proj = nn.Linear(width, dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:          # [B, 1, 28, 28] -> [B, dim], length 1
        h = F.max_pool2d(F.relu(self.conv1(x)), 2)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2)
        return F.normalize(self.img_proj(h.flatten(1)), dim=-1)

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:         # [B, L] -> [B, dim], length 1
        return F.normalize(self.txt_proj(mean_pool(self.emb(ids), ids)), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=MAX_SCALE)

    def forward(self, image: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """One drawing [1, 1, 28, 28] against K captions [K, L]: the K shares (softmax over the captions), as the page."""
        a = self.encode_image(image)[0]
        b = self.encode_text(ids)
        return (self.scale() * (b @ a)).softmax(-1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
