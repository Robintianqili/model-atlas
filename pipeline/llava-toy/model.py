"""llava-toy: a LLaVA-style connector toy (kind A).

Three parts, in the order the page shows them:

1. The frozen TinyViT (pipeline/tinyvit/model.py and model_final.pt, read only; never trained here). Its residual stream
   after block 3 of 4 (the second-to-last block, as LLaVA reads CLIP's second-to-last layer), without the [CLS] row,
   gives 16 image features of 48 numbers, one per 7x7 patch of the drawing.
2. The projector, LLaVA-1.5's "mlp2x_gelu": Linear 48 -> 64, exact GELU, Linear 64 -> 64. It turns each image feature
   into an image token as wide as the language model's word rows.
3. A tiny decoder-only language model (word level, 20 ids, d 64, 2 pre-norm blocks of 4 causal heads, MLP 128 with
   exact GELU, learned positions for 24 rows). It reads the 16 image tokens followed by the question's word rows; the
   last row predicts the one-word answer.

Module names are the layer ids export.py writes (proj.fc1, lm.tok, lm.blocks[i] -> "lm.L{i+1}.*", ...).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def _tinyvit_module():
    """pipeline/tinyvit/model.py (read only), loaded under its own name so it never shadows this file."""
    spec = importlib.util.spec_from_file_location("llava_toy_tinyvit_model", ROOT / "pipeline" / "tinyvit" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


TV = _tinyvit_module()
CLASSES = TV.CLASSES          # TinyCNN's 10 classes (TinyViT's)
ZH = TV.ZH
VIT_BLOCKS = 3                # blocks of TinyViT read: 1..3 of 4 (the second-to-last block's output)
N_IMG = TV.GRID * TV.GRID     # 16 image tokens
D_VIT = TV.D                  # 48

# the language model's words. Id 0, <image>, is the placeholder of the image tokens: the page's sequence starts with 16 of them
# and their word rows are replaced by the image tokens (as LLaVA replaces its <image> placeholder). In training batches id 0
# also fills the end of the 4-word questions, after the answer row, where the causal mask keeps every read row from it.
WORDS = ["<image>", "what", "is", "this", "it", "a", "an", "?", "yes", "no"]
VOCAB = WORDS + CLASSES
ID = {w: i for i, w in enumerate(VOCAB)}
VOWEL = {"apple", "umbrella"}  # "an apple", "an umbrella"
D, HEADS, BLOCKS, MLP, MAX_WORDS = 64, 4, 2, 128, 8
MAX_T = N_IMG + MAX_WORDS     # 24 rows


def article(c: str) -> str:
    return "an" if c in VOWEL else "a"


def load_vit() -> nn.Module:
    """The trained TinyViT on the CPU, frozen (requires_grad off), in eval mode."""
    vit = TV.TinyViT()
    vit.load_state_dict(torch.load(ROOT / "pipeline" / "tinyvit" / "model_final.pt", map_location="cpu"))
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit


def image_features(vit: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """x [B, 1, 28, 28] -> [B, 16, 48]: TinyViT's rows after block 3, the [CLS] row dropped."""
    h = vit.tokens(x)
    for b in vit.blocks[:VIT_BLOCKS]:
        h = b(h)
    return h[:, 1:]


class Projector(nn.Module):
    """LLaVA-1.5's two-layer MLP: 48 -> 64 -> GELU -> 64."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_VIT, D)
        self.fc2 = nn.Linear(D, D)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(f)))


class Block(nn.Module):
    """Pre-norm decoder block with causal attention: x + attn(ln1(x)), then x + mlp(ln2(x))."""

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

    def forward(self, x: torch.Tensor, keep: list | None = None) -> torch.Tensor:
        B, n, _ = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, n, HEADS, D // HEADS).transpose(1, 2)
        k = self.k(h).view(B, n, HEADS, D // HEADS).transpose(1, 2)
        v = self.v(h).view(B, n, HEADS, D // HEADS).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) * (D // HEADS) ** -0.5
        mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
        w = s.masked_fill(mask, float("-inf")).softmax(-1)
        if keep is not None:
            keep.append(w)
        a = (w @ v).transpose(1, 2).reshape(B, n, D)
        x = x + self.proj(a)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), D)
        self.pos = nn.Parameter(torch.zeros(MAX_T, D))
        self.blocks = nn.ModuleList([Block() for _ in range(BLOCKS)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, len(VOCAB))
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, img: torch.Tensor, ids: torch.Tensor, keep: list | None = None) -> torch.Tensor:
        """img [B, 16, 64] image tokens, ids [B, L] -> logits [B, 16 + L, V]."""
        h = torch.cat([img, self.tok(ids)], 1)
        h = h + self.pos[: h.shape[1]]
        for b in self.blocks:
            h = b(h, keep)
        return self.head(self.ln(h))


class LlavaToy(nn.Module):
    """The trainable part (projector + LM); the frozen TinyViT is passed in and never saved with it."""

    def __init__(self):
        super().__init__()
        self.proj = Projector()
        self.lm = TinyLM()
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor, ids: torch.Tensor, keep: list | None = None) -> torch.Tensor:
        return self.lm(self.proj(feats), ids, keep)


class Full(nn.Module):
    """drawing + ids (16 <image> placeholders, then the question) -> next-word shares of every row (what export.py
    records), for the AK103 check. Replacing the placeholders' word rows by the image tokens is the same as joining them."""

    def __init__(self, vit: nn.Module, m: LlavaToy):
        super().__init__()
        self.vit, self.m = vit, m

    def forward(self, x):
        img, ids = x
        assert bool((ids[:, :N_IMG] == 0).all())
        return self.m(image_features(self.vit, img), ids[:, N_IMG:]).softmax(-1)


def encode(words: list[str]) -> list[int]:
    return [ID[w] for w in words]


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    vit, m = load_vit(), LlavaToy()
    ids = torch.tensor([encode("what is this ?".split())])
    print(n_params(m), "trainable params;", sum(p.numel() for b in vit.blocks[:VIT_BLOCKS] for p in b.parameters()), "in the 3 TinyViT blocks read")
    print(Full(vit, m)((torch.zeros(1, 1, 28, 28), ids)).shape)
