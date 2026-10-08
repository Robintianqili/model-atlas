"""tinybert: a small BERT-style encoder trained with masked-language modelling (MLM) on TinyStories (kind A).

Built from TinyGPT's blocks on purpose, so the two pages differ only where the models differ:
  - the same byte-level BPE tokenizer (pipeline/tinygpt/bpe.py, static/models/tinygpt/tokenizer.json): ids 0..1022 are
    text, 1023 is <|endoftext|>; TinyBERT adds three special tokens: 1024 [MASK], 1025 [CLS], 1026 [SEP];
  - the same size: d 76, 4 pre-norm blocks of 4 heads (19 numbers each), MLP 76 -> 304 -> 76 with GELU (tanh form),
    at most 64 positions, the output head tied to the token table (no bias);
  - what differs: no causal mask (every position reads every position, left and right), a segment table (sentence A
    or B, as in BERT), and the training task (fill in hidden tokens, not continue the text).

Input: [CLS] sentence A [SEP] (sentence B [SEP]); segment ids 0 for [CLS], A and its [SEP], 1 for B and the last [SEP].
Unlike the original BERT (post-norm blocks, an extra dense + LayerNorm in the MLM head, next-sentence prediction), this
keeps TinyGPT's pre-norm blocks and plain tied head and trains MLM only (notes/tinybert.md, Decisions).

Module names are the layer ids export.py writes (tok, pos, seg, blocks[i] -> "B{i+1}.*", ln), so the Code tab and the
weight names line up with this file.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

TEXT_VOCAB = 1024              # TinyGPT's tokenizer: 256 bytes + 767 merges + <|endoftext|> (1023)
EOT = 1023
MASK, CLS, SEP = 1024, 1025, 1026
VOCAB = 1027
SPECIAL = {MASK: "[MASK]", CLS: "[CLS]", SEP: "[SEP]"}
D, HEADS, LAYERS, CTX, MLP = 76, 4, 4, 64, 304
DH = D // HEADS
EPS = 1e-5


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU, tanh form (as TinyGPT and GPT-2): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class Block(nn.Module):
    """Pre-norm Transformer block with NO causal mask: x + attn(ln1(x)), then x + mlp(ln2(x))."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D, eps=EPS)
        self.q = nn.Linear(D, D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D, eps=EPS)
        self.fc = nn.Linear(D, MLP)
        self.out = nn.Linear(MLP, D)

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, T, D]; pad [B, T] (True = padding, used only in training batches): padded keys get no weight."""
        B, T, _ = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, T, HEADS, DH).transpose(1, 2)
        k = self.k(h).view(B, T, HEADS, DH).transpose(1, 2)
        v = self.v(h).view(B, T, HEADS, DH).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(DH)          # [B, H, T, T]: every query against every key
        if pad is not None:
            s = s.masked_fill(pad[:, None, None, :], float("-inf"))
        w = s.softmax(-1)
        a = (w @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        return x + self.out(gelu_tanh(self.fc(self.ln2(x))))


class TinyBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D)        # one learned row per token id (text and the 3 special tokens)
        self.pos = nn.Embedding(CTX, D)          # one learned row per position 0..63
        self.seg = nn.Embedding(2, D)            # one learned row per segment: 0 = sentence A, 1 = sentence B
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D, eps=EPS)
        self.apply(self._init)
        for b in self.blocks:  # GPT-2 init, as TinyGPT: residual projections scaled by 1/sqrt(2 * layers)
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * LAYERS))
            nn.init.normal_(b.out.weight, std=0.02 / math.sqrt(2 * LAYERS))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, ids: torch.Tensor, seg: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
        """ids, seg [B, T] -> the last LayerNorm's output [B, T, D]."""
        T = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))[None] + self.seg(seg)
        for b in self.blocks:
            x = b(x, pad)
        return self.ln(x)

    def forward(self, ids: torch.Tensor, seg: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
        """Scores for every vocabulary id at every position, [B, T, 1027] (tied head: h @ tok.weight^T)."""
        return self.hidden(ids, seg, pad) @ self.tok.weight.t()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = TinyBERT()
    print(n_params(m), "params")
    ids = torch.tensor([[CLS, 300, MASK, 400, SEP]])
    print(m(ids, torch.zeros_like(ids)).shape)
