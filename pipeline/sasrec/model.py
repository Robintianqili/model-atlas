"""sasrec: a tiny SASRec (Kang & McAuley 2018) in PyTorch: self-attention over a click history, one block, one head.

    ids [T]            the products clicked, in order (T = 1 … 12)
    item, pos          a row of the item table (31 × 16) for each click + a row of the position table (12 × 16)
    block (pre-norm)   x = x + attention(LayerNorm(x)), causal: position t reads only clicks 0 … t
                       x = x + FFN(LayerNorm(x)), FFN = linear 16 -> 16, ReLU, linear 16 -> 16 (point-wise, as in the paper)
    ln                 a last LayerNorm
    logits [T, 31]     row t · every row of the item table (the same table: weights shared): the score of each product as
                       the click after click t; softmax over the products gives shares.

No user id: the history is the user. Trained with a full softmax over the 31 products at every position (the paper used
one sampled negative and a binary loss; with 31 products the full softmax is cheap and its shares can be read).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ITEMS, DIM, MAX_LEN, FFN = 31, 16, 12, 16


class SASRec(nn.Module):
    def __init__(self, n_items: int = N_ITEMS, d: int = DIM, max_len: int = MAX_LEN, ffn: int = FFN, dropout: float = 0.1):
        super().__init__()
        self.item = nn.Embedding(n_items, d)
        self.pos = nn.Embedding(max_len, d)
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.d = d

    def attention(self, ids: torch.Tensor):
        """[B, T] ids -> (x after the embeddings, attention weights [B, T, T])"""
        T = ids.shape[1]
        x = self.drop(self.item(ids) + self.pos(torch.arange(T, device=ids.device)))
        h = self.ln1(x)
        s = self.q(h) @ self.k(h).transpose(1, 2) / math.sqrt(self.d)
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1), float("-inf"))
        return x, h, s.softmax(-1)

    def hidden(self, ids: torch.Tensor) -> torch.Tensor:
        x, h, w = self.attention(ids)
        x = x + self.drop(w) @ self.v(h)
        x = x + self.fc2(self.drop(F.relu(self.fc1(self.ln2(x)))))
        return self.ln(x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, T] ids -> [B, T, N_ITEMS] scores"""
        return self.hidden(ids) @ self.item.weight.t()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
