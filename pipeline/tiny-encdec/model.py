"""tiny-encdec: a tiny T5-style encoder-decoder Transformer for the date task of tinyseq2seq (kind A).

    source characters -> src_emb + enc_pos -> encoder blocks (pre-norm: LN, bidirectional self-attention, add; LN,
    ReLU feed-forward, add) -> enc_ln -> one vector per character (the "memory")
    decoder input [<s>, y_0 .. y_{L-2}] -> tgt_emb + dec_pos -> decoder blocks (LN, causal self-attention, add; LN,
    cross-attention: queries from the decoder, keys and values from the memory, add; LN, feed-forward, add) -> dec_ln
    -> head -> one score per output symbol at every position
T5-style: an encoder-decoder trained text to text, pre-norm blocks, no biases in the attention and feed-forward
projections, ReLU feed-forward. Unlike T5: learned absolute positions (T5 adds a relative position bias to the scores),
LayerNorm (T5 uses RMSNorm) and scores scaled by 1/sqrt(d_head) (T5 folds the scale into its initialization).
Attention is written out (q, k, v, scores, softmax, mix, o) so that every piece maps onto one kit op.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

BOS, EOS, STEPS = 0, 1, 11
T_MAX = 32


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, xq, xkv, causal: bool = False, key_ok=None):
        """xq [B, Tq, D], xkv [B, Tk, D] -> (out [B, Tq, D], weights [B, H, Tq, Tk]); key_ok [B, Tk] masks padding."""
        B, Tq, D = xq.shape
        Tk, H = xkv.shape[1], self.h
        dh = D // H
        q = self.q(xq).view(B, Tq, H, dh).transpose(1, 2)
        k = self.k(xkv).view(B, Tk, H, dh).transpose(1, 2)
        v = self.v(xkv).view(B, Tk, H, dh).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) * (1 / math.sqrt(dh))
        if causal:
            s = s.masked_fill(torch.ones(Tq, Tk, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
        if key_ok is not None:
            s = s.masked_fill(~key_ok[:, None, None, :], float("-inf"))
        w = torch.softmax(s, dim=-1)
        return self.o((w @ v).transpose(1, 2).reshape(B, Tq, D)), w


class FF(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.up, self.down = nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


class EncBlock(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1, self.sa, self.ln2, self.ff = nn.LayerNorm(d), Attention(d, heads), nn.LayerNorm(d), FF(d, ff)

    def forward(self, x, key_ok=None):
        h = self.ln1(x)
        a, w = self.sa(h, h, key_ok=key_ok)
        x = x + a
        return x + self.ff(self.ln2(x)), w


class DecBlock(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.ln1, self.sa = nn.LayerNorm(d), Attention(d, heads)
        self.ln2, self.ca = nn.LayerNorm(d), Attention(d, heads)
        self.ln3, self.ff = nn.LayerNorm(d), FF(d, ff)

    def forward(self, y, mem, key_ok=None):
        h = self.ln1(y)
        a, ws = self.sa(h, h, causal=True)
        y = y + a
        c, wc = self.ca(self.ln2(y), mem, key_ok=key_ok)
        y = y + c
        return y + self.ff(self.ln3(y)), ws, wc


class EncDec(nn.Module):
    def __init__(self, n_src: int, n_tgt: int, d: int = 32, heads: int = 2, ff: int = 64, n_enc: int = 1, n_dec: int = 1):
        super().__init__()
        self.src_emb = nn.Embedding(n_src, d)
        self.enc_pos = nn.Embedding(T_MAX, d)
        self.enc = nn.ModuleList(EncBlock(d, heads, ff) for _ in range(n_enc))
        self.enc_ln = nn.LayerNorm(d)
        self.tgt_emb = nn.Embedding(n_tgt, d)
        self.dec_pos = nn.Embedding(STEPS, d)
        self.dec = nn.ModuleList(DecBlock(d, heads, ff) for _ in range(n_dec))
        self.dec_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_tgt, bias=False)

    def encode(self, src, key_ok=None):
        """src [B, T] -> memory [B, T, D] and the self-attention weights of every block."""
        x = self.src_emb(src) + self.enc_pos(torch.arange(src.shape[1], device=src.device))
        ws = []
        for b in self.enc:
            x, w = b(x, key_ok)
            ws.append(w)
        return self.enc_ln(x), ws

    def decode(self, y_in, mem, key_ok=None):
        """y_in [B, L] (starts with <s>) -> logits [B, L, V], self- and cross-attention weights of every block."""
        y = self.tgt_emb(y_in) + self.dec_pos(torch.arange(y_in.shape[1], device=y_in.device))
        ws, wc = [], []
        for b in self.dec:
            y, a, c = b(y, mem, key_ok)
            ws.append(a)
            wc.append(c)
        return self.head(self.dec_ln(y)), ws, wc

    def forward(self, src, key_ok, y_in):
        """Teacher forcing: logits [B, L, V]."""
        mem, _ = self.encode(src, key_ok)
        return self.decode(y_in, mem, key_ok)[0]

    @torch.no_grad()
    def greedy(self, src, key_ok=None, steps: int = STEPS):
        """Greedy decoding: the decoder runs once per symbol on everything written so far. Returns the symbols
        [B, steps] and the decoder input [B, steps] (<s> then the first steps - 1 symbols)."""
        mem, _ = self.encode(src, key_ok)
        y_in = torch.full((src.shape[0], 1), BOS, dtype=torch.long, device=src.device)
        out = []
        for t in range(steps):
            logits = self.decode(y_in, mem, key_ok)[0][:, -1]
            out.append(logits.argmax(-1))
            if t < steps - 1:
                y_in = torch.cat([y_in, out[-1][:, None]], dim=1)
        return torch.stack(out, 1), y_in


class GreedyProbs(nn.Module):
    """For export (AK103): unbatched character ids [T] -> the chances of every symbol at every step [STEPS, V]: greedy
    decoding, then one pass of the decoder over <s> and the symbols it wrote (the causal mask makes row t of that pass
    equal to step t of the decoding)."""

    def __init__(self, m: EncDec):
        super().__init__()
        self.m = m

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        _, y_in = self.m.greedy(ids[None])
        mem, _ = self.m.encode(ids[None])
        return torch.softmax(self.m.decode(y_in, mem)[0][0], -1)
