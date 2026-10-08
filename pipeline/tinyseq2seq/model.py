"""tinyseq2seq: a GRU encoder-decoder with Bahdanau (additive) attention (kind A).

    source characters -> src_emb -> enc (nn.GRU, reads left to right) -> encoder states h_0 .. h_{T-1}
    decoder state s starts as the last encoder state h_{T-1}; the first input symbol is <s>
    each decoder step t (Bahdanau et al. 2015, with s_{t-1} the previous decoder state):
        key_j   = W_k h_j + b_k                          (att_k, computed once per input)
        query_t = W_q s_{t-1}                            (att_q)
        e_tj    = v . tanh(query_t + key_j)              (att_v: the attention score of source position j)
        a_t     = softmax_j(e_t)                         (attention weights, add up to 1)
        c_t     = sum_j a_tj h_j                         (context vector)
        s_t     = GRUCell([tgt_emb(y_{t-1}); c_t], s_{t-1})
        logits  = W_o [s_t; c_t] + b_o;  y_t = argmax (greedy)
With attention=False (the bottleneck baseline) every c_t is the last encoder state h_{T-1} (Cho et al. 2014): the
decoder sees the input only through those numbers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

BOS, EOS, STEPS = 0, 1, 11


class Seq2Seq(nn.Module):
    def __init__(self, n_src: int, n_tgt: int, emb: int = 16, hidden: int = 32, attn: int = 32, attention: bool = True):
        super().__init__()
        self.attention = attention
        self.src_emb = nn.Embedding(n_src, emb, padding_idx=0)
        self.enc = nn.GRU(emb, hidden, batch_first=True)
        self.tgt_emb = nn.Embedding(n_tgt, emb)
        self.att_q = nn.Linear(hidden, attn, bias=False)
        self.att_k = nn.Linear(hidden, attn)
        self.att_v = nn.Linear(attn, 1, bias=False)
        self.dec = nn.GRUCell(emb + hidden, hidden)
        self.out = nn.Linear(2 * hidden, n_tgt)

    def encode(self, src: torch.Tensor, lengths: torch.Tensor):
        """src [B, T] (0 = padding) -> states [B, T, H] (zeros after the end), last state [B, H]."""
        packed = pack_padded_sequence(self.src_emb(src), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h_n = self.enc(packed)
        states, _ = pad_packed_sequence(out, batch_first=True, total_length=src.shape[1])
        return states, h_n[0]

    def step(self, y_prev, s, states, keys, mask):
        """One decoder step for a batch: returns logits, new state, attention weights [B, T]."""
        if self.attention:
            e = self.att_v(torch.tanh(self.att_q(s)[:, None, :] + keys))[..., 0]   # [B, T]
            e = e.masked_fill(~mask, float("-inf"))
            a = torch.softmax(e, dim=-1)
            c = torch.bmm(a[:, None, :], states)[:, 0]
        else:
            a = torch.zeros(mask.shape, dtype=s.dtype, device=s.device)
            c = self.last
        s = self.dec(torch.cat([self.tgt_emb(y_prev), c], dim=-1), s)
        return self.out(torch.cat([s, c], dim=-1)), s, a

    def forward(self, src, lengths, y_in):
        """Teacher forcing: y_in [B, L] (starts with <s>) -> logits [B, L, V]."""
        states, s = self.encode(src, lengths)
        self.last = s
        keys = self.att_k(states)
        mask = torch.arange(src.shape[1], device=src.device)[None, :] < lengths[:, None].to(src.device)
        outs = []
        for t in range(y_in.shape[1]):
            logits, s, _ = self.step(y_in[:, t], s, states, keys, mask)
            outs.append(logits)
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def greedy(self, src, lengths, steps: int = STEPS):
        """Greedy decoding: ids [B, steps] and attention weights [B, steps, T]."""
        states, s = self.encode(src, lengths)
        self.last = s
        keys = self.att_k(states)
        mask = torch.arange(src.shape[1], device=src.device)[None, :] < lengths[:, None].to(src.device)
        y = torch.full((src.shape[0],), BOS, dtype=torch.long, device=src.device)
        ids, attn = [], []
        for _ in range(steps):
            logits, s, a = self.step(y, s, states, keys, mask)
            y = logits.argmax(-1)
            ids.append(y)
            attn.append(a)
        return torch.stack(ids, 1), torch.stack(attn, 1)


class GreedyProbs(nn.Module):
    """For export (AK103): one unbatched input ids [T] -> the greedy decoder's probabilities [STEPS, V]."""

    def __init__(self, m: Seq2Seq):
        super().__init__()
        self.m = m

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        m = self.m
        src = ids[None, :]
        lengths = torch.tensor([ids.shape[0]])
        states, s = m.encode(src, lengths)
        m.last = s
        keys = m.att_k(states)
        mask = torch.ones(1, ids.shape[0], dtype=torch.bool)
        y = torch.tensor([BOS])
        probs = []
        for _ in range(STEPS):
            logits, s, _ = m.step(y, s, states, keys, mask)
            probs.append(torch.softmax(logits[0], -1))
            y = logits.argmax(-1)
        return torch.stack(probs)
