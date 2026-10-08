"""TinyGPT: a GPT-2-style decoder small enough that the browser can show every number.

token embedding + learned position embedding
N x [ LN -> causal multi-head self-attention -> residual add ; LN -> MLP (4x, GELU tanh) -> residual add ]
final LN -> logits through the token embedding (tied output head, no bias)

The `trace=True` path computes the same thing step by step with plain tensor ops and records every
intermediate under the layer ids the website uses (tok_emb, pos_emb, embed, L0.ln1, L0.attn.q, ...).
Ideas from nanoGPT / minGPT (Andrej Karpathy, MIT); code written fresh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    context: int = 64
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 64
    ln_eps: float = 1e-5

    def to_dict(self):
        return asdict(self)


def gelu_tanh(x):
    # GELU, tanh approximation (as in GPT-2): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.qkv = nn.Linear(d, 3 * d)  # q, k, v stacked; split on export
        self.attn_proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, eps=cfg.ln_eps)
        self.fc = nn.Linear(d, 4 * d)
        self.mlp_proj = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_head
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, T, H, D // H).transpose(1, 2)
        k = k.view(B, T, H, D // H).transpose(1, 2)
        v = v.view(B, T, H, D // H).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.attn_proj(y)
        x = x + self.mlp_proj(gelu_tanh(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.ln_eps)
        self.apply(self._init)
        for b in self.blocks:  # GPT-2: scale residual projections by 1/sqrt(2 * n_layer)
            nn.init.normal_(b.attn_proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))
            nn.init.normal_(b.mlp_proj.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.context
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.t()  # tied output head
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def trace(self, ids: list[int]):
        """Run one sequence step by step and return (order, {layer id: tensor}) in fp32.

        Shapes: sequences [T, D]; attention scores/masked/weights [n_head, T, T];
        L{i}.attn.scores = q k^T / sqrt(d_head); masked = scores with -inf above the diagonal.
        """
        cfg = self.cfg
        T, D, H = len(ids), cfg.d_model, cfg.n_head
        hd = D // H
        assert T <= cfg.context
        idx = torch.tensor(ids, dtype=torch.long)
        t, order = {}, []

        def rec(name, val):
            order.append(name)
            t[name] = val.detach().float().clone()
            return val

        tok = rec("tok_emb", self.tok_emb.weight[idx])
        pos = rec("pos_emb", self.pos_emb.weight[:T])
        x = rec("embed", tok + pos)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        for i, b in enumerate(self.blocks):
            p = f"L{i}."
            h = rec(p + "ln1", F.layer_norm(x, (D,), b.ln1.weight, b.ln1.bias, cfg.ln_eps))
            Wq, Wk, Wv = b.qkv.weight.split(D, 0)
            bq, bk, bv = b.qkv.bias.split(D, 0)
            q = rec(p + "attn.q", h @ Wq.t() + bq)
            k = rec(p + "attn.k", h @ Wk.t() + bk)
            v = rec(p + "attn.v", h @ Wv.t() + bv)
            qh = q.view(T, H, hd).transpose(0, 1)  # [H, T, hd]
            kh = k.view(T, H, hd).transpose(0, 1)
            vh = v.view(T, H, hd).transpose(0, 1)
            s = rec(p + "attn.scores", qh @ kh.transpose(1, 2) / math.sqrt(hd))
            sm = rec(p + "attn.masked", s.masked_fill(mask, float("-inf")))
            w = rec(p + "attn.weights", torch.softmax(sm, dim=-1))
            heads = rec(p + "attn.heads", (w @ vh).transpose(0, 1).reshape(T, D))
            a = rec(p + "attn.proj", heads @ b.attn_proj.weight.t() + b.attn_proj.bias)
            x = rec(p + "resid1", x + a)
            h2 = rec(p + "ln2", F.layer_norm(x, (D,), b.ln2.weight, b.ln2.bias, cfg.ln_eps))
            f = rec(p + "mlp.fc", h2 @ b.fc.weight.t() + b.fc.bias)
            g = rec(p + "mlp.gelu", gelu_tanh(f))
            m = rec(p + "mlp.proj", g @ b.mlp_proj.weight.t() + b.mlp_proj.bias)
            x = rec(p + "resid2", x + m)
        xf = rec("ln_f", F.layer_norm(x, (D,), self.ln_f.weight, self.ln_f.bias, cfg.ln_eps))
        logits = rec("logits", xf @ self.tok_emb.weight.t())
        rec("probs", torch.softmax(logits, dim=-1))
        return order, t

    # ------------------------------------------------------------------------------------------
    def export_tensors(self):
        """Named fp32 tensors in the order they are written to weights.bin (names = site ids)."""
        D = self.cfg.d_model
        out = [("tok_emb.weight", self.tok_emb.weight), ("pos_emb.weight", self.pos_emb.weight)]
        for i, b in enumerate(self.blocks):
            p = f"L{i}."
            Wq, Wk, Wv = b.qkv.weight.split(D, 0)
            bq, bk, bv = b.qkv.bias.split(D, 0)
            out += [
                (p + "ln1.weight", b.ln1.weight), (p + "ln1.bias", b.ln1.bias),
                (p + "attn.q.weight", Wq), (p + "attn.q.bias", bq),
                (p + "attn.k.weight", Wk), (p + "attn.k.bias", bk),
                (p + "attn.v.weight", Wv), (p + "attn.v.bias", bv),
                (p + "attn.proj.weight", b.attn_proj.weight), (p + "attn.proj.bias", b.attn_proj.bias),
                (p + "ln2.weight", b.ln2.weight), (p + "ln2.bias", b.ln2.bias),
                (p + "mlp.fc.weight", b.fc.weight), (p + "mlp.fc.bias", b.fc.bias),
                (p + "mlp.proj.weight", b.mlp_proj.weight), (p + "mlp.proj.bias", b.mlp_proj.bias),
            ]
        out += [("ln_f.weight", self.ln_f.weight), ("ln_f.bias", self.ln_f.bias)]
        return [(n, t.detach().float().cpu().contiguous()) for n, t in out]


def param_count(d, L, V=1024, ctx=64):
    return V * d + ctx * d + L * (12 * d * d + 13 * d) + 2 * d
