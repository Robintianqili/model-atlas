"""attention-variants: TinyGPT in PyTorch (fp32) from the site's own spec.json + weights.bin, with grouped-query attention.

pool_kv(W, kv) turns TinyGPT's trained multi-head attention (4 key heads and 4 value heads per block) into kv key/value
heads by mean pooling the key and value projections of each group of 4 / kv heads, as Ainslie et al. (2023) convert a
multi-head checkpoint before uptraining. forward(spec, W, ids) then runs true grouped-query attention: K and V have kv
heads and every head is shared by its group of query heads (repeat_interleave). The browser does the same arithmetic by
copying each group's pooled projection to every head of the group (process.ts), so the two implementations differ.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SITE = Path(os.environ.get("ATLAS_SITE") or (ROOT / "site"))
GPT_DIR = SITE / "static" / "models" / "tinygpt"
DATA = ROOT / "pipeline" / "data" / "tinystories"


def load_tinygpt(device="cpu"):
    spec = json.loads((GPT_DIR / "spec.json").read_text(encoding="utf-8"))
    raw = np.fromfile(GPT_DIR / "weights.bin", dtype="<f4")
    W = {}
    for t in spec["tensors"]:
        W[t["name"]] = torch.from_numpy(raw[t["offset"]: t["offset"] + t["length"]].copy()).view(*t["shape"]).to(device)
    return spec, W


def pool_kv(W, n_layer, n_head, kv, layers=None):
    """The k and v projections of the chosen blocks with kv heads: each group's rows averaged ([kv*hd, D], [kv*hd])."""
    out = dict(W)
    g = n_head // kv
    for l in range(n_layer) if layers is None else layers:
        for nm in ("k", "v"):
            w, b = W[f"L{l}.attn.{nm}.weight"], W[f"L{l}.attn.{nm}.bias"]
            D = w.shape[1]
            hd = w.shape[0] // n_head
            out[f"L{l}.attn.{nm}.weight"] = w.view(kv, g, hd, D).mean(1).reshape(kv * hd, D)
            out[f"L{l}.attn.{nm}.bias"] = b.view(kv, g, hd).mean(1).reshape(kv * hd)
    return out


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def forward(spec, W, ids, record=None):
    """ids [B, T] long -> logits [B, T, V]. K/V heads per block = rows of the k projection / dHead (GQA when fewer).
    record: a dict that receives each block's attention scores and weights [B, H, T, T] and mixed values [B, T, D] as
    'L<l>.attn.scores', 'L<l>.attn.weights' and 'L<l>.attn.heads'."""
    B, T = ids.shape
    D, H = spec["dModel"], spec["nHead"]
    hd = D // H
    eps = spec["lnEps"]
    ln = lambda x, p: torch.nn.functional.layer_norm(x, (D,), W[p + ".weight"], W[p + ".bias"], eps)
    lin = lambda x, p: x @ W[p + ".weight"].T + W[p + ".bias"]
    x = W["tok_emb.weight"][ids] + W["pos_emb.weight"][:T][None]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
    for l in range(spec["nLayer"]):
        P = f"L{l}."
        h = ln(x, P + "ln1")
        kv = W[P + "attn.k.weight"].shape[0] // hd
        q = lin(h, P + "attn.q").view(B, T, H, hd).transpose(1, 2)
        k = lin(h, P + "attn.k").view(B, T, kv, hd).transpose(1, 2).repeat_interleave(H // kv, dim=1)
        v = lin(h, P + "attn.v").view(B, T, kv, hd).transpose(1, 2).repeat_interleave(H // kv, dim=1)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        s = s.masked_fill(mask, float("-inf"))
        a = torch.softmax(s, -1)
        mixed = (a @ v).transpose(1, 2).reshape(B, T, D)
        if record is not None:
            record[P + "attn.scores"], record[P + "attn.weights"], record[P + "attn.heads"] = s, a, mixed
        x = x + lin(mixed, P + "attn.proj")
        x = x + lin(gelu_tanh(lin(ln(x, P + "ln2"), P + "mlp.fc")), P + "mlp.proj")
    x = ln(x, "ln_f")
    return x @ W["tok_emb.weight"].T


def windows(n: int, length: int = 65, story_starts: bool = False, split: str = "val"):
    v = np.fromfile(DATA / f"{split}.bin", dtype="<u2").astype(np.int64)
    if not story_starts:
        return torch.from_numpy(v[: n * length].reshape(n, length).copy())
    eot = np.nonzero(v == 1023)[0]
    out = []
    for e in eot:
        nxt = eot[eot > e]
        end = nxt[0] if len(nxt) else len(v)
        if end - (e + 1) >= length:
            out.append(v[e + 1: e + 1 + length])
        if len(out) == n:
            break
    return torch.from_numpy(np.stack(out))


def perplexity(spec, W, ids, chunk=250):
    """Perplexity of the next tokens of [N, 65] windows (64 inputs each), in float64 log-softmax."""
    nll, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], chunk):
            w = ids[i: i + chunk]
            lp = torch.log_softmax(forward(spec, W, w[:, :-1]).double(), -1)
            nll -= lp.gather(-1, w[:, 1:, None]).sum().item()
            n += w[:, 1:].numel()
    return math.exp(nll / n)
