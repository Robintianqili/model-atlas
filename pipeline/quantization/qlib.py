"""quantization: TinyGPT in PyTorch (fp32) from the site's own spec.json + weights.bin, and the weight quantizers the
concept page runs in the browser (quant.ts): symmetric absmax rounding to int8 or int4 per tensor, per output channel
(row) or per block of `block` weights along a row, dequantized back to float ("fake quantization"). torch.round rounds
ties to even, as quant.ts does.
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
VAL_BIN = ROOT / "pipeline" / "data" / "tinystories" / "val.bin"

LINEAR = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj"]


def load_tinygpt(device="cpu"):
    spec = json.loads((GPT_DIR / "spec.json").read_text(encoding="utf-8"))
    raw = np.fromfile(GPT_DIR / "weights.bin", dtype="<f4")
    W = {}
    for t in spec["tensors"]:
        W[t["name"]] = torch.from_numpy(raw[t["offset"]: t["offset"] + t["length"]].copy()).view(*t["shape"]).to(device)
    return spec, W


def matrices(spec):
    """The weight matrices of the linear layers, in the spec's order: L0.attn.q.weight, ..."""
    return [f"L{l}.{n}.weight" for l in range(spec["nLayer"]) for n in LINEAR]


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def forward(spec, W, ids):
    """ids [B, T] long -> logits [B, T, V] (fp32)."""
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
        q = lin(h, P + "attn.q").view(B, T, H, hd).transpose(1, 2)
        k = lin(h, P + "attn.k").view(B, T, H, hd).transpose(1, 2)
        v = lin(h, P + "attn.v").view(B, T, H, hd).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        s = s.masked_fill(mask, float("-inf"))
        a = torch.softmax(s, -1) @ v
        x = x + lin(a.transpose(1, 2).reshape(B, T, D), P + "attn.proj")
        x = x + lin(gelu_tanh(lin(ln(x, P + "ln2"), P + "mlp.fc")), P + "mlp.proj")
    x = ln(x, "ln_f")
    return x @ W["tok_emb.weight"].T


def quantize(w: torch.Tensor, bits: int, gran: str, block: int = 32) -> torch.Tensor:
    """Symmetric absmax: s = max|w| / qmax over the group, q = clamp(round(w / s), -qmax, qmax), back = q * s."""
    qmax = 2 ** (bits - 1) - 1
    out_f, in_f = w.shape
    if gran == "tensor":
        s = w.abs().max() / qmax
        s = torch.where(s > 0, s, torch.ones_like(s))
        return torch.clamp(torch.round(w / s), -qmax, qmax) * s
    if gran == "channel":
        s = w.abs().amax(dim=1, keepdim=True) / qmax
        s = torch.where(s > 0, s, torch.ones_like(s))
        return torch.clamp(torch.round(w / s), -qmax, qmax) * s
    if gran == "block":
        out = torch.empty_like(w)
        for c0 in range(0, in_f, block):
            g = w[:, c0: c0 + block]
            s = g.abs().amax(dim=1, keepdim=True) / qmax
            s = torch.where(s > 0, s, torch.ones_like(s))
            out[:, c0: c0 + block] = torch.clamp(torch.round(g / s), -qmax, qmax) * s
        return out
    raise ValueError(gran)


def metrics(spec, W0, Wq, ids):
    """Perplexity of the next tokens, mean KL(fp32 || quantized) per position, top-1 agreement with fp32."""
    x, y = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        l0 = forward(spec, W0, x).double()
        l1 = forward(spec, Wq, x).double()
    lp0 = torch.log_softmax(l0, -1)
    lp1 = torch.log_softmax(l1, -1)
    nll0 = -lp0.gather(-1, y[..., None]).mean().item()
    nll1 = -lp1.gather(-1, y[..., None]).mean().item()
    kl = (lp0.exp() * (lp0 - lp1)).sum(-1).mean().item()
    agree = (l0.argmax(-1) == l1.argmax(-1)).double().mean().item()
    return {"ppl0": math.exp(nll0), "ppl": math.exp(nll1), "kl": kl, "agree": agree, "maxLogitDiff": (l1 - l0).abs().max().item()}


def windows(n: int, length: int = 65, story_starts: bool = False):
    v = np.fromfile(VAL_BIN, dtype="<u2").astype(np.int64)
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
