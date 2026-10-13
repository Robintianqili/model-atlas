"""number-formats: TinyGPT in PyTorch fp32 (from the site's spec.json + weights.bin) and the number formats of the concept
page, written independently of site/src/lib/explainers/number-formats/fmt.ts:

  fp16, bf16, fp8 e4m3fn, fp8 e5m2   PyTorch's own dtypes (x.to(dtype)), from float32
  fp4 e2m1                           the nearest of its 8 sizes {0, 0.5, 1, 1.5, 2, 3, 4, 6}, ties to the even code,
                                     saturating at 6 (no infinity, no NaN)
  int8, int4                         q = clamp(round(x * qmax), -qmax, qmax) in float64, round = ties to even; value q / qmax
  nf4                                the nearest of 16 levels built from normal quantiles as QLoRA does (create_normal_map,
                                     offset 0.9677083), with statistics.NormalDist instead of scipy

A scaled cast (per tensor or per block of a row) uses s = fl32(amax / target), y = fl32(w / s), out = fl32(cast(y) * s),
target = the format's largest value (floats) or 1 (int4, nf4).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SITE = Path(os.environ.get("ATLAS_SITE") or (ROOT / "site"))
GPT_DIR = SITE / "static" / "models" / "tinygpt"
VAL_BIN = ROOT / "pipeline" / "data" / "tinystories" / "val.bin"

LINEAR = ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc", "mlp.proj"]
FMT_IDS = ["fp32", "fp16", "bf16", "e4m3", "e5m2", "fp4", "int8", "int4", "nf4"]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
BITS = {"fp32": 32, "fp16": 16, "bf16": 16, "e4m3": 8, "e5m2": 8, "fp4": 4, "int8": 8, "int4": 4, "nf4": 4}
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
LARGEST = {"fp32": 3.4028234663852886e38, "fp16": 65504.0, "bf16": 3.3895313892515355e38, "e4m3": 448.0, "e5m2": 57344.0, "fp4": 6.0}


def nf4_levels() -> torch.Tensor:
    off = 0.9677083
    ppf = NormalDist().inv_cdf
    v1 = [ppf(float(p)) for p in torch.linspace(off, 0.5, 9)[:-1]]
    v3 = [-ppf(float(p)) for p in torch.linspace(off, 0.5, 8)[:-1]]
    v = sorted(v1 + [0.0] + v3)
    return torch.tensor([a / max(v) for a in v], dtype=torch.float32)


NF4 = nf4_levels()


def fp4_index(a: torch.Tensor) -> torch.Tensor:
    """Index into FP4 of the nearest size to a >= 0 (float64), ties to the even index, saturating at 6."""
    i = torch.searchsorted(FP4, a).clamp(1, 7)
    lo, hi = i - 1, i
    dlo, dhi = a - FP4[lo], FP4[hi] - a
    pick = torch.where(dlo < dhi, lo, torch.where(dhi < dlo, hi, torch.where(lo % 2 == 0, lo, hi)))
    return torch.where(a >= 6, torch.full_like(pick, 7), pick)


def nf4_index(y: torch.Tensor) -> torch.Tensor:
    return (y.double().clamp(-1, 1)[..., None] - NF4.double()).abs().argmin(-1)


def cast(fmt: str, w: torch.Tensor) -> torch.Tensor:
    """A float32 tensor rounded to a format and back to float32."""
    w = w.float()
    if fmt == "fp32":
        return w
    if fmt in DTYPES:
        return w.to(DTYPES[fmt]).float()
    if fmt == "fp4":
        a = w.double().abs()
        return (torch.sign(w.double()) * FP4[fp4_index(a)]).float()
    if fmt in ("int8", "int4"):
        qmax = 2 ** (BITS[fmt] - 1) - 1
        q = torch.clamp(torch.round(w.double() * qmax), -qmax, qmax)
        return (q / qmax).float()
    if fmt == "nf4":
        return NF4[nf4_index(w)]
    raise ValueError(fmt)


def code(fmt: str, x: float) -> int:
    """The bit pattern of x (a Python float, first rounded to float32) in a format, as an unsigned number."""
    t = torch.tensor([x], dtype=torch.float32)
    if fmt == "fp32":
        return int(t.view(torch.int32).item()) & 0xFFFFFFFF
    if fmt in ("fp16", "bf16"):
        return int(t.to(DTYPES[fmt]).view(torch.int16).item()) & 0xFFFF
    if fmt in ("e4m3", "e5m2"):
        return int(t.to(DTYPES[fmt]).view(torch.uint8).item())
    if fmt == "fp4":
        v = float(t.item())
        sign = 8 if (v < 0 or math.copysign(1.0, v) < 0) else 0
        return sign | int(fp4_index(torch.tensor([abs(v)], dtype=torch.float64)).item())
    if fmt in ("int8", "int4"):
        b = BITS[fmt]
        qmax = 2 ** (b - 1) - 1
        q = int(torch.clamp(torch.round(torch.tensor([x], dtype=torch.float64) * qmax), -qmax, qmax).item())
        return q % (2 ** b)
    if fmt == "nf4":
        return int(nf4_index(torch.tensor([x], dtype=torch.float64)).item())
    raise ValueError(fmt)


def scaled(w: torch.Tensor, fmt: str, scale: str, block: int = 32) -> torch.Tensor:
    """cast() with one scale per tensor or per block of `block` values along each row (scale 'none': plain cast)."""
    if scale == "none":
        return cast(fmt, w)
    target = torch.tensor(LARGEST.get(fmt, 1.0), dtype=torch.float32)

    def group(g):
        amax = g.abs().amax(-1, keepdim=True)
        s = torch.where(amax > 0, amax / target, torch.ones_like(amax))
        return cast(fmt, g / s) * s

    if scale == "tensor":
        return group(w.reshape(1, -1)).reshape(w.shape)
    # blocks of `block` values along each row (the last block of a row may be shorter)
    return torch.cat([group(w[:, c0: c0 + block]) for c0 in range(0, w.shape[1], block)], dim=1)


# ---------------------------------------------------------------- TinyGPT


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


def metrics(spec, W0, Wq, ids):
    """Perplexity of the next tokens, mean KL(fp32 || cast) per position, top-1 agreement with fp32."""
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
    return {"ppl0": math.exp(nll0), "ppl": math.exp(nll1), "kl": kl, "agree": agree}


def windows(n: int, length: int = 65):
    v = np.fromfile(VAL_BIN, dtype="<u2").astype(np.int64)
    return torch.from_numpy(v[: n * length].reshape(n, length).copy())
