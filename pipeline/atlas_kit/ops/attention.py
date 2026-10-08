"""Shared code of the attention mirrors (ops/attention.ts): attention_scores, causal_mask and attention_mix
(multi-head attention in three layers, TinyGPT style). q, k and v are [T, H*dH]; scores and weights are [H, Tq, Tk]."""
import math

import torch


def heads(x: torch.Tensor, H: int, dH: int) -> torch.Tensor:
    """[T, H*dH] -> [H, T, dH]"""
    return x.reshape(x.shape[0], H, dH).transpose(0, 1)


def causal(s: torch.Tensor) -> torch.Tensor:
    """-Infinity above the diagonal of the last two axes."""
    m = torch.ones(s.shape[-2], s.shape[-1], dtype=torch.bool, device=s.device).triu(1)
    return s.masked_fill(m, float("-inf"))


def scores(xs, params, weights):
    H, dH = int(params["nHead"]), int(params["dHead"])
    q, k = xs[0], xs[1]
    sc = params.get("scale")
    sc = float(sc) if sc is not None else 1.0 / math.sqrt(dH)
    s = heads(q, H, dH) @ heads(k, H, dH).transpose(-2, -1) * sc
    return causal(s) if params.get("causal") else s


def mask(xs, params, weights):
    return causal(xs[0])


def mix(xs, params, weights):
    H, dH = int(params["nHead"]), int(params["dHead"])
    w, v = xs[0], xs[1]
    return (w @ heads(v, H, dH)).transpose(0, 1).reshape(w.shape[1], H * dH)
