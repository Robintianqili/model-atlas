"""PyTorch mirror of the package's local op 'greedy' (site/src/lib/explainers/tiny-encdec/ops/greedy.ts).

Greedy decoding with the decoder block, from the encoder's memory [T, D]: the decoder runs once per symbol on <s> and
everything written so far and writes the symbol with the largest score; the output is the decoder input
[<s>, y_0 .. y_{steps-2}] (the ids the decoder layers after it read, as floats). Written from the weight slots alone
(F.layer_norm, F.linear, softmax), not from model.py, so it checks the TS op independently.
"""
import math

import torch
import torch.nn.functional as F

SLOTS = ["emb", "pos", "ln1_w", "ln1_b", "sq", "sk", "sv", "so", "ln2_w", "ln2_b", "cq", "ck", "cv", "co",
         "ln3_w", "ln3_b", "up", "down", "ln_w", "ln_b", "head"]


def attend(xq, xkv, wq, wk, wv, wo, H, causal):
    Tq, D = xq.shape
    dh = D // H
    q, k, v = (F.linear(x, w).view(-1, H, dh).transpose(0, 1) for x, w in ((xq, wq), (xkv, wk), (xkv, wv)))
    s = (q @ k.transpose(-1, -2)) * (1 / math.sqrt(dh))
    if causal:
        s = s.masked_fill(torch.ones(Tq, xkv.shape[0], dtype=torch.bool).triu(1), float("-inf"))
    return F.linear((torch.softmax(s, -1) @ v).transpose(0, 1).reshape(Tq, D), wo)


def decoder(ids, mem, w, H):
    """logits [L, V] of the decoder on ids [L] (a pre-norm block, then the final LayerNorm and the head)."""
    D = mem.shape[1]
    ln = lambda x, a: F.layer_norm(x, (D,), w[a + "_w"], w[a + "_b"])  # noqa: E731
    y = w["emb"][ids] + w["pos"][: len(ids)]
    h = ln(y, "ln1")
    y = y + attend(h, h, w["sq"], w["sk"], w["sv"], w["so"], H, True)
    y = y + attend(ln(y, "ln2"), mem, w["cq"], w["ck"], w["cv"], w["co"], H, False)
    y = y + F.linear(torch.relu(F.linear(ln(y, "ln3"), w["up"])), w["down"])
    return F.linear(ln(y, "ln"), w["head"])


def reference(xs, params, weights):
    mem = xs[0]
    ids = [int(params.get("bos", 0))]
    for _ in range(int(params["steps"]) - 1):
        ids.append(int(decoder(torch.tensor(ids), mem, weights, int(params["nHead"]))[-1].argmax()))
    return torch.tensor(ids, dtype=torch.float32)
