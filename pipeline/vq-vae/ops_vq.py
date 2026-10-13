"""Python mirror of the vq-vae local op 'vq' (site/src/lib/explainers/vq-vae/ops/vq.ts): vector quantization.

z_e [D, H, W] and the codebook [K, D] -> z_q [D, H, W]: each cell's D numbers are replaced by the codebook row at the
smallest squared distance sum_d (z_e[d] - e_k[d])^2 (explicit differences, as model.py's quantize(); argmin keeps the
first of equal distances). Sub-steps: 'dist' [K, H, W] (every squared distance) and 'idx' [H, W] (the code indices).
Used from export.py: b.op("vq", "zq", [ze], weights={"codebook": ...}, ref=ops_vq.reference).
"""
import torch

OP = "vq"
VERSION = 1
CATEGORY = "embedding"
LABEL = ("Nearest code", "最近的码字")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = ["codebook"]
SUBS: list = ["dist", "idx"]


def reference(xs, params, weights):
    ze, cb = xs[0], weights["codebook"]
    dist = ((ze[None] - cb[:, :, None, None]) ** 2).sum(1)
    idx = dist.argmin(0)
    zq = cb[idx].permute(2, 0, 1).contiguous()
    return zq, {"dist": dist, "idx": idx.float()}
