"""Python mirrors of the contrastive-learning local ops (site/src/lib/explainers/contrastive-learning/ops/clip.ts).

    mean_pool   [..., L, E] rows + [..., L] ids -> [..., E]   average of the rows whose id is not 0 (0 when none)
    l2norm      x / max(|x|, 1e-12) on the last axis          F.normalize(x, dim=-1)
    similarity  a [D] or [N, D], b [M, D] -> [N, M]           min(e^t, maxScale) * (a . b)

Used from export.py's describe() as b.op("<op>", id, [...], ref=ops_clip.<op>).
"""
import torch
import torch.nn.functional as F

OPS = {
    "mean_pool": {"version": 1, "category": "pool", "label": ("Average of the words", "词的平均"), "attrs": {}, "weights": []},
    "l2norm": {"version": 1, "category": "norm", "label": ("Scale to length 1", "缩放到长度 1"), "attrs": {}, "weights": []},
    "similarity": {"version": 1, "category": "merge", "label": ("Similarity score", "相似度分数"),
                   "attrs": {"maxScale": {"type": "float", "default": 100}}, "weights": [{"slot": "logit_scale"}]},
}


def mean_pool(xs, params, weights):
    e, ids = xs
    m = (ids != 0).to(e.dtype).unsqueeze(-1)
    return (e * m).sum(-2) / m.sum(-2).clamp(min=1)


def l2norm(xs, params, weights):
    return F.normalize(xs[0], dim=-1)


def similarity(xs, params, weights):
    a, b = xs
    s = weights["logit_scale"].reshape(()).exp().clamp(max=float(params.get("maxScale", 100)))
    return s * (a @ b.t() if b.dim() == 2 else a @ b) if a.dim() == 2 else s * (b @ a if b.dim() == 2 else a @ b)
