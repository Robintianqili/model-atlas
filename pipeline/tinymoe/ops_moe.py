"""Python mirror of the tinymoe local op 'moe' (site/src/lib/explainers/tinymoe/ops/moe.ts).

    inputs  x [T, D] (the normalized token vectors), p [T, E] (the router's chances, a softmax over E experts)
    weights fc [E, H, D], fc_b [E, H], out [E, D, H], out_b [E, D]  (expert e: out[e] relu(fc[e] x + fc_b[e]) + out_b[e])
    params  {"k": k}: experts per token
    gate[t] = the k largest chances of p[t] rescaled to add up to 1, 0 for the other experts   sub-step 'gate' [T, E]
    e[t, j] = expert j's output for token t, computed only where gate[t, j] > 0 (0 elsewhere)  sub-step 'e' [T, E, D]
    y[t] = sum_j gate[t, j] e[t, j]                                                             output [T, D]
Used from export.py's describe() as b.op("moe", id, [x, p], params={"k": 2}, weights=..., ref=reference).
"""
import torch

OP = "moe"
VERSION = 1
CATEGORY = "merge"
LABEL = ("Mixture of experts (top-k)", "混合专家（取前 k 个）")
ATTRS = {"k": {"type": "int", "required": True}}
WEIGHTS = [{"slot": "fc"}, {"slot": "fc_b"}, {"slot": "out"}, {"slot": "out_b"}]
SUBS = ["gate", "e"]


def gates(p: torch.Tensor, k: int) -> torch.Tensor:
    top_p, top_i = p.topk(k, dim=-1)
    return torch.zeros_like(p).scatter(-1, top_i, top_p / top_p.sum(-1, keepdim=True))


def reference(xs, params, weights):
    x, p = xs
    g = gates(p, int(params["k"]))
    T, E = p.shape
    e = torch.zeros(T, E, x.shape[1])
    for j in range(E):
        rows = g[:, j] > 0
        h = torch.relu(x[rows] @ weights["fc"][j].T + weights["fc_b"][j])
        e[rows, j] = h @ weights["out"][j].T + weights["out_b"][j]
    return (g[:, :, None] * e).sum(1), {"gate": g, "e": e}
