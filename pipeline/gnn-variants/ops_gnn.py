"""Python mirrors of the gnn-variants local ops (site/src/lib/explainers/gnn-variants/ops/gnn.ts): the same arithmetic
as model.py, with the sub-steps the page shows.

    gcn_conv   H [n, i], A [n, n] -> Â (H Wᵀ) + b [n, o]; subs ahat (Â) and xw (H Wᵀ)
    sage_conv  H [n, i], A [n, n] -> [H ‖ mean of the friends' rows] Wᵀ + b [n, o]; sub mean
    gin_agg    H [n, i], A [n, n] -> (1 + ε) H + A H [n, i] (the MLP after it is two core linear layers and a ReLU)
    gat_conv   H [n, i], A [n, n] -> Σⱼ αᵢⱼ zⱼ + b [n, o]; subs z (H Wᵀ), e (scores, −inf off the edges), alpha

Used from export.py's describe() as b.op("<op>", id, [h, adj], ..., ref=ops_gnn.<op>).
"""
import torch
import torch.nn.functional as F

from model import SAGEConv, normalize


def gcn_conv(xs, params, weights):
    h, adj = xs
    ahat = normalize(adj)
    xw = h @ weights["weight"].T
    return ahat @ xw + weights["bias"], {"ahat": ahat, "xw": xw}


def sage_conv(xs, params, weights):
    h, adj = xs
    mean = SAGEConv.mean(h, adj)
    return torch.cat([h, mean], -1) @ weights["weight"].T + weights["bias"], {"mean": mean}


def gin_agg(xs, params, weights):
    h, adj = xs
    return (1 + weights["eps"]) * h + adj @ h


def gat_conv(xs, params, weights):
    h, adj = xs
    z = h @ weights["weight"].T
    e = F.leaky_relu((z @ weights["att_dst"])[:, None] + (z @ weights["att_src"])[None, :], params.get("slope", 0.2))
    e = e.masked_fill((adj + torch.eye(adj.shape[-1], dtype=adj.dtype)) == 0, float("-inf"))
    alpha = torch.softmax(e, -1)
    return alpha @ z + weights["bias"], {"z": z, "e": e, "alpha": alpha}
