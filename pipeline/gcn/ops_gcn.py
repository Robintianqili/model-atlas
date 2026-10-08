"""Python mirrors of the gcn local ops (site/src/lib/explainers/gcn/ops/gcn.ts).

    gcn_norm   A [n, n] -> Â = D^-1/2 (A + I) D^-1/2 [n, n]; subs loops (A + I) and deg (d, row sums of A + I)
    gcn_conv   H [n, i], Â [n, n] -> Â (H Wᵀ) + b [n, o]; sub xw (H Wᵀ, every member's transformed row)

Used from export.py's describe() as b.op("<op>", id, [...], ref=ops_gcn.<op>); the same arithmetic as model.py.
"""
from model import normalize

OPS = {
    "gcn_norm": {"version": 1, "category": "graph", "label": ("Normalize the adjacency", "归一化邻接矩阵"), "attrs": {}, "weights": [],
                 "subs": ["loops", "deg"]},
    "gcn_conv": {"version": 1, "category": "graph", "label": ("Graph convolution", "图卷积"), "attrs": {},
                 "weights": [{"slot": "weight"}, {"slot": "bias"}], "subs": ["xw"]},
}


def gcn_norm(xs, params, weights):
    ahat, loops, deg = normalize(xs[0])
    return ahat, {"loops": loops, "deg": deg}


def gcn_conv(xs, params, weights):
    h, ahat = xs
    xw = h @ weights["weight"].T
    return ahat @ xw + weights["bias"], {"xw": xw}
