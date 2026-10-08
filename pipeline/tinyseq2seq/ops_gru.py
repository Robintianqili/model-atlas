"""Python mirror of op 'gru' (site/src/lib/explainers/tinyseq2seq/ops/gru.ts): one GRU layer over x [T, In] -> the
states h [T, H], nn.GRU semantics (rows r, z, n; h_0 = 0; both biases kept):
    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr),  z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn)),  h' = (1 - z) * n + z * h
Sub-steps r, z, n, each [T, H], recorded as "<id>#<sub>". export.py checks this loop against the nn.GRU module.
"""
import torch

OP = "gru"
VERSION = 1
CATEGORY = "recurrent"
LABEL = ("GRU", "GRU（门控循环单元）")
ATTRS = {"hidden": {"type": "int", "required": True}}
ALIASES: dict = {}
WEIGHTS = [{"slot": "weight_ih"}, {"slot": "weight_hh"}, {"slot": "bias_ih"}, {"slot": "bias_hh"}]
SUBS = ["r", "z", "n"]


def cell(x, h, Wih, Whh, bih, bhh):
    """One step (nn.GRUCell). Returns r, z, n, h'."""
    H = h.shape[0]
    gi = Wih @ x + bih
    gh = Whh @ h + bhh
    r = torch.sigmoid(gi[:H] + gh[:H])
    z = torch.sigmoid(gi[H:2 * H] + gh[H:2 * H])
    n = torch.tanh(gi[2 * H:] + r * gh[2 * H:])
    return r, z, n, (1 - z) * n + z * h


def reference(xs, params, weights):
    x = xs[0]
    H = int(params["hidden"])
    h = torch.zeros(H, dtype=x.dtype)
    rows = {k: [] for k in ("r", "z", "n", "h")}
    for t in range(x.shape[0]):
        r, z, n, h = cell(x[t], h, weights["weight_ih"], weights["weight_hh"], weights["bias_ih"], weights["bias_hh"])
        for k, v in zip(("r", "z", "n", "h"), (r, z, n, h)):
            rows[k].append(v)
    out = {k: torch.stack(v) if v else torch.zeros(0, H, dtype=x.dtype) for k, v in rows.items()}
    return out.pop("h"), out
