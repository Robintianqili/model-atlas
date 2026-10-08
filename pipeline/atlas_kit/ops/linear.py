"""Mirror of ops/linear.ts: y = x W^T + b over the last axis (aliases: dense, linear_tied).

`accumulate` only changes the TS summation order ('bias-last' as tinycnn.ts, 'bias-first' as tinygpt.ts); the
PyTorch reference is the same either way. inFeatures/outFeatures (TinyCNN's dense) and nHead/dHead (TinyGPT's q/k/v
projections) are informative attributes: the builder writes inFeatures/outFeatures for `dense` layers, and nHead/dHead
only when asked.
"""
import torch.nn.functional as F

OP = "linear"
VERSION = 1
CATEGORY = "linear"
LABEL = ("Linear (fully connected)", "线性层（全连接）")
ATTRS = {
    "accumulate": {"type": "string", "default": "bias-last"},
    "inFeatures": {"type": "int"},
    "outFeatures": {"type": "int"},
    "nHead": {"type": "int"},
    "dHead": {"type": "int"},
}
ALIASES = {"dense": {}, "linear_tied": {}}
WEIGHTS = [{"slot": "weight"}, {"slot": "bias", "optional": True}]
SUBS: list = []


def reference(xs, params, weights):
    return F.linear(xs[0], weights["weight"], weights.get("bias"))
