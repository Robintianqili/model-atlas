"""Mirror of ops/softmax.ts: exp(x - max) / sum along `axis` (default -1); -Infinity entries get exactly 0."""
import torch

OP = "softmax"
VERSION = 1
CATEGORY = "output"
LABEL = ("Softmax", "Softmax")
ATTRS = {"axis": {"type": "int", "default": -1}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return torch.softmax(xs[0], dim=int(params.get("axis", -1)))
