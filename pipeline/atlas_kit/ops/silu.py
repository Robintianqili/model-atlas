"""Mirror of ops/silu.ts: x * sigmoid(x) (also called swish), elementwise."""
import torch.nn.functional as F

OP = "silu"
VERSION = 1
CATEGORY = "activation"
LABEL = ("SiLU", "SiLU")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return F.silu(xs[0])
