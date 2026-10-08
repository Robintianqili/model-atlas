"""Mirror of ops/gelu.ts: GELU, exact (with erf) or the tanh approximation (alias gelu_tanh, as GPT-2)."""
import torch.nn.functional as F

OP = "gelu"
VERSION = 1
CATEGORY = "activation"
LABEL = ("GELU", "GELU")
ATTRS = {"approx": {"type": "string", "default": "none"}}
ALIASES = {"gelu_tanh": {"approx": "tanh"}}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return F.gelu(xs[0], approximate="tanh" if params.get("approx") == "tanh" else "none")
