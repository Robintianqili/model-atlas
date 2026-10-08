"""Mirror of ops/attention.ts causal_mask: -Infinity above the diagonal (a token never looks at later tokens)."""
from .attention import mask

OP = "causal_mask"
VERSION = 1
CATEGORY = "attention"
LABEL = ("Causal mask", "因果掩码")
ATTRS = {"fill": {"type": "string", "default": "-Infinity"}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return mask(xs, params, weights)
