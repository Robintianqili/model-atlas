"""Mirror of ops/attention.ts attention_scores: q.k / sqrt(dHead) per head -> [H, Tq, Tk] (causal: -Infinity above
the diagonal, a fused mask)."""
from .attention import scores

OP = "attention_scores"
VERSION = 1
CATEGORY = "attention"
LABEL = ("Attention scores", "注意力分数")
ATTRS = {
    "nHead": {"type": "int", "required": True},
    "dHead": {"type": "int", "required": True},
    "scale": {"type": "float"},
    "causal": {"type": "bool", "default": False},
}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return scores(xs, params, weights)
