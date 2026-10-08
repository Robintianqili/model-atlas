"""Mirror of ops/attention.ts attention_mix: per head, the attention weights [Tq, Tk] times the values -> [Tq, H*dH]."""
from .attention import mix

OP = "attention_mix"
VERSION = 1
CATEGORY = "attention"
LABEL = ("Weighted sum of values", "值的加权和")
ATTRS = {"nHead": {"type": "int", "required": True}, "dHead": {"type": "int", "required": True}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return mix(xs, params, weights)
