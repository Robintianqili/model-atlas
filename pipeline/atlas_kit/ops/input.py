"""Mirror of ops/input.ts: a layer that shows a declared input as numbers (layer 0)."""
OP = "input"
VERSION = 1
CATEGORY = "io"
LABEL = ("Input", "输入")
ATTRS = {"name": {"type": "string"}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return xs[0]
