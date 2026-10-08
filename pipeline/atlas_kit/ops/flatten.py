"""Mirror of ops/flatten.ts: every axis laid end to end, row-major (channel by channel, row by row)."""

OP = "flatten"
VERSION = 1
CATEGORY = "shape"
LABEL = ("Flatten", "展平")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return xs[0].reshape(-1)
