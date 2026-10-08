"""Mirror of ops/embedding.ts position_embedding: rows 0..T-1 of the [P, D] table, T = the length of the input."""
from ..errors import fail

OP = "position_embedding"
VERSION = 1
CATEGORY = "embedding"
LABEL = ("Position embedding", "位置嵌入")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS = [{"slot": "weight"}]
SUBS: list = []


def reference(xs, params, weights):
    T = int(xs[0].shape[0])
    W = weights["weight"]
    if T > W.shape[0]:
        fail("AK301", f"position_embedding: {T} positions but the table has only {W.shape[0]} rows")
    return W[:T]
