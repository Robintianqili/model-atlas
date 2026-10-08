"""Mirror of ops/add.ts: a + b with PyTorch broadcasting (residual connections, token + position embeddings)."""

OP = "add"
VERSION = 1
CATEGORY = "merge"
LABEL = ("Add", "相加")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return xs[0] + xs[1]
