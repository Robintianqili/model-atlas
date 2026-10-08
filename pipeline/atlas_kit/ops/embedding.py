"""Mirror of ops/embedding.ts: each integer id picks one row of the [V, D] table -> [T, D]."""
import torch.nn.functional as F

OP = "embedding"
VERSION = 1
CATEGORY = "embedding"
LABEL = ("Embedding", "嵌入")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS = [{"slot": "weight"}]
SUBS: list = []


def reference(xs, params, weights):
    return F.embedding(xs[0].long(), weights["weight"])
