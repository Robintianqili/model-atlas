"""Mirror of ops/relu.ts: max(0, x), elementwise."""
import torch

OP = "relu"
VERSION = 1
CATEGORY = "activation"
LABEL = ("ReLU", "ReLU")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return torch.relu(xs[0])
