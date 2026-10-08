"""Mirror of ops/tanh.ts: tanh(x), elementwise."""
import torch

OP = "tanh"
VERSION = 1
CATEGORY = "activation"
LABEL = ("Tanh", "Tanh")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return torch.tanh(xs[0])
