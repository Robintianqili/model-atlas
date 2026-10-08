"""Mirror of ops/sigmoid.ts: 1 / (1 + exp(-x)), elementwise."""
import torch

OP = "sigmoid"
VERSION = 1
CATEGORY = "activation"
LABEL = ("Sigmoid", "Sigmoid")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return torch.sigmoid(xs[0])
