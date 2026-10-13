"""Python mirror of the crnn-ocr local op 'flip' (site/src/lib/explainers/crnn-ocr/ops/flip.ts).

[T, D] -> [T, D] with the rows in reverse order, torch.flip(x, [0]). Values are copied, never computed.
"""
import torch

OP = "flip"
VERSION = 1
CATEGORY = "shape"
LABEL = ("Reverse the order", "倒过来排")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return torch.flip(xs[0], [0]).contiguous()
