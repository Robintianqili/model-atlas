"""Python mirror of the vq-vae local op 'pixel_shuffle' (site/src/lib/explainers/vq-vae/ops/pixel_shuffle.ts).

[C * r * r, H, W] -> [C, H * r, W * r] as torch.nn.functional.pixel_shuffle: channel c*r*r + a*r + b of cell (i, j)
goes to row i*r + a, column j*r + b. Values are copied, never computed.
"""
import torch.nn.functional as F

OP = "pixel_shuffle"
VERSION = 1
CATEGORY = "shape"
LABEL = ("Lay out as blocks", "排成小块")
ATTRS: dict = {"factor": {"type": "int", "required": True}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return F.pixel_shuffle(xs[0][None], int(params["factor"]))[0].contiguous()
