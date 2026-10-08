"""Python mirror of the ddpm-unet local op 'upsample' (site/src/lib/explainers/ddpm-unet/ops/upsample.ts).

Nearest-neighbour upsampling of [C, H, W] by an integer scale: F.interpolate(x, scale_factor=2, mode="nearest").
"""
import torch.nn.functional as F

OP = "upsample"
VERSION = 1
CATEGORY = "shape"
LABEL = ("Upsample (copy each number into a 2x2 block)", "上采样（每个数复制成 2×2 的块）")
ATTRS: dict = {"scale": {"type": "int", "default": 2}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    s = int(params.get("scale", 2))
    return F.interpolate(xs[0].unsqueeze(0), scale_factor=s, mode="nearest")[0]
