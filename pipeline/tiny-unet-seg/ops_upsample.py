"""Python mirror of the tiny-unet-seg local op 'upsample' (site/src/lib/explainers/tiny-unet-seg/ops/upsample.ts):
bilinear upsampling of [C, H, W] maps by an integer scale, align_corners=False (as torch.nn.functional.interpolate)."""
import torch.nn.functional as F

OP = "upsample"
VERSION = 1
CATEGORY = "shape"
LABEL = ("Upsample (bilinear)", "上采样（双线性）")
ATTRS: dict = {"scale": {"type": "int", "default": 2}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    s = int(params.get("scale", 2))
    return F.interpolate(xs[0].unsqueeze(0), scale_factor=s, mode="bilinear", align_corners=False)[0].contiguous()
