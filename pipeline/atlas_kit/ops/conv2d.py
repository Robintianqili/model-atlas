"""Mirror of ops/conv2d.ts: a square kernel slides over a [C, H, W] grid (stride, zero padding).

`accumulate` only changes the TS summation order; the PyTorch reference is the same either way.
inChannels/outChannels are informative (they must equal the weight's shape).
"""
import torch.nn.functional as F

OP = "conv2d"
VERSION = 1
CATEGORY = "conv"
LABEL = ("Convolution", "卷积")
ATTRS = {
    "kernel": {"type": "int", "required": True},
    "stride": {"type": "int", "default": 1},
    "padding": {"type": "int", "default": 0},
    "inChannels": {"type": "int"},
    "outChannels": {"type": "int"},
    "accumulate": {"type": "string", "default": "bias-last"},
}
ALIASES: dict = {}
WEIGHTS = [{"slot": "weight"}, {"slot": "bias", "optional": True}]
SUBS: list = []


def reference(xs, params, weights):
    return F.conv2d(xs[0].unsqueeze(0), weights["weight"], weights.get("bias"),
                    stride=int(params.get("stride", 1)), padding=int(params.get("padding", 0)))[0]
