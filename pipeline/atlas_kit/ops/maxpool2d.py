"""Mirror of ops/maxpool2d.ts: the largest value in each kernel x kernel window of a [C, H, W] grid (stride defaults
to kernel; no padding; the output size rounds down)."""
import torch.nn.functional as F

OP = "maxpool2d"
VERSION = 1
CATEGORY = "pool"
LABEL = ("Max pooling", "最大池化")
ATTRS = {"kernel": {"type": "int", "default": 2}, "stride": {"type": "int"}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    k = int(params.get("kernel", 2))
    s = params.get("stride")
    return F.max_pool2d(xs[0].unsqueeze(0), k, stride=int(s) if s is not None else k)[0]
