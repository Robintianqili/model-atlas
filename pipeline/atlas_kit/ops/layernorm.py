"""Mirror of ops/layernorm.ts: (x - mean) / sqrt(var + eps) * weight + bias over the last axis (biased variance)."""
import torch.nn.functional as F

OP = "layernorm"
VERSION = 1
CATEGORY = "norm"
LABEL = ("Layer normalization", "层归一化")
ATTRS = {"eps": {"type": "float", "default": 1e-05}}
ALIASES: dict = {}
WEIGHTS = [{"slot": "weight", "optional": True}, {"slot": "bias", "optional": True}]
SUBS: list = []


def reference(xs, params, weights):
    x = xs[0]
    return F.layer_norm(x, (x.shape[-1],), weights.get("weight"), weights.get("bias"), float(params.get("eps", 1e-5)))
