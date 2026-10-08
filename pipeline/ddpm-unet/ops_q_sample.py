"""Python mirror of the ddpm-unet local op 'q_sample' (site/src/lib/explainers/ddpm-unet/ops/q_sample.ts).

x_t = sqrt(abar[t - 1]) * (2 x0 - 1) + sqrt(1 - abar[t - 1]) * eps, cell by cell, as model.py Noised.forward() computes it.
The table abar (abar_t at index t - 1, the float32 values of model.py's buffer) is an attribute, not a weight: it is
the schedule, not learned. Used from export.py: b.op("q_sample", "xt", [x0, eps, t], params={"abar": [...]}, ref=...).
"""
import torch

OP = "q_sample"
VERSION = 1
CATEGORY = "merge"
LABEL = ("Add noise: x_t = sqrt(abar_t) (2 x0 - 1) + sqrt(1 - abar_t) eps", "加噪：x_t = √ᾱ_t·(2x₀ − 1) + √(1 − ᾱ_t)·ε")
ATTRS: dict = {"abar": {"type": "float[]", "required": True}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    x0, eps, t = xs
    ab = torch.tensor(params["abar"][int(round(float(t.reshape(-1)[0]))) - 1], dtype=torch.float32)
    return torch.sqrt(ab) * (2 * x0 - 1) + torch.sqrt(1 - ab) * eps
