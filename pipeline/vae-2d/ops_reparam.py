"""Python mirror of the vae-2d local op 'reparam' (site/src/lib/explainers/vae-2d/ops/reparam.ts).

z = mu + exp(0.5 * logvar) * eps, cell by cell, exactly as model.py's forward(); sub-step 'sigma' = exp(0.5 * logvar).
Used from export.py: b.op("reparam", "z", [mu, logvar, eps], ref=ops_reparam.reference).
"""
import torch

OP = "reparam"
VERSION = 1
CATEGORY = "merge"
LABEL = ("Sample z = mu + sigma * eps", "采样 z = μ + σ·ε")
ATTRS: dict = {}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = ["sigma"]


def reference(xs, params, weights):
    mu, logvar, eps = xs
    sigma = torch.exp(0.5 * logvar)
    return mu + sigma * eps, {"sigma": sigma}
