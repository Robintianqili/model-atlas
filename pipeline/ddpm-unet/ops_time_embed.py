"""Python mirror of the ddpm-unet local op 'time_embed' (site/src/lib/explainers/ddpm-unet/ops/time_embed.ts).

t [1] -> [dim, 1, 1]: sin(t f_k) then cos(t f_k), f_k = 10000^(-k / (dim/2)), in float64, then float32 (model.time_embed).
"""
import torch

from model import time_embed

OP = "time_embed"
VERSION = 1
CATEGORY = "embedding"
LABEL = ("Timestep as waves (sin, cos)", "把时间步变成波（sin、cos）")
ATTRS: dict = {"dim": {"type": "int", "required": True}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return time_embed(xs[0].reshape(1), int(params["dim"]))[0].reshape(-1, 1, 1)
