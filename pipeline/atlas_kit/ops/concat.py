"""Mirror of ops/concat.ts: torch.cat along `axis` (default 0; negative counts from the end)."""
import torch

OP = "concat"
VERSION = 1
CATEGORY = "merge"
LABEL = ("Concatenate", "拼接")
ATTRS = {"axis": {"type": "int", "default": 0}}
ALIASES: dict = {}
WEIGHTS: list = []
SUBS: list = []


def reference(xs, params, weights):
    return torch.cat(list(xs), dim=int(params.get("axis", 0)))
