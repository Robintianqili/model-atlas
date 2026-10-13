"""transe: the PyTorch float64 twin of trainer 'transe' (kind B parity).

`npm run atlas -- parity transe` runs the TS trainer in fine detail on the page's start input (the 33 start facts; and
once per other option set of a train source, such as source 'base' with start: true), writes its init params and
batches to site/.atlas/transe/train_fixture.transe.<optimizer>.json, then runs `python -m atlas_kit.train_reference`
with this twin: losses, step-1 gradients and final params must agree over 50 steps (plain SGD <= 1e-10 relative; the
page trains with plain SGD too).

The model is site/src/lib/explainers/transe/trainer.ts (TransE, Bordes et al. 2013): ent.weight [34, dim] (a point per
entity) and rel.weight [3, dim] (an arrow per relation). A batch is one fact (h, r, t) and its corrupted copy
(h2, r, t2); d(a, b) = sqrt(sum((ent[a] + rel[r] - ent[b])^2)) and the loss is relu(gamma + d(h, t) - d(h2, t2)). The
facts and the corruptions are TS data here (the batches come from the TS fixture).
"""
from __future__ import annotations

import torch
import torch.nn as nn

NE, NR = 34, 3  # kg.ts: entities and relations


class TransE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ent = nn.Embedding(NE, dim)
        self.rel = nn.Embedding(NR, dim)


def build(cfg: dict) -> nn.Module:
    """The TS model for this run's options (dim, default 8); the init values come from the TS fixture."""
    return TransE(int(cfg.get("dim", 8))).double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'h', 'r', 't', 'h2', 't2'}: [1] float64 each (ids arrive as floats). No fact gives a zero loss."""
    h, r, t, h2, t2 = (batch[k].long() for k in ("h", "r", "t", "h2", "t2"))
    if h.numel() == 0:
        return model.ent.weight.sum() * 0, {}
    E, rv = model.ent.weight, model.rel.weight[r]

    def d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(((E[a] + rv - E[b]) ** 2).sum())

    d_pos, d_neg = d(h, t), d(h2, t2)
    return torch.relu(float(cfg.get("gamma", 1)) + d_pos - d_neg), {"dPos": d_pos, "dNeg": d_neg}
