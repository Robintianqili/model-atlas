"""node2vec: the PyTorch float64 twin of trainer 'sg' (kind B parity, design §4.11).

`npm run atlas -- parity node2vec` runs the TS trainer in fine detail on the page's start input (the karate club,
p = 1, q = 0.5; and once per other option set of a train source, such as the fixed p and q of source 'base'), writes its
init params and batches to site/.atlas/node2vec/train_fixture.sg.<optimizer>.json, then runs
`python -m atlas_kit.train_reference` with this twin: losses, step-1 gradients and final params must agree over 50 steps
(plain SGD <= 1e-10 relative; the page trains with plain SGD too).

The model is site/src/lib/explainers/node2vec/trainer.ts (skip-gram, as the word2vec package's twin): two tables of
N = 34 rows (one per member) and `dim` columns, emb.weight (W_in, the member vectors) and out.weight (W_out, one output
vector per member). A batch is one (center, context) pair of members from the walks: h = emb.weight[c],
scores = h @ out.weight.T (34 scores), loss = cross-entropy of the scores against the context member. The walks
themselves are TS data here (the batches come from the TS fixture); their sampler is checked by reference.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N = 34  # Zachary's karate club: atlas/data/karate.ts N


class SkipGram(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.emb = nn.Embedding(N, dim)
        self.out = nn.Linear(dim, N, bias=False)


def build(cfg: dict) -> nn.Module:
    """The TS model for this run's options (dim, default 8); the init values come from the TS fixture."""
    return SkipGram(int(cfg.get("dim", 8))).double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'c': [1], 'o': [1]} float64 (ids arrive as floats). An empty pair list gives a zero loss."""
    c, o = batch["c"].long(), batch["o"].long()
    if c.numel() == 0:
        return model.emb.weight.sum() * 0, {}
    h = model.emb.weight[c]
    return F.cross_entropy(h @ model.out.weight.T, o), {}
