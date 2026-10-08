"""word2vec: the PyTorch float64 twin of trainer 'sg' (kind B parity, design §4.11).

`npm run atlas -- parity word2vec` runs the TS trainer in fine detail on the page's start corpus (the pets preset),
writes its init params and batches to site/.atlas/word2vec/train_fixture.sg.sgd.json, then runs
`python -m atlas_kit.train_reference` with this twin: losses, step-1 gradients and final params must agree over 50
steps (plain SGD <= 1e-10 relative; the page trains with plain SGD too).

The model is site/src/lib/explainers/word2vec/trainer.ts: two tables of VMAX = 40 rows (corpus.ts VMAX) and `dim`
columns, emb.weight (W_in) and out.weight (W_out, one output vector per row). A batch is one (center, context) pair
and V, the learner's vocabulary size: h = emb.weight[c], scores = h @ out.weight[:V].T, loss = cross-entropy of the
scores against the context word. Only the first V rows ever take part.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VMAX = 40  # site/src/lib/explainers/word2vec/corpus.ts VMAX


class SkipGram(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.emb = nn.Embedding(VMAX, dim)
        self.out = nn.Linear(dim, VMAX, bias=False)


def build(cfg: dict) -> nn.Module:
    """The TS model for this run's options (dim, default 5); the init values come from the TS fixture."""
    return SkipGram(int(cfg.get("dim", 5))).double()


def loss(model: nn.Module, batch: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    """batch: {'c': [1], 'o': [1], 'V': [1]} float64 (ids arrive as floats). An empty pair list gives a zero loss."""
    c, o, V = batch["c"].long(), batch["o"].long(), int(batch["V"][0].item())
    if c.numel() == 0:
        return model.emb.weight.sum() * 0, {}
    h = model.emb.weight[c]
    return F.cross_entropy(h @ model.out.weight[:V].T, o), {}
