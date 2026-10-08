"""gcn: the PyTorch model (kind A), a graph convolutional network (Kipf and Welling, 2017) on Zachary's karate club.

    adjacency A [n, n] (0/1, no self-loops)  ->  Â = D^-1/2 (A + I) D^-1/2       (D: degrees of A + I)
    features X [n, 34] (one-hot: member i is row i)
    gc1:  H1 = tanh(Â (X W1ᵀ) + b1)        34 -> 4
    gc2:  H2 = tanh(Â (H1 W2ᵀ) + b2)        4 -> 2   (the 2-D embedding)
    head: Z = H2 Whᵀ + bh                   2 -> 2   (faction scores: 0 = Mr. Hi, 1 = Officer)

156 parameters. state_dict names gc1.weight [4, 34], gc1.bias, gc2.weight [2, 4], gc2.bias, head.weight, head.bias:
the TS trainer (site/src/lib/explainers/gcn/trainer.ts) and twin.py use the same names. `Deep` is the same network
with 8 graph layers (34 -> 4 -> ... -> 4 -> 2), measured once for the limits step (train.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def normalize(adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(Â, A + I, degrees): every member also listens to itself; edge i-j is weighted 1 / sqrt(d_i d_j)."""
    loops = adj + torch.eye(adj.shape[-1], dtype=adj.dtype)
    deg = loops.sum(-1)
    r = deg.pow(-0.5)
    return r[:, None] * loops * r[None, :], loops, deg


class GCNConv(nn.Module):
    """H' = Â (H Wᵀ) + b: transform every member's row, then each member adds up its neighbours' rows weighted by Â."""

    def __init__(self, i: int, o: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(o, i))
        self.bias = nn.Parameter(torch.empty(o))
        bound = 1 / math.sqrt(i)  # nn.Linear's initialization
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, h: torch.Tensor, ahat: torch.Tensor) -> torch.Tensor:
        return ahat @ (h @ self.weight.T) + self.bias


class GCN(nn.Module):
    def __init__(self, nodes: int = 34, hidden: int = 4, emb: int = 2, classes: int = 2):
        super().__init__()
        self.gc1 = GCNConv(nodes, hidden)
        self.gc2 = GCNConv(hidden, emb)
        self.head = nn.Linear(emb, classes)

    def embed(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ahat = normalize(adj)[0]
        h = torch.tanh(self.gc1(x, ahat))
        return torch.tanh(self.gc2(h, ahat))

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(adj, x))


class Deep(nn.Module):
    """The same recipe with `layers` graph layers (limits step: over-smoothing)."""

    def __init__(self, layers: int = 8, nodes: int = 34, hidden: int = 4, emb: int = 2, classes: int = 2):
        super().__init__()
        sizes = [nodes] + [hidden] * (layers - 1) + [emb]
        self.gcs = nn.ModuleList([GCNConv(a, b) for a, b in zip(sizes[:-1], sizes[1:])])
        self.head = nn.Linear(emb, classes)

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ahat = normalize(adj)[0]
        h = x
        for g in self.gcs:
            h = torch.tanh(g(h, ahat))
        return self.head(h)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = GCN()
    print(m, sum(p.numel() for p in m.parameters()), "params")
