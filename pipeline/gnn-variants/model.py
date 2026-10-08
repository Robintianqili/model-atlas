"""gnn-variants: the PyTorch models (kind A). Four graph networks on Zachary's karate club, one per way of listening
to the neighbours, each with 2 graph layers (34 -> 4 -> 2, tanh after each) and a 2 -> 2 head, as in the gcn package:

    GCN   H' = Â (H Wᵀ) + b, Â = D^-1/2 (A + I) D^-1/2                 (Kipf and Welling 2017: normalized sum)
    SAGE  H' = [H ‖ mean of the friends' rows] Wᵀ + b                   (Hamilton et al. 2017: mean + concat)
    GIN   H' = MLP((1 + ε) H + A H), MLP = Linear, ReLU, Linear       (Xu et al. 2019: sum + MLP, ε learned)
    GAT   z = H Wᵀ; eᵢⱼ = LeakyReLU(a_dst·zᵢ + a_src·zⱼ); αᵢⱼ = softmax of eᵢⱼ over j in N(i) ∪ {i};
          H'ᵢ = Σⱼ αᵢⱼ zⱼ + b                                          (Veličković et al. 2018: one head, as PyG GATConv)

Inputs: the adjacency A [n, n] (0/1, no self-loops) and the one-hot features X [n, 34]. A member with no friends:
SAGE's mean is 0, GIN's sum is (1 + ε) hᵢ, GCN and GAT keep only the self-loop.
state_dict names: <variant>.gc1.*, <variant>.gc2.*, <variant>.head.*; export.py writes the layer ids <variant>.gc1, ...
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SLOPE = 0.2  # GAT's LeakyReLU slope (PyG's default)
VARIANTS = ("gcn", "sage", "gin", "gat")


def normalize(adj: torch.Tensor) -> torch.Tensor:
    """Â = D^-1/2 (A + I) D^-1/2 (d counts the friends plus the self-loop)."""
    loops = adj + torch.eye(adj.shape[-1], dtype=adj.dtype)
    r = loops.sum(-1).pow(-0.5)
    return r[:, None] * loops * r[None, :]


def _uniform(t: torch.Tensor, fan_in: int) -> None:
    b = 1 / math.sqrt(fan_in)  # nn.Linear's initialization
    nn.init.uniform_(t, -b, b)


class GCNConv(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(o, i))
        self.bias = nn.Parameter(torch.empty(o))
        _uniform(self.weight, i)
        _uniform(self.bias, i)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return normalize(adj) @ (h @ self.weight.T) + self.bias


class SAGEConv(nn.Module):
    """W [o, 2i]: the first i columns read the member's own row, the last i the mean of its friends' rows."""

    def __init__(self, i: int, o: int):
        super().__init__()
        self.lin = nn.Linear(2 * i, o)

    @staticmethod
    def mean(h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return (adj @ h) / adj.sum(-1, keepdim=True).clamp(min=1)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.cat([h, self.mean(h, adj)], -1))


class GINConv(nn.Module):
    def __init__(self, i: int, hid: int, o: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(i, hid), nn.ReLU(), nn.Linear(hid, o))

    def agg(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return (1 + self.eps) * h + adj @ h

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.agg(h, adj))


class GATConv(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(o, i))
        self.att_src = nn.Parameter(torch.empty(o))
        self.att_dst = nn.Parameter(torch.empty(o))
        self.bias = nn.Parameter(torch.zeros(o))
        nn.init.xavier_uniform_(self.weight)
        for a in (self.att_src, self.att_dst):  # glorot on a [1, o] tensor, as PyG
            b = math.sqrt(6 / (1 + o))
            nn.init.uniform_(a, -b, b)

    def attend(self, h: torch.Tensor, adj: torch.Tensor):
        """(z, scores e with −inf off the edges, attention α)."""
        z = h @ self.weight.T
        e = F.leaky_relu((z @ self.att_dst)[:, None] + (z @ self.att_src)[None, :], SLOPE)
        e = e.masked_fill((adj + torch.eye(adj.shape[-1], dtype=adj.dtype)) == 0, float("-inf"))
        return z, e, torch.softmax(e, -1)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        z, _, alpha = self.attend(h, adj)
        return alpha @ z + self.bias


def conv(kind: str, i: int, o: int) -> nn.Module:
    if kind == "gcn":
        return GCNConv(i, o)
    if kind == "sage":
        return SAGEConv(i, o)
    if kind == "gin":
        return GINConv(i, o, o)
    return GATConv(i, o)


class Net(nn.Module):
    """One variant: two graph layers with tanh, then a linear head (faction scores: 0 = Mr. Hi, 1 = the Officer)."""

    def __init__(self, kind: str, nodes: int = 34, hidden: int = 4, emb: int = 2, classes: int = 2):
        super().__init__()
        self.kind = kind
        self.gc1 = conv(kind, nodes, hidden)
        self.gc2 = conv(kind, hidden, emb)
        self.head = nn.Linear(emb, classes)

    def embed(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.gc2(torch.tanh(self.gc1(x, adj)), adj))

    def forward(self, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(adj, x))


class Variants(nn.Module):
    """All four on the same input; returns their faction scores in VARIANTS order."""

    def __init__(self):
        super().__init__()
        for k in VARIANTS:
            setattr(self, k, Net(k))

    def forward(self, adj: torch.Tensor, x: torch.Tensor):
        return tuple(getattr(self, k)(adj, x) for k in VARIANTS)


if __name__ == "__main__":
    torch.manual_seed(0)
    for k in VARIANTS:
        print(k, sum(p.numel() for p in Net(k).parameters()), "params")
