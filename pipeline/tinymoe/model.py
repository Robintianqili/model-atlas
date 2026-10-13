"""tinymoe: a tiny character-level Transformer whose MLPs are mixtures of experts (kind A), and the dense twin it is
compared with. Shared by train, stats, export and reference.

Each block: x + attention(LayerNorm(x)), then x + MoE(LayerNorm(x)). The MoE layer has a router (a linear layer with no
bias: one score per expert), a softmax over the 4 scores, the top 2 experts kept and their two chances rescaled to add
up to 1 (the gates), and 4 experts (each a small MLP 32 -> 64 -> 32 with ReLU); the output is the gate-weighted sum of
the two chosen experts' outputs. That is the top-k routing of GShard and Switch Transformer with Mixtral's k = 2
(softmax over the kept scores = the kept chances rescaled). No capacity limit in training ("dropless", as Mixtral and
MegaBlocks): the page simulates the capacity rule on the routing (process.ts).

The dense twin has the same blocks with one MLP 32 -> 128 -> 32: as many parameters as the two experts a token uses
(the "active" parameters), half of what the four experts hold together.

Module names are the layer ids export.py writes (tok, pos, blocks[i] -> "B{i+1}.*", ln, head).

    npm run atlas -- py pipeline/tinymoe/model.py        (smoke test: shapes and the parameter counts)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

UNK = "�"  # the "unknown character" symbol; id 0
# The alphabet of char-rnn: the unknown symbol, space and newline, the 52 letters, 9 punctuation marks and the 10
# digits (74). TinyStories' curly quotes and dashes become plain ones first (normalize), as a phone keyboard types them.
ALPHABET = [UNK, " ", "\n"] + [chr(c) for c in range(ord("a"), ord("z") + 1)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] \
    + list(".,\"'!?-:;") + [str(d) for d in range(10)]
INDEX = {c: i for i, c in enumerate(ALPHABET)}
NORMALIZE = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "...", " ": " "}
V, D, HEADS, LAYERS, CTX = len(ALPHABET), 32, 2, 2, 64
E, K, H = 4, 2, 64            # experts, experts per token, hidden units per expert
DENSE_H = K * H               # the dense twin's MLP: as wide as the two experts a token uses
EPS = 1e-5


def normalize(text: str) -> str:
    for a, b in NORMALIZE.items():
        text = text.replace(a, b)
    return text


def encode(text: str) -> list[int]:
    return [INDEX.get(c, 0) for c in normalize(text)]


class Expert(nn.Module):
    """One expert: an ordinary Transformer MLP, 32 -> 64 -> 32 with ReLU."""

    def __init__(self, hidden: int = H):
        super().__init__()
        self.fc = nn.Linear(D, hidden)
        self.out = nn.Linear(hidden, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.fc(x)))


def route(scores: torch.Tensor, k: int = K) -> tuple[torch.Tensor, torch.Tensor]:
    """Router scores [..., E] -> (chances p = softmax(scores), gates): the top k chances rescaled to add up to 1,
    0 for the other experts."""
    p = scores.softmax(-1)
    top_p, top_i = p.topk(k, dim=-1)
    gates = torch.zeros_like(p).scatter(-1, top_i, top_p / top_p.sum(-1, keepdim=True))
    return p, gates


def balance_loss(p: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Switch Transformer's load-balancing loss for top-k: E * sum_e f_e * P_e, with f_e the share of the token slots
    sent to expert e (each token fills k slots) and P_e the router's mean chance for e. 1 when perfectly balanced."""
    f = (gates > 0).float().reshape(-1, E).mean(0) / K
    P = p.reshape(-1, E).mean(0)
    return E * (f * P).sum()


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(D, E, bias=False)
        self.experts = nn.ModuleList([Expert() for _ in range(E)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [..., D] -> (y [..., D], p, gates). Training runs every expert on every token and multiplies the ones
        not chosen by a gate of 0 (the same numbers; the page computes only the two chosen experts)."""
        p, gates = route(self.router(x))
        y = sum(gates[..., e:e + 1] * ex(x) for e, ex in enumerate(self.experts))
        return y, p, gates


class Block(nn.Module):
    """Pre-norm causal Transformer block: x + attn(ln1(x)), then x + mlp(ln2(x)); mlp is a MoE or a dense MLP."""

    def __init__(self, dense: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(D, eps=EPS)
        self.q = nn.Linear(D, D)
        self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D, eps=EPS)
        self.dense = dense
        self.mlp = Expert(DENSE_H) if dense else MoE()

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        h = self.ln1(x)
        q, k, v = (m(h).view(B, T, HEADS, D // HEADS).transpose(1, 2) for m in (self.q, self.k, self.v))
        s = (q @ k.transpose(-2, -1)) / math.sqrt(D // HEADS)
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        a = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        if self.dense:
            return x + self.mlp(self.ln2(x)), None, None
        y, p, gates = self.mlp(self.ln2(x))
        return x + y, p, gates


class TinyMoE(nn.Module):
    def __init__(self, dense: bool = False):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block(dense) for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D, eps=EPS)
        self.head = nn.Linear(D, V)

    def run(self, ids: torch.Tensor):
        """ids [B, T] -> (scores [B, T, V], [(p, gates) per block])."""
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        routes = []
        for b in self.blocks:
            x, p, g = b(x)
            routes.append((p, g))
        return self.head(self.ln(x)), routes

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.run(ids)[0]


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def active_params(m: TinyMoE) -> int:
    """Parameters one token uses: everything except the experts it does not use (K of E experts per block)."""
    n = n_params(m)
    for b in m.blocks:
        if not b.dense:
            n -= (E - K) * n_params(b.mlp.experts[0])
    return n


if __name__ == "__main__":
    for dense in (False, True):
        m = TinyMoE(dense)
        z, routes = m.run(torch.tensor([encode("Once upon a time")]))
        print({"dense": dense, "params": n_params(m), "active": active_params(m), "out": list(z.shape)})
