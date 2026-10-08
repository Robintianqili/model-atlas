"""alphazero-ttt: the network in float64 numpy and Monte Carlo tree search with PUCT, as the page runs it.

The page (site/src/lib/explainers/alphazero-ttt/search.ts) does the same arithmetic in TypeScript float64 from the same
float32 weights; reference.py records this search frame by frame so `atlas parity` can compare the two.

Tree: node 0 is the root (the position searched). Every node stores N (visits), W (total value) and P (its prior), all
from the side of the player who made the move INTO the node, so Q = W / N is how good that move looks to the player
choosing it. A simulation:
  select   from the root, while the node has children: pick the child with the largest
           U = Q + c * P * sqrt(N_parent) / (1 + N_child)    (Q = 0 for a child never visited; ties: lowest cell)
  expand   at a leaf: a finished game gives its result; otherwise the network gives a prior for each legal move (the
           children) and a value v, both for the player to move at the leaf
  backup   along the path from the leaf up: N += 1; W += -v at the leaf, +v at its parent, -v above, ...
The root is expanded first (N = 1). After `sims` simulations the move played is the child with the most visits (ties:
lowest cell).
"""
from __future__ import annotations

import math

import numpy as np

import game as G

C_PUCT = 1.5


class NumpyNet:
    """The PVNet forward pass in float64 from the float32 weights (as the page computes it)."""

    def __init__(self, state_dict):
        g = lambda k: state_dict[k].detach().cpu().numpy().astype(np.float32).astype(np.float64)
        self.W1, self.b1, self.W2, self.b2 = g("fc1.weight"), g("fc1.bias"), g("fc2.weight"), g("fc2.bias")
        self.Wp, self.bp, self.Wv, self.bv = g("pol.weight"), g("pol.bias"), g("val.weight"), g("val.bias")

    def __call__(self, b):
        x = np.array(G.view(b), dtype=np.float64)
        h = np.maximum(self.W1 @ x + self.b1, 0.0)
        h = np.maximum(self.W2 @ h + self.b2, 0.0)
        s = self.Wp @ h + self.bp
        leg = [i for i in range(9) if b[i] == 0]
        m = max(s[i] for i in leg)
        e = np.zeros(9)
        for i in leg:
            e[i] = math.exp(s[i] - m)
        tot = 0.0
        for i in leg:
            tot += e[i]
        p = e / tot
        v = math.tanh(float(self.Wv[0] @ h + self.bv[0]))
        return p, v


class Tree:
    def __init__(self, cap: int):
        self.N = np.zeros(cap)
        self.W = np.zeros(cap)
        self.P = np.zeros(cap)
        self.parent = np.full(cap, -1.0)
        self.move = np.full(cap, -1.0)
        self.kids: list[list[int]] = [[] for _ in range(cap)]
        self.board: list = [None] * cap
        self.count = 1


def capacity(sims: int) -> int:
    return 1 + 9 * (sims + 1)


def score(t: Tree, node: int, kid: int, c: float):
    n = t.N[kid]
    q = t.W[kid] / n if n > 0 else 0.0
    bonus = c * t.P[kid] * math.sqrt(t.N[node]) / (1 + n)
    return q, bonus, q + bonus


def select_child(t: Tree, node: int, c: float) -> int:
    best, bu = -1, -math.inf
    for k in t.kids[node]:
        u = score(t, node, k, c)[2]
        if u > bu:
            best, bu = k, u
    return best


def expand(t: Tree, node: int, net):
    """Evaluate a leaf: (v for the player to move there, terminal flag, new child ids)."""
    b = t.board[node]
    if G.terminal(b):
        return G.result_for_mover(b), True, []
    p, v = net(b)
    new = []
    for cell in G.legal(b):
        k = t.count
        t.count += 1
        t.P[k], t.parent[k], t.move[k], t.board[k] = p[cell], node, cell, G.play(b, cell)
        t.kids[node].append(k)
        new.append(k)
    return v, False, new


def backup(t: Tree, path: list, v: float):
    sign = -1.0
    for node in reversed(path):
        t.N[node] += 1
        t.W[node] += sign * v
        sign = -sign


def most_visited(t: Tree, node: int = 0) -> int:
    best = -1
    for k in t.kids[node]:
        if best < 0 or t.N[k] > t.N[best]:
            best = k
    return best


def search(b, net, sims: int = 50, c: float = C_PUCT, noise=None):
    """Plain search (no frames): the tree after `sims` simulations. noise: (eps, alpha, rng) mixes Dirichlet noise into
    the root priors (self-play training only; the page never does this)."""
    t = Tree(capacity(sims))
    t.board[0] = b
    v, _, kids = expand(t, 0, net)
    if noise is not None and kids:
        eps, alpha, rng = noise
        d = rng.dirichlet([alpha] * len(kids))
        for k, dk in zip(kids, d):
            t.P[k] = (1 - eps) * t.P[k] + eps * dk
    backup(t, [0], v)
    for _ in range(sims):
        path = [0]
        while t.kids[path[-1]]:
            path.append(select_child(t, path[-1], c))
        v, _, _ = expand(t, path[-1], net)
        backup(t, path, v)
    return t
