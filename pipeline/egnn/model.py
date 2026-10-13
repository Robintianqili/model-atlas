"""egnn: a small E(n)-equivariant graph network (Satorras, Hoogeboom and Welling, 2021) on toy molecules (kind A).

Input: n atoms, positions x [n, 2] and one-hot kinds [n, 3]; every atom is joined to every other (a full graph). An
embedding turns the kind into F numbers h. Each of the two EGNN layers then, for every pair (i, j):
    d2_ij = |x_i - x_j|^2                                  (pair_dist2: the only thing the network learns about geometry)
    m_ij  = SiLU(e2(SiLU(e1([h_i, h_j, d2_ij]))))          (edge_concat, the message network)
    w_ij  = cw(m_ij)                                       (one number per pair: how hard j pushes i)
    v_i   = sum_{j != i} (x_i - x_j) w_ij / (n - 1)        (vec_sum: the push on atom i, an arrow)
    x_i  <- x_i + v_i                                      (the coordinate update)
    h_i  <- h_i + n2(SiLU(n1([h_i, sum_{j != i} m_ij])))   (nb_sum, the node network, a residual)
The outputs: the energy E = sum_i o2(SiLU(o1(h_i))) (sum_atoms) and a force per atom f_i = v_i(layer 1) + v_i(layer 2),
the total distance the network moved the atom. Distances do not change when the molecule turns, shifts or is mirrored,
so E does not; the pushes are sums of differences x_i - x_j, so f turns (and mirrors) with the molecule.

The functions take any leading batch dims ([..., n, 2]); export.py traces one molecule. Module names are the layer ids.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

K, F, M, L = 3, 16, 16, 2


def pair_dist2(x: torch.Tensor) -> torch.Tensor:
    """[..., n, D] -> [..., n, n]"""
    return ((x.unsqueeze(-2) - x.unsqueeze(-3)) ** 2).sum(-1)


def edge_concat(h: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """h [..., n, F], d2 [..., n, n] -> [..., n, n, 2F + 1]: row (i, j) = h_i, h_j, d2_ij"""
    n, f = h.shape[-2], h.shape[-1]
    hi = h.unsqueeze(-2).expand(*h.shape[:-2], n, n, f)
    hj = h.unsqueeze(-3).expand(*h.shape[:-2], n, n, f)
    return torch.cat([hi, hj, d2.unsqueeze(-1)], -1)


def vec_sum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [..., n, D], w [..., n, n, 1] -> [..., n, D]: sum_{j != i} (x_i - x_j) w_ij / max(1, n - 1)"""
    n = x.shape[-2]
    return ((x.unsqueeze(-2) - x.unsqueeze(-3)) * w).sum(-2) / max(1, n - 1)


def nb_sum(m: torch.Tensor) -> torch.Tensor:
    """[..., n, n, M] -> [..., n, M]: sum over j != i"""
    n = m.shape[-2]
    off = 1 - torch.eye(n, dtype=m.dtype, device=m.device)
    return (m * off.unsqueeze(-1)).sum(-2)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = nn.Linear(2 * F + 1, M)   # message network, layer 1
        self.e2 = nn.Linear(M, M)           # message network, layer 2
        self.cw = nn.Linear(M, 1)           # one push weight per pair
        self.n1 = nn.Linear(F + M, F)       # node network, layer 1
        self.n2 = nn.Linear(F, F)           # node network, layer 2

    def forward(self, h, x):
        m = Fn.silu(self.e2(Fn.silu(self.e1(edge_concat(h, pair_dist2(x))))))
        v = vec_sum(x, self.cw(m))
        h = h + self.n2(Fn.silu(self.n1(torch.cat([h, nb_sum(m)], -1))))
        return h, x + v, v


class EGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(K, F)
        self.layers = nn.ModuleList([Layer() for _ in range(L)])
        self.o1 = nn.Linear(F, F)
        self.o2 = nn.Linear(F, 1)

    def forward(self, pos: torch.Tensor, kind: torch.Tensor):
        """pos [..., n, 2], kind [..., n, 3] -> energy [..., 1], force [..., n, 2]"""
        h, x, f = self.embed(kind), pos, 0
        for layer in self.layers:
            h, x, v = layer(h, x)
            f = f + v
        e = self.o2(Fn.silu(self.o1(h))).sum(-2)
        return e, f


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    m = EGNN()
    x, k = torch.randn(5, 2), torch.eye(3)[torch.tensor([0, 1, 2, 0, 1])]
    c, s = 0.6, 0.8
    R = torch.tensor([[c, -s], [s, c]])
    e0, f0 = m(x, k)
    e1, f1 = m(x @ R.T, k)
    print(n_params(m), "params; turned: |dE|", float((e1 - e0).abs().max()), "max |f(Rx) - R f(x)|", float((f1 - f0 @ R.T).abs().max()))
