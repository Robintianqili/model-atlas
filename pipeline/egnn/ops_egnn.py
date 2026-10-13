"""Python mirrors of the egnn local ops (site/src/lib/explainers/egnn/ops/egnn.ts), the same arithmetic as model.py.

    pair_dist2   x [n, D] -> [n, n]                 squared distance of every pair
    edge_concat  h [n, F], d2 [n, n] -> [n, n, 2F+1] the message network's input for every pair (h_i, h_j, d2_ij)
    vec_sum      x [n, D], w [n, n, 1] -> [n, D]    sum_{j != i} (x_i - x_j) w_ij / max(1, n - 1)
    nb_sum       m [n, n, M] -> [n, M]              sum_{j != i} m_ij
    sum_atoms    y [n, C] -> [C]                    sum over the atoms

Used from export.py's describe() as b.op("<op>", id, [...], ref=ops_egnn.<op>).
"""
import model as md

OPS = {
    "pair_dist2": {"version": 1, "category": "graph", "label": ("Squared distances of every pair", "每对原子的距离平方"), "attrs": {}, "weights": [], "subs": []},
    "edge_concat": {"version": 1, "category": "graph", "label": ("Inputs of every message", "每条消息的输入"), "attrs": {}, "weights": [], "subs": []},
    "vec_sum": {"version": 1, "category": "graph", "label": ("Push every atom along its pairs", "沿每一对推动原子"), "attrs": {}, "weights": [], "subs": []},
    "nb_sum": {"version": 1, "category": "graph", "label": ("Add up the messages each atom receives", "把每个原子收到的消息加起来"), "attrs": {}, "weights": [], "subs": []},
    "sum_atoms": {"version": 1, "category": "pool", "label": ("Add up the shares of all atoms", "把所有原子的份额加起来"), "attrs": {}, "weights": [], "subs": []},
}


def pair_dist2(xs, params, weights):
    return md.pair_dist2(xs[0])


def edge_concat(xs, params, weights):
    return md.edge_concat(xs[0], xs[1])


def vec_sum(xs, params, weights):
    return md.vec_sum(xs[0], xs[1])


def nb_sum(xs, params, weights):
    return md.nb_sum(xs[0])


def sum_atoms(xs, params, weights):
    return xs[0].sum(0)
