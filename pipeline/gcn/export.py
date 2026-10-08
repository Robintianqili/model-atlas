"""gcn: export the trained GCN to site/static/models/gcn/ (kind A, design §4.11).

    npm run atlas -- py pipeline/gcn/train.py     # first: model_final.pt and train_metrics.json
    npm run atlas -- export gcn                   # this file, then parity --write and the lesson draft

describe() names every step of model.py's GCN.forward: the two declared inputs (the adjacency A and the one-hot
features X), the local ops gcn_norm (Â with the sub-steps A + I and the degrees) and gcn_conv (with the sub-step
X Wᵀ), tanh, the 2 -> 2 head and a softmax for the shares. Reference cases: Zachary's club, the club with member 11's
only friendship removed (an isolated member), no friendships at all, and one extra friendship across the factions.
Also writes karate.json: the graph from networkx (Zachary 1977, unweighted), the factions and an offline layout
(Kamada-Kawai) for the drawings; claims.test.ts checks that graph.ts holds the same graph.
No "classes" extra and no "class" axis on the outputs: the kit would read the per-member [34, 2] output as a sequence
and print its last row as "the answer" (notes/gcn.md, Kit issues); the faction names are in the "factions" extra.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import networkx as nx
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ops_gcn as ops  # noqa: E402
from model import GCN  # noqa: E402

model = GCN()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
SRC = (HERE / "model.py").read_text(encoding="utf-8").splitlines()
KARATE = ak.data.karate()
N = KARATE["n"]


def code(pattern: str) -> tuple[str, int]:
    return ("pipeline/gcn/model.py", next(i + 1 for i, s in enumerate(SRC) if pattern in s))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    adj, x = case.input
    a = b.input("adj", adj, adapter="graph", kind="graph", axes="n,k", label=("Friendships (adjacency A)", "朋友关系（邻接矩阵 A）"),
                desc=("Row i has a 1 in column j when members i and j are friends.", "成员 i 和 j 是朋友时，第 i 行第 j 列是 1。"))
    f = b.input("x", x, adapter="graph", kind="graph", axes="n,f", label=("One-hot features X", "独热特征 X"),
                desc=("Row i is all 0 except a 1 in column i: member i knows only who it is.", "第 i 行除了第 i 列是 1 以外全是 0：成员 i 只知道自己是谁。"))
    ah = b.op("gcn_norm", "norm", [a], ref=ops.gcn_norm, axes="n,k", label=("Normalized adjacency Â", "归一化邻接矩阵 Â"),
              code=code("def normalize"))
    h = b.op("gcn_conv", "gc1", [f, ah], weights={"weight": model.gc1.weight, "bias": model.gc1.bias}, ref=ops.gcn_conv,
             axes="n,d", label=("Graph layer 1", "图卷积层 1"), code=code("self.gc1 ="))
    h = b.tanh("act1", h, axes="n,d", label=("Tanh 1", "Tanh 1"), code=code("h = torch.tanh(self.gc1"))
    h = b.op("gcn_conv", "gc2", [h, ah], weights={"weight": model.gc2.weight, "bias": model.gc2.bias}, ref=ops.gcn_conv,
             axes="n,d", label=("Graph layer 2", "图卷积层 2"), code=code("self.gc2 ="))
    e = b.tanh("act2", h, axes="n,d", label=("2-D embedding", "二维嵌入"), code=code("return torch.tanh(self.gc2"))
    z = b.linear("head", e, model.head, axes="n,d", label=("Faction scores", "派别分数"), code=code("self.head ="))
    b.output(b.softmax("probs", z, axes="n,d", label=("Faction shares", "派别份额")))


class WithSoftmax(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, inp):
        return torch.softmax(self.m(*inp), dim=-1)


def adjacency(edges) -> torch.Tensor:
    a = torch.zeros(N, N)
    for i, j in edges:
        a[i, j] = a[j, i] = 1.0
    return a


def cases() -> list:
    e = [tuple(p) for p in KARATE["edges"]]
    x = torch.eye(N)
    return [
        ak.Case("karate", "Zachary's karate club", (adjacency(e), x)),
        ak.Case("isolated", "member 11 without friends", (adjacency([p for p in e if 11 not in p]), x), edge=True),
        ak.Case("none", "no friendships", (adjacency([]), x), edge=True),
        ak.Case("bridge", "one more friendship, 16-33", (adjacency(e + [(16, 33)]), x)),
    ]


def write_karate() -> None:
    g = nx.karate_club_graph()
    pos = nx.kamada_kawai_layout(g)
    xs = [float(p[0]) for p in pos.values()]
    ys = [float(p[1]) for p in pos.values()]
    lo_x, hi_y = min(xs), max(ys)
    s = max(max(xs) - lo_x, hi_y - min(ys))
    xy = [[round((float(pos[v][0]) - lo_x) / s, 3), round((hi_y - float(pos[v][1])) / s, 3)] for v in sorted(g.nodes())]
    out = {"format": "gcn-karate/1", "n": N, "edges": [list(p) for p in KARATE["edges"]], "labels": KARATE["labels"].tolist(),
           "clubs": list(KARATE["clubs"]), "pos": xy, "layout": "networkx kamada_kawai_layout, scaled to [0, 1] (y down)",
           "source": "networkx karate_club_graph(), unweighted (ak.data.karate)", "citation": KARATE["citation"]}
    p = ak.site_path("static", "models", "gcn", "karate.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    mean = lambda v: round(sum(v) / len(v), 6)  # noqa: E731
    src = "pipeline/gcn/train.py (CPU float32, seed 0)"
    src10 = "pipeline/gcn/train.py (the same recipe, seeds 0-9, mean share of the 32 unlabelled members)"
    facts = {
        "steps": {"value": tm["steps"], "unit": "steps", "source": src},
        "lr": {"value": tm["lr"], "unit": "learning rate (Adam)", "source": src},
        "lossFinal": {"value": round(tm["loss_final"], 6), "unit": "cross-entropy", "source": src},
        "trainAcc": {"value": tm["unlabelled_acc"], "unit": "share of the 32 unlabelled members", "source": src},
        "shallowAcc": {"value": mean(tm["seeds_shallow_acc"]), "unit": "share", "source": src10},
        "deepAcc": {"value": mean(tm["seeds_deep_acc"]), "unit": "share", "source": src10 + ", 8 graph layers"},
        "seeds": {"value": len(tm["seeds_deep_acc"]), "unit": "random starts", "source": src10},
    }
    write_karate()
    report = ak.export_model(
        describe, cases(), slug="gcn", name=("Graph convolutional network (GCN)", "图卷积网络（GCN）"), kinds=("trace", "train", "process"),
        module=WithSoftmax(model), module_input=lambda c: c.input,
        extras={"rowNames": [str(i) for i in range(N)], "factions": list(KARATE["clubs"])},
        facts=facts,
        training={"dataset": "Zachary's karate club (networkx, unweighted), labels of members 0 and 33 only", "steps": tm["steps"],
                  "optimizer": f"Adam lr {tm['lr']}, full graph every step, seed {tm['seed']}", "hardware": "CPU float32",
                  "metrics": {"unlabelled_acc": tm["unlabelled_acc"], "wrong": tm["wrong"], "loss_final": tm["loss_final"]}},
    )
    print(report)
