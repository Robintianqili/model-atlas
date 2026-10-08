"""gnn-variants: export the four trained networks to site/static/models/gnn-variants/ (kind A, design §4.11).

    npm run atlas -- py pipeline/gnn-variants/train.py     # first: model_final.pt and train_metrics.json
    npm run atlas -- export gnn-variants                   # this file, then parity --write and the lesson draft

One spec holds all four networks side by side on the same two inputs (the adjacency A and the one-hot features X), so
one trace runs GCN, GraphSAGE, GIN and GAT on the learner's graph. describe() names every step of model.py:
local ops gcn_conv, sage_conv, gin_agg and gat_conv (ops_gnn.py; sub-steps for the page), core linear, relu and tanh.
Reference cases: Zachary's club, the club with member 11's only friendship removed (an isolated member), no
friendships at all, and one extra friendship across the factions. Also writes karate.json (the graph from networkx,
Zachary 1977, unweighted, the factions and the gcn package's Kamada-Kawai layout); claims.test.ts checks graph.ts.
No "classes" extra and no "class" axis on the outputs: the kit would read a per-member [34, 2] output as a sequence
(notes/gcn.md, Kit issues); the faction names are in the "factions" extra.
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
import ops_gnn as ops  # noqa: E402
from model import SLOPE, VARIANTS, Variants  # noqa: E402

model = Variants()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
SRC = (HERE / "model.py").read_text(encoding="utf-8").splitlines()
KARATE = ak.data.karate()
N = KARATE["n"]
NAME = {"gcn": ("GCN", "GCN"), "sage": ("GraphSAGE", "GraphSAGE"), "gin": ("GIN", "GIN"), "gat": ("GAT", "GAT")}


def code(pattern: str) -> tuple[str, int]:
    return ("pipeline/gnn-variants/model.py", next(i + 1 for i, s in enumerate(SRC) if pattern in s))


def lab(v: str, en: str, zh: str) -> tuple[str, str]:
    return (f"{NAME[v][0]}: {en}", f"{NAME[v][1]}：{zh}")


def graph_layer(b: ak.SpecBuilder, v: str, k: int, h, a):
    g = getattr(getattr(model, v), f"gc{k}")
    L = lab(v, f"graph layer {k}", f"图网络层 {k}")
    if v == "gcn":
        return b.op("gcn_conv", f"gcn.gc{k}", [h, a], weights={"weight": g.weight, "bias": g.bias}, ref=ops.gcn_conv, axes="n,d", label=L,
                    code=code("class GCNConv"))
    if v == "sage":
        return b.op("sage_conv", f"sage.gc{k}", [h, a], weights={"weight": g.lin.weight, "bias": g.lin.bias}, ref=ops.sage_conv, axes="n,d",
                    label=L, code=code("class SAGEConv"))
    if v == "gat":
        return b.op("gat_conv", f"gat.gc{k}", [h, a], params={"slope": SLOPE}, ref=ops.gat_conv, axes="n,d", label=L, code=code("class GATConv"),
                    weights={"weight": g.weight, "att_src": g.att_src, "att_dst": g.att_dst, "bias": g.bias})
    s = b.op("gin_agg", f"gin.gc{k}.agg", [h, a], weights={"eps": g.eps}, ref=ops.gin_agg, axes="n,f",
             label=lab(v, f"sum of layer {k}", f"第 {k} 层的求和"), code=code("def agg"))
    s = b.linear(f"gin.gc{k}.mlp.0", s, g.mlp[0], axes="n,d", label=lab(v, f"MLP {k}, linear 1", f"MLP {k} 的线性层 1"), code=code("self.mlp ="))
    s = b.relu(f"gin.gc{k}.mlp.1", s, axes="n,d", label=lab(v, f"MLP {k}, ReLU", f"MLP {k} 的 ReLU"))
    return b.linear(f"gin.gc{k}.mlp.2", s, g.mlp[2], axes="n,d", label=lab(v, f"MLP {k}, linear 2", f"MLP {k} 的线性层 2"))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    adj, x = case.input
    a = b.input("adj", adj, adapter="graph", kind="graph", axes="n,k", label=("Friendships (adjacency A)", "朋友关系（邻接矩阵 A）"),
                desc=("Row i has a 1 in column j when members i and j are friends.", "成员 i 和 j 是朋友时，第 i 行第 j 列是 1。"))
    f = b.input("x", x, adapter="graph", kind="graph", axes="n,f", label=("One-hot features X", "独热特征 X"),
                desc=("Row i is all 0 except a 1 in column i: member i knows only who it is.", "第 i 行除了第 i 列是 1 以外全是 0：成员 i 只知道自己是谁。"))
    heads = []
    for v in VARIANTS:
        h = f
        for k in (1, 2):
            h = graph_layer(b, v, k, h, a)
            h = b.tanh(f"{v}.act{k}", h, axes="n,d", label=lab(v, "2-D embedding" if k == 2 else "tanh 1", "二维嵌入" if k == 2 else "tanh 1"),
                       code=code("return torch.tanh(self.gc2"))
        heads.append(b.linear(f"{v}.head", h, getattr(model, v).head, axes="n,d", label=lab(v, "faction scores", "派别分数"), code=code("self.head =")))
    b.output(*heads)


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, inp):
        return self.m(*inp)


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
    out = {"format": "gnn-karate/1", "n": N, "edges": [list(p) for p in KARATE["edges"]], "labels": KARATE["labels"].tolist(),
           "clubs": list(KARATE["clubs"]), "pos": xy, "layout": "networkx kamada_kawai_layout, scaled to [0, 1] (y down)",
           "source": "networkx karate_club_graph(), unweighted (ak.data.karate)", "citation": KARATE["citation"]}
    p = ak.site_path("static", "models", "gnn-variants", "karate.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    src = "pipeline/gnn-variants/train.py (CPU float32, seed 0)"
    src10 = f"pipeline/gnn-variants/train.py (the same recipe, seeds 0-{tm['seeds'] - 1}; share of the 32 unlabelled members)"
    facts = {
        "steps": {"value": tm["steps"], "unit": "steps", "source": src},
        "lr": {"value": tm["lr"], "unit": "learning rate (Adam)", "source": src},
        "seeds": {"value": tm["seeds"], "unit": "random starts", "source": src10},
    }
    for v, m in tm["variants"].items():
        facts[f"{v}Params"] = {"value": m["params"], "unit": "parameters", "source": src}
        facts[f"{v}Acc"] = {"value": m["unlabelled_acc"], "unit": "share of the 32 unlabelled members", "source": src}
        facts[f"{v}Loss"] = {"value": round(m["loss_final"], 6), "unit": "cross-entropy of the 2 labelled members", "source": src}
        for key in ("mean", "min", "max"):
            facts[f"{v}{key.capitalize()}"] = {"value": round(m[f"seeds_{key}"], 6), "unit": "share", "source": src10}
    write_karate()
    report = ak.export_model(
        describe, cases(), slug="gnn-variants", name=("GAT / GraphSAGE / GIN", "GAT / GraphSAGE / GIN"), kinds=("trace", "process"),
        module=Wrap(model), module_input=lambda c: c.input,
        extras={"rowNames": [str(i) for i in range(N)], "factions": list(KARATE["clubs"]), "variants": list(VARIANTS)},
        facts=facts,
        training={"dataset": "Zachary's karate club (networkx, unweighted), labels of members 0 and 33 only", "steps": tm["steps"],
                  "optimizer": f"Adam lr {tm['lr']}, full graph every step, seed {tm['seed']}, the same for all four networks",
                  "hardware": "CPU float32",
                  "metrics": {v: {"unlabelled_acc": m["unlabelled_acc"], "wrong": m["wrong"], "loss_final": m["loss_final"]} for v, m in tm["variants"].items()}},
    )
    print(report)
