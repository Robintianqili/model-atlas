"""egnn: export the trained EGNN to site/static/models/egnn/ (kind A).

    npm run atlas -- export egnn          # runs this file, then parity --write and the lesson draft

describe() names every step of EGNN.forward (model.py): the two declared inputs (positions and one-hot kinds), the
embedding, per EGNN layer the local ops pair_dist2, edge_concat, vec_sum and nb_sum (ops_egnn.py) around the message,
push and node networks, then the energy head and sum_atoms. Outputs: the energy [1] and the force [n, 2]. Reference
cases: the page's presets and edge fixtures through the page's own adapter (make_data.ts), the bent molecule turned by
90 degrees and mirrored, and a random molecule of 12 atoms. facts.json: stats.py and train_metrics.json.
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

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
import ops_egnn as ops  # noqa: E402
from model import EGNN, K  # noqa: E402
from stats import measure  # noqa: E402

model = EGNN()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
SRC = (HERE / "model.py").read_text(encoding="utf-8").splitlines()


def code(pattern: str) -> tuple[str, int]:
    return ("pipeline/egnn/model.py", next(i + 1 for i, s in enumerate(SRC) if pattern in s))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    pos, kind = case.input
    x = b.input("pos", pos, adapter="points2d", kind="points2d", axes="n,xy", shape=["N", 2],
                dims={"N": (1, data.meta()["maxAtoms"], ("atoms", "原子"))}, label=("Atom positions", "原子位置"),
                desc=("x and y of every atom, in molecule units.", "每个原子的 x 和 y，单位是分子单位。"))
    k = b.input("kind", kind, adapter="points2d", kind="points2d", axes="n,f", shape=["N", K], label=("Atom kinds (one-hot)", "原子种类（独热）"),
                desc=("Row i has a 1 in the column of atom i's kind.", "第 i 行在原子 i 的种类那一列是 1。"))
    h = b.linear("embed", k, model.embed, axes="n,f", label=("Atom numbers h", "原子的数 h"), code=code("self.embed ="))
    f = None
    for l, layer in enumerate(model.layers, start=1):
        p, last = f"L{l}.", l == len(model.layers)
        d2 = b.op("pair_dist2", p + "d2", [x], ref=ops.pair_dist2, axes="n,k", label=(f"Layer {l}: squared distances", f"第 {l} 层：距离平方"), code=code("def pair_dist2"))
        e = b.op("edge_concat", p + "edge", [h, d2], ref=ops.edge_concat, axes="n,k,f", label=(f"Layer {l}: message inputs", f"第 {l} 层：消息的输入"), code=code("def edge_concat"))
        m = b.linear(p + "e1", e, layer.e1, axes="n,k,f", label=(f"Layer {l}: message network 1", f"第 {l} 层：消息网络 1"), code=code("self.e1 ="))
        m = b.silu(p + "e1act", m, axes="n,k,f", label=(f"Layer {l}: SiLU", f"第 {l} 层：SiLU"), code=code("m = Fn.silu"))
        m = b.linear(p + "e2", m, layer.e2, axes="n,k,f", label=(f"Layer {l}: message network 2", f"第 {l} 层：消息网络 2"), code=code("self.e2 ="))
        m = b.silu(p + "msg", m, axes="n,k,f", label=(f"Layer {l}: messages", f"第 {l} 层：消息"), code=code("m = Fn.silu"))
        w = b.linear(p + "cw", m, layer.cw, axes="n,k,f", label=(f"Layer {l}: push weights", f"第 {l} 层：推力权重"), code=code("self.cw ="))
        v = b.op("vec_sum", p + "push", [x, w], ref=ops.vec_sum, axes="n,xy", label=(f"Layer {l}: pushes", f"第 {l} 层：推动"), code=code("v = vec_sum"))
        if not last:
            x = b.add(p + "pos", x, v, axes="n,xy", label=(f"Layer {l}: moved positions", f"第 {l} 层：移动后的位置"), code=code("return h, x + v, v"))
        a = b.op("nb_sum", p + "agg", [m], ref=ops.nb_sum, axes="n,f", label=(f"Layer {l}: summed messages", f"第 {l} 层：消息之和"), code=code("nb_sum(m)], -1)"))
        u = b.concat(p + "cat", [h, a], axis=1, axes="n,f", label=(f"Layer {l}: h and the message sum", f"第 {l} 层：h 与消息之和"), code=code("nb_sum(m)], -1)"))
        u = b.linear(p + "n1", u, layer.n1, axes="n,f", label=(f"Layer {l}: node network 1", f"第 {l} 层：节点网络 1"), code=code("self.n1 ="))
        u = b.silu(p + "n1act", u, axes="n,f", label=(f"Layer {l}: SiLU", f"第 {l} 层：SiLU"), code=code("nb_sum(m)], -1)"))
        u = b.linear(p + "n2", u, layer.n2, axes="n,f", label=(f"Layer {l}: node network 2", f"第 {l} 层：节点网络 2"), code=code("self.n2 ="))
        h = b.add(p + "h", h, u, axes="n,f", label=(f"Layer {l}: new atom numbers h", f"第 {l} 层：新的原子数 h"), code=code("h = h + self.n2"))
        f = v if f is None else b.add("force", f, v, axes="n,xy", label=("Forces: the pushes of both layers", "力：两层推动之和"), code=code("f = f + v"))
    y = b.linear("o1", h, model.o1, axes="n,f", label=("Energy head 1", "能量头 1"), code=code("self.o1 ="))
    y = b.silu("o1act", y, axes="n,f", label=("Energy head: SiLU", "能量头：SiLU"), code=code("e = self.o2"))
    y = b.linear("share", y, model.o2, axes="n,f", label=("Energy share of each atom", "每个原子的能量份额"), code=code("self.o2 ="))
    E = b.op("sum_atoms", "energy", [y], ref=ops.sum_atoms, axes="f", label=("Energy", "能量"), code=code("e = self.o2"))
    b.output(E, f)


class Both(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, inp):
        return self.m(*inp)


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    mt = data.meta()
    eye = torch.eye(K)
    cases = [ak.Case(c["id"], c["label"], (c["pos"], eye[c["kind"]]), edge=c["edge"]) for c in data.cases()]
    src = "pipeline/egnn/stats.py (CPU float32, the 5,000 test molecules of make_data.ts)"
    facts = {k: {"value": v, "unit": u, "source": src} for k, (v, u) in measure(model).items()}
    for k, key, u in (("train_n", "trainN", "molecules"), ("epochs", "epochs", "epochs"), ("best_epoch", "bestEpoch", "epoch")):
        facts[key] = {"value": tm[k], "unit": u, "source": "pipeline/egnn/train.py"}
    facts["nMin"] = {"value": mt["nmin"], "unit": "atoms", "source": "pipeline/egnn/make_data.ts (fewest atoms of a training molecule)"}
    facts["nMax"] = {"value": mt["nmax"], "unit": "atoms", "source": "pipeline/egnn/make_data.ts (most atoms of a training molecule)"}
    report = ak.export_model(
        describe, cases, slug="egnn", name=("Small EGNN", "小型 EGNN"), kinds=("trace",),
        module=Both(model), module_input=lambda c: c.input,
        facts=facts,
        training={"dataset": "synthetic 2-D toy molecules (pipeline/egnn/make_data.ts with the page's mol.ts): 3 atom kinds, "
                             f"{mt['nmin']}-{mt['nmax']} atoms, a made-up Morse pair energy and its exact forces",
                  "steps": tm["steps"], "batchSize": tm["batch"], "epochs": tm["epochs"], "bestEpoch": tm["best_epoch"],
                  "optimizer": f"Adam lr {tm['lr']} (cosine to 0); loss mse(E)/var(E) + mse(F)/var(F); batches of equal atom counts",
                  "hardware": f"{tm['device']} (training, {tm['seconds']} s); CPU float32 (evaluation, export)",
                  "metrics": {k: facts[k]["value"] for k in ("energyMAE", "forceMAE", "energyMAD", "forceMAD", "testN")}},
    )
    print(report)
