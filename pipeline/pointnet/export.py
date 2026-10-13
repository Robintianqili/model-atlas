"""pointnet: export the trained tiny PointNet to site/static/models/pointnet/ (kind A).

    npm run atlas -- export pointnet          # runs this file, then parity --write and the lesson draft

describe() names every step of PointNet.forward (model.py) with a builder call: the shared MLP on every point (fc1,
ReLU, fc2, ReLU), the local op max_points (ops_max_points.py) and the classifier (fc3, ReLU, fc4, softmax).
export_model() checks it against the module (AK103) and writes spec.json, weights.bin, reference.json (the page's five
presets, one point, a straight line and the largest cloud, 1,024 points; all made by the page's own adapter in
make_data.ts) and facts.json (stats.py and train_metrics.json). The presets come from the adapter (presetsFromAdapter).
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
from data import cases as data_cases, meta  # noqa: E402
from model import CLASSES, ZH, PointNet  # noqa: E402
from ops_max_points import reference as max_points  # noqa: E402
from stats import measure  # noqa: E402

model = PointNet()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
SRC = (HERE / "model.py").read_text(encoding="utf-8").splitlines()
MAX_N = 1024


def code(pattern: str) -> tuple[str, int]:
    """("pipeline/pointnet/model.py", line) of the first line of model.py that contains `pattern` (the Code tab)."""
    return ("pipeline/pointnet/model.py", next(i + 1 for i, s in enumerate(SRC) if pattern in s))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    x = b.input("input", case.input, adapter="points2d", kind="points2d", axes="k,xy", shape=["N", 2],
                dims={"N": (1, MAX_N, ("points", "点"))}, label=("Your points", "你的点"),
                desc=("x and y of every point, centred and scaled so the farthest point is at distance 1.",
                      "每个点的 x 和 y，已经居中并缩放，最远的点距离为 1。"))
    h = b.linear("fc1", x, model.fc1, label=("Shared layer 1", "共享层 1"), code=code("self.fc1 ="), axes="k,f",
                 desc=("The same 2 -> 32 weighted sums for every point.", "每个点都用同一组 2 -> 32 的加权和。"))
    h = b.relu("relu1", h, label=("ReLU 1", "ReLU 1"), code=code("F.relu(self.fc2(F.relu"), axes="k,f")
    h = b.linear("fc2", h, model.fc2, label=("Shared layer 2", "共享层 2"), code=code("self.fc2 ="), axes="k,f",
                 desc=("The same 32 -> 32 weighted sums for every point.", "每个点都用同一组 32 -> 32 的加权和。"))
    h = b.relu("relu2", h, label=("Point features", "点特征"), code=code("F.relu(self.fc2(F.relu"), axes="k,f")
    g = b.op("max_points", "pool", [h], ref=max_points, label=("Max over the points", "对所有点取最大值"),
             code=code(".max(dim=-2)"), axes="f")
    y = b.linear("fc3", g, model.fc3, label=("Classifier layer", "分类层"), code=code("self.fc3 ="), axes="f")
    y = b.relu("relu3", y, label=("ReLU 3", "ReLU 3"), code=code("return self.fc4(F.relu"), axes="f")
    y = b.linear("fc4", y, model.fc4, label=("Class scores", "类别分数"), code=code("self.fc4 ="), axes="class")
    b.output(b.softmax("softmax", y, label=("Probabilities", "概率"), axes="class"))


class WithSoftmax(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=-1)


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    mt = meta()
    cases = [ak.Case(c["id"], c["label"], c["xy"], edge=c["edge"]) for c in data_cases()]
    src = "pipeline/pointnet/stats.py (CPU float32, 5,000 test clouds of make_data.ts)"
    facts = {k: {"value": v, "unit": u, "source": src} for k, (v, u) in measure(model).items()}
    for k, u in (("train_n", "clouds"), ("epochs", "epochs"), ("best_epoch", "epoch")):
        facts[{"train_n": "trainN", "epochs": "epochs", "best_epoch": "bestEpoch"}[k]] = {"value": tm[k], "unit": u, "source": "pipeline/pointnet/train.py"}
    facts["nMin"] = {"value": mt["nmin"], "unit": "points", "source": "pipeline/pointnet/make_data.ts (fewest points of a training cloud)"}
    facts["nMax"] = {"value": mt["nmax"], "unit": "points", "source": "pipeline/pointnet/make_data.ts (most points of a training cloud)"}
    facts["stray"] = {"value": mt["stray"], "unit": "radii", "source": "shapes.ts STRAY (where the limits step puts the stray point)"}
    report = ak.export_model(
        describe, cases, slug="pointnet", name=("Tiny PointNet", "微型 PointNet"), kinds=("trace",),
        module=WithSoftmax(model), module_input=lambda c: c.input.unsqueeze(0),
        extras={"classes": [{"id": c, "en": c, "zh": ZH[c]} for c in CLASSES]},
        facts=facts,
        training={"dataset": "synthetic 2-D point clouds (pipeline/pointnet/make_data.ts with the page's shapes.ts): 5 shapes, "
                             f"{mt['nmin']}-{mt['nmax']} points, any rotation, stretch 0.85-1.15, jitter, uneven spacing, small gaps",
                  "steps": tm["steps"], "batchSize": tm["batch"], "epochs": tm["epochs"], "bestEpoch": tm["best_epoch"],
                  "optimizer": f"Adam lr {tm['lr']} (cosine to 0), cross-entropy; clouds padded to 256 points by repeating their own points",
                  "hardware": "RTX 5090 Laptop GPU (training); CPU float32 (evaluation, export)",
                  "metrics": {"test_acc": tm["test_acc"], "val_acc": tm["val_acc"], "test_n": tm["test_n"],
                              "per_class_test_acc": tm["per_class_test_acc"], "confusion": tm["confusion"]}},
    )
    print(report)
