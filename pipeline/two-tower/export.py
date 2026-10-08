"""two-tower: export the two towers to site/static/models/two-tower/ (kind A, design §4.11).

    npm run atlas -- run two-tower logs --opt '{"n":6000}' --save      # the TS logs (features, likes, fixed inputs)
    npm run atlas -- py pipeline/two-tower/train.py                       # model_final.pt, train_metrics.json
    npm run atlas -- export two-tower                                     # this file, then parity --write and the draft

One graph with two inputs: `item` [8] (the learner's new movie) and `user` [15] (the learner's genres and picks). The
item tower on the new movie; the user tower on the learner; then `scores` [36]: the learner's vector against every row
of the item table, a linear layer without bias whose weight IS the table (the item tower's outputs for the 36 movies,
computed here once, as a real service computes them offline). Reference cases: the presets and edge inputs whose
features TS wrote into frame 0 of the logs tape, so parity runs on the bytes the page computes.
Spec extras: `likes` (each movie's likes in the training logs) and `ranker` (train.py), `rankerRef` (each case's
shortlist reordered by the ranker, computed here in float64: a claim test compares the TS ranker with it).
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

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import TwoTower  # noqa: E402

M = "pipeline/two-tower/model.py"
model = TwoTower()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
TM = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))

h, blob = ak.tape.read_tape(ak.site_path(".atlas", "two-tower", "logs"))
fx = {k: ak.tape.decode(v, blob) for k, v in next(f for f in h["frames"] if f["phase"] == "fixed")["state"].items()}
ITEMS = torch.tensor(np.asarray(fx["items"], dtype="<f4"))
with torch.no_grad():
    TABLE = model.item(ITEMS).detach().clone()   # [36, 8]: the item tower, run once for every movie


class Page(torch.nn.Module):
    """module(x) for AK103: x = (user [1, 15], item [1, 8]) -> (the 36 scores, the new movie's vector), as recorded."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return (self.m.user(x[0])[0] @ self.m.item(ITEMS).t(), self.m.item(x[1])[0])


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    it = b.input("item", case.input["item"], adapter="prefs", kind="ratings", axes="f", label=("Your new movie: 8 numbers", "你的新电影：8 个数"),
                 desc=("Its genre as 6 switches (1 = this genre), then its year and its length, scaled to about −1 … 1.", "它的类型用 6 个开关表示（1 = 这个类型），然后是它的年份和片长，缩放到大约 −1 … 1。"))
    us = b.input("user", case.input["user"], adapter="prefs", kind="ratings", axes="f", label=("You: 15 numbers", "你：15 个数"),
                 desc=("The genres you like (6 switches), the genre shares of the movies you picked (6), their mean year and length, and 1 if you picked any.",
                       "你喜欢的类型（6 个开关）、你所选电影的类型占比（6 个）、它们的平均年份和片长，以及选过电影时为 1。"))
    h = b.linear("item.fc1", it, model.item.fc1, label=("Item tower: layer 1", "物品塔：第 1 层"), code=(M, 25))
    h = b.relu("item.relu", h, label=("Item tower: ReLU", "物品塔：ReLU"), code=(M, 29))
    v = b.linear("item.fc2", h, model.item.fc2, label=("Item tower: the movie’s vector", "物品塔：电影的向量"), code=(M, 29))
    h = b.linear("user.fc1", us, model.user.fc1, label=("User tower: layer 1", "用户塔：第 1 层"), code=(M, 25))
    h = b.relu("user.relu", h, label=("User tower: ReLU", "用户塔：ReLU"), code=(M, 29))
    u = b.linear("user.fc2", h, model.user.fc2, label=("User tower: your vector", "用户塔：你的向量"), code=(M, 29))
    s = b.linear("scores", u, weight=TABLE, axes="k", label=("Scores: you against every movie", "分数：你对每部电影"),
                 desc=("One dot product per movie: your vector times each row of the item table, the 36 movies’ vectors computed in advance.",
                       "每部电影一个点积：你的向量乘以物品表的每一行，也就是提前算好的 36 部电影的向量。"), code=(M, 40))
    b.output(s, v)


IDS = ["space", "laughs", "classic", "scary", "none", "genres-only", "one-pick", "twelve", "twin"]
EDGE = {"none", "genres-only", "one-pick", "twelve", "twin"}
U = torch.tensor(np.asarray(fx["fixedUsers"], dtype="<f4"))
V = torch.tensor(np.asarray(fx["fixedItems"], dtype="<f4"))
PICKS = [[int(x) for x in row if x >= 0] for row in np.asarray(fx["fixedPicks"])]
assert len(IDS) == len(U), f"{len(U)} fixed inputs in the tape, {len(IDS)} ids here (world.ts FIXED)"
CASES = [ak.Case(k, f"{k}: picks {PICKS[n]}", {"user": U[n], "item": V[n]}, edge=k in EDGE) for n, k in enumerate(IDS)]


def ranker_ref():
    """each case's shortlist (the best 8 unpicked movies) reordered by the ranker, in float64 (recs.ts reranked)"""
    r = TM["ranker"]
    lp = np.log1p(np.asarray(TM["likes"], dtype=np.float64))
    pop = (lp - r["popMean"]) / r["popStd"]
    out = []
    with torch.no_grad():
        for n, k in enumerate(IDS):
            s = (model.user(U[n]) @ TABLE.t()).double().numpy()
            top = [i for i in sorted(range(36), key=lambda i: (-s[i], i)) if i not in set(PICKS[n])][:8]
            z = [r["w"][0] * s[i] + r["w"][1] * pop[i] + r["b"] for i in top]
            order = sorted(range(8), key=lambda j: (-z[j], j))
            out.append({"case": k, "shortlist": top, "order": [top[j] for j in order], "z": [z[j] for j in order]})
    return out


if __name__ == "__main__":
    src = "pipeline/two-tower/train.py (CPU float32)"
    F = lambda v, unit, source=src: {"value": v, "unit": unit, "source": source}  # noqa: E731
    facts = {
        "params": F(TM["params"], "count", "spec"), "trainViewers": F(TM["trainViewers"], "viewers"), "trainPairs": F(TM["trainPairs"], "pairs"),
        "testViewers": F(TM["testViewers"], "viewers"), "valViewers": F(TM["valViewers"], "viewers"), "epochs": F(TM["epochs"], "epochs"),
        "bestEpoch": F(max(TM["history"], key=lambda h: h["valRecall8"])["epoch"], "epoch", src + ", best validation recall@8"), "batch": F(TM["bs"], "pairs"),
        "uniqueCols": F(TM["uniqueCols"], "movies per batch", src + ", first epoch"),
        "recall8": F(TM["testRecall8"], "share", src + ", test viewers"), "recall8Pop": F(TM["testRecall8Pop"], "share", src + ", test viewers"),
        "recall8Random": F(TM["testRecall8Random"], "share", src + ", test viewers"),
        "hits3Before": F(TM["hits3Before"], "movies", src + ", test viewers"), "hits3After": F(TM["hits3After"], "movies", src + ", test viewers"),
        "rankFitRows": F(TM["ranker"]["fitRows"], "rows", src + ", validation viewers"),
        "twinLikesA": F(TM["twinLikes"][0], "likes", src), "twinLikesB": F(TM["twinLikes"][1], "likes", src),
    }
    report = ak.export_model(
        describe, CASES, slug="two-tower", name=("Two-tower retrieval", "双塔召回"), kinds=("trace", "process"),
        module=Page(model), module_input=lambda c: (c.input["user"].unsqueeze(0), c.input["item"].unsqueeze(0)),
        facts=facts, extras={"likes": TM["likes"], "ranker": {k: TM["ranker"][k] for k in ("w", "b", "popMean", "popStd")}, "rankerRef": ranker_ref()},
        training={"dataset": f"synthetic logs from world.ts: {TM['trainViewers']:,} viewers, {TM['trainPairs']:,} (viewer, liked movie) pairs", "steps": TM["steps"],
                  "metrics": {"test_recall8": TM["testRecall8"], "test_recall8_pop": TM["testRecall8Pop"], "test_recall8_random": TM["testRecall8Random"]}},
    )
    print(report)
