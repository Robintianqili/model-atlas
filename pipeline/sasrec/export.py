"""sasrec: export the tiny SASRec to site/static/models/sasrec/ (kind A, design §4.11).

    npm run atlas -- run sasrec logs --opt '{"n":10000}' --save      # the TS logs (sessions, fixed inputs)
    npm run atlas -- py pipeline/sasrec/train.py                       # model_final.pt, train_metrics.json
    npm run atlas -- export sasrec                                     # this file, then parity --write and the draft

The graph on the learner's clicks (ids [T], T = 1 … 12): item rows + position rows, one pre-norm block with one causal
attention head (scores, mask, softmax, mix) and a point-wise FFN, a last LayerNorm, then the score of every product as
the dot product with its row of the same item table (tied weights), and a softmax. Reference cases: the presets and
edge inputs whose ids TS wrote into frame 0 of the logs tape. The product names (spec extra `classes`) are read from
world.ts with Node, so TS stays the one source of truth.
"""
from __future__ import annotations

import json
import os
import subprocess
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
from model import MAX_LEN, SASRec  # noqa: E402

M = "pipeline/sasrec/model.py"
SRC = (HERE / "model.py").read_text(encoding="utf-8").splitlines()


def code(pattern: str) -> tuple[str, int]:
    return (M, next(i + 1 for i, s in enumerate(SRC) if pattern in s))


model = SASRec(dropout=0.0)
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
TM = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))

h, blob = ak.tape.read_tape(ak.site_path(".atlas", "sasrec", "logs"))
FIXED = np.asarray(ak.tape.decode(next(f for f in h["frames"] if f["phase"] == "fixed")["state"]["fixed"], blob))
WORLD = Path(ak.site_path("src", "lib", "explainers", "sasrec", "world.ts")).resolve().as_uri()
ITEMS = json.loads(subprocess.run(["node", "-e", f"import('{WORLD}').then((m) => console.log(JSON.stringify(m.ITEMS)))"],
                                  capture_output=True, text=True, check=True, env=os.environ).stdout)
CLASSES = [{"id": f"p{i}", "en": x["name"]["en"], "zh": x["name"]["zh"]} for i, x in enumerate(ITEMS)]


class Probs(torch.nn.Module):
    """module(x) for AK103: x = ids [1, T] -> shares [T, 31], as recorded."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=-1)[0]


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    ids = b.input("ids", case.input, adapter="clicks", kind="ratings", axes="t", shape=["T"], dtype="i32",
                  dims={"T": (1, MAX_LEN, ("clicks", "点击数"))}, label=("Your clicks as product ids", "你的点击：商品编号"))
    e = b.embedding("item", ids, model.item, label=("Item rows", "商品行"), code=code("self.item ="), axes="t,d")
    p = b.position_embedding("pos", ids, model.pos, label=("Position rows", "位置行"), code=code("self.pos ="), axes="t,d")
    x = b.add("x", e, p, label=("Item + position", "商品 + 位置"), code=code("x = self.drop(self.item(ids)"), axes="t,d")
    a = b.layernorm("ln1", x, model.ln1, label=("LayerNorm 1", "层归一化 1"), code=code("h = self.ln1(x)"), axes="t,d")
    q = b.linear("q", a, model.q, n_head=1, label=("Queries Q", "查询 Q"), code=code("s = self.q(h)"), axes="t,d")
    k = b.linear("k", a, model.k, n_head=1, label=("Keys K", "键 K"), code=code("s = self.q(h)"), axes="t,d")
    v = b.linear("v", a, model.v, n_head=1, label=("Values V", "值 V"), code=code("x = x + self.drop(w) @ self.v(h)"), axes="t,d")
    s = b.attention_scores("att.scores", q, k, 1, label=("Attention scores QKᵀ/√d", "注意力分数 QKᵀ/√d"), code=code("s = self.q(h)"), axes="h,tq,tk")
    s = b.causal_mask("att.masked", s, label=("Causal mask", "因果掩码"), code=code("s = s.masked_fill"), axes="h,tq,tk")
    w = b.softmax("att.w", s, label=("Attention weights", "注意力权重"), code=code("return x, h, s.softmax(-1)"), axes="h,tq,tk")
    m = b.attention_mix("att.mix", w, v, 1, label=("Weighted sum of values", "值的加权和"), code=code("x = x + self.drop(w) @ self.v(h)"), axes="t,d")
    x = b.add("add1", x, m, label=("Add attention", "加上注意力"), code=code("x = x + self.drop(w) @ self.v(h)"), axes="t,d")
    f = b.layernorm("ln2", x, model.ln2, label=("LayerNorm 2", "层归一化 2"), code=code("self.ln2(x)"), axes="t,d")
    f = b.linear("ffn.fc1", f, model.fc1, label=("FFN: layer 1", "前馈网络：第 1 层"), code=code("self.fc1 ="), axes="t,d")
    f = b.relu("ffn.relu", f, label=("FFN: ReLU", "前馈网络：ReLU"), code=code("F.relu(self.fc1"), axes="t,d")
    f = b.linear("ffn.fc2", f, model.fc2, label=("FFN: layer 2", "前馈网络：第 2 层"), code=code("self.fc2 ="), axes="t,d")
    x = b.add("add2", x, f, label=("Add the FFN", "加上前馈网络"), code=code("x = x + self.fc2"), axes="t,d")
    o = b.layernorm("ln", x, model.ln, label=("Last LayerNorm", "最后的层归一化"), code=code("return self.ln(x)"), axes="t,d")
    y = b.linear("logits", o, tied_to="item.weight", label=("Score of every product", "每件商品的分数"), code=code("@ self.item.weight.t()"), axes="t,vocab",
                 desc=("Row t times every row of the item table (the same table the clicks were looked up in): the score of each product as the next click.",
                       "第 t 行乘以商品表的每一行（就是查点击时用的那张表）：每件商品作为下一次点击的分数。"))
    b.output(b.softmax("probs", y, label=("Shares of the next click", "下一次点击的份额"), code=code("[B, T, N_ITEMS] scores"), axes="t,vocab"))


IDS = ["trip", "coffee", "console", "back", "baker", "one", "twelve", "drone"]
EDGE = {"one", "twelve", "drone"}
assert len(IDS) == len(FIXED), f"{len(FIXED)} fixed inputs in the tape, {len(IDS)} ids here (world.ts FIXED)"
CASES = [ak.Case(k, f"{k}: clicks {[int(i) for i in FIXED[n] if i >= 0]}", torch.tensor([int(i) for i in FIXED[n] if i >= 0], dtype=torch.int32), edge=k in EDGE)
         for n, k in enumerate(IDS)]


if __name__ == "__main__":
    src = "pipeline/sasrec/train.py (CPU float32)"
    test = src + ", 1,000 test sessions, last click held out"
    F = lambda v, unit, source=src: {"value": v, "unit": unit, "source": source}  # noqa: E731
    facts = {
        "params": F(TM["params"], "count", "spec"), "trainSessions": F(TM["trainSessions"], "sessions"), "trainTargets": F(TM["trainTargets"], "next clicks"),
        "trainClicks": F(TM["trainClicks"], "clicks"), "valSessions": F(TM["valSessions"], "sessions"), "testSessions": F(TM["testSessions"], "sessions"),
        "epochs": F(TM["epochs"], "epochs"), "bestEpoch": F(TM["bestEpoch"], "epoch", src + ", best validation hit rate @5"), "batch": F(TM["bs"], "sessions"),
        "hit5": F(TM["testHit5"], "share", test), "hit5Pop": F(TM["testHit5Pop"], "share", test), "hit5Markov": F(TM["testHit5Markov"], "share", test),
        "hit5Shuffled": F(TM["testHit5Shuffled"], "share", test), "hit5LastOnly": F(TM["testHit5LastOnly"], "share", test),
        "popShareRec": F(TM["popShareRec"], "share", test), "popShareTarget": F(TM["popShareTarget"], "share", test),
        "newItemInTop5": F(TM["newItemInTop5"], "sessions", test), "newItemPlaceMean": F(TM["newItemPlaceMean"], "place", test),
        "attLastIsMax": F(TM["attLastIsMax"], "share", test + ", histories of 2 clicks or more"), "attSessions": F(TM["attSessions"], "sessions", test),
    }
    report = ak.export_model(
        describe, CASES, slug="sasrec", name=("Sequential recommender (tiny SASRec)", "序列推荐（小型 SASRec）"), kinds=("trace", "process"),
        module=Probs(model), module_input=lambda c: c.input.long().unsqueeze(0),
        facts=facts, extras={"classes": CLASSES, "clicks": TM["clicks"], "popTop5": TM["popTop5"]},
        training={"dataset": f"synthetic shop sessions from world.ts: {TM['trainSessions']:,} sessions, {TM['trainTargets']:,} next clicks", "steps": TM["steps"],
                  "metrics": {"test_hit5": TM["testHit5"], "test_hit5_markov": TM["testHit5Markov"], "test_hit5_pop": TM["testHit5Pop"], "test_hit5_shuffled": TM["testHit5Shuffled"]}},
    )
    print(report)
