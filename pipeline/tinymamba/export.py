"""tinymamba: export the trained model to site/static/models/tinymamba/ (kind A, DEVKIT section 5).

    npm run atlas -- export tinymamba

Layers (per block i = 1, 2): B<i>.norm (LayerNorm), B<i>.in (linear d -> D), B<i>.u (SiLU), B<i>.dt, B<i>.B, B<i>.C (linears), B<i>.scan (the
local op ssm_scan, sub-steps #delta and #h), B<i>.out (linear D -> d), B<i>.res (the residual add); then head and
probs (after a last LayerNorm, norm). The input is a row of token ids of symbolic length T (1..40); the cases include T = 1 and T = 40. Writes
spec.json, weights.bin, reference.json, facts.json (train_metrics.json + stats.json) and presets.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import ALPHABET, BODY_MAX, BODY_MIN, K_MAX  # noqa: E402
from model import TinyMamba  # noqa: E402
from ops_ssm_scan import ssm_scan  # noqa: E402

T_MAX = 40
metrics = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
stats = json.loads((HERE / "stats.json").read_text(encoding="utf-8")) if (HERE / "stats.json").is_file() else {}
CFG = metrics["config"]
model = TinyMamba(**CFG)
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
INDEX = {c: i for i, c in enumerate(ALPHABET)}

PRESETS = [
    ("start", "Three letters", "三个字母", ".c..a....e..???"),
    ("one", "One letter", "一个字母", "....g......?"),
    ("four", "Four letters (the most in training)", "四个字母（训练里最多的）", ".h.b..f..d....????"),
    ("twice", "A letter twice", "同一个字母两次", "..d...d.b...???"),
    ("six", "Six letters (more than in training)", "六个字母（比训练时多）", "a.c.e.g.b.d??????"),
    ("far", "One letter, far back", "一个字母，离得很远", "b" + "." * 30 + "?"),
]


def encode(text: str) -> torch.Tensor:
    return torch.tensor([INDEX[c] for c in text[-T_MAX:]], dtype=torch.int64)


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    x = b.input("input", case.input, adapter="seq", kind="text", axes="t", shape=["T"],
                dims={"T": (1, T_MAX, ("tokens", "词元"))}, label=("Your row", "你的一行"))
    x = b.embedding("emb", x, model.emb, label=("Token vectors", "词元向量"))
    N = CFG["N"]
    for i, blk in enumerate(model.blocks, 1):
        p, en, zh = f"B{i}", f"Block {i}: ", f"第 {i} 块："
        z = b.layernorm(f"{p}.norm", x, blk.norm, label=(en + "layer normalization", zh + "层归一化"))
        u = b.linear(f"{p}.in", z, blk.inp, label=(en + "into channels", zh + "进入通道"))
        u = b.silu(f"{p}.u", u, label=(en + "channels u", zh + "通道 u"))
        dt = b.linear(f"{p}.dt", u, blk.dt, label=(en + "step size before softplus", zh + "softplus 之前的步长"))
        Bm = b.linear(f"{p}.B", u, blk.B, label=(en + "write strengths B", zh + "写入强度 B"))
        Cm = b.linear(f"{p}.C", u, blk.C, label=(en + "read strengths C", zh + "读出强度 C"))
        y = b.op("ssm_scan", f"{p}.scan", [u, dt, Bm, Cm], params={"state": N}, weights={"A_log": blk.A_log, "D": blk.Dskip},
                 ref=ssm_scan, label=(en + "selective scan", zh + "选择性扫描"))
        o = b.linear(f"{p}.out", y, blk.out, label=(en + "back to the row", zh + "回到这一行"))
        x = b.add(f"{p}.res", x, o, label=(en + "add to the input", zh + "加回输入"))
    x = b.layernorm("norm", x, model.norm, label=("Layer normalization", "层归一化"))
    z = b.linear("head", x, model.head, label=("Scores of the 10 tokens", "10 个词元的分数"))
    b.output(b.softmax("probs", z, label=("Chances of the 10 tokens", "10 个词元的概率")))


class WithSoftmax(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids):
        return torch.softmax(self.m(ids), dim=-1)


CASES = [ak.Case("ask", "a single ?", encode("?"), edge=True)]
CASES += [ak.Case(k, t, encode(t)) for k, _, _, t in PRESETS[:5]]
CASES += [ak.Case("max", f"{T_MAX} tokens", encode(".a...b..c....d" + "." * 22 + "????"), edge=True)]


def fact(v, source, unit=None):
    return {"value": v, "source": source, **({"unit": unit} if unit else {})}


FACTS = {
    "params": fact(sum(p.numel() for p in model.parameters()), "export.py"),
    "steps": fact(metrics["steps"], "train.py"), "batch": fact(metrics["batch"], "train.py"),
    "trainExamples": fact(metrics["train"]["examples"], "train.py"),
    "valAcc": fact(metrics["valAcc"], "train.py: 5,000 new rows (seed 2), answers at the ?", "share"),
    "valExact": fact(metrics["valExact"], "train.py: rows with every answer right", "share"),
    "kMax": fact(K_MAX, "data.py"), "bodyMin": fact(BODY_MIN, "data.py"), "bodyMax": fact(BODY_MAX, "data.py"),
    "valExamples": fact(metrics["val"]["examples"], "train.py"),
}
for k, v in stats.items():
    FACTS[k] = fact(v, "stats.py")

if __name__ == "__main__":
    report = ak.export_model(
        describe, CASES, slug="tinymamba", name=("Tiny Mamba (state-space model)", "小型 Mamba（状态空间模型）"),
        kinds=("trace", "process"), module=WithSoftmax(model), module_input=lambda c: c.input, facts=FACTS,
        presets={"items": [{"id": k, "en": en, "zh": zh, "text": t} for k, en, zh, t in PRESETS]},
        training={"dataset": "selective copy (synthetic, task.ts / data.py)", "steps": metrics["steps"],
                  "metrics": {"valAcc": metrics["valAcc"], "valExact": metrics["valExact"]}},
        extras={"alphabet": ALPHABET, "context": T_MAX, "config": CFG})
    print(report)
