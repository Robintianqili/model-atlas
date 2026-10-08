"""tinyvit: export the trained TinyViT to site/static/models/tinyvit/ (kind A, design §4.11).

    npm run atlas -- export tinyvit          # runs this file, then parity --write and the lesson draft

describe() names every step of TinyViT.forward (model.py) with a builder call: the patch convolution, the three local
ops (ops_tokens.py: to_tokens, cls_token, select_token), the position table, four pre-norm blocks and the class head.
export_model() checks it against the module (AK103) and writes spec.json, weights.bin, reference.json (5 test
drawings plus the blank and one-dot edge cases), facts.json and presets.json.

facts.json holds numbers measured here on the CPU in float32 over the 15,000 test drawings (TinyCNN's split): the test
accuracy of both models, how often they agree, and where the [CLS] row of each block looks (stats.py computes them).
presets.json is TinyCNN's own presets file (the same Quick, Draw! test drawings), so both pages offer the same ones.
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
import ops_tokens as ops  # noqa: E402
from model import BLOCKS, CLASSES, HEADS, ZH, TinyViT  # noqa: E402
from stats import measure  # noqa: E402

CKPT = HERE / "model_final.pt"
model = TinyViT()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model = model.cpu().eval()
SRC = (HERE / "model.py").read_text(encoding="utf-8").splitlines()


def code(pattern: str) -> tuple[str, int]:
    """("pipeline/tinyvit/model.py", line) of the first line of model.py that contains `pattern` (the Code tab)."""
    return ("pipeline/tinyvit/model.py", next(i + 1 for i, s in enumerate(SRC) if pattern in s))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    x = b.input("input", case.input, adapter="drawing28", kind="drawing", axes="c,y,x",
                label=("Your drawing", "你的画"), desc=("28x28 numbers: 0 is empty, 1 is full ink.", "28x28 个数：0 是空白，1 是满墨。"))
    h = b.conv2d("patch", x, model.patch, label=("Patch embedding", "图块嵌入"), code=code("self.patch ="), axes="c,y,x",
                 desc=("Each 7x7 patch (49 pixels) becomes 48 numbers.", "每个 7x7 图块（49 个像素）变成 48 个数。"))
    h = b.op("to_tokens", "tokens", [h], ref=ops.to_tokens, label=("Patches become rows", "图块变成行"), axes="t,d",
             code=code("flatten(2).transpose"))
    h = b.op("cls_token", "cls", [h], weights={"weight": model.cls}, ref=ops.cls_token, axes="t,d",
             label=("[CLS] row in front", "前面加上 [CLS] 行"), code=code("self.cls ="))
    p = b.position_embedding("pos", h, weight=model.pos, label=("Position rows", "位置行"), code=code("self.pos ="), axes="t,d")
    h = b.add("embed", h, p, label=("Rows + positions", "行 + 位置"), code=code("+ self.pos"), axes="t,d", teach={"narration": {
        "what": {"en": "Each row gets the learned position row of its place added, so the same stroke in another patch gives other numbers. "
                       "The position rows are the same for every drawing; only their count follows the rows. The result has shape {outShape}.",
                 "zh": "每一行都加上它所在位置的那一行学来的位置数，所以同样的笔画放在别的图块里会得到不同的数。"
                       "位置行对每幅画都一样，只有它们的行数跟着输入走。结果的形状是 {outShape}。"}}})
    for i, blk in enumerate(model.blocks):
        n, en, zh = f"B{i + 1}", f"Block {i + 1} · ", f"第 {i + 1} 块 · "
        a = b.layernorm(f"{n}.ln1", h, blk.ln1, label=(en + "LayerNorm 1", zh + "层归一化 1"), code=code("h = self.ln1(x)"), axes="t,d")
        q = b.linear(f"{n}.q", a, blk.q, n_head=HEADS, label=(en + "Queries Q", zh + "查询 Q"), code=code("q = self.q(h)"), axes="t,d")
        k = b.linear(f"{n}.k", a, blk.k, n_head=HEADS, label=(en + "Keys K", zh + "键 K"), code=code("k = self.k(h)"), axes="t,d")
        v = b.linear(f"{n}.v", a, blk.v, n_head=HEADS, label=(en + "Values V", zh + "值 V"), code=code("v = self.v(h)"), axes="t,d")
        s = b.attention_scores(f"{n}.scores", q, k, HEADS, label=(en + "Attention scores", zh + "注意力分数"), code=code("s = (q @ k"), axes="h,tq,tk")
        w = b.softmax(f"{n}.attn", s, label=(en + "Attention weights", zh + "注意力权重"), code=code("s.softmax(-1)"), axes="h,tq,tk")
        m = b.attention_mix(f"{n}.mix", w, v, HEADS, label=(en + "Mix of values", zh + "值的混合"), code=code("a = (w @ v)"), axes="t,d")
        o = b.linear(f"{n}.proj", m, blk.proj, label=(en + "Attention output", zh + "注意力输出"), code=code("self.proj(a)"), axes="t,d")
        h = b.add(f"{n}.add1", h, o, label=(en + "Add attention", zh + "加上注意力"), code=code("x = x + self.proj(a)"), axes="t,d")
        c = b.layernorm(f"{n}.ln2", h, blk.ln2, label=(en + "LayerNorm 2", zh + "层归一化 2"), code=code("self.ln2(x)"), axes="t,d")
        f = b.linear(f"{n}.fc1", c, blk.fc1, label=(en + "MLP up (192)", zh + "MLP 放大（192）"), code=code("self.fc1 ="), axes="t,d")
        f = b.gelu(f"{n}.gelu", f, label=(en + "GELU", zh + "GELU"), code=code("F.gelu"), axes="t,d")
        f = b.linear(f"{n}.fc2", f, blk.fc2, label=(en + "MLP down (48)", zh + "MLP 缩回（48）"), code=code("self.fc2 ="), axes="t,d")
        h = b.add(f"{n}.add2", h, f, label=(en + "Add MLP", zh + "加上 MLP"), code=code("return x + self.fc2"), axes="t,d")
    h = b.layernorm("ln", h, model.ln, label=("Final LayerNorm", "最后的层归一化"), code=code("self.ln = "), axes="t,d")
    h = b.op("select_token", "clsout", [h], params={"index": 0}, ref=ops.select_token, axes="d",
             label=("The [CLS] row", "[CLS] 行"), code=code("[:, 0]"))
    y = b.linear("head", h, model.head, label=("Class scores", "类别分数"), code=code("self.head ="), axes="class")
    b.output(b.softmax("softmax", y, label=("Probabilities", "概率"), axes="class"))


class WithSoftmax(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=-1)


def cases() -> list:
    """5 test drawings (TinyCNN's preset drawings of apple, car, fish, star and umbrella) + blank + one dot."""
    pre = json.loads(ak.site_path("static", "models", "tinycnn", "presets.json").read_text(encoding="utf-8"))
    out = []
    for c in pre["classes"]:
        if c["id"] in ("apple", "car", "fish", "star", "umbrella"):
            item = c["items"][0]
            px = np.frombuffer(__import__("base64").b64decode(item["b64"]), np.uint8).reshape(1, 28, 28)
            out.append(ak.Case(f"{c['id']}-{item['fileIndex']}", c["id"], torch.tensor(px, dtype=torch.float32) / 255))
    dot = torch.zeros(1, 28, 28)
    dot[0, 14, 14] = 1.0
    return out + [ak.Case("blank", "blank", torch.zeros(1, 28, 28), edge=True), ak.Case("dot", "one dot", dot, edge=True)]


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    presets = json.loads(ak.site_path("static", "models", "tinycnn", "presets.json").read_text(encoding="utf-8"))
    presets["note"] = "TinyCNN's presets (static/models/tinycnn/presets.json): " + presets.get("note", "")
    m = measure(model)
    src = "pipeline/tinyvit/stats.py (CPU float32, 15,000 test drawings)"
    facts = {k: {"value": v, "unit": u, "source": src} for k, (v, u) in m.items()}
    eps, k = tm["label_smoothing"], len(CLASSES)
    facts["smoothTarget"] = {"value": round(1 - eps + eps / k, 4), "unit": "share",
                             "source": "pipeline/tinyvit/train.py (label smoothing: the training target of the right class)"}
    facts["bestEpoch"] = {"value": tm["best_epoch"], "unit": "epoch", "source": "pipeline/tinyvit/train.py"}
    facts["epochs"] = {"value": tm["epochs"], "unit": "epoch", "source": "pipeline/tinyvit/train.py"}
    report = ak.export_model(
        describe, cases(), slug="tinyvit", name=("TinyViT", "TinyViT"), kinds=("trace",),
        module=WithSoftmax(model), module_input=lambda c: c.input.unsqueeze(0),
        extras={"classes": [{"id": c, "en": c, "zh": ZH[c]} for c in CLASSES], "heldOutClasses": ["ladder", "envelope"]},
        facts=facts, presets=presets,
        training={"dataset": "Quick, Draw! numpy bitmaps (TinyCNN's 10 classes and split)", "steps": tm["steps"],
                  "batchSize": tm["batch"], "epochs": tm["epochs"], "bestEpoch": tm["best_epoch"],
                  "optimizer": f"AdamW lr {tm['lr']} (5 warm-up epochs, cosine), weight decay {tm['weight_decay']} on matrices, label smoothing {tm['label_smoothing']}",
                  "augmentation": "TinyCNN's augment(): shift, scale 0.8-1.1, rotation +-12 deg, stroke thickness jitter",
                  "hardware": "RTX 5090 Laptop GPU (training); CPU float32 (evaluation, export)",
                  "metrics": {"test_acc": tm["test_acc"], "val_acc": tm["val_acc"], "test_n": tm["test_n"],
                              "per_class_test_acc": tm["per_class_test_acc"], "confusion": tm["confusion"]}},
    )
    print(report)
