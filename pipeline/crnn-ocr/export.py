"""crnn-ocr: export the trained tiny CRNN to site/static/models/crnn-ocr/ (kind A).

    CRNN_OCR_DATA=<data folder> npm run atlas -- export crnn-ocr      # then parity --write and the lesson draft

describe() names every step of CRNN.forward (model.py) with a builder call: two convolutions (ReLU, one max pool), the
local op `columns` (ops_columns.py: feature maps -> 32 columns of 128 features), the forward LSTM, the backward LSTM
as flip -> LSTM -> flip (local op `flip`, ops_flip.py), the two joined per column, 11 scores per column and their
softmax. export_model() checks it against the module (AK103) and writes spec.json, weights.bin, reference.json (the
page's presets and edge fixtures, made by the page's own adapter in make_data.ts) and facts.json (train_metrics.json
and the test-set measurements of measure()). The presets come from the adapter (presetsFromAdapter).
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
from model import CRNN, SYMBOLS, cases as data_cases, greedy, load  # noqa: E402
from ops_columns import reference as columns_ref  # noqa: E402
from ops_flip import reference as flip_ref  # noqa: E402

model = CRNN()
model.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
model = model.cpu().eval()
SRC = (HERE / "model.py").read_text(encoding="utf-8").splitlines()


def code(pattern: str) -> tuple[str, int]:
    """("pipeline/crnn-ocr/model.py", line) of the first line of model.py that contains `pattern` (the Code tab)."""
    return ("pipeline/crnn-ocr/model.py", next(i + 1 for i, s in enumerate(SRC) if pattern in s))


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    x = b.input("input", case.input, adapter="word-strokes", kind="drawing", axes="c,y,x", label=("Your writing", "你写的字"),
                desc=("16 x 64 ink values: 0 is paper, 1 is full ink.", "16 x 64 个墨迹值：0 是纸，1 是满墨。"))
    h = b.conv2d("conv1", x, model.conv1, label=("Convolution 1", "卷积 1"), code=code("self.conv1 ="),
                 desc=("8 kernels of 3 x 3 slide over the strip.", "8 个 3 x 3 的卷积核在长条上滑动。"))
    h = b.relu("relu1", h, label=("ReLU 1", "ReLU 1"), code=code("F.max_pool2d(F.relu"))
    h = b.maxpool2d("pool1", h, kernel=2, label=("Max pooling", "最大池化"), code=code("F.max_pool2d(F.relu"))
    h = b.conv2d("conv2", h, model.conv2, label=("Convolution 2", "卷积 2"), code=code("self.conv2 ="),
                 desc=("16 kernels of 3 x 3 x 8 over the pooled maps.", "16 个 3 x 3 x 8 的卷积核作用在池化后的图上。"))
    h = b.relu("relu2", h, label=("Feature maps", "特征图"), code=code("cols = columns("))
    cols = b.op("columns", "cols", [h], ref=columns_ref, label=("Columns", "列"), code=code("def columns("), axes="t,f")
    hf = b.lstm("fwd", cols, model.fwd, label=("LSTM, left to right", "LSTM，从左往右"), code=code("hf, _ = self.fwd"))
    rv = b.op("flip", "rev", [cols], ref=flip_ref, label=("Columns, right to left", "列，从右往左"), code=code("hb, _ = self.bwd"), axes="t,f")
    hb = b.lstm("bwd", rv, model.bwd, label=("LSTM, right to left", "LSTM，从右往左"), code=code("hb, _ = self.bwd"))
    hb = b.op("flip", "bwd_back", [hb], ref=flip_ref, label=("Backward states, in column order", "反向状态，按列的顺序"),
              code=code("torch.flip(hb"), axes="t,d")
    both = b.concat("both", [hf, hb], axis=1, label=("Both directions", "两个方向"), code=code("torch.cat("), axes="t,d")
    z = b.linear("head", both, model.head, label=("Scores per column", "每列的分数"), code=code("self.head ="), axes="x,class")
    b.output(b.softmax("probs", z, label=("Chances per column", "每列的概率"), axes="x,class"))


class WithSoftmax(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=-1)[:, 0]


@torch.no_grad()
def measure() -> dict:
    """Test-set numbers (CPU float32): the share of columns whose top symbol is the blank, and the words read wrong."""
    x, _, _, texts = load("test")
    scores = model(x)
    blank = (scores.argmax(-1) == 0).float().mean().item()
    wrong = sum(t != r for t, r in zip(texts, greedy(scores)))
    return {"blankShare": (blank, "share of the 32 columns of 7,000 test words whose top symbol is the blank"),
            "testWrong": (wrong, "test words read wrong")}


if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    cases = [ak.Case(c["id"], c["label"], c["ink"], edge=c["edge"]) for c in data_cases()]
    src = "pipeline/crnn-ocr/export.py measure() (CPU float32, 7,000 test words of make_data.ts)"
    facts = {k: {"value": v, "unit": u, "source": src} for k, (v, u) in measure().items()}
    tr = "pipeline/crnn-ocr/train.py (CPU float32)"
    for k, key, u in (("params", "params", "parameters"), ("train_words", "trainWords", "words"), ("epochs", "epochs", "epochs"),
                      ("best_epoch", "bestEpoch", "epoch (counting from 0)"), ("test_acc", "testAcc", "share of test words read exactly"),
                      ("val_acc", "valAcc", "share of validation words read exactly"), ("test_words", "testWords", "words"),
                      ("wild_acc", "wildAcc", "share of 7,000 messier words (glyphs.ts wild 2.2) read exactly")):
        facts[key] = {"value": tm[k], "unit": u, "source": tr}
    for n, v in tm["test_acc_by_length"].items():
        facts[f"acc{n}"] = {"value": v, "unit": f"share of the 1,000 test words of {n} digits read exactly", "source": tr}
    facts["maxLen"] = {"value": max(int(k) for k in tm["test_acc_by_length"]), "unit": "digits (longest training word)",
                       "source": "pipeline/crnn-ocr/make_data.ts"}
    report = ak.export_model(
        describe, cases, slug="crnn-ocr", name=("Tiny CRNN + CTC", "微型 CRNN + CTC"), kinds=("trace", "process"),
        module=WithSoftmax(model), module_input=lambda c: c.input.unsqueeze(0),
        extras={"classes": [{"id": "blank", "en": "blank (–)", "zh": "空白（–）"}] + [{"id": s, "en": s, "zh": s} for s in SYMBOLS[1:]]},
        facts=facts,
        training={"dataset": "synthetic handwriting-like numbers (pipeline/crnn-ocr/make_data.ts with the page's glyphs.ts): "
                             f"{tm['train_words']:,} words of 1-7 digits, jittered shape, size, slant, tilt, spacing and pen width",
                  "epochs": tm["epochs"], "bestEpoch": tm["best_epoch"], "batchSize": tm["batch"],
                  "optimizer": f"{tm['optimizer']}, lr {tm['lr']}; CTC loss (blank = class 0)",
                  "hardware": "RTX 5090 Laptop GPU (training); CPU float32 (evaluation, export)",
                  "metrics": {"test_acc": tm["test_acc"], "val_acc": tm["val_acc"], "wild_acc": tm["wild_acc"],
                              "test_acc_by_length": tm["test_acc_by_length"]}},
    )
    print(report)
