"""hello-xor-2: export the trained XorNet (sigmoid output) to site/static/models/hello-xor-2/ (kind A).

    npm run atlas -- export hello-xor-2        # from site/: runs this file, then parity --write and the lesson draft
    python pipeline/hello-xor-2/export.py      # also works (PYTHONPATH=pipeline, or the fallback below)

describe() is the graph, written once: fc1 -> tanh -> fc2 -> sigmoid. export_model() calls it once per case, proves it
equals module(x) (AK103) and writes spec.json, weights.bin (9 floats), reference.json, facts.json and presets.json.
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import X, Y, XorNet  # noqa: E402

CKPT = HERE / "model_final.pt"   # written by train.py (torch seed 0, 1,500 Adam steps)

model = XorNet()
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model = model.cpu().eval()


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    x = b.input("input", case.input, adapter="xor-point", kind="points2d", axes="f",
                label=("Your point", "你的点"), desc=("The two numbers x1 and x2 of the point.", "这个点的两个数 x1 和 x2。"))
    h = b.linear("fc1", x, model.fc1, label=("Hidden layer", "隐藏层"),
                 desc=("Two weighted sums of x1 and x2, each plus a bias.", "x1 和 x2 的两个加权和，各自再加一个偏置。"),
                 code=("pipeline/hello-xor-2/model.py", 22))
    h = b.tanh("act1", h, label=("Tanh", "Tanh"), desc=("Squashes each sum into the range -1 to 1.", "把每个和压到 -1 到 1 之间。"))
    z = b.linear("fc2", h, model.fc2, label=("Output score", "输出分数"),
                 desc=("One weighted sum of the two hidden values: any number, positive for XOR true.",
                       "两个隐藏值的一个加权和：可以是任意数，正数表示 XOR 为真。"),
                 code=("pipeline/hello-xor-2/model.py", 23))
    y = b.sigmoid("out", z, label=("Sigmoid: the probability", "Sigmoid：概率"),
                  desc=("Squashes the score into 0 to 1: the chance that XOR is true.", "把分数压到 0 到 1 之间：XOR 为真的概率。"),
                  code=("pipeline/hello-xor-2/model.py", 29))
    b.output(y)


# The four XOR points, plus two edge cases away from the training points.
CASES = [
    ak.Case("p00", "(0, 0)", torch.tensor([0.0, 0.0]), edge=True),
    ak.Case("p01", "(0, 1)", torch.tensor([0.0, 1.0])),
    ak.Case("p10", "(1, 0)", torch.tensor([1.0, 0.0])),
    ak.Case("p11", "(1, 1)", torch.tensor([1.0, 1.0])),
    ak.Case("mid", "(0.5, 0.5)", torch.tensor([0.5, 0.5]), edge=True),
    ak.Case("far", "(3, -2)", torch.tensor([3.0, -2.0]), edge=True),
]

PRESETS = {"format": "hello-xor-presets/1", "points": [
    {"id": "p00", "en": "(0, 0): XOR is 0", "zh": "(0, 0)：XOR 为 0", "x": [0, 0]},
    {"id": "p01", "en": "(0, 1): XOR is 1", "zh": "(0, 1)：XOR 为 1", "x": [0, 1]},
    {"id": "p10", "en": "(1, 0): XOR is 1", "zh": "(1, 0)：XOR 为 1", "x": [1, 0]},
    {"id": "p11", "en": "(1, 1): XOR is 0", "zh": "(1, 1)：XOR 为 0", "x": [1, 1]},
]}

if __name__ == "__main__":
    mfile = HERE / "train_metrics.json"
    tm = json.loads(mfile.read_text(encoding="utf-8")) if mfile.is_file() else {}
    with torch.no_grad():
        loss = float(F.binary_cross_entropy_with_logits(model.logits(X), Y))
        correct = int((((model(X) > 0.5).float()) == Y).sum())
    report = ak.export_model(
        describe, CASES, slug="hello-xor-2", name=("Hello XOR 2 (sigmoid output)", "你好 XOR 2（sigmoid 输出）"),
        kinds=("trace", "train", "process"),
        module=model, module_input=lambda c: c.input.unsqueeze(0),
        facts={
            "trainLoss": {"value": loss, "unit": "bce", "source": "pipeline/hello-xor-2/export.py (model_final.pt on the 4 XOR points)"},
            "correct": {"value": correct, "unit": "points of 4", "source": "pipeline/hello-xor-2/export.py"},
            "trainSteps": {"value": int(tm.get("steps", 1500)), "unit": "steps", "source": "pipeline/hello-xor-2/train.py"},
        },
        presets=PRESETS,
        training={"dataset": "the XOR truth table (4 points)", "steps": int(tm.get("steps", 1500)),
                  "optimizer": "Adam lr 0.05, full batch, binary cross-entropy", "metrics": {"final_loss": loss, "correct": correct}},
        extras={"target": "XOR(x1, x2) for x1, x2 in {0, 1}"},
    )
    print(report)
