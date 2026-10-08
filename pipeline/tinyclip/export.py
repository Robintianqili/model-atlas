"""tinyclip: export TinyCLIP to site/static/models/tinyclip/ (kind A, design §4.11).

    npm run atlas -- run tinyclip synth --opt '{"n":24000}' --save      # the TS-rendered pairs and fixed drawings
    npm run atlas -- py pipeline/tinyclip/train.py --run v1               # model_final.pt
    npm run atlas -- export tinyclip                                      # this file, then parity --write and the draft

One graph with two inputs: `image` [1, 28, 28] (the learner's drawing) and `text` [K, L] word ids (K captions of at
most L words, 0 pads). Both towers, then the score of the drawing against each caption and the softmax over captions.
The reference cases are the fixed drawings TS rendered (frame 0 of the synth tape: presets, partner pairs, edge cases),
so parity runs on exactly the bytes the page computes. facts.json: stats.py's test numbers and the training run.
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
import ops_clip  # noqa: E402
from model import VOCAB, TinyCLIP  # noqa: E402
from stats import measure  # noqa: E402
from train import load_fixed  # noqa: E402

M = "pipeline/tinyclip/model.py"
def load(name: str) -> TinyCLIP:
    m = TinyCLIP()
    m.load_state_dict(torch.load(HERE / name, map_location="cpu"))
    return m.cpu().eval()


model = load("model_final.pt")


MODEL = [model]   # describe() records this model (the trained one, then the starting weights)


class Pair(torch.nn.Module):
    """module(x) for AK103: x = (image [1, 1, 28, 28], ids [K, L]) -> the K shares, as describe() records them."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x[0], x[1])


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    model = MODEL[0]
    img = b.input("image", case.input["image"], adapter="shapes-captions", kind="drawing", axes="c,y,x",
                  label=("Your drawing", "你的画"), desc=("28×28 numbers: the frame and your strokes, 0 paper to 1 ink.", "28×28 个数：画框和你的笔画，0 是纸，1 是墨。"))
    ids = b.input("text", case.input["text"], adapter="shapes-captions", kind="text", axes="k,t", shape=["K", "L"], dtype="i32",
                  dims={"K": (1, 3, ("captions", "句描述")), "L": (1, 8, ("words", "个词"))})
    h = b.conv2d("img.conv1", img, model.conv1, label=("Image: convolution 1", "图像：卷积 1"), code=(M, 43))
    h = b.relu("img.relu1", h, label=("Image: ReLU 1", "图像：ReLU 1"))
    h = b.maxpool2d("img.pool1", h, 2, label=("Image: max pooling 1", "图像：最大池化 1"))
    h = b.conv2d("img.conv2", h, model.conv2, label=("Image: convolution 2", "图像：卷积 2"), code=(M, 44))
    h = b.relu("img.relu2", h, label=("Image: ReLU 2", "图像：ReLU 2"))
    h = b.maxpool2d("img.pool2", h, 2, label=("Image: max pooling 2", "图像：最大池化 2"))
    h = b.flatten("img.flat", h, label=("Image: flatten", "图像：展平"))
    h = b.linear("img.proj", h, model.img_proj, label=("Image vector", "图像向量"), code=(M, 45))
    a = b.op("l2norm", "img.norm", [h], ref=ops_clip.l2norm, label=("Image vector, length 1", "图像向量，长度 1"), axes="d", code=(M, 53))
    e = b.embedding("txt.emb", ids, model.emb, label=("Text: word rows", "文本：词行"), axes="k,t,d", code=(M, 46))
    p = b.op("mean_pool", "txt.pool", [e, ids], ref=ops_clip.mean_pool, label=("Text: average of the words", "文本：词的平均"), axes="k,d", code=(M, 34))
    p = b.linear("txt.proj", p, model.txt_proj, label=("Caption vectors", "描述向量"), axes="k,d", code=(M, 47))
    t = b.op("l2norm", "txt.norm", [p], ref=ops_clip.l2norm, label=("Caption vectors, length 1", "描述向量，长度 1"), axes="k,d", code=(M, 56))
    s = b.op("similarity", "sim", [a, t], weights={"logit_scale": model.logit_scale.detach().reshape(1)}, ref=ops_clip.similarity,
             label=("Scores: drawing vs each caption", "分数：画对每句描述"), axes="k", code=(M, 65))
    b.output(b.softmax("probs", s, label=("Shares over your captions", "在你的描述之间的份额"), axes="k"))


def text_of(ids: torch.Tensor) -> torch.Tensor:
    """[3, 8] ids of a fixed drawing -> [K, L]: captions without a known word dropped, columns trimmed to the longest."""
    rows = [r for r in ids if int((r != 0).sum())]
    L = max(int((r != 0).sum()) for r in rows)
    return torch.stack(rows)[:, :L].contiguous()


X, IDS, CAPS, FIXED_HASH = load_fixed()
NAMES = ["circle", "square", "triangle", "star", "cross", "stack", "p-triangle", "p-square", "p-star", "p-cross", "p-pair", "blank", "dot", "small"]
PICK = ["circle", "square", "triangle", "star", "cross", "stack", "p-square", "p-pair", "blank", "dot", "small"]
CASES = [ak.Case(n, f"{n}: " + " | ".join(c for c in CAPS[NAMES.index(n)] if c), {"image": X[NAMES.index(n)], "text": text_of(IDS[NAMES.index(n)])},
                 edge=n in ("blank", "dot", "small")) for n in PICK]
CASES.append(ak.Case("no-words", "circle: a caption with no known word (all padding)", {"image": X[0], "text": torch.zeros(1, 1, dtype=torch.long)}, edge=True))

if __name__ == "__main__":
    tm = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))
    st = measure(model)
    st0 = measure(load("model_init.pt"))
    src = "pipeline/tinyclip/stats.py (CPU float32, TF32 off)"
    F = lambda v, unit, source=src: {"value": v, "unit": unit, "source": source}  # noqa: E731
    facts = {
        "params": F(st["params"], "count", "spec"), "vocab": F(len(VOCAB) - 1, "words", "model.py VOCAB (without the pad)"),
        "trainN": F(20000, "pairs", "train.py"), "valN": F(2000, "pairs", "train.py"), "testN": F(st["testN"], "pairs", src),
        "epochs": F(tm["epochs"], "epochs", "train.py"), "batch": F(tm["bs"], "pairs", "train.py"),
        "scale": F(st["scale"], "s = e^t", "model_final.pt"),
        "rowAcc6": F(st["rowAcc6"], "share", src), "colAcc6": F(st["colAcc6"], "share", src), "loss6": F(st["loss6"], "nats", src),
        "batches6": F(st["batches6"], "batches", src), "loss6Init": F(st0["loss6"], "nats", src + ", starting weights"),
        "rowAcc6Init": F(st0["rowAcc6"], "share", src + ", starting weights"),
        "oneN": F(st["oneN"], "drawings", src), "shapeAcc": F(st["shapeAcc"], "share", src), "sizeAcc": F(st["sizeAcc"], "share", src),
        "placeAcc": F(st["placeAcc"], "share", src),
        "twoN": F(st["twoN"], "drawings", src), "orderTies": F(st["orderTies"], "drawings", src), "orderMaxGap": F(st["orderMaxGap"], "score", src),
        "chunkHash0": F(st["chunkHash0"], "crc32", "synth tape, chunk 0, as read by Python"),
        "fixedHash": F(FIXED_HASH, "crc32", "synth tape, fixed drawings, as read by Python"),
    }
    report = ak.export_model(
        describe, CASES, slug="tinyclip", name=("TinyCLIP", "TinyCLIP"), kinds=("trace", "process"),
        module=Pair(model), module_input=lambda c: (c.input["image"].unsqueeze(0), c.input["text"]),
        facts=facts, extras={"vocab": VOCAB},
        training={"dataset": "24,000 synthetic drawing-caption pairs from scenes.ts (20,000 for training)", "steps": tm["epochs"] * (20000 // tm["bs"]),
                  "metrics": {"test, batches of 6: own caption first": st["rowAcc6"], "test, batches of 6: own drawing first": st["colAcc6"],
                              "test, zero-shot shape (5 captions)": st["shapeAcc"]}},
    )
    print(report)
    # the same graph with the starting weights (model_init.pt): the lesson's "before training" batch reads it
    init = load("model_init.pt")
    MODEL[0] = init
    print(ak.export_model(describe, CASES[:1] + CASES[-4:], slug="tinyclip", name=("TinyCLIP before training", "训练前的 TinyCLIP"), kinds=("trace",),
                          out=str(ak.site_path("static", "models", "tinyclip", "init")), module=Pair(init),
                          module_input=lambda c: (c.input["image"].unsqueeze(0), c.input["text"]), extras={"vocab": VOCAB},
                          note="the starting weights of the training run (train.py, seed 0), before any update"))
