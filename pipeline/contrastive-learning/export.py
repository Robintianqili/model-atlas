"""contrastive-learning: export the checkpoints of the TinyCLIP training re-run, and the page's facts.

    npm run atlas -- run tinyclip synth --opt '{"n":24000}' --save                # the TS-rendered pairs (tinyclip's process)
    npm run atlas -- py pipeline/contrastive-learning/train.py --run v1             # runs/v1/ck-<update>.pt + metrics.json
    npm run atlas -- py pipeline/contrastive-learning/export.py --run v1            # this file

Every checkpoint in PAGE becomes static/models/contrastive-learning/ck/<update>/ (spec.json, weights.bin and a small
reference.json): the graph of /m/tinyclip/ (both towers, the score and the softmax over captions; describe() is copied
from pipeline/tinyclip/export.py) with that checkpoint's weights. The page loads them as trace sources ({ assets }) and
runs the learner's batch through each; `npm run atlas -- parity contrastive-learning --dir
static/models/contrastive-learning/ck/<update>` checks the TS forward pass against PyTorch on the reference cases.
facts.json (static/models/contrastive-learning/): the re-run's measured test numbers at the first and last checkpoint,
and the temperature the shipped TinyCLIP learned (static/models/tinyclip/facts.json).
"""
from __future__ import annotations

import argparse
import datetime
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
from train import load_fixed  # noqa: E402

M = "pipeline/contrastive-learning/model.py"
# the checkpoints the page shows (batch.ts CKPTS must list the same updates)
PAGE = [0, 10, 20, 30, 50, 100, 300, 4680]
MODEL: list[TinyCLIP] = []


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="v1")
    a = ap.parse_args()
    run = HERE / "runs" / a.run
    tm = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    X, IDS, CAPS, _ = load_fixed()
    names = ["circle", "square", "triangle", "star", "cross", "stack", "p-triangle", "p-square", "p-star", "p-cross", "p-pair", "blank", "dot", "small"]
    k = names.index("circle")
    cases = [ak.Case("circle", "circle: " + " | ".join(c for c in CAPS[k] if c), {"image": X[k], "text": text_of(IDS[k])}),
             ak.Case("stack", "stack: two shapes", {"image": X[names.index("stack")], "text": text_of(IDS[names.index("stack")])}),
             ak.Case("no-words", "circle: a caption with no known word (all padding)", {"image": X[k], "text": torch.zeros(1, 1, dtype=torch.long)}, edge=True)]
    for u in PAGE:
        m = TinyCLIP()
        m.load_state_dict(torch.load(run / f"ck-{u}.pt", map_location="cpu"))
        m.eval()
        MODEL[:] = [m]
        out = ak.site_path("static", "models", "contrastive-learning", "ck", str(u))
        print(ak.export_model(describe, cases, slug="contrastive-learning", name=(f"TinyCLIP after {u} updates", f"训练 {u} 次更新后的 TinyCLIP"),
                              kinds=("trace",), out=str(out), module=Pair(m), module_input=lambda c: (c.input["image"].unsqueeze(0), c.input["text"]),
                              extras={"vocab": VOCAB}, note=f"a re-run of TinyCLIP's training (train.py, seed 0, CPU): the weights after {u} of {tm['updates']} updates"))
    cks = {c["step"]: c for c in tm["checkpoints"]}
    first, last = cks[PAGE[0]], cks[PAGE[-1]]
    clip = json.loads(ak.site_path("static", "models", "tinyclip", "facts.json").read_text(encoding="utf-8"))["facts"]
    today = datetime.date.today().isoformat()
    src = f"pipeline/contrastive-learning/train.py --run {a.run} (CPU, seed 0), 2,000 test pairs in batches of 6"
    F = lambda v, unit, source=src: {"value": v, "unit": unit, "source": source, "measured": today}  # noqa: E731
    facts = {
        "updates": F(tm["updates"], "updates", "train.py"), "epochs": F(tm["epochs"], "epochs", "train.py"), "batch": F(tm["bs"], "pairs", "train.py"),
        "trainN": F(20000, "pairs", "train.py"), "testN": F(2000, "pairs", "train.py"), "batches6": F(first["batches"], "batches", src),
        "checkpoints": F(len(PAGE), "checkpoints", "export.py PAGE"),
        "loss6First": F(first["loss6"], "nats", src), "loss6Last": F(last["loss6"], "nats", src),
        "acc6First": F(first["rowAcc6"], "share", src), "acc6Last": F(last["rowAcc6"], "share", src),
        "tauFirst": F(1 / first["scale"], "τ = 1/s", src), "tauLast": F(1 / last["scale"], "τ = 1/s", src),
        "tauClip": F(1 / clip["scale"]["value"], "τ = 1/s", "static/models/tinyclip/facts.json scale (the shipped TinyCLIP)"),
        "params": F(tm["params"], "count", "train.py"),
    }
    path = ak.site_path("static", "models", "contrastive-learning", "facts.json")
    path.write_text(json.dumps({"format": "atlas-facts/1", "facts": facts}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}: {', '.join(facts)}")


if __name__ == "__main__":
    main()
