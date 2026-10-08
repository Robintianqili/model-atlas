"""hello-xor: Python reference for process 'sgdStep' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/hello-xor/process.ts in plain Python floats: the same
starting weights (ak.Rng is bit-identical to the TS rng for next()), the same forward pass, loss, gradients and update.
`npm run atlas -- parity hello-xor` records the TS tape, runs this file and compares the two tapes frame by frame
(every float within 1e-12 relative, the rest exact; AK204 names the first difference).

    npm run atlas -- parity hello-xor                             # the usual way (it passes the arguments below)
    python pipeline/hello-xor/reference.py --seed 1 --out site/.atlas/hello-xor/sgdStep.ref

Arguments (the CLI contract for every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json>
--input <json> --out <stem>. This process draws only with next(), so it ignores --draws and --input.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np

XS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
YS = [0.0, 1.0, 1.0, 0.0]


def _t(*shape_and_values):
    *shape, values = shape_and_values
    return np.array(values, dtype=np.float64).reshape(shape)


def _schema() -> dict:
    def f(kind, en, zh, **kw):
        return {"kind": kind, "label": (en, zh), **kw}
    return {
        "W1": f("tensor", "Hidden-layer weights", "隐藏层权重", shape=[2, 2], role="param", scale="diverging", tol=1e-12),
        "b1": f("tensor", "Hidden-layer biases", "隐藏层偏置", shape=[2], role="param", scale="diverging", tol=1e-12),
        "W2": f("tensor", "Output weights", "输出层权重", shape=[1, 2], role="param", scale="diverging", tol=1e-12),
        "b2": f("tensor", "Output bias", "输出层偏置", shape=[1], role="param", scale="diverging", tol=1e-12),
        "x": f("tensor", "The point", "这个点", shape=[2], role="data", tol="exact"),
        "z1": f("tensor", "Hidden sums", "隐藏层的和", shape=[2], role="activation", scale="diverging", tol=1e-12),
        "h": f("tensor", "Hidden values (after tanh)", "隐藏值（tanh 之后）", shape=[2], role="activation", scale="diverging", tol=1e-12),
        "yhat": f("scalar", "Network output", "网络输出", role="output", tol=1e-12),
        "target": f("scalar", "Right answer", "正确答案", role="target", tol="exact"),
        "loss": f("scalar", "Loss (mean squared error)", "损失（均方误差）", role="output", tol=1e-12),
        "gW1": f("tensor", "Gradient of the hidden weights", "隐藏层权重的梯度", shape=[2, 2], role="grad", scale="diverging", tol=1e-12),
        "gb1": f("tensor", "Gradient of the hidden biases", "隐藏层偏置的梯度", shape=[2], role="grad", scale="diverging", tol=1e-12),
        "gW2": f("tensor", "Gradient of the output weights", "输出层权重的梯度", shape=[1, 2], role="grad", scale="diverging", tol=1e-12),
        "gb2": f("tensor", "Gradient of the output bias", "输出层偏置的梯度", shape=[1], role="grad", scale="diverging", tol=1e-12),
        "lossAfter": f("scalar", "Loss after the step", "这一步之后的损失", role="output", tol=1e-12),
    }


PHASES = {
    "init": {"label": ("Starting weights", "起始权重"), "kind": "other"},
    "forward": {"label": ("Forward pass, one point", "前向计算：一个点"), "kind": "forward"},
    "loss": {"label": ("Loss", "损失"), "kind": "loss"},
    "backward": {"label": ("Backward pass", "反向传播"), "kind": "backward"},
    "update": {"label": ("Update", "更新"), "kind": "update"},
    "check": {"label": ("Did it help?", "有用吗？"), "kind": "observe"},
}


def forward(w: dict, x) -> tuple[list, list, float]:
    z1 = [w["W1"][j][0] * x[0] + w["W1"][j][1] * x[1] + w["b1"][j] for j in range(2)]
    h = [math.tanh(z) for z in z1]
    yhat = w["W2"][0] * h[0] + w["W2"][1] * h[1] + w["b2"]
    return z1, h, yhat


def params(w: dict) -> dict:
    return {"W1": _t(2, 2, [w["W1"][0][0], w["W1"][0][1], w["W1"][1][0], w["W1"][1][1]]), "b1": _t(2, w["b1"]),
            "W2": _t(1, 2, w["W2"]), "b2": _t(1, [w["b2"]])}


def run(seed: int = 1, lr: float = 0.5, scale: float = 1.0) -> "ak.TapeWriter":
    tape = ak.TapeWriter("sgdStep", _schema(), seed=seed, opt={"lr": lr, "scale": scale}, phases=PHASES, provenance="live")
    rng = ak.Rng(seed)

    def draw() -> float:
        return scale * (2 * rng.next() - 1)

    w = {"W1": [[0.0, 0.0], [0.0, 0.0]], "b1": [0.0, 0.0], "W2": [0.0, 0.0], "b2": 0.0}
    w["W1"][0][0] = draw()
    w["W1"][0][1] = draw()
    w["W1"][1][0] = draw()
    w["W1"][1][1] = draw()
    w["b1"][0] = draw()
    w["b1"][1] = draw()
    w["W2"][0] = draw()
    w["W2"][1] = draw()
    w["b2"] = draw()
    tape.frame("init", params(w), metrics={"count": 9, "lo": -scale, "hi": scale})

    fw = [forward(w, x) for x in XS]
    for i, (z1, h, yhat) in enumerate(fw):
        tape.frame("forward", {"x": _t(2, XS[i]), "z1": _t(2, z1), "h": _t(2, h), "yhat": yhat, "target": YS[i]}, step=0, sub=i,
                   metrics={"x1": XS[i][0], "x2": XS[i][1], "h1": h[0], "h2": h[1], "w1": w["W2"][0], "w2": w["W2"][1], "b": w["b2"],
                            "yhat": yhat, "target": YS[i]},
                   focus=[{"field": "x", "role": "in"}, {"field": "yhat", "role": "out"}])

    s = 0.0
    for i in range(4):
        e = fw[i][2] - YS[i]
        s += e * e
    loss = s / 4
    tape.frame("loss", {"loss": loss}, step=0, metrics={"loss": loss})

    gW1 = [[0.0, 0.0], [0.0, 0.0]]
    gb1 = [0.0, 0.0]
    gW2 = [0.0, 0.0]
    gb2 = 0.0
    for i in range(4):
        _, h, yhat = fw[i]
        d = (2 * (yhat - YS[i])) / 4
        gW2[0] += d * h[0]
        gW2[1] += d * h[1]
        gb2 += d
        for j in range(2):
            dz = d * w["W2"][j] * (1 - h[j] * h[j])
            gW1[j][0] += dz * XS[i][0]
            gW1[j][1] += dz * XS[i][1]
            gb1[j] += dz
    tape.frame("backward", {"gW1": _t(2, 2, [gW1[0][0], gW1[0][1], gW1[1][0], gW1[1][1]]), "gb1": _t(2, gb1),
                            "gW2": _t(1, 2, gW2), "gb2": _t(1, [gb2])}, step=0, metrics={"g": gW2[0], "loss": loss})

    old = w["W2"][0]
    for j in range(2):
        w["W1"][j][0] -= lr * gW1[j][0]
        w["W1"][j][1] -= lr * gW1[j][1]
        w["b1"][j] -= lr * gb1[j]
        w["W2"][j] -= lr * gW2[j]
    w["b2"] -= lr * gb2
    s = 0.0
    for i in range(4):
        e = forward(w, XS[i])[2] - YS[i]
        s += e * e
    loss_after = s / 4
    tape.frame("update", params(w), step=0, metrics={"lr": lr, "old": old, "g": gW2[0], "new": w["W2"][0], "loss": loss, "lossAfter": loss_after})

    tape.frame("check", {"lossAfter": loss_after}, step=0, metrics={"loss": loss, "lossAfter": loss_after})
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="sgdStep")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran (lr, scale)")
    ap.add_argument("--draws", default=None, help="draws.json written by the TS run (unused: this process draws with next() only)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (unused)")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/hello-xor/sgdStep.ref)")
    a = ap.parse_args()
    if a.process != "sgdStep":
        sys.exit(f"reference.py knows process 'sgdStep', not {a.process!r}")
    opt = {"lr": 0.5, "scale": 1.0, **json.loads(a.opt)}
    t = run(seed=a.seed, lr=float(opt["lr"]), scale=float(opt["scale"]))
    jpath, _ = t.write(a.out or ak.site_path(".atlas", "hello-xor", "sgdStep.ref"))
    print(f"wrote {jpath} ({len(t.frames)} frames)")
