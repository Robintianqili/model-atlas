"""loss-gradient-descent: Python reference for the processes 'gd' and 'rates' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/loss-gradient-descent/process.ts in plain Python floats: the
same eight points, the same error score (the mean of the squared errors), its gradient by the chain rule written out
(dL/dw = mean(2·e·x), dL/db = mean(2·e)) instead of the kit's autograd, and the same updates w <- w - lr·dL/dw.
`npm run atlas -- parity loss-gradient-descent` records the TS tape, runs this file and compares the two tapes frame by
frame (every float within 1e-12 relative; AK204 names the first difference). Parity records the coarse detail (one frame
per update), which is what this file writes.

    npm run atlas -- parity loss-gradient-descent

Arguments (the CLI contract for every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json>
--input <json> --out <stem>. These processes draw no random numbers, so --draws and --seed only travel into the header.
The input is the learner's dials, {"w": .., "b": ..}.
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

XS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
YS = [-1.25, -1.0, -0.1, 0.3, 1.25, 1.45, 2.25, 2.7]
N = len(XS)
W_RANGE = (-2.0, 4.0)
B_RANGE = (-2.0, 3.0)


def dials(raw) -> tuple[float, float]:
    """data.ts dialsOf: { w, b } or [w, b], clamped into the dial ranges; anything else is (0, 0)."""
    def fin(v, d=0.0):
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else d
    if isinstance(raw, list):
        w, b = fin(raw[0] if raw else None), fin(raw[1] if len(raw) > 1 else None)
    elif isinstance(raw, dict):
        w, b = fin(raw.get("w")), fin(raw.get("b"))
    else:
        w, b = 0.0, 0.0
    return min(W_RANGE[1], max(W_RANGE[0], w)), min(B_RANGE[1], max(B_RANGE[0], b))


def score_and_gradient(w: float, b: float) -> tuple[float, float, float]:
    errs = [w * x + b - y for x, y in zip(XS, YS)]
    loss = sum(e * e for e in errs) / N
    gw = sum(2 * e * x / N for e, x in zip(errs, XS))
    gb = sum(2 * e / N for e in errs)
    return loss, gw, gb


SCHEMA = {
    "w": {"kind": "scalar", "label": ("Weight w", "权重 w"), "role": "param", "tol": 1e-12},
    "b": {"kind": "scalar", "label": ("Bias b", "偏置 b"), "role": "param", "tol": 1e-12},
    "gw": {"kind": "scalar", "label": ("Slope under w (dL/dw)", "w 方向的斜率 (dL/dw)"), "role": "grad", "tol": 1e-12},
    "gb": {"kind": "scalar", "label": ("Slope under b (dL/db)", "b 方向的斜率 (dL/db)"), "role": "grad", "tol": 1e-12},
    "path": {"kind": "tensor", "label": ("The dials visited so far (w, b)", "到目前为止走过的旋钮位置 (w, b)"), "shape": ["k", 2], "role": "param", "tol": 1e-12},
}
PHASES = {
    "start": {"label": ("Your dials", "你的旋钮"), "kind": "input"},
    "loss": {"label": ("The error score", "误差分数"), "kind": "loss"},
    "grad": {"label": ("The gradient", "梯度"), "kind": "backward"},
    "update": {"label": ("One step downhill", "向下走一步"), "kind": "update"},
    "step": {"label": ("Gradient descent", "梯度下降"), "kind": "update"},
    "end": {"label": ("Where it stopped", "停在哪里"), "kind": "observe"},
}


def coarse(raw, lr: float, steps: int) -> list[tuple[str, int, dict, dict]]:
    """The coarse frames of process 'gd': start, then one 'step' frame per update (step = updates done)."""
    w, b = dials(raw)
    pts = [[w, b]]
    loss, gw, gb = score_and_gradient(w, b)
    out = [("start", 0, {"w": w, "b": b, "path": np.array(pts, dtype=np.float64)}, {"step": 0, "w": w, "b": b, "loss": loss, "lr": lr})]
    for k in range(steps):
        w1 = w - lr * gw
        b1 = b - lr * gb
        loss1, gw1, gb1 = score_and_gradient(w1, b1)
        pts.append([w1, b1])
        out.append(("step", k + 1, {"w": w1, "b": b1, "gw": gw, "gb": gb, "path": np.array(pts, dtype=np.float64)},
                    {"step": k + 1, "w": w1, "b": b1, "loss": loss1, "gw": gw, "gb": gb, "lr": lr}))
        w, b, loss, gw, gb = w1, b1, loss1, gw1, gb1
    return out


def run(process: str, raw, opt: dict, seed: int) -> "ak.TapeWriter":
    tape = ak.TapeWriter(process, SCHEMA, seed=seed, opt=opt, phases=PHASES, provenance="live")
    steps = int(opt.get("steps", 20 if process == "gd" else 30))
    lr = float(opt.get("lr", 0.1))
    main = coarse(raw, lr, steps)
    if process == "gd":
        for phase, step, state, metrics in main:
            tape.frame(phase, state, step=step, metrics=metrics)
    elif process == "rates":
        lr_slow = float(opt.get("lrSlow", 0.01))
        lr_big = float(opt.get("lrBig", 0.7))
        slow = coarse(raw, lr_slow, steps)
        big = coarse(raw, lr_big, steps)
        for i, (phase, step, state, metrics) in enumerate(main):
            m = {**metrics, "lrSlow": lr_slow, "lossSlow": slow[i][3]["loss"], "lrBig": lr_big, "lossBig": big[i][3]["loss"]}
            tape.frame(phase, state, step=step, metrics=m)
    else:
        raise SystemExit(f"reference.py: no reference for process '{process}' (gd and rates only)")
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="gd")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null", help="JSON input of the TS run, or @<file> (read it with ak.json_arg)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = ak.json_arg(a.input)
    tape = run(a.process, raw, json.loads(a.opt), a.seed)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "loss-gradient-descent", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
