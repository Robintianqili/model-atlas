"""realnvp-2d: the Python reference for processes 'flow' and 'sample' (kind C parity): "the layer-by-layer tape equals
a plain PyTorch float64 run of the same flow".

It repeats the TS runs of site/src/lib/explainers/realnvp-2d/process.ts independently of the TS code: the twin flow
(twin.py build(), torch float64) gets the trained weights from --input ({params, xy}), then
  flow     your points and a grid over the data plane (13 + 13 lines of 49 points over +-1.5) through layers 0..7, one
           frame per layer ('start', then 'couple' k = 0..7), then 'density': log p = log N(z; 0, I) + sum s on a 40 x 40
           grid over +-1.5 and the mean -log p of the points
  sample   n draws z ~ N(0, I) (the TS draws, replayed from --draws) and a grid over the noise plane (+-3), pulled back
           through layers 7..0 ('noise', then 'uncouple' j = 0..7); at the end the samples' mean distance to the nearest
           point, the share farther than 0.15 from every point, and the share with 0.6 < |x| < 0.75 (the rings' gap)
Metrics follow point 0 (flow) and sample 0 (sample), as the TS frames do; positions and metrics are compared at 1e-9.

    npm run atlas -- parity realnvp-2d   # records the TS tapes (flow, sample, ringSample) and runs this file on each

CLI contract (every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
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
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from twin import LAYERS, build  # noqa: E402

DSPAN, ZSPAN, DG, LINES, PER = 1.5, 3.0, 40, 13, 49
GAP = (0.6, 0.75)
LOG2PI = math.log(2 * math.pi)


def t2(label, shape=("N", 2), role="activation"):
    return {"kind": "tensor", "label": label, "shape": list(shape), "role": role, "tol": 1e-9}


FLOW_SCHEMA = {"pts": t2(("Points", "点")), "grid": t2(("Grid", "网格"), ("G", 2)),
               "logp": t2(("log p on a grid", "网格上的 log p"), ("M",), "output")}
SAMPLE_SCHEMA = {"x": t2(("Samples", "样本"), role="output"), "grid": t2(("Grid", "网格"), ("G", 2))}
FLOW_PHASES = {"start": {"label": ("Your points", "你的点"), "kind": "input"},
               "couple": {"label": ("One coupling layer", "一个耦合层"), "kind": "forward"},
               "density": {"label": ("The exact log-likelihood", "精确的对数似然"), "kind": "output"}}
SAMPLE_PHASES = {"noise": {"label": ("Draw noise", "抽取噪声"), "kind": "sample"},
                 "uncouple": {"label": ("Undo one coupling layer", "撤销一个耦合层"), "kind": "forward"}}


def load(inp: dict):
    model = build({})
    sd = {k: torch.tensor(np.array(v["data"], dtype=np.float64).reshape(v["shape"])) for k, v in inp["params"].items()}
    model.load_state_dict(sd)
    data = np.array(inp["xy"], dtype=np.float64).reshape(-1, 2)
    return model, data


def grid(G: int, span: float) -> np.ndarray:
    i = np.arange(G * G)
    step = 2 * span / G
    return np.stack([-span + (i % G + 0.5) * step, -span + (i // G + 0.5) * step], 1)


def lines(span: float) -> np.ndarray:
    a = -span + np.arange(LINES) * 2 * span / (LINES - 1)
    b = -span + np.arange(PER) * 2 * span / (PER - 1)
    vert = np.stack([np.repeat(a, PER), np.tile(b, LINES)], 1)
    horiz = np.stack([np.tile(b, LINES), np.repeat(a, PER)], 1)
    return np.concatenate([vert, horiz], 0)


def moments(p: np.ndarray) -> dict:
    m = p.mean(0)
    s = np.sqrt(((p - m) ** 2).mean(0))
    return {"mx": m[0], "my": m[1], "sx": s[0], "sy": s[1]}


def nearest(x: np.ndarray, data: np.ndarray) -> np.ndarray:
    return np.sqrt(((x[:, None, :] - data[None, :, :]) ** 2).sum(-1)).min(1)


def run_flow(model, data: np.ndarray) -> "ak.TapeWriter":
    w = ak.TapeWriter("flow", FLOW_SCHEMA, seed=1, opt={}, phases=FLOW_PHASES, provenance="live")
    n = len(data)
    pts, g = torch.tensor(data), torch.tensor(lines(DSPAN))
    ld = torch.zeros(n, dtype=torch.float64)
    w.frame("start", {"pts": pts.numpy().copy(), "grid": g.numpy().copy()}, step=0,
            metrics={"n": n, "px": float(pts[0, 0]), "py": float(pts[0, 1]), **moments(pts.numpy())})
    with torch.no_grad():
        for k in range(LAYERS):
            c = k % 2
            u, v = float(pts[0, c]), float(pts[0, 1 - c])
            o = getattr(model, f"c{k}")(pts[:, c:c + 1])
            s0, t0 = float(torch.tanh(o[0, 0])), float(o[0, 1])
            pts, s = model.couple(k, pts)
            g, _ = model.couple(k, g)
            ld = ld + s
            w.frame("couple", {"pts": pts.numpy().copy(), "grid": g.numpy().copy()}, step=k,
                    metrics={"k": k, "keep": c, "u": u, "v": v, "s": s0, "t": t0, "es": math.exp(s0), "v2": float(pts[0, 1 - c]),
                             "ld": float(ld[0]), "px": float(pts[0, 0]), "py": float(pts[0, 1]), **moments(pts.numpy())})
        z, lds = model(torch.tensor(grid(DG, DSPAN)))
        lp = (-0.5 * (z ** 2).sum(1) - LOG2PI + lds).numpy()
        zall, ldall = model(torch.tensor(data))
        nll = float(-(-0.5 * (zall ** 2).sum(1) - LOG2PI + ldall).mean())
    ln = -0.5 * float(pts[0, 0] ** 2 + pts[0, 1] ** 2) - LOG2PI
    w.frame("density", {"pts": pts.numpy().copy(), "grid": g.numpy().copy(), "logp": lp}, step=LAYERS,
            metrics={"zx": float(pts[0, 0]), "zy": float(pts[0, 1]), "logN": ln, "ld": float(ld[0]), "logp": ln + float(ld[0]),
                     "nll": nll, "lpMax": float(lp.max()), "lpMin": float(lp.min())})
    return w


def run_sample(model, data: np.ndarray, opt: dict, draws: list) -> "ak.TapeWriter":
    w = ak.TapeWriter("sample", SAMPLE_SCHEMA, seed=1, opt=opt, phases=SAMPLE_PHASES, provenance="live")
    n = max(1, int(opt["n"]))
    x = torch.tensor(np.array(draws[: 2 * n], dtype=np.float64).reshape(n, 2))
    g = torch.tensor(lines(ZSPAN))
    w.frame("noise", {"x": x.numpy().copy(), "grid": g.numpy().copy()}, step=0,
            metrics={"n": n, "px": float(x[0, 0]), "py": float(x[0, 1]), "logN": -0.5 * float(x[0, 0] ** 2 + x[0, 1] ** 2) - LOG2PI})
    with torch.no_grad():
        for j in range(LAYERS):
            k = LAYERS - 1 - j
            c = k % 2
            v2 = float(x[0, 1 - c])
            o = getattr(model, f"c{k}")(x[:, c:c + 1])
            s0, t0 = float(torch.tanh(o[0, 0])), float(o[0, 1])
            x, _ = model.couple(k, x, inverse=True)
            g, _ = model.couple(k, g, inverse=True)
            m = {"k": k, "keep": c, "u": float(x[0, c]), "v2": v2, "s": s0, "t": t0, "v": float(x[0, 1 - c]), "px": float(x[0, 0]), "py": float(x[0, 1])}
            if j == LAYERS - 1:
                xs = x.numpy()
                d = nearest(xs, data)
                r = np.sqrt((xs ** 2).sum(1))
                m.update({"dist": float(d.mean()), "far": float((d >= 0.15).mean()), "gap": float(((r > GAP[0]) & (r < GAP[1])).mean())})
            w.frame("uncouple", {"x": x.numpy().copy(), "grid": g.numpy().copy()}, step=j, metrics=m)
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="flow")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options the TS source ran with")
    ap.add_argument("--draws", default=None, help="draws.json from the TS run (replayed)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (or @file)")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/realnvp-2d/<process>.ref)")
    a = ap.parse_args()
    inp = ak.json_arg(a.input)
    opt = {"n": 500, **json.loads(a.opt)} if a.process == "sample" else json.loads(a.opt)
    if not inp or not inp.get("params") or not inp.get("xy"):
        tape = ak.TapeWriter(a.process, FLOW_SCHEMA if a.process == "flow" else SAMPLE_SCHEMA, seed=a.seed, opt=opt, provenance="live")
    else:
        model, data = load(inp)
        if a.process == "flow":
            tape = run_flow(model, data)
        else:
            frames = json.loads(Path(a.draws).read_text(encoding="utf-8"))["frames"] if a.draws else []
            tape = run_sample(model, data, opt, [v for f in frames for v in (f or [])])
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "realnvp-2d", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
