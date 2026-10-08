"""score-2d: the Python reference for process 'langevin' (kind C parity): "the Langevin tape equals a Python loop with
the same seed".

It repeats the TS run of site/src/lib/explainers/score-2d/process.ts as a plain loop, independently of the TS code: the
twin network (twin.py build(), torch float64) gets the trained weights from --input, and the randomness comes from
ak.Rng with the same seed, drawn in the same order (no replayed draws):
  starts   2 x chains draws of next(): x = (2u - 1) * 2.5, chain by chain (x then y)
  step k   2 x chains draws of normal(): z, chain by chain; then x <- x + (alpha / 2) * s(x, sigma) + noise * sqrt(alpha) * z
           for every chain, with s = d(x, log sigma) / sigma and alpha = opt.alpha * (sigma / 0.04)^2 at each level
Frames: 'start' (step 0; metrics chains, dist, far, cover, spread), then 'move' per step (sigma, level, alpha, chain 0's
px, py, sx, sy, zx, zy, nx, ny; dist, far, cover and spread at the end of each level). ak.Rng's normal() may differ from
the TS one in the last bit (libm), so the positions and metrics are compared at 1e-9 (the process's tol and metricTol).

    npm run atlas -- parity score-2d    # records the TS tapes (walk, plain, annealed, still) and runs this file on each

CLI contract (every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
"""
from __future__ import annotations

import argparse
import importlib.util
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
SPAN, SMIN = 2.5, 0.04
SCHEMA = {"x": {"kind": "tensor", "label": ("Chains", "链"), "shape": ["N", 2], "role": "output", "tol": 1e-9}}
PHASES = {"start": {"label": ("Start anywhere", "从任意位置出发"), "kind": "input"},
          "move": {"label": ("One Langevin step", "朗之万法走一步"), "kind": "update"}}
DEFAULTS = {"sigmas": [0.16], "T": 100, "alpha": 2e-4, "noise": 1, "chains": 300}


def load_twin():
    spec = importlib.util.spec_from_file_location("score2d_twin", HERE / "twin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def tensor(v) -> torch.Tensor:
    """A TS tensor as JSON: {shape, data} with data a list (or an index -> value object)."""
    data = v["data"]
    if isinstance(data, dict):
        data = [data[str(i)] for i in range(len(data))]
    return torch.tensor(data, dtype=torch.float64).reshape(v["shape"])


def stats(x: np.ndarray, data: np.ndarray) -> dict:
    d2 = ((x[:, None, :] - data[None, :, :]) ** 2).sum(-1)          # [chains, points]
    near = np.sqrt(d2.min(1))
    mean = x.mean(0)
    return {"dist": float(near.mean()), "far": float((d2 >= 0.2 ** 2).all(1).mean()),
            "cover": float((d2 < 0.1 ** 2).any(0).mean()), "spread": float(np.sqrt(((x - mean) ** 2).sum(1)).mean())}


def run(opt: dict, seed: int, params: dict, data: np.ndarray) -> "ak.TapeWriter":
    w = ak.TapeWriter("langevin", SCHEMA, seed=seed, opt=opt, phases=PHASES, provenance="live")
    net = load_twin().build(opt)
    net.load_state_dict({k: tensor(v) for k, v in params.items()})
    net.eval()
    rng = ak.Rng(seed)
    N, T = max(1, int(opt["chains"])), max(1, int(opt["T"]))
    x = np.array([(2 * rng.next() - 1) * SPAN for _ in range(2 * N)]).reshape(N, 2)
    w.frame("start", {"x": x.copy()}, step=0, metrics={"chains": N, **stats(x, data)})
    k = 0
    for level, s in enumerate(opt["sigmas"]):
        a = float(opt["alpha"]) * (s / SMIN) ** 2
        for t in range(T):
            z = np.array([rng.normal() for _ in range(2 * N)]).reshape(N, 2)
            with torch.no_grad():
                inp = torch.cat([torch.from_numpy(x), torch.full((N, 1), math.log(s), dtype=torch.float64)], 1)
                g = (net(inp) / s).numpy()
            m = {"sigma": s, "level": level, "alpha": a, "px": x[0, 0], "py": x[0, 1], "sx": g[0, 0], "sy": g[0, 1],
                 "zx": z[0, 0], "zy": z[0, 1]}
            x = x + (a / 2 * g + float(opt["noise"]) * math.sqrt(a) * z)
            m.update(nx=x[0, 0], ny=x[0, 1])
            if (k + 1) % T == 0:
                m.update(stats(x, data))
            w.frame("move", {"x": x.copy()}, step=k, metrics={key: float(v) for key, v in m.items()})
            k += 1
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="langevin")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options the TS source ran with")
    ap.add_argument("--draws", default=None, help="unused: this reference draws from ak.Rng with the same seed")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (or @file): {params, xy}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/score-2d/langevin.ref)")
    a = ap.parse_args()
    opt = {**DEFAULTS, **ak.json_arg(a.opt, {})}
    inp = ak.json_arg(a.input) or {}
    data = np.array(inp.get("xy") or [], dtype=np.float64).reshape(-1, 2)
    if not inp.get("params") or data.shape[0] == 0:  # the TS run records no frames without weights or points
        tape = ak.TapeWriter("langevin", SCHEMA, seed=a.seed, opt=opt, phases=PHASES, provenance="live")
    else:
        tape = run(opt, a.seed, inp["params"], data)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "score-2d", "langevin.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
