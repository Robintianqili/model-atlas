"""fno: Python reference for process 'solve' (kind C parity).

An independent implementation of the 'solve' process in site/src/lib/explainers/fno/process.ts with numpy float64: the
explicit finite-difference rule for the heat equation on a ring of 64 points, u_j <- u_j + r (u_{j-1} - 2 u_j + u_{j+1})
with r = 0.1, for 50 steps of 0.02, one frame per step (the curve, its highest and lowest point, their spread and the steepest rise between neighbours).
`npm run atlas -- parity fno` records the TS tape, runs this file with the same input and compares the tapes frame by
frame (AK204). It draws nothing.

CLI contract: --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json | @file> --out <stem>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np

N, STEPS, R, DT = 64, 50, 0.1, 0.02
SCHEMA = {"u": {"kind": "tensor", "label": ("The curve: 64 heights", "曲线：64 个高度"), "tol": 1e-12}}
PHASES = {
    "start": {"label": ("The starting curve, time 0", "起始曲线，时间 0"), "kind": "input"},
    "step": {"label": ("One time step: every point moves toward the average of its neighbours", "一个时间步：每个点都向两边邻居的平均值靠拢"), "kind": "update"},
}


def stats(u: np.ndarray) -> dict:
    top, lo = int(np.argmax(u)), int(np.argmin(u))   # first index of the extreme, as the TS loop keeps
    steep = float(np.max(np.abs(np.roll(u, -1) - u)))   # |u_{j+1} - u_j| around the ring
    return {"max": float(u[top]), "min": float(u[lo]), "top": top, "spread": float(u[top] - u[lo]), "steep": steep}


def run_solve(seed: int, opt: dict, u0) -> ak.TapeWriter:
    w = ak.TapeWriter("solve", SCHEMA, seed=seed, opt=opt, phases=PHASES, provenance="live")
    if not u0 or len(u0) != N:
        return w
    u = np.asarray(u0, dtype=np.float64)
    w.frame("start", {"u": u.copy()}, step=0, metrics={"t": 0, **stats(u)})
    for s in range(1, STEPS + 1):
        a, c = np.roll(u, 1), np.roll(u, -1)        # a_j = u_{j-1}, c_j = u_{j+1}
        u = u + R * (a - 2 * u + c)
        w.frame("step", {"u": u.copy()}, step=s, metrics={"t": s * DT, **stats(u)})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="solve", choices=["solve"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran (none are used)")
    ap.add_argument("--draws", default=None, help="unused: the process draws nothing")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (the 64 starting heights), or @file")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/fno/solve.ref)")
    a = ap.parse_args()
    tape = run_solve(a.seed, json.loads(a.opt), ak.json_arg(a.input))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "fno", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
