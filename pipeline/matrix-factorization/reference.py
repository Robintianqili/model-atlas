"""matrix-factorization: Python reference for process 'nudge' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/matrix-factorization/process.ts in plain Python floats: one
step of stochastic gradient descent on one rating, slowed down into start -> error -> gradient -> update. The starting
numbers are the trainer's: ak.Rng(seed).fork('init') (bit-identical to the TS rng for next() and fork()), uniform in
+-init, U [25, 2] first, row by row, then V [20, 2]. The rating: the learner's rating with the largest size (ties: the
first movie); when the learner rated nothing, person 0's rating with the largest size, which the TS process passes in its options
(opt.fallback = [person, movie, rating]: the other people's ratings are TS data, made with normal() draws).
`npm run atlas -- parity matrix-factorization` records the TS tape, runs this file and compares the tapes frame by frame
(every float within 1e-12 relative, the rest exact; AK204 names the first difference).

CLI contract (every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>.
This process draws only with next(), so it ignores --draws.
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

N_USERS, N_ITEMS, YOU = 25, 20, 24
SCHEMA = {
    "u": {"kind": "tensor", "label": ("The person's two numbers", "这个人的两个数"), "shape": [2], "role": "param", "tol": 1e-12},
    "v": {"kind": "tensor", "label": ("The movie's two numbers", "这部电影的两个数"), "shape": [2], "role": "param", "tol": 1e-12},
}
PHASES = {
    "start": {"label": ("Starting numbers", "起始的数"), "kind": "input"},
    "error": {"label": ("Prediction and error", "预测与误差"), "kind": "loss"},
    "gradient": {"label": ("Gradients", "梯度"), "kind": "backward"},
    "update": {"label": ("Update", "更新"), "kind": "update"},
}


def pick(ratings: list) -> tuple[int, int, float] | None:
    """The learner's rating with the largest size, ties to the first movie; None when nothing is rated."""
    best = -1
    for i, r in enumerate(ratings):
        if r is not None and (best < 0 or abs(r) > abs(ratings[best])):
            best = i
    return (YOU, best, float(ratings[best])) if best >= 0 else None


def run(seed: int, lr: float, init: float, ratings: list, fallback: tuple[int, int, float]) -> "ak.TapeWriter":
    w = ak.TapeWriter("nudge", SCHEMA, seed=seed, opt={"lr": lr, "init": init}, phases=PHASES, provenance="live")
    rng = ak.Rng(seed).fork("init")
    U = [(2 * rng.next() - 1) * init for _ in range(N_USERS * 2)]
    V = [(2 * rng.next() - 1) * init for _ in range(N_ITEMS * 2)]
    user, item, r = pick(ratings) or fallback
    u1, u2, v1, v2 = U[2 * user], U[2 * user + 1], V[2 * item], V[2 * item + 1]
    m = {"user": user, "item": item, "r": r, "u1": u1, "u2": u2, "v1": v1, "v2": v2, "lr": lr}

    def vec(a: float, b: float) -> np.ndarray:
        return np.array([a, b], dtype=np.float64)

    w.frame("start", {"u": vec(u1, u2), "v": vec(v1, v2)}, step=0, metrics=dict(m))
    p1, p2 = u1 * v1, u2 * v2
    pred = p1 + p2
    e = pred - r
    m.update(p1=p1, p2=p2, pred=pred, e=e, loss=e * e)
    w.frame("error", {"u": vec(u1, u2), "v": vec(v1, v2)}, step=0, metrics=dict(m))
    gu1, gu2, gv1, gv2 = 2 * e * v1, 2 * e * v2, 2 * e * u1, 2 * e * u2
    m.update(gu1=gu1, gu2=gu2, gv1=gv1, gv2=gv2)
    w.frame("gradient", {"u": vec(u1, u2), "v": vec(v1, v2)}, step=0, metrics=dict(m))
    nu1, nu2, nv1, nv2 = u1 - lr * gu1, u2 - lr * gu2, v1 - lr * gv1, v2 - lr * gv2
    pa = nu1 * nv1 + nu2 * nv2
    ea = pa - r
    m.update(nu1=nu1, nu2=nu2, nv1=nv1, nv2=nv2, predAfter=pa, eAfter=ea, lossAfter=ea * ea)
    w.frame("update", {"u": vec(nu1, nu2), "v": vec(nv1, nv2)}, step=0, metrics=dict(m))
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="nudge")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options the TS source ran with")
    ap.add_argument("--draws", default=None, help="ignored: this process draws with next() only")
    ap.add_argument("--input", default="null", help="JSON input of the TS run: {ratings: [20 x (int | null)]}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/matrix-factorization/nudge.ref)")
    a = ap.parse_args()
    opt = {"lr": 0.05, "init": 0.5, **json.loads(a.opt)}
    raw = json.loads(a.input) or {}
    ratings = [r if isinstance(r, (int, float)) and not isinstance(r, bool) else None for r in (raw.get("ratings") or [])][:N_ITEMS]
    ratings += [None] * (N_ITEMS - len(ratings))
    out = a.out or ak.site_path(".atlas", "matrix-factorization", "nudge.ref")
    fb = opt.get("fallback") or [0, 0, 0]
    tape = run(a.seed, float(opt["lr"]), float(opt["init"]), ratings, (int(fb[0]), int(fb[1]), float(fb[2])))
    jpath, _ = tape.write(out)
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
