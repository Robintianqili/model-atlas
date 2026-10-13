"""transe: the Python reference of process 'rank' (site/src/lib/explainers/transe/process.ts; kind C parity).

Link prediction for one question (h, r, ?) on the trained TransE tables the TS process received as its input (the
entity and relation rows of the train source `run` at its last frame, flat and row-major, `dim` numbers per row): one
'score' frame per candidate tail x = 0 .. 33 with d = ||h + r - x|| (summed over k = 0, 1, ... in that order, then the
square root, exactly as space.ts distFlat) and the best candidate so far (the first smallest), then one 'result' frame
with the three nearest (ties: the smaller id) and the rank of the first true tail with the other true tails left out
(0: the facts know no tail). No randomness: every metric must agree exactly.

    npm run atlas -- parity transe       # records the TS tape, runs this file, compares frame by frame

CLI contract (every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

NE = 34  # kg.ts: 11 countries, 11 capitals, 4 continents, 8 languages
PHASES = {
    "score": {"label": ("Score one candidate", "给一个候选打分"), "kind": "observe"},
    "result": {"label": ("The ranking", "排名"), "kind": "output"},
}


def dist(E: list[float], R: list[float], dim: int, h: int, r: int, t: int) -> float:
    s = 0.0
    for k in range(dim):
        v = E[h * dim + k] + R[r * dim + k] - E[t * dim + k]
        s += v * v
    return math.sqrt(s)


def rank(inp, seed: int, opt: dict) -> "ak.TapeWriter":
    w = ak.TapeWriter("rank", {}, seed=seed, opt=opt, phases=PHASES, provenance="live")
    if not isinstance(inp, dict):
        return w
    E, R, dim, h, r, tails = inp["E"], inp["R"], int(inp["dim"]), int(inp["h"]), int(inp["r"]), [int(x) for x in inp["tails"]]
    ds: list[float] = []
    best = -1
    for x in range(NE):
        d = dist(E, R, dim, h, r, x)
        ds.append(d)
        if best < 0 or d < ds[best]:
            best = x
        w.frame("score", {}, step=x, metrics={"x": x, "d": d, "best": best, "bestD": ds[best], "truth": 1 if x in tails else 0})
    order = sorted(range(NE), key=lambda x: (ds[x], x))
    t0 = tails[0] if tails else -1
    others = set(tails[1:])
    truth_rank = 0 if t0 < 0 else [x for x in order if x not in others].index(t0) + 1
    w.frame("result", {}, step=NE, metrics={"t1": order[0], "d1": ds[order[0]], "t2": order[1], "d2": ds[order[1]], "t3": order[2],
                                            "d3": ds[order[2]], "truth": t0, "truthRank": truth_rank})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", required=True, choices=["rank"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None, help="unused: the process draws nothing")
    ap.add_argument("--input", default="null")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tape = rank(ak.json_arg(a.input), a.seed, ak.json_arg(a.opt, {}))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "transe", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
