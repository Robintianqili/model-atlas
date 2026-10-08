"""gcn: Python references for the processes of site/src/lib/explainers/gcn/process.ts (kind C parity, design §4.11).

    mp      message passing into the member in focus (layer 1). Its input is what the TS source hands the process: the
            traced numbers of the exported model (degrees, the focus row of Â, X Wᵀ, b, the layer output). This file
            re-derives the events from them: the self-loop, the friend with the most and the fewest friends, one
            message per incoming edge (Âᵢⱼ × row j) with the running sums, the sum + b and tanh.
    smooth  over-smoothing: Â from the learner's graph (the raw input: friendships), then the one-hot rows averaged
            with Â k = 0 … rounds times in float64; per round the cosine similarity of Mr. Hi's (0) and the Officer's
            (33) rows and each member's lean (cos with 33 − cos with 0). The loops run in the TS order, so the floats
            agree bit for bit.

    npm run atlas -- parity gcn       # records the TS tapes, runs this file per process source, compares frame by frame

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

N, F = 34, 4
L = lambda en, zh: (en, zh)  # noqa: E731
MP_SCHEMA = {
    "node": {"kind": "int", "label": L("Member in focus", "关注的成员"), "tol": "exact"},
    "sum": {"kind": "tensor", "label": L("Messages added so far", "目前加起来的消息"), "shape": [F], "tol": 1e-12},
}
MP_PHASES = {
    "loop": {"label": L("Add a self-loop", "加上自环"), "kind": "input"},
    "norm": {"label": L("Weigh each edge (Â)", "给每条边定权重（Â）"), "kind": "other"},
    "transform": {"label": L("Transform every member (X Wᵀ)", "变换每个成员（X Wᵀ）"), "kind": "forward"},
    "message": {"label": L("A message along one edge", "沿一条边的消息"), "kind": "message"},
    "aggregate": {"label": L("Add up the messages and the bias", "把消息和偏置加起来"), "kind": "aggregate"},
    "activate": {"label": L("Tanh", "Tanh"), "kind": "forward"},
}
SMOOTH_SCHEMA = {"lean": {"kind": "tensor", "label": L("Lean toward the Officer (+) or Mr. Hi (−)", "偏向干事（+）还是 Mr. Hi（−）"), "shape": [N], "tol": 1e-12}}
SMOOTH_PHASES = {"round": {"label": L("Average with the neighbours", "和邻居平均"), "kind": "aggregate"}}


def vec(p: str, v) -> dict:
    return {f"{p}{k}": float(x) for k, x in enumerate(v)}


def mp(inp: dict | None, seed: int, opt: dict) -> "ak.TapeWriter":
    w = ak.TapeWriter("mp", MP_SCHEMA, seed=seed, opt=opt, phases=MP_PHASES, provenance="live")
    if not inp:
        return w
    i, nb, deg, ahat, xw = inp["node"], inp["nb"], inp["deg"], inp["ahat"], inp["xw"]
    st = lambda s: {"node": i, "sum": np.array(s, dtype=np.float64)}  # noqa: E731
    zero = [0.0] * F
    w.frame("loop", st(zero), step=0, metrics={"node": i, "friends": len(nb), "d": deg[i]})
    hub = low = nb[0] if nb else i
    for j in nb:  # first best wins (ties: the smaller member)
        if deg[j] > deg[hub]:
            hub = j
        if deg[j] < deg[low]:
            low = j
    w.frame("norm", st(zero), step=0, metrics={"node": i, "d": deg[i], "self": ahat[i], "hub": hub, "dHub": deg[hub], "aHub": ahat[hub],
                                              "low": low, "dLow": deg[low], "aLow": ahat[low], "friends": len(nb)})
    w.frame("transform", st(zero), step=0, metrics={"node": i, **vec("x", xw[i * F:(i + 1) * F])})
    s = [0.0] * F
    senders = [i] + list(nb)
    for k, j in enumerate(senders):
        x = xw[j * F:(j + 1) * F]
        m = [ahat[j] * v for v in x]
        s = [a + b for a, b in zip(s, m)]
        w.frame("message", st(s), step=k + 1, metrics={"node": i, "j": j, "k": k + 1, "of": len(senders), "a": ahat[j], "dj": deg[j],
                                                  **vec("x", x), **vec("m", m), **vec("s", s)})
    w.frame("aggregate", st(s), step=0, metrics={"node": i, **vec("s", s), **vec("b", inp["b"]), **vec("z", inp["z"])})
    w.frame("activate", st(s), step=0, metrics={"node": i, **vec("s", s), **vec("b", inp["b"]), **vec("z", inp["z"]), **vec("h", inp["h"])})
    return w


def smooth(raw: dict | None, seed: int, opt: dict) -> "ak.TapeWriter":
    w = ak.TapeWriter("smooth", SMOOTH_SCHEMA, seed=seed, opt=opt, phases=SMOOTH_PHASES, provenance="live")
    edges = (raw or {}).get("edges")
    if edges is None:  # graph.ts graphOf(): no friendships list -> Zachary's club
        edges = ak.data.karate()["edges"]
    a = [[0.0] * N for _ in range(N)]
    for p, q in edges:
        if p != q:
            a[p][q] = a[q][p] = 1.0
    loops = [[a[r][c] + (1.0 if r == c else 0.0) for c in range(N)] for r in range(N)]
    deg = [0.0] * N
    for r in range(N):
        for c in range(N):
            deg[r] += loops[r][c]
    rs = [1 / math.sqrt(d) for d in deg]
    ahat = [[rs[r] * loops[r][c] * rs[c] for c in range(N)] for r in range(N)]
    z = [[1.0 if r == c else 0.0 for c in range(N)] for r in range(N)]

    def cos(p: int, q: int) -> float:
        ab = aa = bb = 0.0
        for c in range(N):
            ab += z[p][c] * z[q][c]
            aa += z[p][c] * z[p][c]
            bb += z[q][c] * z[q][c]
        return ab / math.sqrt(aa * bb)

    for k in range(int(opt.get("rounds", 8)) + 1):
        if k > 0:
            nxt = [[0.0] * N for _ in range(N)]
            for r in range(N):
                for j in range(N):
                    if ahat[r][j]:
                        for c in range(N):
                            nxt[r][c] += ahat[r][j] * z[j][c]
            z = nxt
        lean = [cos(r, 33) - cos(r, 0) for r in range(N)]
        spread = 0.0
        for v in lean:
            spread += abs(v)
        w.frame("round", {"lean": np.array(lean)}, step=k, metrics={"k": k, "leaders": cos(0, 33), "spread": spread / N})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", required=True, choices=["mp", "smooth"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None, help="unused: these processes draw no random numbers")
    ap.add_argument("--input", default="null")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    opt = json.loads(a.opt)
    tape = (mp if a.process == "mp" else smooth)(json.loads(a.input), a.seed, opt)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "gcn", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
