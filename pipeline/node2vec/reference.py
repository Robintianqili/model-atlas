"""node2vec: the Python reference of process 'walks' (site/src/lib/explainers/node2vec/process.ts; kind C parity).

The biased random walks of node2vec (Grover & Leskovec 2016) on the learner's graph, written again from the rules in
walks.ts: WALKS = 10 rounds, a walk from every member 0..33 in turn, LEN - 1 = 9 moves each (a walk stops at a member
without friends). A move from v, having come from t, weighs each friend x of v (in increasing order): 1/p when x = t,
1 when x is also a friend of t, 1/q otherwise; every friend 1 on a walk's first move. The draw is u = rng.next() and the
first x whose running sum of weights exceeds u * (the sum of all weights). ak.Rng is bit-identical to the TS rng, so the
walks, the draws and every metric of every frame must agree (ints exactly, floats within 1e-12).

    npm run atlas -- parity node2vec       # records the TS tape, runs this file, compares frame by frame

CLI contract (every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

N, WALKS, LEN = 34, 10, 10
PQ = (0.25, 0.5, 1, 2, 4)
START_P, START_Q, FOCUS = 1, 0.5, 8
PHASES = {"move": {"label": ("One move of a walk", "游走的一步"), "kind": "sample"}}


def graph_of(raw) -> tuple[list[list[int]], int, float, float]:
    """karate.ts graphOf + walks.ts inputOf: friendships (a < b, no self-loops, deduplicated), focus, p and q."""
    raw = raw if isinstance(raw, dict) else {}
    edges = raw.get("edges")
    if not isinstance(edges, list):
        edges = [list(e) for e in ak.data.karate()["edges"]]
    member = lambda i: isinstance(i, int) and not isinstance(i, bool) and 0 <= i < N  # noqa: E731
    nb: list[set[int]] = [set() for _ in range(N)]
    for e in edges:
        if isinstance(e, list) and len(e) >= 2 and member(e[0]) and member(e[1]) and e[0] != e[1]:
            nb[e[0]].add(e[1])
            nb[e[1]].add(e[0])
    focus = raw.get("focus") if member(raw.get("focus")) else FOCUS
    pick = lambda v, d: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v in PQ else d  # noqa: E731
    return [sorted(s) for s in nb], focus, pick(raw.get("p"), START_P), pick(raw.get("q"), START_Q)


def move_of(nb, t: int, v: int, p: float, q: float):
    x = nb[v]
    kind = [1 if t < 0 else 0 if y == t else 1 if y in nb[t] else 2 for y in x]
    a = [1 / p if k == 0 else 1.0 if k == 1 else 1 / q for k in kind]
    total = 0.0
    for w in a:
        total += w
    return x, kind, a, [w / total for w in a], total


def walks(raw, seed: int, opt: dict) -> "ak.TapeWriter":
    w = ak.TapeWriter("walks", {}, seed=seed, opt=opt, phases=PHASES, provenance="live")
    nb, focus, p, q = graph_of(raw)
    rng = ak.Rng(seed)
    evs = []
    for r in range(WALKS):
        for s in range(N):
            t, v = -1, s
            for k in range(LEN - 1):
                if not nb[v]:
                    break
                x, kind, a, prob, total = move_of(nb, t, v, p, q)
                u = rng.next()
                target, cum, i = u * total, 0.0, 0
                while i < len(a) - 1:
                    cum += a[i]
                    if target < cum:
                        break
                    i += 1
                by = {}
                for kk in (0, 1, 2):
                    n, pp = 0, 0.0
                    for j, kj in enumerate(kind):
                        if kj == kk:
                            n += 1
                            pp += prob[j]
                    by[kk] = (n, pp)
                evs.append({"w": r * N + s, "s": s, "k": k, "t": t, "v": v, "x": x[i], "u": u, "i": i, "deg": len(x), "total": total,
                            "ux": u * total, "a": a[i], "px": prob[i], "kx": kind[i], "n0": by[0][0], "p0": by[0][1], "n1": by[1][0],
                            "p1": by[1][1], "n2": by[2][0], "p2": by[2][1], "_kinds": len(set(kind))})
                t, v = v, x[i]
    # the lesson's walk (walks.ts exampleWalk): the focus member's first walk, else the first walk that moves
    lw = focus if nb[focus] else next((s for s in range(N) if nb[s]), -1)
    best = None
    for e in evs:  # its first move with a previous member and the most kinds of candidates
        if e["w"] == lw and e["t"] >= 0 and (best is None or e["_kinds"] > best["_kinds"]):
            best = e
    for step, e in enumerate(evs):
        m = {k: v for k, v in e.items() if k != "_kinds"}
        m["fw"] = 1 if e["w"] == lw else 0
        m["ex"] = 1 if e is best else 0
        w.frame("move", {}, step=step, sub=e["k"] + 1, metrics=m)
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", required=True, choices=["walks"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None, help="unused: the draws are made again with ak.Rng")
    ap.add_argument("--input", default="null")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tape = walks(ak.json_arg(a.input), a.seed, ak.json_arg(a.opt, {}))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "node2vec", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
