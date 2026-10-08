"""classic-ml: Python reference for the processes 'knn', 'knnMap', 'tree' and 'kmeans' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/classic-ml/process.ts in plain Python floats (no sklearn):
k nearest neighbours (one new point, and the decision map for several k), a CART decision tree with Gini impurity
(hold-out split, every node breadth first, the depth sweep) and k-means (Lloyd's algorithm, plus the k sweep).
`npm run atlas -- parity classic-ml` records the TS tape of every process source, runs this file with the same input
(the plain numbers of ml.ts dataOf()) and compares the tapes frame by frame: phase, step and sub exactly, every stored
field within its tol, every metric within 1e-12 (AK204). The tree's shuffle and k-means' starting points use ak.Rng,
bit-identical to the TS rng for next(), int() and fork().

    npm run atlas -- parity classic-ml --fixtures     # the usual way (it passes the arguments below)

CLI contract: --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json or @file> --out <stem>
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

B, R, DMAX, MAXIT = 1.25, 40, 8, 100
KS = [1, 3, 5, 9, 15, 25]


def lab(en, zh=None):
    return (en, zh or en)


def tensor(rows, shape=None):
    a = np.array(rows, dtype=np.float64)
    return a.reshape(shape) if shape is not None else a


def cell_x(c):
    return -B + ((c + 0.5) * 2 * B) / R


def cell_y(r):
    return B - ((r + 0.5) * 2 * B) / R


def gini(counts, m):
    s = 0.0
    for c in counts:
        p = c / m
        s += p * p
    return 1 - s


def majority(counts):
    best = 0
    for c in range(1, len(counts)):
        if counts[c] > counts[best]:
            best = c
    return best


def patches(grid):
    seen, count = set(), 0
    for s in range(R * R):
        if s in seen:
            continue
        count += 1
        seen.add(s)
        stack = [s]
        while stack:
            c = stack.pop()
            r, x = divmod(c, R)
            for rr, xx in ((r - 1, x), (r + 1, x), (r, x - 1), (r, x + 1)):
                t = rr * R + xx
                if 0 <= rr < R and 0 <= xx < R and t not in seen and grid[t] == grid[c]:
                    seen.add(t)
                    stack.append(t)
    return count


def per(prefix, values):
    return {f"{prefix}{i}": v for i, v in enumerate(values)}


class Data:
    def __init__(self, d: dict):
        self.xy = [float(v) for v in d["xy"]]
        self.label = [int(v) for v in d["label"]]
        self.n = len(self.label)
        self.classes = int(d["classes"])
        self.q = [float(v) for v in d["q"]]
        self.k, self.kc = int(d["k"]), int(d["kc"])

    def x(self, i):
        return self.xy[2 * i]

    def y(self, i):
        return self.xy[2 * i + 1]

    def points(self):
        return {"xy": tensor(self.xy, (self.n, 2)), "label": tensor(self.label)}


def schema(fields):
    return {k: {"kind": "tensor", "label": lab(k), "tol": tol} for k, tol in fields.items()}


def phases(names):
    return {p: {"label": lab(p), "kind": "other"} for p in names}


# ------------------------------------------------------------------ k nearest neighbours
def run_knn(seed, opt, D: Data):
    w = ak.TapeWriter("knn", schema({"xy": "exact", "label": "exact", "q": "exact", "d": 1e-12, "nn": "exact"}), seed=seed, opt=opt,
                      phases=phases(["query", "distances", "vote"]), provenance="live")
    if not D.n:
        return w
    qx, qy = D.q
    w.frame("query", {**D.points(), "q": tensor(D.q)}, metrics={"n": D.n, "qx": qx, "qy": qy, "classes": D.classes})
    d = [math.sqrt((D.x(i) - qx) * (D.x(i) - qx) + (D.y(i) - qy) * (D.y(i) - qy)) for i in range(D.n)]
    order = sorted(range(D.n), key=lambda i: (d[i], i))
    i0 = order[0]
    w.frame("distances", {"d": tensor(d)}, metrics={"n": D.n, "qx": qx, "qy": qy, "i0": i0, "x0": D.x(i0), "y0": D.y(i0), "d0": d[i0], "c0": D.label[i0]})
    k = min(D.k, D.n)
    votes = [0] * D.classes
    for i in order[:k]:
        votes[D.label[i]] += 1
    top = max(votes)
    w.frame("vote", {"nn": tensor(order[:k])}, metrics={"k": k, "kAsked": D.k, "r": d[order[k - 1]], **per("v", votes), "pred": majority(votes),
                                                         "tie": 1 if votes.count(top) > 1 else 0})
    return w


def run_knn_map(seed, opt, D: Data):
    w = ak.TapeWriter("knnMap", schema({"grid": "exact"}), seed=seed, opt=opt, phases=phases(["map"]), provenance="live")
    ks = [k for k in KS if k <= D.n]
    if not D.n:
        return w
    grids = [[0.0] * (R * R) for _ in ks]
    for r in range(R):
        for c in range(R):
            px, py = cell_x(c), cell_y(r)
            near = sorted(((D.x(i) - px) * (D.x(i) - px) + (D.y(i) - py) * (D.y(i) - py), i) for i in range(D.n))
            for s, k in enumerate(ks):
                votes = [0] * D.classes
                for _, i in near[:k]:
                    votes[D.label[i]] += 1
                grids[s][r * R + c] = majority(votes)
    for s, k in enumerate(ks):
        w.frame("map", {"grid": tensor(grids[s], (R, R))}, step=k, metrics={"k": k, "patches": patches(grids[s])}, caption=(f"k = {k}", f"k = {k}"))
    return w


# ------------------------------------------------------------------ decision tree
def run_tree(seed, opt, D: Data):
    w = ak.TapeWriter("tree", schema({"xy": "exact", "label": "exact", "test": "exact", "cand": 1e-12, "grid": "exact"}), seed=seed, opt=opt,
                      phases=phases(["holdout", "gini", "node", "depth"]), provenance="live")
    n, C = D.n, D.classes
    if not n:
        return w
    rng = ak.Rng(seed)
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.int(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    n_test = max(1, n // 4) if n >= 2 else 0
    test = [0] * n
    for i in perm[:n_test]:
        test[i] = 1
    train = [i for i in range(n) if not test[i]]

    def count(idx):
        c = [0] * C
        for i in idx:
            c[D.label[i]] += 1
        return c

    def coord(i, axis):
        return D.xy[2 * i + axis]

    w.frame("holdout", {**D.points(), "test": tensor(test)}, metrics={"n": n, "nTrain": len(train), "nTest": n_test, "classes": C})
    root = count(train)
    w.frame("gini", {}, metrics={"m": len(train), "gini": gini(root, len(train)), **per("c", root), "pred": majority(root)})
    nodes = [{"depth": 0, "idx": train, "box": [-B, B, -B, B], "pred": majority(root), "axis": -1, "thr": 0.0, "kids": []}]
    nid = 0
    while nid < len(nodes):
        nd = nodes[nid]
        m, counts = len(nd["idx"]), count(nd["idx"])
        g = gini(counts, m)
        if nd["depth"] >= DMAX:
            nid += 1
            continue
        cand, best = [], None  # best = (w, axis, thr, cut, order)
        for axis in ((0, 1) if g > 0 else ()):
            order = sorted(nd["idx"], key=lambda i: (coord(i, axis), i))
            left = [0] * C
            for i in range(1, m):
                left[D.label[order[i - 1]]] += 1
                a, b = coord(order[i - 1], axis), coord(order[i], axis)
                if a == b:
                    continue
                right = [counts[c] - left[c] for c in range(C)]
                wg = (i / m) * gini(left, i) + ((m - i) / m) * gini(right, m - i)
                thr = (a + b) / 2
                cand += [axis, thr, wg]
                if best is None or wg < best[0]:
                    best = (wg, axis, thr, i, order)
        box = nd["box"]
        base = {"node": nid, "depth": nd["depth"], "m": m, "gini": g, **per("c", counts), "pred": nd["pred"], "x0": box[0], "x1": box[1], "y0": box[2], "y1": box[3]}
        if best is None or not (g - best[0] > 1e-12):
            w.frame("node", {}, step=nid, metrics={**base, "axis": -1})
            nid += 1
            continue
        wg, axis, thr, cut, order = best
        sides = [order[:cut], order[cut:]]
        cs = [count(s) for s in sides]
        boxes = [[box[0], thr, box[2], box[3]], [thr, box[1], box[2], box[3]]] if axis == 0 else [[box[0], box[1], box[2], thr], [box[0], box[1], thr, box[3]]]
        nd.update(axis=axis, thr=thr, kids=[len(nodes), len(nodes) + 1])
        for s in range(2):
            nodes.append({"depth": nd["depth"] + 1, "idx": sides[s], "box": boxes[s], "pred": majority(cs[s]), "axis": -1, "thr": 0.0, "kids": []})
        nl, nr = len(sides[0]), len(sides[1])
        w.frame("node", {"cand": tensor(cand, (len(cand) // 3, 3))}, step=nid,
                metrics={**base, "axis": axis, "thr": thr, "nL": nl, "nR": nr, "gL": gini(cs[0], nl), "gR": gini(cs[1], nr), "w": wg, "gain": g - wg,
                         "left": nd["kids"][0], "right": nd["kids"][1], **per("l", cs[0]), **per("r", cs[1])})
        nid += 1

    def leaf(x, y, d):
        nd = nodes[0]
        while nd["depth"] < d and nd["kids"]:
            nd = nodes[nd["kids"][0] if (x if nd["axis"] == 0 else y) <= nd["thr"] else nd["kids"][1]]
        return nd

    for d in range(1, DMAX + 1):
        grid = [float(leaf(cell_x(c), cell_y(r), d)["pred"]) for r in range(R) for c in range(R)]
        ok = [0, 0]
        for i in range(n):
            if leaf(D.x(i), D.y(i), d)["pred"] == D.label[i]:
                ok[test[i]] += 1
        leaves = sum(1 for nd in nodes if nd["depth"] == d or (nd["depth"] < d and not nd["kids"]))
        w.frame("depth", {"grid": tensor(grid, (R, R))}, step=d, metrics={"d": d, "trainAcc": ok[0] / len(train), "testAcc": ok[1] / n_test if n_test else -1,
                                                                            "leaves": leaves, "nTrain": len(train), "nTest": n_test})
    return w


# ------------------------------------------------------------------ k-means
def lloyd(D: Data, k, rng, w=None):
    n = D.n
    pool, pick = list(range(n)), []
    for j in range(k):
        r = rng.int(n - j)
        pick.append(pool[r])
        pool[r] = pool[n - 1 - j]
    cen = []
    for i in pick:
        cen += [D.x(i), D.y(i)]
    assign = [-1] * n

    def d2(i, j):
        dx, dy = D.x(i) - cen[2 * j], D.y(i) - cen[2 * j + 1]
        return dx * dx + dy * dy

    def inertia():
        s = 0
        for i in range(n):
            s += d2(i, assign[i])
        return s

    mx = 0
    for j in range(k):
        mx += cen[2 * j]
    my = 0
    for j in range(k):
        my += cen[2 * j + 1]
    mx, my = mx / k, my / k
    p = 0
    for i in range(1, n):
        if (D.x(i) - mx) * (D.x(i) - mx) + (D.y(i) - my) * (D.y(i) - my) < (D.x(p) - mx) * (D.x(p) - mx) + (D.y(p) - my) * (D.y(p) - my):
            p = i
    if w:
        w.frame("init", {**D.points(), "C": tensor(cen, (k, 2)), "assign": tensor(assign)}, step=0, metrics={"k": k, "n": n, "p": p, **per("p", pick)})
    for t in range(MAXIT):
        changed = 0
        for i in range(n):
            b = 0
            for j in range(1, k):
                if d2(i, j) < d2(i, b):
                    b = j
            if b != assign[i]:
                changed += 1
            assign[i] = b
        e = inertia()
        if w:
            w.frame("assign", {"assign": tensor(assign)}, step=t, metrics={"t": t, "changed": changed, "inertia": e, "p": p, "px": D.x(p), "py": D.y(p), "j": assign[p],
                                                                          **per("d", [math.sqrt(d2(p, j)) for j in range(k)])})
        if not changed or t == MAXIT - 1:
            if w:
                mix = [sum(1 for i in range(n) if assign[i] == j and D.label[i] == c) for j in range(k) for c in range(D.classes)]
                w.frame("done", {}, step=t, metrics={"k": k, "rounds": t, "inertia": e, "converged": 0 if changed else 1, **per("m", mix)})
            return e, t
        old, moved, first = cen[:], 0, None
        for j in range(k):
            sx = sy = 0
            c = 0
            for i in range(n):
                if assign[i] == j:
                    sx += D.x(i)
                    sy += D.y(i)
                    c += 1
            if c:
                cen[2 * j], cen[2 * j + 1] = sx / c, sy / c
            dx, dy = cen[2 * j] - old[2 * j], cen[2 * j + 1] - old[2 * j + 1]
            moved = max(moved, math.sqrt(dx * dx + dy * dy))
            if j == 0:
                first = (sx, sy, c)
        if w:
            w.frame("update", {"C": tensor(cen, (k, 2))}, step=t, metrics={"t": t, "moved": moved, "inertia": inertia(), "cnt": first[2], "sx": first[0], "sy": first[1],
                                                                           "ox": old[0], "oy": old[1], "nx": cen[0], "ny": cen[1]})
    return float("nan"), MAXIT


def run_kmeans(seed, opt, D: Data):
    o = {"sweep": 0, **opt}
    w = ak.TapeWriter("kmeans", schema({"xy": "exact", "label": "exact", "C": 1e-12, "assign": "exact"}), seed=seed, opt=o,
                      phases=phases(["init", "assign", "update", "done", "sweep"]), provenance="live")
    if not D.n:
        return w
    rng = ak.Rng(seed)
    lloyd(D, min(D.kc, D.n), rng, w)
    for k in range(1, int(o["sweep"]) + 1):
        e, t = lloyd(D, min(k, D.n), rng.fork(f"k{k}"))
        w.frame("sweep", {}, step=k, metrics={"k": k, "inertia": e, "rounds": t})
    return w


RUN = {"knn": run_knn, "knnMap": run_knn_map, "tree": run_tree, "kmeans": run_kmeans}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="knn", choices=list(RUN))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran")
    ap.add_argument("--draws", default=None, help="unused: these processes draw only with next()/int(), which ak.Rng repeats exactly")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (ml.ts dataOf: xy, label, classes, q, k, depth, kc), or @<file>")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/classic-ml/<process>.ref)")
    a = ap.parse_args()
    raw = ak.json_arg(a.input)  # inline JSON, or @<file> for a long input (K1: atlas parity passes 200 points that way)
    if not isinstance(raw, dict) or "xy" not in raw:
        sys.exit("reference: --input must be the plain numbers of ml.ts dataOf() (the sources pass them)")
    tape = RUN[a.process](a.seed, json.loads(a.opt), Data(raw))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "classic-ml", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
