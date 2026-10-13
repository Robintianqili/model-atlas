"""gaussian-process: Python reference for process 'gp' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/gaussian-process/process.ts (and gp.ts) with numpy's own
linear algebra (LAPACK Cholesky and solves, vectorised kernels): Gaussian process regression with an RBF kernel
k(a, b) = sf2 exp(-(a - b)^2 / (2 ls^2)) and noise variance sn2 on the learner's points. Frames: prior, kernel, one
add per point (in the order added), sample, evidence, done. `npm run atlas -- parity gaussian-process` records the TS
tape of every process source, runs this file with the same input (the plain numbers of gp.ts dataOf(): x, y, ls, sf2,
sn2 and the standard-normal draws zPrior, zPost) and compares the tapes frame by frame: phase and step exactly, every
stored field within its tol (1e-9; the sample functions 1e-6, because their 81 x 81 grid covariance is nearly singular
and LAPACK factors it in another order), every metric within 1e-9 (AK204). The draws are made in TS and arrive here as
numbers: this process draws nothing.

    npm run atlas -- parity gaussian-process --fixtures     # the usual way (it passes the arguments below)

CLI contract: --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json | @file> --out <stem>
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

# gp.ts: the plot's x range, the grid, the samples, the slider of length-scales, the jitter and the "large" data set
XLO, XHI, G, NS = -5, 5, 81, 3
GRID = np.array([XLO + (g * (XHI - XLO)) / (G - 1) for g in range(G)], dtype=np.float64)
LS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3]
JIT = 1e-6
BIG = 10000


def lab(en, zh):
    return (en, zh)


SCHEMA = {
    "x": {"kind": "tensor", "label": lab("x of each point, in the order added", "每个点的 x（按加入的顺序）"), "role": "data", "tol": "exact"},
    "y": {"kind": "tensor", "label": lab("y of each point", "每个点的 y"), "role": "target", "tol": "exact"},
    "mean": {"kind": "tensor", "label": lab("Posterior mean on the grid", "网格上的后验均值"), "tol": 1e-9},
    "sd": {"kind": "tensor", "label": lab("Posterior standard deviation on the grid", "网格上的后验标准差"), "tol": 1e-9},
    "samples": {"kind": "tensor", "label": lab("Sample functions on the grid", "网格上的样本函数"), "tol": 1e-6},
    "K": {"kind": "tensor", "label": lab("Kernel matrix K + σn² I", "核矩阵 K + σn² I"), "tol": 1e-12},
    "alpha": {"kind": "tensor", "label": lab("Weights α = (K + σn² I)⁻¹ y", "权重 α = (K + σn² I)⁻¹ y"), "tol": 1e-9},
    "lmls": {"kind": "tensor", "label": lab("Log marginal likelihood for each length-scale on the slider", "滑块上每个长度尺度的对数边缘似然"), "tol": 1e-9},
}
PHASES = {
    "prior": {"label": lab("Before any data: the prior", "还没有数据：先验"), "kind": "input"},
    "kernel": {"label": lab("The kernel matrix and the weights", "核矩阵和权重"), "kind": "forward"},
    "add": {"label": lab("A point joins: the posterior", "加入一个点：后验"), "kind": "update"},
    "sample": {"label": lab("Functions drawn from the posterior", "从后验中抽取函数"), "kind": "sample"},
    "evidence": {"label": lab("The marginal likelihood", "边缘似然"), "kind": "loss"},
    "done": {"label": lab("Summary", "总结"), "kind": "output"},
}


# ------------------------------------------------------------------ the model (numpy)
def kern(a, b, ls, sf2):
    """RBF kernel matrix between the 1-D arrays a and b"""
    d = np.subtract.outer(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return sf2 * np.exp(-(d * d) / (2 * ls * ls))


def fit(x, y, ls, sf2, sn2):
    K = kern(x, x, ls, sf2) + sn2 * np.eye(len(x))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.asarray(y, dtype=np.float64)))
    return K, L, alpha


def posterior(x, y, ls, sf2, sn2, at=GRID):
    K, L, alpha = fit(x, y, ls, sf2, sn2)
    Ks = kern(x, at, ls, sf2)  # n x m
    V = np.linalg.solve(L, Ks)
    mean = Ks.T @ alpha
    sd = np.sqrt(np.maximum(sf2 - np.sum(V * V, axis=0), 0.0))
    return {"K": K, "L": L, "alpha": alpha, "mean": mean, "sd": sd, "V": V}


def evidence(x, y, ls, sf2, sn2):
    _, L, alpha = fit(x, y, ls, sf2, sn2)
    n = len(x)
    f = -0.5 * float(np.dot(np.asarray(y, dtype=np.float64), alpha))
    c = -float(np.sum(np.log(np.diag(L))))
    k = -0.5 * n * math.log(2 * math.pi)
    return {"fit": f, "cpx": c, "cst": k, "lml": f + c + k}


def draw(mean, cov, z, sf2):
    C = np.linalg.cholesky(cov + JIT * sf2 * np.eye(G))
    Z = np.asarray(z, dtype=np.float64).reshape(NS, G)
    return mean[None, :] + Z @ C.T


def spread(s, g):
    return float(np.max(s[:, g]) - np.min(s[:, g]))


# ------------------------------------------------------------------ the process
def run_gp(seed, opt, d):
    w = ak.TapeWriter("gp", SCHEMA, seed=seed, opt=opt, phases=PHASES, provenance="live")
    x, y, ls, sf2, sn2 = [float(v) for v in d["x"]], [float(v) for v in d["y"]], d["ls"], d["sf2"], d["sn2"]
    n = len(x)
    if not n:
        return w
    sd0, kd = math.sqrt(sf2), sf2 + sn2
    prior = draw(np.zeros(G), kern(GRID, GRID, ls, sf2), d["zPrior"], sf2)
    w.frame("prior", {"x": np.array(x), "y": np.array(y), "mean": np.zeros(G), "sd": np.full(G, sd0), "samples": prior}, step=0,
            metrics={"n": n, "ls": ls, "sf2": sf2, "sn2": sn2, "sd0": sd0, "band0": 2 * sd0})
    allp = posterior(x, y, ls, sf2, sn2)
    pm = {}
    if n > 1:
        iN = jN = iF = jF = 0
        dN, dF = math.inf, -1.0
        for i in range(n):
            for j in range(i + 1, n):
                dd = abs(x[i] - x[j])
                if dd < dN:
                    iN, jN, dN = i, j, dd
                if dd > dF:
                    iF, jF, dF = i, j, dd
        pm = {"iN": iN, "jN": jN, "dN": dN, "iF": iF, "jF": jF, "dF": dF,
              "kN": float(kern([x[iN]], [x[jN]], ls, sf2)[0, 0]), "kF": float(kern([x[iF]], [x[jF]], ls, sf2)[0, 0])}
    k0 = kern([0.0], x, ls, sf2)[0]
    terms = allp["alpha"] * k0
    top = 0
    for i in range(n):
        if abs(terms[i]) > abs(terms[top]) * (1 + 1e-9):
            top = i
    w.frame("kernel", {"K": allp["K"], "alpha": allp["alpha"]}, step=0,
            metrics={"n": n, "ls": ls, "sf2": sf2, "sn2": sn2, "kd": kd, **pm, "mq": float(np.sum(terms)), "top": top,
                     "topA": float(allp["alpha"][top]), "topK": float(k0[top]), "topX": x[top], "topTerm": float(terms[top])})
    post = allp
    for i in range(n):
        xs, ys = x[: i + 1], y[: i + 1]
        post = posterior(xs, ys, ls, sf2, sn2)
        at = posterior(xs, ys, ls, sf2, sn2, np.array(xs))
        max_res = float(np.max(np.abs(at["mean"] - np.array(ys))))
        avg_band = float(np.sum(2 * post["sd"]) / G)
        first = {}
        if i == 0:
            xp = x[0] + ls if x[0] + ls <= GRID[G - 1] else x[0] - ls
            kp = float(kern([xp], [x[0]], ls, sf2)[0, 0])
            first = {"kd": kd, "w": sf2 / kd, "xp": xp, "kp": kp, "mp": kp * y[0] / kd, "sdp": math.sqrt(sf2 - kp * kp / kd)}
        w.frame("add", {"mean": post["mean"], "sd": post["sd"]}, step=i,
                metrics={"i": i, "used": i + 1, "xi": x[i], "yi": y[i], "mAt": float(at["mean"][i]), "sdAt": float(at["sd"][i]), "sd0": sd0,
                         "avgBand": avg_band, "maxRes": max_res, **first})
    gaps = np.min(np.abs(np.subtract.outer(GRID, np.array(x))), axis=1)
    g_far = 0
    for g in range(G):
        if gaps[g] > gaps[g_far]:
            g_far = g
    g_near = 0
    for g in range(G):
        if abs(GRID[g] - x[0]) < abs(GRID[g_near] - x[0]):
            g_near = g
    cov = kern(GRID, GRID, ls, sf2) - post["V"].T @ post["V"]
    s = draw(post["mean"], cov, d["zPost"], sf2)
    w.frame("sample", {"samples": s}, step=n - 1,
            metrics={"n": n, "xFar": float(GRID[g_far]), "dFar": float(gaps[g_far]), "sdFar": float(post["sd"][g_far]), "spreadFar": spread(s, g_far),
                     "xNear": float(GRID[g_near]), "sdNear": float(post["sd"][g_near]), "spreadNear": spread(s, g_near), "sd0": sd0})
    ev = [evidence(x, y, l, sf2, sn2) for l in LS]
    cur = evidence(x, y, ls, sf2, sn2)
    best = 0
    for i in range(len(LS)):
        if ev[i]["lml"] > ev[best]["lml"]:
            best = i
    w.frame("evidence", {"lmls": np.array([e["lml"] for e in ev])}, step=n - 1,
            metrics={"n": n, "ls": ls, "sf2": sf2, "fit": cur["fit"], "cpx": cur["cpx"], "cst": cur["cst"], "lml": cur["lml"], "bestI": best,
                     "bestLs": LS[best], "bestLml": ev[best]["lml"], "drop": ev[best]["lml"] - cur["lml"]})
    xs = sorted(set(x))
    ga = gb = 0.0
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] > gb - ga:
            ga, gb = xs[i - 1], xs[i]
    gm = {}
    if len(xs) > 1:
        p3 = posterior(x, y, ls, sf2, sn2, np.array([ga, (ga + gb) / 2, gb]))
        gm = {"ga": ga, "gb": gb, "gw": gb - ga, "gwl": (gb - ga) / ls, "gmid": (ga + gb) / 2, "sdA": float(p3["sd"][0]), "sdB": float(p3["sd"][2]),
              "sdMid": float(p3["sd"][1])}
    last = posterior(x, y, ls, sf2, sn2, np.array(x))
    peak, ymax = float(np.max(post["mean"])), max(y)
    w.frame("done", {}, step=n - 1,
            metrics={"n": n, "ls": ls, "sf2": sf2, "sd0": sd0, **gm, "ops": n * n * n / 3, "big": BIG, "opsBig": BIG * BIG * BIG / 3, "mem": n * n,
                     "memBig": BIG * BIG, "peak": peak, "ymax": ymax, "maxRes": float(np.max(np.abs(last["mean"] - np.array(y)))),
                     "avgBand": float(np.sum(2 * post["sd"]) / G)})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="gp", choices=["gp"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran (none are used)")
    ap.add_argument("--draws", default=None, help="unused: the process draws nothing (the normal draws are in the input)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (gp.ts dataOf: x, y, ls, sf2, sn2, zPrior, zPost), or @file")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/gaussian-process/gp.ref)")
    a = ap.parse_args()
    data = ak.json_arg(a.input)
    if not isinstance(data, dict) or "x" not in data:
        sys.exit("reference: --input must be the plain numbers of gp.ts dataOf() (the sources pass them)")
    tape = run_gp(a.seed, json.loads(a.opt), data)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "gaussian-process", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
