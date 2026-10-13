"""hmm: Python reference for the processes 'forward', 'viterbi', 'sample' and 'spells' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/hmm/process.ts (and the algorithms in hmm.ts) with numpy: a
hidden Markov model with two hidden states (0 sunny, 1 rainy) and three observations (0 walk, 1 shop, 2 clean).
`npm run atlas -- parity hmm` records the TS tape of every source, runs this file with the same input (the plain numbers
of hmm.ts dataOf(): the activities obs, the probabilities pi, A, B and the starting tables base; the sampling source gets
pi, A, B; the spells source the made-up stretch of weather) and compares the tapes frame by frame: phase, step and sub
exactly, every stored field within its tol, every metric within 1e-12 (AK204). The sampling process draws its uniform
numbers from ak.Rng (bit-identical to the TS rng for next()), so it needs no recorded draws.

    npm run atlas -- parity hmm          # the usual way (it passes the arguments below)

CLI contract: --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
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

K = 2  # hidden states


def lab(en, zh):
    return (en, zh)


OBS = {"kind": "tensor", "label": lab("The activity of each day (0 walk, 1 shop, 2 clean)", "每天的活动（0 散步，1 购物，2 打扫）"), "role": "data", "tol": "exact"}
SCHEMAS = {
    "forward": {
        "obs": OBS,
        "guess": {"kind": "tensor", "label": lab("The guessed weather of each day (0 sunny, 1 rainy)", "猜测的每天天气（0 晴天，1 雨天）"), "tol": "exact"},
        "alpha": {"kind": "tensor", "label": lab("Forward probabilities α[day, weather]", "前向概率 α[天, 天气]"), "scale": "sequential", "tol": 1e-12},
    },
    "viterbi": {
        "obs": OBS,
        "delta": {"kind": "tensor", "label": lab("Best-path probabilities δ[day, weather]", "最优路径概率 δ[天, 天气]"), "scale": "sequential", "tol": 1e-12},
        "psi": {"kind": "tensor", "label": lab("Back-pointers: the weather the day before (−1 on day 0)", "回溯指针：前一天的天气（第 0 天为 −1）"), "tol": "exact"},
        "path": {"kind": "tensor", "label": lab("The decoded weather, traced back (−1 not yet)", "回溯得到的天气（−1 表示还没到）"), "tol": "exact"},
        "naive": {"kind": "tensor", "label": lab("Each day’s weather from its activity alone", "只看当天活动得出的天气"), "tol": "exact"},
        "path0": {"kind": "tensor", "label": lab("The decoded weather with the starting tables", "用起始表格解码出的天气"), "tol": "exact"},
    },
    "sample": {
        "weather": {"kind": "tensor", "label": lab("The drawn weather (0 sunny, 1 rainy; −1 not yet)", "抽到的天气（0 晴天，1 雨天；−1 还没抽）"), "tol": "exact"},
        "acts": {"kind": "tensor", "label": lab("The drawn activity (0 walk, 1 shop, 2 clean; −1 not yet)", "抽到的活动（0 散步，1 购物，2 打扫；−1 还没抽）"), "tol": "exact"},
    },
    "spells": {
        "year": {"kind": "tensor", "label": lab("The made-up weather (0 sunny, 1 rainy)", "编出来的天气（0 晴天，1 雨天）"), "role": "data", "tol": "exact"},
        "model": {"kind": "tensor", "label": lab("Share of rainy spells of each length, as the fitted chain implies", "拟合的马尔可夫链所隐含的各长度连雨天比例"), "scale": "sequential", "tol": 1e-12},
        "truth": {"kind": "tensor", "label": lab("Share of rainy spells of each length, as they happened", "实际各长度连雨天的比例"), "scale": "sequential", "tol": 1e-12},
    },
}
PHASES = {
    "forward": {"guess": ("input", "One guess of the weather, one product"), "init": ("forward", "Day 0: start probability times emission"),
                "cell": ("forward", "One cell: sum the ways in, times the emission"), "end": ("output", "The probability of all the activities")},
    "viterbi": {"init": ("forward", "Day 0: start probability times emission"), "cell": ("forward", "One cell: the best way in, times the emission"),
                "end": ("aggregate", "The best last cell"), "back": ("other", "Follow the back-pointer one day back"), "done": ("output", "The decoded weather")},
    "sample": {"day": ("sample", "Draw the weather, then the activity"), "done": ("output", "The made-up days, counted")},
    "spells": {"count": ("aggregate", "Count the transitions"), "fit": ("output", "The spell lengths the counted table implies")},
}


def phases(pid):
    return {k: {"label": (en, en), "kind": kind} for k, (kind, en) in PHASES[pid].items()}


def vec(x):
    return np.array(x, dtype=np.float64)


def grid(rows, n):
    g = np.zeros((n, K), dtype=np.float64)
    for t, r in enumerate(rows):
        g[t, :len(r)] = r
    return g


def argmax(xs):
    """the index of the largest value (ties to the smallest index)"""
    b = 0
    for i in range(1, len(xs)):
        if xs[i] > xs[b]:
            b = i
    return b


def total(xs):
    s = 0.0
    for x in xs:
        s = s + x
    return s


def two(name, xs):
    return {f"{name}{i}": v for i, v in enumerate(xs)}


def first_diff(a, b):
    return next((t for t in range(len(a)) if a[t] != b[t]), -1)


# ------------------------------------------------------------------ the algorithms (plain floats, in the TS order of operations)
def naive(d):
    return [argmax([d["B"][j][o] for j in range(K)]) for o in d["obs"]]


def joint(d, path):
    p = d["pi"][path[0]] * d["B"][path[0]][d["obs"][0]]
    for t in range(1, len(path)):
        p *= d["A"][path[t - 1]][path[t]] * d["B"][path[t]][d["obs"][t]]
    return p


def forward_table(d):
    """alpha[t][j] = (sum_i alpha[t-1][i] A[i][j]) B[j][o_t];  alpha[0][j] = pi[j] B[j][o_0]"""
    al = []
    for t, o in enumerate(d["obs"]):
        row = []
        for j in range(K):
            s = total([al[t - 1][i] * d["A"][i][j] for i in range(K)]) if t else d["pi"][j]
            row.append(s * d["B"][j][o])
        al.append(row)
    return al


def viterbi_table(d):
    """delta[t][j] = max_i delta[t-1][i] A[i][j] * B[j][o_t], psi the argmax; the path traced back from the best last state"""
    delta, psi = [], []
    for t, o in enumerate(d["obs"]):
        row, frm = [], []
        for j in range(K):
            c = [delta[t - 1][i] * d["A"][i][j] for i in range(K)] if t else [d["pi"][j]]
            i = argmax(c)
            row.append(c[i] * d["B"][j][o])
            frm.append(i if t else -1)
        delta.append(row)
        psi.append(frm)
    n = len(d["obs"])
    path = [0] * n
    if n:
        path[n - 1] = argmax(delta[n - 1])
    for t in range(n - 1, 0, -1):
        path[t - 1] = psi[t][path[t]]
    return delta, psi, path


# ------------------------------------------------------------------ the processes
def run_forward(w, d):
    n = len(d["obs"])
    if not n:
        return
    g, al = naive(d), forward_table(d)
    jt, paths = joint(d, g), float(K ** n)
    head = {"g1": g[1], "a1": d["A"][g[0]][g[1]], "b1": d["B"][g[1]][d["obs"][1]], "two": joint(d, g[:2])} if n > 1 else {}
    w.frame("guess", {"obs": vec(d["obs"]), "guess": vec(g), "alpha": grid([], n)}, step=0,
            metrics={"n": n, "paths": paths, "g0": g[0], "pi": d["pi"][g[0]], "b0": d["B"][g[0]][d["obs"][0]], **head, "joint": jt})
    for t in range(n):
        for j in range(K):
            o, b = d["obs"][t], d["B"][j][d["obs"][t]]
            cells = al[:t] + [[v if i <= j else 0.0 for i, v in enumerate(al[t])]]
            if not t:
                w.frame("init", {"alpha": grid(cells, n)}, step=0, sub=j, metrics={"t": t, "j": j, "o": o, "pi": d["pi"][j], "b": b, "alpha": al[t][j]})
            else:
                ins = [al[t - 1][i] * d["A"][i][j] for i in range(K)]
                w.frame("cell", {"alpha": grid(cells, n)}, step=t, sub=j,
                        metrics={"t": t, "j": j, "o": o, **two("p", al[t - 1]), **two("a", [d["A"][i][j] for i in range(K)]), **two("in", ins),
                                 "sum": total(ins), "b": b, "alpha": al[t][j]})
    ev = total(al[n - 1])
    w.frame("end", {}, step=n - 1, metrics={"n": n, "paths": paths, "evidence": ev, **two("s", [v / ev for v in al[n - 1]]), "joint": jt, "guessShare": jt / ev})


def run_viterbi(w, d):
    n = len(d["obs"])
    if not n:
        return
    delta, psi, vpath = viterbi_table(d)
    ev = total(forward_table(d)[n - 1])
    none = [-1] * n
    g = naive(d)
    p0 = viterbi_table({**d["base"], "obs": d["obs"]})[2]
    for t in range(n):
        for j in range(K):
            o, b = d["obs"][t], d["B"][j][d["obs"][t]]
            cut_d = delta[:t] + [[x if i <= j else 0.0 for i, x in enumerate(delta[t])]]
            cut_p = psi[:t] + [[x if i <= j else -1 for i, x in enumerate(psi[t])]] + [[-1] * K] * (n - t - 1)
            state = {"delta": grid(cut_d, n), "psi": grid(cut_p, n)}
            if not t:
                if not j:
                    state.update({"obs": vec(d["obs"]), "path": vec(none), "naive": vec(g), "path0": vec(p0)})
                w.frame("init", state, step=0, sub=j, metrics={"t": t, "j": j, "o": o, "pi": d["pi"][j], "b": b, "delta": delta[t][j]})
            else:
                c = [delta[t - 1][i] * d["A"][i][j] for i in range(K)]
                w.frame("cell", state, step=t, sub=j,
                        metrics={"t": t, "j": j, "o": o, **two("p", delta[t - 1]), **two("a", [d["A"][i][j] for i in range(K)]), **two("c", c),
                                 "from": psi[t][j], "best": c[psi[t][j]], "b": b, "delta": delta[t][j]})
    last = argmax(delta[n - 1])
    p_best = delta[n - 1][last]
    w.frame("end", {}, step=n - 1, metrics={"n": n, "last": last, "pBest": p_best, "evidence": ev, "share": p_best / ev, **two("d", delta[n - 1])})
    path = list(none)
    for t in range(n - 1, -1, -1):
        path[t] = vpath[t]
        w.frame("back", {"path": vec(path)}, step=t, metrics={"t": t, "s": path[t], "from": psi[t][path[t]] if t else -1})
    w.frame("done", {}, step=n - 1,
            metrics={"n": n, "pBest": p_best, "share": p_best / ev, "diff": sum(1 for t in range(n) if vpath[t] != g[t]), "firstDiff": first_diff(vpath, g),
                     "changed": sum(1 for t in range(n) if vpath[t] != p0[t]), "firstChanged": first_diff(vpath, p0), "rainy": sum(1 for s in vpath if s == 1)})


def draw(row, u):
    """the index whose interval [lo, hi) of the cumulative row holds u (the last interval ends at 1)"""
    lo = 0.0
    for i in range(len(row)):
        hi = 1.0 if i == len(row) - 1 else lo + row[i]
        if u < hi:
            return i, lo, hi
        lo = hi
    return len(row) - 1, lo, 1.0


def run_sample(w, tb, rng, opt):
    n = max(1, int(opt.get("days", 12)))
    wt, ac = [-1] * n, [-1] * n
    for t in range(n):
        u = rng.next()
        s, lo, hi = draw(tb["A"][wt[t - 1]] if t else tb["pi"], u)
        v = rng.next()
        a, alo, ahi = draw(tb["B"][s], v)
        wt[t], ac[t] = s, a
        w.frame("day", {"weather": vec(wt), "acts": vec(ac)}, step=t,
                metrics={"t": t, "u": u, "s": s, "lo": lo, "hi": hi, "v": v, "act": a, "alo": alo, "ahi": ahi, "prev": wt[t - 1] if t else -1})
    stays, run, longest = 0, 1, 1
    for t in range(1, n):
        same = wt[t] == wt[t - 1]
        stays += 1 if same else 0
        run = run + 1 if same else 1
        longest = max(longest, run)
    w.frame("done", {}, step=n - 1, metrics={"n": n, "stays": stays, "switches": n - 1 - stays, "rainy": sum(1 for s in wt if s == 1), "longest": longest})


def run_spells(w, inp):
    y, L = inp["year"], int(inp["maxLen"])
    n = len(y)
    c = np.zeros((2, 2), dtype=np.int64)
    for t in range(1, n):
        c[y[t - 1], y[t]] += 1
    lens = []
    for t in range(n):
        if y[t] != 1 or (t and y[t - 1] == 1):
            continue
        e = t
        while e < n and y[e] == 1:
            e += 1
        lens.append(e - t)
    a_ss = int(c[0, 0]) / int(c[0, 0] + c[0, 1])
    a_rr = int(c[1, 1]) / int(c[1, 0] + c[1, 1])
    w.frame("count", {"year": vec(y), "model": np.zeros(L), "truth": np.zeros(L)}, step=0,
            metrics={"n": n, "nSS": int(c[0, 0]), "nSR": int(c[0, 1]), "nRS": int(c[1, 0]), "nRR": int(c[1, 1]), "aSS": a_ss, "aRR": a_rr,
                     "spells": len(lens), "mean": total([float(x) for x in lens]) / len(lens)})
    model = [a_rr ** k * (1 - a_rr) for k in range(L)]
    truth = [sum(1 for x in lens if x == k + 1) / len(lens) for k in range(L)]
    w.frame("fit", {"model": vec(model), "truth": vec(truth)}, step=0,
            metrics={"aRR": a_rr, "meanModel": 1 / (1 - a_rr), **{f"p{k + 1}": p for k, p in enumerate(model)}, **{f"q{k + 1}": p for k, p in enumerate(truth)},
                     "mode": argmax(model) + 1, "modeTrue": argmax(truth) + 1, "tail": 1 - total(model)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="forward", choices=list(PHASES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran")
    ap.add_argument("--draws", default=None, help="unused: ak.Rng repeats the TS draws exactly")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (hmm.ts dataOf, or its tables, or the stretch of weather)")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/hmm/<process>.ref)")
    a = ap.parse_args()
    data, opt = ak.json_arg(a.input), json.loads(a.opt)
    w = ak.TapeWriter(a.process, SCHEMAS[a.process], seed=a.seed, opt=opt, phases=phases(a.process), provenance="live")
    if a.process == "forward":
        run_forward(w, data)
    elif a.process == "viterbi":
        run_viterbi(w, data)
    elif a.process == "sample":
        run_sample(w, data, ak.Rng(a.seed), opt)
    else:
        run_spells(w, data)
    jpath, _ = w.write(a.out or ak.site_path(".atlas", "hmm", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(w.frames)} frames)")
