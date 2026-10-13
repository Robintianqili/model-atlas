"""tiny-tcn: the Python reference of process 'arima' (kind C parity, design §4.11).

An independent run of explainers/tiny-tcn/arima.ts and process.ts in plain Python floats, frame by frame: the series and
its changes, the ARIMA(p, 1, q)-lite fit (Hannan-Rissanen: a long AR(8) by least squares for the errors, then least
squares for every p <= 5, q <= 2 with at most N/3 coefficients, unstable ones dropped, the smallest AIC kept), and the
forecast of the held-out part. The loops add in the same order as the TS, so the numbers agree to the last bits.
The TS source passes the series as plain numbers (--input), so both sides read the same values.

CLI contract (every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import math
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

RF = 16
MAX_HOLD = 24
MIN_POS = 16
P_MAX, Q_MAX, M_LONG, RIDGE = 5, 2, 8, 1e-8
T = lambda en, zh: {"en": en, "zh": zh}  # noqa: E731
SCHEMA = {
    "series": {"kind": "tensor", "label": T("The series", "序列"), "role": "data", "tol": 1e-12},
    "changes": {"kind": "tensor", "label": T("The changes", "变化量"), "role": "data", "tol": 1e-12},
    "coef": {"kind": "tensor", "label": T("μ, φ…, θ…", "μ、φ…、θ…"), "role": "param", "tol": 1e-9},
    "path": {"kind": "tensor", "label": T("ARIMA forecast", "ARIMA 的预测"), "role": "output", "tol": 1e-9},
}
PHASES = {
    "series": {"label": T("The series and its changes", "序列和它的变化量"), "kind": "input"},
    "fit": {"label": T("Fit the coefficients", "拟合系数"), "kind": "update"},
    "forecast": {"label": T("Forecast the held-out part", "预测留出的部分"), "kind": "output"},
}


def data_of(values):
    """data.ts dataOf() for a list of numbers (sine(96) when too short)."""
    y = [float(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0 for v in values]
    n = len(y)
    if n - min(MAX_HOLD, n // 4) - 1 - RF < MIN_POS:
        y = [math.sin((2 * math.pi * i) / 24 + 0) for i in range(96)]
        n = 96
    hold = min(MAX_HOLD, n // 4)
    c = [0.0 if t == 0 else y[t] - y[t - 1] for t in range(n)]
    return y, c, n, hold, n - hold


def lstsq(rows, y):
    k = len(rows[0])
    A = [[RIDGE if j == i else 0.0 for j in range(k + 1)] for i in range(k)]
    for r in range(len(rows)):
        for i in range(k):
            for j in range(k):
                A[i][j] += rows[r][i] * rows[r][j]
            A[i][k] += rows[r][i] * y[r]
    for i in range(k):
        p = i
        for r in range(i + 1, k):
            if abs(A[r][i]) > abs(A[p][i]):
                p = r
        A[i], A[p] = A[p], A[i]
        for r in range(i + 1, k):
            f = A[r][i] / A[i][i]
            for j in range(i, k + 1):
                A[r][j] -= f * A[i][j]
    x = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = A[i][k]
        for j in range(i + 1, k):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]
    return x


def stable(a):
    c = list(a)
    for k in range(len(c), 0, -1):
        kap = c[k - 1]
        if not abs(kap) <= 1 + 1e-9:
            return False
        if abs(kap) >= 1 - 1e-12:
            return True
        c = [(c[j] + kap * c[k - 2 - j]) / (1 - kap * kap) for j in range(k - 1)]
    return True


def lag_row(z, e, t, p, q):
    return [1.0] + [z[t - i] for i in range(1, p + 1)] + [e[t - j] for j in range(1, q + 1)]


def dot(row, b):
    s = 0.0
    for i, x in enumerate(row):
        s += x * b[i]
    return s


def fit_arima(z):
    long = lstsq([lag_row(z, [], t, M_LONG, 0) for t in range(M_LONG, len(z))], z[M_LONG:])
    e = [0.0 if t < M_LONG else v - dot(lag_row(z, [], t, M_LONG, 0), long) for t, v in enumerate(z)]
    t0 = M_LONG + Q_MAX
    ys = z[t0:]
    N = len(ys)
    best = None
    for p in range(P_MAX + 1):
        for q in range(Q_MAX + 1):
            if 3 * (p + q + 1) > N:
                continue
            rows = [lag_row(z, e, r + t0, p, q) for r in range(N)]
            b = lstsq(rows, ys)
            rss = 0.0
            for r, row in enumerate(rows):
                f = dot(row, b)
                rss += (ys[r] - f) * (ys[r] - f)
            sigma2 = rss / N
            aic = N * math.log(sigma2 + 1e-12) + 2 * (p + q + 1)
            if not stable(b[1:1 + p]) or not stable([-v for v in b[1 + p:]]):
                continue
            if best is None or aic < best["aic"]:
                best = {"p": p, "q": q, "mu": b[0], "phi": b[1:1 + p], "theta": b[1 + p:], "sigma2": sigma2, "aic": aic, "N": N}
    return best


def forecast_arima(f, z, last, hold):
    zz = list(z)
    e = []

    def at(a, i):
        return a[i] if 0 <= i < len(a) else 0.0

    def predict(t):
        s1 = 0.0
        for i, ph in enumerate(f["phi"]):
            s1 += ph * at(zz, t - 1 - i)
        s2 = 0.0
        for j, th in enumerate(f["theta"]):
            s2 += th * at(e, t - 1 - j)
        return f["mu"] + s1 + s2

    for t in range(len(z)):
        e.append(z[t] - predict(t))
    path = []
    y = last
    for _ in range(hold):
        v = predict(len(zz))
        zz.append(v)
        e.append(0.0)
        y += v
        path.append(y)
    return path


def run(process: str, raw, seed: int, opt: dict) -> ak.TapeWriter:
    if process != "arima":
        raise SystemExit(f"reference.py: unknown process {process!r} (arima)")
    y, c, n, hold, split = data_of(raw if isinstance(raw, list) else [])
    w = ak.TapeWriter("arima", SCHEMA, seed=seed, opt=opt, phases=PHASES, provenance="live")
    w.frame("series", {"series": np.array(y), "changes": np.array(c)}, metrics={"n": n, "hold": hold, "split": split, "last": c[split - 1]})
    z = c[1:split]
    f = fit_arima(z)
    w.frame("fit", {"coef": np.array([f["mu"], *f["phi"], *f["theta"]])},
            metrics={"p": f["p"], "q": f["q"], "mu": f["mu"], "phi1": f["phi"][0] if f["phi"] else 0.0, "sigma": math.sqrt(f["sigma2"]),
                     "N": f["N"], "pMax": P_MAX, "qMax": Q_MAX, "k": f["p"] + f["q"] + 1})
    path = forecast_arima(f, z, y[split - 1], hold)
    s = 0.0
    for h, v in enumerate(path):
        s += abs(v - y[split + h])
    w.frame("forecast", {"path": np.array(path)}, metrics={"mae": s / hold, "hold": hold, "first": path[0] if path else float("nan")})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="arima")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tape = run(a.process, ak.json_arg(a.input), a.seed, json.loads(a.opt))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "tiny-tcn", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
