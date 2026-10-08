"""kalman-filter: Python reference for process 'kf' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/kalman-filter/process.ts (and the filter in track.ts) in plain
Python floats: a Kalman filter with a constant-velocity motion model (F = [[1, 1], [0, 1]], H = [1, 0],
Q = q [[1/4, 1/2], [1/2, 1]]) follows the sensor readings of a cart on a rail. Frames: start, then per second
predict -> measure -> gain -> update, then done (the run scored against the truth, and the best q and r on the
sliders' grid). `npm run atlas -- parity kalman-filter` records the TS tape of every process source, runs this file
with the same input (the plain numbers of track.ts dataOf(): true positions, readings, q, r, w) and compares the tapes
frame by frame: phase and step exactly, every stored field within its tol, every metric within 1e-12 (AK204).
The readings' noise is drawn in TS (dataOf, seeded) and arrives here as numbers: this process draws nothing.

    npm run atlas -- parity kalman-filter --fixtures     # the usual way (it passes the arguments below)

CLI contract: --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
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

# the sliders' grid (track.ts QS, RS) and the velocity doubt of the first guess (V0)
QS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
RS = [0.1, 0.25, 0.5, 1, 2, 4, 8]
V0 = 1


def lab(en, zh):
    return (en, zh)


SCHEMA = {
    "truth": {"kind": "tensor", "label": lab("True position each second (the filter never reads it)", "每秒的真实位置（滤波器从不读取）"), "role": "target", "tol": "exact"},
    "z": {"kind": "tensor", "label": lab("Sensor reading each second", "每秒的传感器读数"), "role": "data", "tol": "exact"},
    "ma": {"kind": "tensor", "label": lab("Moving average of the last readings", "最近几个读数的移动平均"), "tol": 1e-12},
    "x": {"kind": "tensor", "label": lab("The estimate: position (m) and velocity (m/s)", "估计：位置（米）和速度（米/秒）"), "tol": 1e-12},
    "P": {"kind": "tensor", "label": lab("Its covariance P (the doubt)", "它的协方差 P（不确定程度）"), "tol": 1e-12},
}
PHASES = {
    "start": {"label": lab("The first reading is the first guess", "第一个读数就是第一个猜测"), "kind": "input"},
    "predict": {"label": lab("Predict: move on with the estimated speed", "预测：按估计的速度往前走"), "kind": "forward"},
    "measure": {"label": lab("A reading arrives", "读数到了"), "kind": "observe"},
    "gain": {"label": lab("The Kalman gain: how much to trust the reading", "卡尔曼增益：读数信几分"), "kind": "other"},
    "update": {"label": lab("Update: move part of the way to the reading", "更新：朝读数挪一部分"), "kind": "update"},
    "done": {"label": lab("The whole track, scored", "整条轨迹的成绩"), "kind": "output"},
}


# ------------------------------------------------------------------ the filter (one belief = p, v, P00, P01, P11)
def kf_start(z0, r):
    return {"p": z0, "v": 0.0, "P00": r, "P01": 0.0, "P11": V0}


def kf_predict(b, q):
    """x <- F x;  P <- F P F^T + Q  with F = [[1, 1], [0, 1]], Q = q [[1/4, 1/2], [1/2, 1]]"""
    return {"p": b["p"] + b["v"], "v": b["v"], "P00": b["P00"] + 2 * b["P01"] + b["P11"] + q / 4,
            "P01": b["P01"] + b["P11"] + q / 2, "P11": b["P11"] + q}


def kf_update(b, z, r):
    """H = [1, 0]: S = P00 + r, K = P H^T / S, x <- x + K y, P <- (I - K H) P"""
    y = z - b["p"]
    S = b["P00"] + r
    K = b["P00"] / S
    Kv = b["P01"] / S
    nb = {"p": b["p"] + K * y, "v": b["v"] + Kv * y, "P00": (1 - K) * b["P00"], "P01": (1 - K) * b["P01"], "P11": b["P11"] - Kv * b["P01"]}
    return nb, y, S, K, Kv


def kf_run(z, q, r):
    b = kf_start(z[0], r)
    est = [b["p"]]
    for k in range(1, len(z)):
        b = kf_update(kf_predict(b, q), z[k], r)[0]
        est.append(b["p"])
    return est


def moving_average(z, w):
    out = []
    for k in range(len(z)):
        s = 0.0
        lo = max(0, k - w + 1)
        for j in range(lo, k + 1):
            s += z[j]
        out.append(s / (k - lo + 1))
    return out


def rmse(a, b):
    s = 0.0
    for k in range(len(a)):
        s += (a[k] - b[k]) * (a[k] - b[k])
    return math.sqrt(s / len(a))


def vec(*v):
    return np.array(v, dtype=np.float64)


def cov(b):
    return np.array([[b["P00"], b["P01"]], [b["P01"], b["P11"]]], dtype=np.float64)


# ------------------------------------------------------------------ the process
def run_kf(seed, opt, d):
    w = ak.TapeWriter("kf", SCHEMA, seed=seed, opt=opt, phases=PHASES, provenance="live")
    z, truth, q, r, win = d["z"], d["truth"], d["q"], d["r"], d["w"]
    n = len(z)
    if not n:
        return w
    ma = moving_average(z, win)
    b = kf_start(z[0], r)
    w.frame("start", {"truth": vec(*truth), "z": vec(*z), "ma": vec(*ma), "x": vec(b["p"], b["v"]), "P": cov(b)}, step=0,
            metrics={"n": n, "k": 0, "z0": z[0], "p": b["p"], "v": b["v"], "P00": b["P00"], "P11": b["P11"], "q": q, "r": r, "w": win, "rmseZ": rmse(z, truth)})
    est, sd = [b["p"]], [math.sqrt(b["P00"])]
    for k in range(1, n):
        bp = kf_predict(b, q)
        w.frame("predict", {"x": vec(bp["p"], bp["v"]), "P": cov(bp)}, step=k,
                metrics={"k": k, "p": b["p"], "v": b["v"], "P00": b["P00"], "P01": b["P01"], "P11": b["P11"], "pp": bp["p"], "vp": bp["v"],
                         "Pp00": bp["P00"], "Pp01": bp["P01"], "Pp11": bp["P11"], "q": q, "sd": math.sqrt(b["P00"]), "sdp": math.sqrt(bp["P00"])})
        nb, y, S, K, Kv = kf_update(bp, z[k], r)
        w.frame("measure", {}, step=k, metrics={"k": k, "pp": bp["p"], "z": z[k], "y": y, "r": r, "Pp00": bp["P00"], "sdp": math.sqrt(bp["P00"]),
                                                 "sdz": math.sqrt(r), "truth": truth[k]})
        w.frame("gain", {}, step=k, metrics={"k": k, "Pp00": bp["P00"], "r": r, "S": S, "K": K, "Kv": Kv, "pp": bp["p"], "z": z[k]})
        b = nb
        est.append(b["p"])
        sd.append(math.sqrt(b["P00"]))
        err = b["p"] - truth[k]
        w.frame("update", {"x": vec(b["p"], b["v"]), "P": cov(b)}, step=k,
                metrics={"k": k, "pp": bp["p"], "vp": bp["v"], "z": z[k], "y": y, "K": K, "Kv": Kv, "p": b["p"], "v": b["v"], "Pp00": bp["P00"],
                         "P00": b["P00"], "P01": b["P01"], "P11": b["P11"], "sdp": math.sqrt(bp["P00"]), "sd": math.sqrt(b["P00"]), "sdz": math.sqrt(r),
                         "truth": truth[k], "err": err, "errZ": z[k] - truth[k], "errMa": ma[k] - truth[k],
                         "out": 1 if abs(err) > 2 * math.sqrt(b["P00"]) else 0})
    worst, worst_k, cover = 0.0, 0, 0
    for k in range(n):
        e = abs(est[k] - truth[k])
        if e > worst:
            worst, worst_k = e, k
        if e <= 2 * sd[k]:
            cover += 1
    best = (math.inf, QS[0], RS[0])
    for qq in QS:
        for rr in RS:
            e = rmse(kf_run(z, qq, rr), truth)
            if e < best[0]:
                best = (e, qq, rr)
    w.frame("done", {}, step=n - 1, metrics={"n": n, "q": q, "r": r, "w": win, "rmseKf": rmse(est, truth), "rmseZ": rmse(z, truth), "rmseMa": rmse(ma, truth),
                                             "worst": worst, "worstK": worst_k, "cover": cover, "bestQ": best[1], "bestR": best[2], "bestRmse": best[0]})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="kf", choices=["kf"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran (none are used)")
    ap.add_argument("--draws", default=None, help="unused: the process draws nothing (the noise is in the input)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (track.ts dataOf: truth, z, q, r, w)")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/kalman-filter/kf.ref)")
    a = ap.parse_args()
    data = json.loads(a.input)
    if not isinstance(data, dict) or "z" not in data:
        sys.exit("reference: --input must be the plain numbers of track.ts dataOf() (the sources pass them)")
    tape = run_kf(a.seed, json.loads(a.opt), data)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "kalman-filter", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
