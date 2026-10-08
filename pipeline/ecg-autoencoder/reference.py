"""ecg-autoencoder: Python reference for process 'sweep' (kind C parity, design §4.11).

An independent implementation of the 'sweep' process in site/src/lib/explainers/ecg-autoencoder/process.ts, in plain
Python floats: the threshold slides down over the test set's anomaly scores (testset.json, written by export.py), one
beat at a time from the highest score (ties: the smaller index first). Each step flags one more beat, counted as caught
(an odd beat) or as a false alarm (a normal beat); the counts give one point of the ROC curve (false-alarm rate, catch
rate), and the area under the curve is added up by the trapezoid rule. `npm run atlas -- parity ecg-autoencoder` records
the TS tape, runs this file with the same input and compares the tapes frame by frame (AK204). It draws nothing.

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

SCHEMA = {"point": {"kind": "tensor", "label": ("Where the curve is: false-alarm rate, catch rate", "曲线当前的位置：误报率、检出率"), "tol": 1e-12}}
PHASES = {
    "start": {"label": ("Threshold above every score: nothing flagged", "阈值高于所有分数：什么都不标记"), "kind": "input"},
    "flag": {"label": ("Lower the threshold to the next score: one more beat flagged", "把阈值降到下一个分数：多标记一个心跳"), "kind": "other"},
    "area": {"label": ("The area under the curve", "曲线下的面积"), "kind": "output"},
}


def pt(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float64)


def run_sweep(seed: int, opt: dict, d: dict | None) -> ak.TapeWriter:
    w = ak.TapeWriter("sweep", SCHEMA, seed=seed, opt=opt, phases=PHASES, provenance="live")
    if not d or not d.get("scores"):
        return w
    scores, kinds = [float(s) for s in d["scores"]], [int(k) for k in d["kinds"]]
    n = len(scores)
    odd = [1 if k > 0 else 0 for k in kinds]
    n_pos = sum(odd)
    n_neg = n - n_pos
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    w.frame("start", {"point": pt(0.0, 0.0)}, step=0, metrics={"n": n, "nPos": n_pos, "nNeg": n_neg, "tp": 0, "fp": 0})
    tp = fp = 0
    auc = x0 = y0 = 0.0
    for k, i in enumerate(order):
        if odd[i]:
            tp += 1
        else:
            fp += 1
        x = fp / n_neg if n_neg else 0.0
        y = tp / n_pos if n_pos else 0.0
        auc += 0.5 * (y0 + y) * (x - x0)
        x0, y0 = x, y
        w.frame("flag", {"point": pt(x, y)}, step=k + 1,
                metrics={"k": k + 1, "beat": i, "kind": kinds[i], "thr": scores[i], "tp": tp, "fp": fp, "fpr": x, "tpr": y,
                         "precision": tp / (tp + fp), "recall": y})
    w.frame("area", {}, step=n + 1, metrics={"auc": auc, "nPos": n_pos, "nNeg": n_neg})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="sweep", choices=["sweep"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran (none are used)")
    ap.add_argument("--draws", default=None, help="unused: the process draws nothing")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (scores and kinds of testset.json), or @file")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/ecg-autoencoder/sweep.ref)")
    a = ap.parse_args()
    tape = run_sweep(a.seed, json.loads(a.opt), ak.json_arg(a.input))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "ecg-autoencoder", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
