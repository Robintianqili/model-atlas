"""tinyforecaster: the Python reference of the three processes (kind C parity, design §4.11).

An independent run of explainers/tinyforecaster/process.ts in Python, frame by frame:
  windows   the series (with the held-out part), then every training window and its target
  forecast  trainer 'lstm' (the PyTorch float64 twin, twin.py; Adam from atlas_kit.train_reference) with the same
            ak.Rng streams as the TS trainer (init weights from fork('init'), batches from fork('batch')), then the
            rollout over the held-out part (process/library/rollout.ts frames) and the seasonal-naive baseline
  walks     the same training on each of the given random walks, one frame per walk
The TS sources pass every series as plain numbers (--input), so both sides read the same values.
`npm run atlas -- parity tinyforecaster` compares the tapes: phase/step/sub exactly, stored fields within their tol,
metrics within 1e-12. 200 Adam steps drift from TS by up to about 1e-10, so the TS schema compares every stored value
computed from trained weights (loss, params, pred, err, path, one) at 1e-9; for the same reason this reference writes
only the metrics that do not depend on the trained weights (step, lr, the data, the baseline and the counts): the others
(loss, gradNorm, mse, mae, lstmMae, early, late, one, fit and their averages) follow from the stored fields it checks.

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
import torch

from atlas_kit.train_reference import default_optimizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin import Forecaster  # noqa: E402

L = 24
MAX_HOLD = 24
MIN_WINDOWS = 4
NAN = float("nan")
T = lambda en, zh: {"en": en, "zh": zh}  # noqa: E731

DEFAULTS = {"seed": 1, "lr": 0.01, "batch": 16, "steps": 200, "snapshotEvery": 10, "optimizer": "adam"}
TRAIN_SCHEMA = {
    "step": {"kind": "int", "label": T("Step", "步数")},
    "loss": {"kind": "scalar", "label": T("Loss", "损失"), "role": "output"},
    "lr": {"kind": "scalar", "label": T("Learning rate", "学习率"), "role": "optimizer"},
    "optimizer": {"kind": "text", "label": T("Optimizer", "优化器"), "role": "optimizer"},
    "params": {"kind": "tensor", "label": T("Weights", "权重"), "role": "param"},
}
ROLL_SCHEMA = {
    "window": {"kind": "tensor", "label": T("Values the model reads", "模型读到的数值"), "tol": 1e-12},
    "pred": {"kind": "scalar", "label": T("Prediction", "预测"), "tol": 1e-9},
    "truth": {"kind": "scalar", "label": T("True value", "真实值"), "tol": "exact"},
    "err": {"kind": "scalar", "label": T("Error", "误差"), "tol": 1e-9},
    "path": {"kind": "tensor", "label": T("Predictions so far", "目前的预测"), "track": "changes", "tol": 1e-9},
    "h": {"kind": "int", "label": T("Steps ahead", "向前几步"), "tol": "exact"},
}
SCHEMAS = {
    "windows": {
        "series": {"kind": "tensor", "label": T("The series", "序列"), "tol": 1e-12},
        "window": {"kind": "tensor", "label": T("The window (model input)", "窗口（模型输入）"), "tol": 1e-12},
        "target": {"kind": "tensor", "label": T("The next values (target)", "接下来的值（目标）"), "tol": 1e-12},
        "start": {"kind": "int", "label": T("Where the window starts", "窗口的起点"), "tol": "exact"},
    },
    "forecast": {**TRAIN_SCHEMA, **ROLL_SCHEMA,
                 "naive": {"kind": "tensor", "label": T("Seasonal-naive forecast", "季节性朴素预测"), "tol": 1e-12},
                 "one": {"kind": "tensor", "label": T("One-step forecasts", "单步预测"), "tol": 1e-12}},
    "walks": {"series": {"kind": "tensor", "label": T("The walk", "随机游走"), "tol": 1e-12},
              "one": {"kind": "tensor", "label": T("One-step forecasts", "单步预测"), "tol": 1e-12}},
}
PHASES = {
    "windows": {"series": {"label": T("The whole series", "整条序列"), "kind": "input"},
                "window": {"label": T("One window and its target", "一个窗口和它的目标"), "kind": "observe"}},
    "forecast": {"init": {"label": T("Start", "开始"), "kind": "other"},
                 "step": {"label": T("Training step", "训练步"), "kind": "update"},
                 "snapshot": {"label": T("Snapshot", "快照"), "kind": "observe"},
                 "start": {"label": T("Forecast from the split", "从分界处开始预测"), "kind": "input"},
                 "predict": {"label": T("Predict, append, repeat", "预测、追加、重复"), "kind": "output"},
                 "naive": {"label": T("The seasonal-naive baseline", "季节性朴素基准"), "kind": "output"}},
    "walks": {"walk": {"label": T("One random walk", "一条随机游走"), "kind": "output"}},
}


class Data:
    """data.ts dataOf() for a list of numbers: the held-out part, the split, the training windows, the sine fallback."""

    def __init__(self, values):
        y = np.array([float(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0 for v in values], dtype=np.float64)
        self.given = len(y)
        if self._windows(len(y)) < MIN_WINDOWS:  # sine(96): amplitude 1, period 24
            y = np.array([1 * math.sin((2 * math.pi * i) / 24 + 0) for i in range(96)], dtype=np.float64)
        self.y = y
        self.n = len(y)
        self.hold = min(MAX_HOLD, self.n // 4)
        self.split = self.n - self.hold
        self.windows = self.split - L

    @staticmethod
    def _windows(n: int) -> int:
        return n - min(MAX_HOLD, n // 4) - L


def init_net(rng: ak.Rng) -> Forecaster:
    """nn.ts LSTMCell('lstm', 1, 16) then Linear('head', 16, 1): U(−bound, bound) row-major, bound 1/√16 for both."""
    m = Forecaster(16).double()
    b = 1 / math.sqrt(16)
    with torch.no_grad():
        for p in (m.lstm.weight_ih, m.lstm.weight_hh, m.lstm.bias_ih, m.lstm.bias_hh, m.head.weight, m.head.bias):
            vals = [(2 * rng.next() - 1) * b for _ in range(p.numel())]
            p.copy_(torch.tensor(vals, dtype=torch.float64).reshape(p.shape))
    return m


def named(m: Forecaster) -> dict:
    return {k: v.detach().clone().numpy() for k, v in m.state_dict().items()}


def forecast_rows(m: Forecaster, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0)
    with torch.no_grad():
        return m(torch.tensor(x, dtype=torch.float64)).reshape(-1).numpy()


def training_set(d: Data):
    x = np.stack([d.y[k:k + L] for k in range(d.windows)])
    return x, d.y[L:L + d.windows].copy()


def fit_of(m: Forecaster, d: Data) -> dict:
    x, y = training_set(d)
    e = forecast_rows(m, x) - y
    return {"mse": float(np.sum(e * e)) / len(y), "mae": float(np.sum(np.abs(e))) / len(y)}


def train(d: Data, seed: int, opt: dict, frame=None, hold: int = 0) -> Forecaster:
    """trainer.ts trainingProcess, coarse: init frame, then for k = 0..steps a 'step' or 'snapshot' frame."""
    root = ak.Rng(seed)
    m = init_net(root.fork("init"))
    optim = default_optimizer(list(m.parameters()), opt)
    rng = root.fork("batch")
    x_all, y_all = training_set(d)
    if frame:
        frame("init", {"step": 0, "params": named(m), "lr": opt["lr"], "optimizer": "adam", "path": np.full(hold, NAN)},
              metrics={"step": 0, "lr": opt["lr"]})
    steps, every = int(opt["steps"]), max(1, int(opt["snapshotEvery"]))
    for k in range(steps + 1):
        idx = [rng.int(d.windows) for _ in range(int(opt["batch"]))]
        xb = torch.tensor(x_all[idx], dtype=torch.float64)
        yb = torch.tensor(y_all[idx], dtype=torch.float64).reshape(-1, 1)
        optim.zero_grad()
        loss = torch.mean((m(xb) - yb) ** 2)
        lv = loss.item()
        last = k >= steps
        metrics = {"step": k, "lr": opt["lr"]}
        if not last:
            loss.backward()
        snap = last or k % every == 0
        state = {"step": k, "loss": lv}
        if snap:
            state["params"] = named(m)
        if frame:
            frame("snapshot" if snap else "step", state, step=k, metrics=metrics)
        if last:
            break
        optim.step()
    return m


def seasonal_naive(d: Data):
    m, fit = 1, math.inf
    for lag in range(1, L + 1):
        s = 0.0
        for t in range(L, d.split):
            s += abs(d.y[t] - d.y[t - lag])
        s /= d.split - L
        if s < fit - 1e-12:
            fit, m = s, lag
    path = np.array([d.y[d.split + k - m * (k // m + 1)] for k in range(d.hold)])
    e = 0.0
    for k in range(d.hold):
        e += abs(path[k] - d.y[d.split + k])
    return m, path, e / d.hold


def one_step(m: Forecaster, d: Data):
    def rows(a, b):
        return forecast_rows(m, np.stack([d.y[t - L:t] for t in range(a, b)]))

    def mae(p, a):
        s = 0.0
        for k, v in enumerate(p):
            s += abs(v - d.y[a + k])
        return s / len(p)

    def persist(a, b):
        s = 0.0
        for t in range(a, b):
            s += abs(d.y[t - 1] - d.y[t])
        return s / (b - a)

    path = rows(d.split, d.n)
    return path, {"one": mae(path, d.split), "persist": persist(d.split, d.n), "fit": mae(rows(L, d.split), L), "persistFit": persist(L, d.split)}


def run(process: str, raw, seed: int, opt: dict) -> "ak.TapeWriter":
    w = ak.TapeWriter(process, SCHEMAS[process], seed=seed, opt=opt, phases=PHASES[process], provenance="live")
    if process == "windows":
        d = Data(raw if isinstance(raw, list) else [])
        W, H = int(opt.get("L", L)), int(opt.get("H", 1))
        count = max(0, d.split - W - H + 1)
        w.frame("series", {"series": d.y.copy()}, metrics={"n": d.n, "hold": d.hold, "split": d.split, "count": count, "L": W, "H": H})
        for k in range(count):
            w.frame("window", {"window": d.y[k:k + W].copy(), "target": d.y[k + W:k + W + H].copy(), "start": k}, step=k,
                    metrics={"k": k, "start": k, "end": k + W - 1, "target": float(d.y[k + W]), "count": count, "L": W, "H": H})
        return w
    o = {**DEFAULTS, **opt}
    if process == "forecast":
        d = Data(raw if isinstance(raw, list) else [])
        m = train(d, seed, o, w.frame, d.hold)
        hist = list(d.y[:d.split])
        path = np.full(d.hold, NAN)
        w.frame("start", {"path": path.copy(), "h": 0}, metrics={"start": d.split, "horizon": d.hold})
        abs_sum, early, late, q = 0.0, 0.0, 0.0, max(1, d.hold // 3)
        abs_lib = 0.0
        for h in range(d.hold):
            pred = float(forecast_rows(m, np.array([hist[-L:]]))[0])
            truth = float(d.y[d.split + h])
            err = pred - truth
            path[h] = pred
            hist.append(pred)
            abs_lib += abs(err)
            w.frame("predict", {"pred": pred, "truth": truth, "err": err, "h": h + 1}, step=h,
                    changes=[{"field": "path", "index": [h], "from": NAN, "to": pred}],
                    metrics={"truth": truth})
            a = abs(err)
            abs_sum += a
            if h < q:
                early += a / q
            if h >= d.hold - q:
                late += a / q
        nm, npath, nmae = seasonal_naive(d)
        opath, omae = one_step(m, d)
        w.frame("naive", {"naive": npath, "one": opath},
                metrics={"m": nm, "naiveMae": nmae, "hold": d.hold, "q": q, "persist": omae["persist"], "persistFit": omae["persistFit"]})
        return w
    if process == "walks":
        walks = raw if isinstance(raw, list) else []
        total = {"fit": 0.0, "persistFit": 0.0, "one": 0.0, "persist": 0.0}
        wins = 0
        for k, values in enumerate(walks):
            d = Data(values)
            m = train(d, seed, {**o, "snapshotEvery": o["steps"]})  # as process.ts: no per-snapshot metrics
            opath, omae = one_step(m, d)
            for key in total:
                total[key] += omae[key]
            if omae["one"] < omae["persist"]:
                wins += 1
            n = k + 1
            w.frame("walk", {"series": d.y.copy(), "one": opath}, step=k, sub=n,
                    metrics={"persist": omae["persist"], "persistFit": omae["persistFit"], "walk": n, "walks": len(walks),
                             "split": d.split, "avgPersistFit": total["persistFit"] / n, "avgPersist": total["persist"] / n,
                             "wins": wins})
        return w
    raise SystemExit(f"reference.py: unknown process {process!r} (windows, forecast or walks)")


if __name__ == "__main__":
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="windows")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tape = run(a.process, json.loads(a.input), a.seed, json.loads(a.opt))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "tinyforecaster", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
