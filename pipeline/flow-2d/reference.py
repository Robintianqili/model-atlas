"""flow-2d: the Python reference for process 'eulerCheck' (kind C parity): "Euler frames equal the twin".

It repeats the TS run of site/src/lib/explainers/flow-2d/process.ts in PyTorch float64, independently of the TS code:
the twin network (twin.py build()) starts from the TS initial weights, trains with torch.optim.Adam on the same pairs,
then samples with Euler's method. Randomness is replayed from the TS draws (--draws), in the order the TS run drew it:
  init      8,706 uniform draws u: every Linear's weight (row-major) then bias, (2u - 1) / sqrt(fan_in)
  step k    256 pairs, each: a data index (int), noise x0, y0 (normal, normal), a time t (next)
  noise     2 x particles normal draws (x, y per particle), the same noise for the K-step and the Kfew-step runs
Frames (the same fields and metrics as the TS tape):
  init; then for k = 0..steps a 'snapshot' (every snapshotEvery steps and at the end, with params) or a 'step' frame
  (loss, lr, gradNorm before the update); then 'noise' + K 'move' frames (x <- x + h v(x, t), h = 1/K) and 'noiseFew' +
  Kfew 'moveFew' frames; metric dist (mean distance to the nearest data point) on the noise and last frames.

    npm run atlas -- parity flow-2d    # records the TS tape and its draws, runs this file, compares frame by frame

CLI contract (every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
"""
from __future__ import annotations

import argparse
import importlib.util
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

HERE = Path(__file__).resolve().parent
SCHEMA = {
    "step": {"kind": "int", "label": ("Step", "步数")},
    "loss": {"kind": "scalar", "label": ("Loss", "损失")},
    "lr": {"kind": "scalar", "label": ("Learning rate", "学习率")},
    "optimizer": {"kind": "text", "label": ("Optimizer", "优化器")},
    "params": {"kind": "tensor", "label": ("Weights", "权重")},
    "x": {"kind": "tensor", "label": ("Sampled points", "采样的点"), "tol": 1e-9},
}
PHASES = {k: {"label": (k, k), "kind": kind} for k, kind in
          [("init", "other"), ("step", "update"), ("snapshot", "observe"), ("noise", "input"), ("move", "update"),
           ("noiseFew", "input"), ("moveFew", "update")]}
DEFAULTS = {"seed": 1, "lr": 1e-3, "batch": 256, "steps": 1500, "snapshotEvery": 100, "optimizer": "adam",
            "particles": 400, "K": 50, "Kfew": 4, "sample": True}


def load_twin():
    spec = importlib.util.spec_from_file_location("flow2d_twin", HERE / "twin.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class Draws:
    """The TS draws of every frame, consumed in order."""

    def __init__(self, frames: list[list[float]]):
        self.flat = [v for f in frames for v in f]
        self.i = 0

    def take(self) -> float:
        v = self.flat[self.i]
        self.i += 1
        return v


def mean_nearest(x: np.ndarray, data: np.ndarray) -> float:
    total = 0.0
    for i in range(x.shape[0]):
        d = (x[i, 0] - data[:, 0]) ** 2 + (x[i, 1] - data[:, 1]) ** 2
        total += math.sqrt(float(d.min()))
    return total / x.shape[0]


def params_of(model) -> dict:
    return {k: v.detach().clone().numpy() for k, v in model.state_dict().items()}


def run(opt: dict, seed: int, draws: Draws, data: np.ndarray) -> "ak.TapeWriter":
    torch.set_default_dtype(torch.float64)
    w = ak.TapeWriter("eulerCheck", SCHEMA, seed=seed, opt=opt, phases=PHASES, provenance="live")
    model = load_twin().build(opt).double()
    with torch.no_grad():  # the TS init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)), weight then bias, layer by layer
        for m in model.net:
            if isinstance(m, torch.nn.Linear):
                bound = 1 / math.sqrt(m.in_features)
                for p in (m.weight, m.bias):
                    flat = p.view(-1)
                    for i in range(flat.numel()):
                        flat[i] = (2 * draws.take() - 1) * bound
    optim = default_optimizer(model.parameters(), opt)
    lr = float(opt["lr"])
    w.frame("init", {"step": 0, "params": params_of(model), "lr": lr, "optimizer": str(opt["optimizer"])}, metrics={"step": 0, "lr": lr})
    n = data.shape[0]
    B, steps, every = int(opt["batch"]), int(opt["steps"]), max(1, int(opt["snapshotEvery"]))
    for k in range(steps + 1):
        inp = np.zeros((B, 3))
        tgt = np.zeros((B, 2))
        for i in range(B):
            j = int(draws.take())
            x0, y0, t = draws.take(), draws.take(), draws.take()
            x1, y1 = data[j % n]
            inp[i] = ((1 - t) * x0 + t * x1, (1 - t) * y0 + t * y1, t)
            tgt[i] = (x1 - x0, y1 - y0)
        optim.zero_grad(set_to_none=False)
        loss = torch.mean((model(torch.from_numpy(inp)) - torch.from_numpy(tgt)) ** 2)
        last = k >= steps
        metrics = {"step": k, "loss": float(loss.item()), "lr": lr}
        if not last:
            loss.backward()
            metrics["gradNorm"] = math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.parameters()))
        state = {"step": k, "loss": float(loss.item())}
        snap = last or k % every == 0
        if snap:
            state["params"] = params_of(model)
        w.frame("snapshot" if snap else "step", state, step=k, metrics=metrics)
        if last:
            break
        optim.step()
    if not opt.get("sample", True):
        return w
    N = max(1, int(opt["particles"]))
    noise = np.array([draws.take() for _ in range(2 * N)]).reshape(N, 2)
    for K, (start, move) in ((int(opt["K"]), ("noise", "move")), (int(opt["Kfew"]), ("noiseFew", "moveFew"))):
        K = max(1, K)
        h = 1 / K
        x = torch.from_numpy(noise.copy())
        w.frame(start, {"x": x.numpy().copy()}, step=0, metrics={"t": 0, "h": h, "K": K, "dist": mean_nearest(x.numpy(), data)})
        with torch.no_grad():
            for k in range(K):
                t = torch.full((N, 1), k * h)
                x = x + h * model(torch.cat([x, t], 1))
                m = {"t": (k + 1) * h, "h": h}
                if k == K - 1:
                    m["dist"] = mean_nearest(x.numpy(), data)
                w.frame(move, {"x": x.numpy().copy()}, step=k, metrics=m)
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="eulerCheck")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options the TS source ran with")
    ap.add_argument("--draws", required=True, help="draws.json from the TS run (replayed)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run: {\"xy\": [x0, y0, x1, y1, ...]}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/flow-2d/eulerCheck.ref)")
    a = ap.parse_args()
    opt = {**DEFAULTS, **json.loads(a.opt)}
    xy = (json.loads(a.input) or {}).get("xy") or []
    data = np.array(xy, dtype=np.float64).reshape(-1, 2)
    if data.shape[0] == 0:  # the TS run records no frames without points
        tape = ak.TapeWriter("eulerCheck", SCHEMA, seed=a.seed, opt=opt, phases=PHASES, provenance="live")
    else:
        frames = json.loads(Path(a.draws).read_text(encoding="utf-8"))["frames"]
        tape = run(opt, a.seed, Draws(frames), data)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "flow-2d", "eulerCheck.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
