"""ddpm-unet: the Python reference for processes 'sample' and 'forward' (kind C parity): "the sampling tape equals a Python
loop with the same seed".

It repeats the TS runs of site/src/lib/explainers/ddpm-unet/process.ts independently of the TS code:
  sample   the U-Net of model_final.pt in PyTorch (CPU, float32, TF32 off), DDPM ancestral sampling in float64 numpy:
           x_T = the seed's 784 normal draws; then for each kept step t (T, T - stride, ..., stride):
           eps_hat = UNet(float32(x_t), t); alpha = abar_t / abar_next, beta = 1 - alpha;
           x <- (x - beta / sqrt(1 - abar_t) * eps_hat) / sqrt(alpha) + sqrt(beta) * z   (no z on the last step)
           The normal draws (x_T, then 784 z per step) are replayed from the TS run (--draws), in the order it drew them.
  forward  the input drawing (stretched to -1..1) noised at t = 0..T with the one-jump formula and the input noise (no draws).
Frames have the same phases, fields and metrics as the TS tape. The spec and weights in --input (the page's model) are
not read: the reference loads the PyTorch checkpoint export.py exported them from.

    npm run atlas -- parity ddpm-unet   # records the TS tapes and draws, runs this file, compares frame by frame

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CH, T, UNet, schedule  # noqa: E402

N, PROBE = 784, 14 * 28 + 14
PIC = {"kind": "tensor", "shape": [28, 28], "scale": "diverging", "role": "output"}
SCHEMA = {"x": {**PIC, "label": ("The picture x_t", "图 x_t"), "tol": 1e-3}, "eps": {**PIC, "label": ("Predicted noise", "预测的噪声 ε̂"), "tol": 1e-3}}
PHASES = {"noise": {"label": ("Start from pure noise", "从纯噪声出发"), "kind": "input"},
          "step": {"label": ("One denoising step", "去噪一步"), "kind": "update"},
          "noisy": {"label": ("Your drawing at step t", "第 t 步的画"), "kind": "forward"}}
DEFAULTS = {"stride": 1, "probe": PROBE, "n": 12}


class Draws:
    """The TS draws of every frame, consumed in order."""

    def __init__(self, frames: list):
        self.flat = [v for f in frames for v in (f or [])]
        self.i = 0

    def take(self, n: int) -> np.ndarray:
        v = np.array(self.flat[self.i:self.i + n], dtype=np.float64)
        self.i += n
        return v


def sample(seed: int, opt: dict, draws: Draws, learner_seed: int) -> "ak.TapeWriter":
    w = ak.TapeWriter("sample", SCHEMA, seed=seed, opt=opt, phases={k: PHASES[k] for k in ("noise", "step")}, provenance="live")
    unet = UNet(CH)
    unet.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    unet.eval()
    stride, p = max(1, int(opt["stride"])), int(opt["probe"])
    steps = [T - i * stride for i in range(T // stride)]
    ab = schedule()["abar"]
    x = draws.take(N)
    f32 = lambda a: a.astype(np.float32).reshape(28, 28)  # noqa: E731
    w.frame("noise", {"x": f32(x)}, step=0, metrics={"t": T, "seed": learner_seed, "K": len(steps)})
    for i, t in enumerate(steps):
        last = i == len(steps) - 1
        abt, prev = ab[t - 1], (1.0 if last else ab[steps[i + 1] - 1])
        a = abt / prev
        b = 1 - a
        coef, inv, sigma = b / math.sqrt(1 - abt), 1 / math.sqrt(a), (0.0 if last else math.sqrt(b))
        with torch.no_grad():
            eps = unet(torch.from_numpy(x.astype(np.float32)).view(1, 1, 28, 28), torch.tensor([float(t)])).view(-1).numpy()
        mean = (x - coef * eps.astype(np.float64)) * inv
        z = np.zeros(N) if last else draws.take(N)
        nxt = mean + sigma * z
        w.frame("step", {"x": f32(nxt), "eps": eps.reshape(28, 28).copy()}, step=i,
                metrics={"t": t, "tNext": 0 if last else steps[i + 1], "k": i + 1, "K": len(steps), "abar": abt, "alpha": a,
                         "beta": b, "coef": coef, "inv": inv, "sigma": sigma, "xt": float(x[p]), "eps": float(eps[p]),
                         "mean": float(mean[p]), "z": float(z[p]), "x": float(nxt[p])})
        x = nxt
    return w


def forward(opt: dict, inp: dict) -> "ak.TapeWriter":
    w = ak.TapeWriter("forward", {"x": {**PIC, "label": ("The picture x_t", "图 x_t"), "tol": 1e-12}}, seed=1, opt=opt,
                      phases={"noisy": PHASES["noisy"]}, provenance="live")
    x0, eps = np.array(inp["x0"], dtype=np.float64), np.array(inp["eps"], dtype=np.float64)
    s = 2 * x0 - 1
    ab = schedule()["abar"]
    p = int(opt["probe"])
    for t in range(T + 1):
        abt = ab[t - 1] if t else 1.0
        a, b = math.sqrt(abt), math.sqrt(1 - abt)
        x = a * s + b * eps
        w.frame("noisy", {"x": x.reshape(28, 28)}, step=t, metrics={"t": t, "abar": abt, "a": a, "b": b, "x0": float(x0[p]), "eps": float(eps[p]), "x": float(x[p])})
    return w


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="sample")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options the TS source ran with")
    ap.add_argument("--draws", default=None, help="draws.json from the TS run (replayed)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (or @file)")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/ddpm-unet/<process>.ref)")
    a = ap.parse_args()
    opt = {**DEFAULTS, **json.loads(a.opt)}
    inp = ak.json_arg(a.input)
    if a.process == "forward":
        tape = forward(opt, inp) if inp else ak.TapeWriter("forward", SCHEMA, seed=a.seed, opt=opt, provenance="live")
    else:
        frames = json.loads(Path(a.draws).read_text(encoding="utf-8"))["frames"] if a.draws else []
        tape = sample(a.seed, opt, Draws(frames), int(inp.get("seed", 0))) if inp else ak.TapeWriter("sample", SCHEMA, seed=a.seed, opt=opt, provenance="live")
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "ddpm-unet", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
