"""lora: Python reference for the processes 'lora' and 'ranks' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/lora/process.ts: the same toy world rebuilt from the same
seeded generator (ak.Rng, bit-identical to the TS rng for next() and fork(); data.ts uses uniform draws and + - * / sqrt
only), and the same LoRA training written with PyTorch float64 autograd and torch.optim.Adam (single-tensor, foreach
off) instead of the kit's autograd and Adam. `npm run atlas -- parity lora` records the TS tapes, runs this file and
compares the frames (state within each field's tol, metrics within the process's metricTol, 1e-7 relative).

    npm run atlas -- parity lora

Arguments (the CLI contract for every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json>
--input <json> --out <stem>. The draws are not needed: the uniform draws are replayed by ak.Rng itself.
The input is the learner's rank, {"r": ..}.
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

import torch

D, N, ALPHA, R_MIN, R_MAX = 16, 64, 8, 1, 8
SWEEP = [1, 2, 4, 8]


def spectrum() -> list[float]:
    s = [2.0, 1.2]
    v = 0.3
    while len(s) < D:
        s.append(v)
        v *= 0.75
    return s


SPECTRUM = spectrum()


def basis(nxt) -> list[list[float]]:
    """data.ts basis(): modified Gram-Schmidt on uniform vectors in [-1, 1), the same operations in the same order."""
    out: list[list[float]] = []
    while len(out) < D:
        v = [2 * nxt() - 1 for _ in range(D)]
        for u in out:
            d = 0.0
            for i in range(D):
                d += u[i] * v[i]
            for i in range(D):
                v[i] -= d * u[i]
        n = 0.0
        for i in range(D):
            n += v[i] * v[i]
        n = math.sqrt(n)
        for i in range(D):
            v[i] /= n
        out.append(v)
    return out


def world():
    rng = ak.Rng(7)
    nxt = rng.next
    W0 = [nxt() - 0.5 for _ in range(D * D)]
    U = basis(nxt)
    V = basis(nxt)
    dWs = [0.0] * (D * D)
    for o in range(D):
        for i in range(D):
            s = 0.0
            for k in range(D):
                s += SPECTRUM[k] * U[k][o] * V[k][i]
            dWs[o * D + i] = s
    X = [0.0] * (N * D)
    for b in range(N // D):
        Q = basis(nxt)
        for r in range(D):
            for i in range(D):
                X[(b * D + r) * D + i] = 4 * Q[r][i]
    Y = [0.0] * (N * D)
    for n in range(N):
        for o in range(D):
            s = 0.0
            for i in range(D):
                s += X[n * D + i] * (W0[o * D + i] + dWs[o * D + i])
            Y[n * D + o] = s
    t = lambda a, shape: torch.tensor(a, dtype=torch.float64).reshape(shape)
    return t(W0, (D, D)), t(dWs, (D, D)), t(X, (N, D)), t(Y, (N, D))


W0, DWS, X, Y = world()
BASE_OUT = X @ W0.T


def rank_of(raw) -> int:
    v = raw.get("r") if isinstance(raw, dict) else raw
    r = round(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else 2
    return min(R_MAX, max(R_MIN, int(r)))


def floor_of(r: int) -> float:
    s = 0.0
    for k in range(max(0, r), D):
        s += SPECTRUM[k] * SPECTRUM[k]
    return s / D


def fit_run(r, opt: dict, nxt):
    """process.ts fitRun: one snapshot per update (step 0 first)."""
    full = r == "full"
    k = 0 if full else r
    scale = 1.0 if full else opt["alpha"] / k
    bound = 1 / math.sqrt(D)
    if full:
        dW = torch.zeros(D, D, dtype=torch.float64, requires_grad=True)
        params = [dW]
    else:
        A = torch.tensor([(2 * nxt() - 1) * bound for _ in range(k * D)], dtype=torch.float64).reshape(k, D).requires_grad_(True)
        B = torch.zeros(D, k, dtype=torch.float64, requires_grad=True)
        params = [A, B]
    adam = torch.optim.Adam(params, lr=opt["lr"], foreach=False)

    def grads() -> float:
        for p in params:
            p.grad = None
        change = X @ dW.T if full else (X @ A.T @ B.T) * scale
        L = ((BASE_OUT + change - Y) ** 2).mean()
        L.backward()
        return L.item()

    def snap(step, loss, dA, dB):
        if full:
            return {"step": step, "loss": loss, "gA": 0.0, "gB": dW.grad.norm().item(), "dA": dA, "dB": dB}
        return {"step": step, "loss": loss, "gA": A.grad.norm().item(), "gB": B.grad.norm().item(), "dA": dA, "dB": dB,
                "A": A.detach().clone(), "B": B.detach().clone(), "dW": scale * (B.detach() @ A.detach())}

    loss = grads()
    yield snap(0, loss, 0.0, 0.0)
    for s in range(1, opt["steps"] + 1):
        a0 = None if full else A.detach().clone()
        b0 = (dW if full else B).detach().clone()
        adam.step()
        dA = 0.0 if full else (A.detach() - a0).abs().max().item()
        dB = ((dW if full else B).detach() - b0).abs().max().item()
        loss = grads()
        yield snap(s, loss, dA, dB)


SCHEMA = {
    "A": {"kind": "tensor", "label": ("A (r × 16), trainable", "A（r × 16），可训练"), "shape": ["r", D], "role": "param", "scale": "diverging", "tol": 1e-7},
    "B": {"kind": "tensor", "label": ("B (16 × r), trainable, starts at 0", "B（16 × r），可训练，从 0 开始"), "shape": [D, "r"], "role": "param", "scale": "diverging", "tol": 1e-7},
    "dW": {"kind": "tensor", "label": ("The change (α/r)·B·A", "改变量 (α/r)·B·A"), "shape": [D, D], "role": "param", "scale": "diverging", "tol": 1e-7},
}
PHASES = {
    "start": {"label": ("Before training: B = 0", "训练之前：B = 0"), "kind": "input"},
    "step": {"label": ("One Adam update of A and B", "A 和 B 的一次 Adam 更新"), "kind": "update"},
}
RANK_PHASES = {
    "start": {"label": ("Before training", "训练之前"), "kind": "input"},
    "step": {"label": ("Five runs, one update each", "五次运行，各更新一次"), "kind": "update"},
}


def run(process: str, raw, opt: dict, seed: int) -> "ak.TapeWriter":
    o = {"steps": 100, "lr": 0.02, "alpha": ALPHA, **opt}
    o["steps"] = int(o["steps"])
    rng = ak.Rng(seed)
    if process == "lora":
        tape = ak.TapeWriter(process, SCHEMA, seed=seed, opt=opt, phases=PHASES, provenance="live")
        r = rank_of(raw)
        fr = rng.fork(f"r{r}")
        fixed = {"r": r, "alpha": o["alpha"], "scale": o["alpha"] / r, "lr": o["lr"], "floor": floor_of(r), "base": floor_of(0),
                 "trainable": r * (D + D), "full": D * D}
        for s in fit_run(r, o, fr.next):
            state = {k: s[k].numpy() for k in ("A", "B", "dW")}
            normB = math.sqrt(sum(v * v for v in s["B"].flatten().tolist()))
            m = {"step": s["step"], "loss": s["loss"], "gA": s["gA"], "gB": s["gB"], "dA": s["dA"], "dB": s["dB"], "normB": normB, **fixed}
            tape.frame("start" if s["step"] == 0 else "step", state, step=s["step"], metrics=m)
        return tape
    if process == "ranks":
        tape = ak.TapeWriter(process, {}, seed=seed, opt=opt, phases=RANK_PHASES, provenance="live")
        runs = []
        for r in SWEEP:
            fr = rng.fork(f"r{r}")
            runs.append(fit_run(r, o, fr.next))
        for f in fit_run("full", o, lambda: 0.0):
            m = {"step": f["step"], "lfull": f["loss"]}
            for r, g in zip(SWEEP, runs):
                m[f"l{r}"] = next(g)["loss"]
            tape.frame("start" if f["step"] == 0 else "step", {}, step=f["step"], metrics=m)
        return tape
    raise SystemExit(f"reference.py: no reference for process '{process}' (lora and ranks only)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="lora")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null", help="JSON input of the TS run, or @<file> (read it with ak.json_arg)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = ak.json_arg(a.input)
    tape = run(a.process, raw, json.loads(a.opt), a.seed)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "lora", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
