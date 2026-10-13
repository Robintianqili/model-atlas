"""contrastive-learning: Python reference for process 'nce' (kind C parity, design §4.11), in torch float64 with autograd.

An independent implementation of the `nce` process in site/src/lib/explainers/contrastive-learning/process.ts: from the
N image vectors and N caption vectors of the learner's batch (the --input JSON: I, T, N, D, tau; the vectors come from
the shipped TinyCLIP, batch.ts) it computes the N x N cosines and scores cos / tau, the row and column softmax, the
symmetric InfoNCE loss (torch.nn.functional.cross_entropy) and the gradient of the loss on the scores (torch.autograd),
and writes the same three frames. `atlas parity contrastive-learning` compares them with the TS tape (every field and
metric within 1e-12). Adapted from pipeline/tinyclip/reference.py (which scales by the learned s instead of 1 / tau).

The `progress` and `simclr` processes take the models themselves as input (run on the main thread), so they have no
Python reference (reference: false); the checkpoints' forward pass is checked by `parity --dir` (export.py).

    npm run atlas -- parity contrastive-learning            # records the TS tape, runs this file, compares

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

import torch
import torch.nn.functional as F

SCHEMA = {
    "C": {"kind": "tensor", "label": ("Cosines (drawing i with caption j)", "余弦（画 i 与描述 j）"), "role": "activation", "tol": 1e-12},
    "L": {"kind": "tensor", "label": ("Scores: cosine ÷ τ", "分数：余弦 ÷ τ"), "role": "activation", "tol": 1e-12},
    "I": {"kind": "tensor", "label": ("Image vectors (length 1)", "图像向量（长度 1）"), "role": "data", "tol": 1e-12},
    "T": {"kind": "tensor", "label": ("Caption vectors (length 1)", "描述向量（长度 1）"), "role": "data", "tol": 1e-12},
    "P": {"kind": "tensor", "label": ("Row softmax: which caption fits each drawing", "行 softmax：每幅画配哪句描述"), "role": "output", "tol": 1e-12},
    "Q": {"kind": "tensor", "label": ("Column softmax: which drawing fits each caption", "列 softmax：每句描述配哪幅画"), "role": "output", "tol": 1e-12},
    "G": {"kind": "tensor", "label": ("Gradient of the loss on each score", "损失对每个分数的梯度"), "role": "grad", "tol": 1e-12},
}
PHASES = {
    "scores": {"label": ("Cosines and scores of every pair", "每一对的余弦和分数"), "kind": "forward"},
    "softmax": {"label": ("Softmax along rows and columns", "沿行和列做 softmax"), "kind": "loss"},
    "grad": {"label": ("Gradient on the scores", "分数上的梯度"), "kind": "backward"},
}


def off_max(A: torch.Tensor):
    """the largest entry off the diagonal and its row and column (the first one on a tie, row by row)"""
    N = A.shape[0]
    best = (-math.inf, 0, 1)
    for i in range(N):
        for j in range(N):
            if i != j and float(A[i, j]) > best[0]:
                best = (float(A[i, j]), i, j)
    return best


def wins(C: torch.Tensor):
    N = C.shape[0]
    rows = sum(all(float(C[i, j]) < float(C[i, i]) for j in range(N) if j != i) for i in range(N))
    cols = sum(all(float(C[j, i]) < float(C[i, i]) for j in range(N) if j != i) for i in range(N))
    return rows, cols


def run(b: dict | None, seed: int = 1) -> "ak.TapeWriter":
    w = ak.TapeWriter("nce", SCHEMA, seed=seed, opt={}, phases=PHASES, provenance="live")
    if not b or b["N"] < 2:
        return w
    N, D, tau = int(b["N"]), int(b["D"]), float(b["tau"])
    I = torch.tensor(b["I"], dtype=torch.float64).reshape(N, D)
    T = torch.tensor(b["T"], dtype=torch.float64).reshape(N, D)
    C = I @ T.t()
    L = (C / tau).requires_grad_(True)
    Ld = L.detach()
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    off = [float(C[i, j]) for i in range(N) for j in range(N) if i != j]
    hv, hi, hj = off_max(C)
    w.frame("scores", {"C": C.numpy(), "L": Ld.numpy(), "I": I.numpy(), "T": T.numpy()}, metrics={
        "N": N, "D": D, "tau": tau, "cos00": float(C[0, 0]), "l00": float(Ld[0, 0]), "diagMean": mean([float(C[i, i]) for i in range(N)]),
        "offMean": mean(off), "hard": hv, "hardI": hi, "hardJ": hj})
    P, Q = torch.softmax(Ld, 1), torch.softmax(Ld, 0)
    y = torch.arange(N)
    loss = (F.cross_entropy(L, y) + F.cross_entropy(L.t(), y)) / 2
    r, c = wins(C)
    w.frame("softmax", {"P": P.numpy(), "Q": Q.numpy()}, metrics={
        "p00": float(P[0, 0]), "q00": float(Q[0, 0]), "l0": -math.log(float(P[0, 0])),
        "rowLoss": -mean([math.log(float(P[i, i])) for i in range(N)]), "colLoss": -mean([math.log(float(Q[i, i])) for i in range(N)]),
        "loss": float(loss), "chance": math.log(N), "rowRight": r, "colRight": c})
    loss.backward()
    G = L.grad
    gv, gi, gj = off_max(G)
    w.frame("grad", {"G": G.numpy()}, metrics={"g00": float(G[0, 0]), "gOffMax": gv, "gOffI": gi, "gOffJ": gj})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="nce")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null", help="JSON input of the TS run: {I, T, N, D, tau}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/contrastive-learning/nce.ref)")
    a = ap.parse_args()
    if a.process != "nce":
        raise SystemExit(f"no Python reference for process {a.process!r} (only 'nce'; 'synth' is checked by its CRC-32 in train.py)")
    tape = run(json.loads(a.input), seed=a.seed)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "contrastive-learning", "nce.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
