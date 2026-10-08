"""tinyclip: Python reference for process 'nce' (kind C parity, design §4.11), in torch float64 with autograd.

An independent implementation of the `nce` process in site/src/lib/explainers/tinyclip/process.ts: from the N image
vectors and N caption vectors of the page's batch (the --input JSON: I, T, N, D, scale; the vectors come from the
exported model, batch.ts) it computes the N x N scores s * cos, the row and column softmax, the symmetric InfoNCE loss
(torch.nn.functional.cross_entropy) and the gradient of the loss on the scores (torch.autograd), and writes the same
three frames. `atlas parity tinyclip` compares them with the TS tape (every field and metric within 1e-12).

The `synth` process (the training pairs) has no Python reference: its drawings are rendered by the TS
preprocessStrokes, and train.py checks the CRC-32 of the bytes it reads instead.

    npm run atlas -- parity tinyclip            # records the TS tape, runs this file, compares

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

import torch
import torch.nn.functional as F

SCHEMA = {
    "L": {"kind": "tensor", "label": ("Scores (scale × cosine)", "分数（缩放 × 余弦）"), "role": "activation", "tol": 1e-12},
    "P": {"kind": "tensor", "label": ("Row softmax: which caption fits each drawing", "行 softmax：每幅画配哪句描述"), "role": "output", "tol": 1e-12},
    "Q": {"kind": "tensor", "label": ("Column softmax: which drawing fits each caption", "列 softmax：每句描述配哪幅画"), "role": "output", "tol": 1e-12},
    "G": {"kind": "tensor", "label": ("Gradient of the loss on each score", "损失对每个分数的梯度"), "role": "grad", "tol": 1e-12},
}
PHASES = {
    "scores": {"label": ("Scores of every pair", "每一对的分数"), "kind": "forward"},
    "softmax": {"label": ("Softmax along rows and columns", "沿行和列做 softmax"), "kind": "loss"},
    "grad": {"label": ("Gradient on the scores", "分数上的梯度"), "kind": "backward"},
}


def run(b: dict | None, seed: int = 1) -> "ak.TapeWriter":
    w = ak.TapeWriter("nce", SCHEMA, seed=seed, opt={}, phases=PHASES, provenance="live")
    if not b or b["N"] < 1:
        return w
    N, D, s = int(b["N"]), int(b["D"]), float(b["scale"])
    I = torch.tensor(b["I"], dtype=torch.float64).reshape(N, D)
    T = torch.tensor(b["T"], dtype=torch.float64).reshape(N, D)
    cos = I @ T.t()
    L = (s * cos).requires_grad_(True)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    Ld = L.detach()
    off = [float(Ld[i, j]) for i in range(N) for j in range(N) if i != j]
    w.frame("scores", {"L": Ld.numpy()}, metrics={"N": N, "D": D, "scale": s, "cos00": float(cos[0, 0]), "l00": float(Ld[0, 0]),
                                                  "diagMean": mean([float(Ld[i, i]) for i in range(N)]), "offMean": mean(off) if off else 0.0})
    P, Q = torch.softmax(Ld, 1), torch.softmax(Ld, 0)
    y = torch.arange(N)
    loss = (F.cross_entropy(L, y) + F.cross_entropy(L.t(), y)) / 2
    w.frame("softmax", {"P": P.numpy(), "Q": Q.numpy()}, metrics={
        "p00": float(P[0, 0]), "q00": float(Q[0, 0]), "rowLoss": -mean([float(torch.log(P[i, i])) for i in range(N)]),
        "colLoss": -mean([float(torch.log(Q[i, i])) for i in range(N)]), "loss": float(loss), "chance": float(torch.log(torch.tensor(float(N), dtype=torch.float64)))})
    loss.backward()
    G = L.grad
    offG = [float(G[i, j]) for i in range(N) for j in range(N) if i != j]
    w.frame("grad", {"G": G.numpy()}, metrics={"g00": float(G[0, 0]), "gMin": min(float(G[i, i]) for i in range(N)),
                                               "gOffMax": max(offG) if offG else 0.0, "gSum": float(G.sum())})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="nce")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null", help="JSON input of the TS run: {I, T, N, D, scale}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/tinyclip/nce.ref)")
    a = ap.parse_args()
    if a.process != "nce":
        raise SystemExit(f"no Python reference for process {a.process!r} (only 'nce'; 'synth' is checked by its CRC-32 in train.py)")
    tape = run(json.loads(a.input), seed=a.seed)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "tinyclip", "nce.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
