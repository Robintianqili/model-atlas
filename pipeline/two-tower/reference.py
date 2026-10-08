"""two-tower: Python reference for process 'inbatch' (kind C parity, design §4.11), in torch float64 with autograd.

An independent implementation of the `inbatch` process in site/src/lib/explainers/two-tower/process.ts: from the B user
vectors and B movie vectors of the page's batch (the --input JSON: U, V, B, D; the vectors come from the exported
towers, recs.ts batchOf) it computes the B x B scores, the softmax along each row, the loss (cross-entropy of each
viewer's own movie, torch.nn.functional.cross_entropy, averaged over the rows) and its gradient on the scores
(torch.autograd), and writes the same three frames. `atlas parity two-tower` compares them with the TS tape.

The `logs` process has no Python reference (reference: false): train.py reads its bytes and checks every frame's sum.

    npm run atlas -- parity two-tower            # records the TS tape, runs this file, compares

CLI contract: --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
"""
from __future__ import annotations

import argparse
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
    "S": {"kind": "tensor", "label": ("Scores (dot products)", "分数（点积）"), "role": "activation", "tol": 1e-12},
    "P": {"kind": "tensor", "label": ("Row softmax: which movie of the batch fits each viewer", "行 softmax：批里哪部电影最配每位观众"), "role": "output", "tol": 1e-12},
    "G": {"kind": "tensor", "label": ("Gradient of the loss on each score", "损失对每个分数的梯度"), "role": "grad", "tol": 1e-12},
}
PHASES = {
    "scores": {"label": ("Every viewer against every movie of the batch", "批里每位观众对每部电影"), "kind": "forward"},
    "softmax": {"label": ("Softmax along each row, and the loss", "沿每一行做 softmax，以及损失"), "kind": "loss"},
    "grad": {"label": ("Gradient on the scores", "分数上的梯度"), "kind": "backward"},
}


def run(b: dict | None, seed: int = 1) -> "ak.TapeWriter":
    w = ak.TapeWriter("inbatch", SCHEMA, seed=seed, opt={}, phases=PHASES, provenance="live")
    if not b or b["B"] < 1:
        return w
    B, D = int(b["B"]), int(b["D"])
    U = torch.tensor(b["U"], dtype=torch.float64).reshape(B, D)
    V = torch.tensor(b["V"], dtype=torch.float64).reshape(B, D)
    S = (U @ V.t()).requires_grad_(True)
    Sd = S.detach()
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    off = [float(Sd[i, j]) for i in range(B) for j in range(B) if i != j]
    w.frame("scores", {"S": Sd.numpy()}, metrics={"B": B, "D": D, "s00": float(Sd[0, 0]), "diagMean": mean([float(Sd[i, i]) for i in range(B)]),
                                                  "offMean": mean(off) if off else 0.0})
    P = torch.softmax(Sd, 1)
    y = torch.arange(B)
    loss = F.cross_entropy(S, y)
    hits = sum(1 for i in range(B) if float(Sd[i, i]) >= float(Sd[i].max()))
    w.frame("softmax", {"P": P.numpy()}, metrics={"p00": float(P[0, 0]), "loss0": -float(torch.log(P[0, 0])), "loss": float(loss.detach()),
                                                  "chance": float(torch.log(torch.tensor(float(B), dtype=torch.float64))), "hits": hits})
    loss.backward()
    G = S.grad
    offG = [float(G[i, j]) for i in range(B) for j in range(B) if i != j]
    w.frame("grad", {"G": G.numpy()}, metrics={"g00": float(G[0, 0]), "g01": float(G[0, 1]) if B > 1 else 0.0,
                                               "gMin": min(float(G[i, i]) for i in range(B)), "gOffMax": max(offG) if offG else 0.0, "gSum": float(G.sum())})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="inbatch")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null", help="JSON input of the TS run: {U, V, B, D}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/two-tower/inbatch.ref)")
    a = ap.parse_args()
    if a.process != "inbatch":
        raise SystemExit(f"no Python reference for process {a.process!r} (only 'inbatch'; 'logs' is checked by its sums in train.py)")
    tape = run(ak.json_arg(a.input), seed=a.seed)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "two-tower", "inbatch.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
