"""tinymamba: Python reference for the process 'scan' (kind C parity, DEVKIT section 7.2).

An independent float64 implementation of site/src/lib/explainers/tinymamba/{mamba,process}.ts: the trained model
(model_final.pt, the same float32 weights as weights.bin) cast to float64, its blocks written out with plain Python
floats (math.exp, math.log1p), one frame per token of the chosen block's scan: the whole state h [D, N] and the
arithmetic of one channel and slot. `npm run atlas -- parity tinymamba` records the TS tape, runs this file and compares
the tapes frame by frame (phase and step exactly, h within 1e-12, every metric within 1e-12).

    npm run atlas -- parity tinymamba --fixtures
    npm run atlas -- py pipeline/tinymamba/reference.py --input "{\"ids\": [0, 3, 0, 9], \"block\": 1, \"ch\": 2, \"slot\": 0}"

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
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ASK = 9
SCHEMA = {"h": {"kind": "tensor", "label": ("state h (channels × slots)", "状态 h（通道 × 槽）"), "shape": ["D", "N"],
                "scale": "diverging", "tol": 1e-12}}
PHASES = {"start": {"label": ("Empty state", "空状态"), "kind": "input"},
          "token": {"label": ("Read one token", "读一个词元"), "kind": "forward"}}

W = {k: v.double().tolist() for k, v in torch.load(HERE / "model_final.pt", map_location="cpu").items()}


def softplus(x: float) -> float:
    return x if x > 20 else math.log1p(math.exp(x))


def linear(rows, Wm, b=None):
    return [[sum(Wm[o][i] * r[i] for i in range(len(r))) + (b[o] if b else 0.0) for o in range(len(Wm))] for r in rows]


def norm(row, g, b):
    n = len(row)
    mean = sum(v / n for v in row)
    var = sum((v - mean) * (v - mean) / n for v in row)
    s = math.sqrt(var + 1e-5)
    return [(v - mean) / s * g[i] + b[i] for i, v in enumerate(row)]


def blocks(ids):
    x = [list(W["emb.weight"][i]) for i in ids]
    out = []
    L = sum(1 for k in W if k.endswith(".A_log"))
    for b in range(L):
        p = f"blocks.{b}."
        z = [norm(r, W[p + "norm.weight"], W[p + "norm.bias"]) for r in x]
        u = [[v / (1 + math.exp(-v)) for v in r] for r in linear(z, W[p + "inp.weight"], W[p + "inp.bias"])]
        delta = [[softplus(v) for v in r] for r in linear(u, W[p + "dt.weight"], W[p + "dt.bias"])]
        Bm, Cm = linear(u, W[p + "B.weight"]), linear(u, W[p + "C.weight"])
        A = [[-math.exp(v) for v in r] for r in W[p + "A_log"]]
        run = {"u": u, "delta": delta, "B": Bm, "C": Cm, "A": A, "D": W[p + "Dskip"]}
        out.append(run)
        D, N = len(A), len(A[0])
        h = [[0.0] * N for _ in range(D)]
        ys = [step(run, t, h) for t in range(len(ids))]
        o = linear(ys, W[p + "out.weight"], W[p + "out.bias"])
        x = [[x[t][i] + o[t][i] for i in range(len(x[t]))] for t in range(len(ids))]
    return out


def step(r, t, h):
    """h <- exp(delta A) h + delta B u for every channel and slot (in place); returns y."""
    y = []
    for c in range(len(h)):
        dl, s = r["delta"][t][c], 0.0
        for n in range(len(h[c])):
            h[c][n] = math.exp(dl * r["A"][c][n]) * h[c][n] + dl * r["B"][t][n] * r["u"][t][c]
            s += r["C"][t][n] * h[c][n]
        y.append(s + r["D"][c] * r["u"][t][c])
    return y


def run(raw, seed: int = 1, opt: dict | None = None) -> "ak.TapeWriter":
    w = ak.TapeWriter("scan", SCHEMA, seed=seed, opt=opt or {}, phases=PHASES, provenance="live")
    ids = [int(i) for i in raw["ids"]]
    blk, c, sl = int(raw["block"]), int(raw["ch"]), int(raw["slot"])
    D, N = len(W["blocks.0.A_log"]), len(W["blocks.0.A_log"][0])
    h = [[0.0] * N for _ in range(D)]
    if not ids:
        return w
    w.frame("start", {"h": np.array(h, dtype=np.float64)}, metrics={"T": len(ids), "D": D, "N": N, "block": blk, "ch": c, "slot": sl})
    r = blocks(ids)[blk - 1]
    for t in range(len(ids)):
        old = h[c][sl]
        y = step(r, t, h)
        dl = r["delta"][t][c]
        m = {"t": t, "id": ids[t], "letter": 1 if 1 <= ids[t] <= 8 else 0, "ask": 1 if ids[t] == ASK else 0, "delta": dl,
             "A": r["A"][c][sl], "decay": math.exp(dl * r["A"][c][sl]), "B": r["B"][t][sl], "u": r["u"][t][c],
             "write": dl * r["B"][t][sl] * r["u"][t][c], "hOld": old, "hNew": h[c][sl], "C": r["C"][t][sl], "y": y[c],
             "meanDelta": sum(v / D for v in r["delta"][t])}
        w.frame("token", {"h": np.array(h, dtype=np.float64)}, step=t, metrics=m)
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="scan")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None, help="not needed: the scan draws nothing")
    ap.add_argument("--input", default="null", help="JSON input of the TS run ({ids, block, ch, slot})")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tape = run(json.loads(a.input), seed=a.seed, opt=json.loads(a.opt))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "tinymamba", "scan.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
