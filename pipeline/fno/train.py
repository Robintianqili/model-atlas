"""fno: train the Fourier neural operator on pairs made in TS (GPU if available; fixed seed).

    npm run atlas -- run fno data --opt '{"nTrain":4000,"nVal":400,"nTest":400}' --save   # first
    npm run atlas -- py pipeline/fno/train.py [--steps 6000] [--seed 0]

Data: $ATLAS_SITE/.atlas/fno/data.tape.json + .tape.bin (atlas-tape/1), written by the package's TS process `data`
(heat.ts): 4,000 training pairs, 400 validation and 400 test pairs; a pair is a random starting curve of 64 heights (one
of four families in turn: waves, pulses, bumps, lines) and the numerical solver's curve 50 time steps later. Every
chunk's CRC-32 is recomputed here and must equal the hash TS recorded, so the model trains on the bytes the page computes.

Recipe: mean squared error over the 64 points, Adam lr 1e-2 with a cosine decay to 0, batch 64. The checkpoint with the
lowest validation error is kept, evaluated on the CPU in float32 (TF32 off) and saved to model_final.pt; its numbers go
to train_metrics.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import zlib
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import FNO1d, MODES, N, WIDTH, n_params  # noqa: E402

SETS = ["train", "val", "test"]


def load_sets(check: bool = True) -> dict:
    """{'train': (U0 [n, 64, 1], UT [n, 64, 1], fam [n]), 'val': ..., 'test': ...} from the TS tape; CRC-32 checked."""
    h, blob = ak.tape.read_tape(ak.site_path(".atlas", "fno", "data"))
    parts: dict = {s: ([], [], []) for s in SETS}
    for f in h["frames"]:
        st = {k: ak.tape.decode(v, blob) for k, v in f["state"].items()}
        u0, uT = (np.ascontiguousarray(st[k], dtype="<f4") for k in ("u0", "uT"))
        if check and (zlib.crc32(u0.tobytes()) != int(f["metrics"]["hash"]) or zlib.crc32(uT.tobytes()) != int(f["metrics"]["hashT"])):
            raise SystemExit(f"chunk {f['step']}: CRC-32 of the bytes read differs from the TS hash")
        s = SETS[int(f["metrics"]["set"])]
        parts[s][0].append(u0)
        parts[s][1].append(uT)
        parts[s][2].append(np.asarray(st["fam"], dtype=np.int64))
    out = {s: (torch.tensor(np.concatenate(p[0]))[..., None], torch.tensor(np.concatenate(p[1]))[..., None], torch.tensor(np.concatenate(p[2])))
           for s, p in parts.items()}
    out["opt"], out["seed"] = h["opt"], h["seed"]
    return out


@torch.no_grad()
def errors(m, U0, UT, keep=None) -> dict:
    """mean squared error, the mean relative L2 error and the mean largest |error| per curve"""
    m.eval()
    P = m(U0, keep)
    d = (P - UT)[..., 0]
    rel = d.norm(dim=1) / UT[..., 0].norm(dim=1).clamp_min(1e-12)
    return {"mse": float(d.pow(2).mean()), "rel": float(rel.mean()), "max": float(d.abs().amax(1).mean())}


def fit(D, *, steps=6000, seed=0, lr=1e-2, bs=64, log=print):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    U0, UT = (t.to(dev) for t in D["train"][:2])
    V0, VT = (t.to(dev) for t in D["val"][:2])
    m = FNO1d().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator(device=dev).manual_seed(seed)
    best, best_state, best_step = float("inf"), None, -1
    t0 = time.time()
    for s in range(steps):
        m.train()
        i = torch.randint(0, len(U0), (bs,), device=dev, generator=gen)
        loss = (m(U0[i]) - UT[i]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (s + 1) % 100 == 0:
            v = errors(m, V0, VT)["mse"]
            if v < best:
                best, best_state, best_step = v, copy.deepcopy(m.state_dict()), s + 1
            if (s + 1) % 1000 == 0:
                log(f"step {s + 1}: train {float(loss):.3e} val {v:.3e} {time.time() - t0:.0f}s")
    m.load_state_dict(best_state)
    return m.cpu().eval(), {"best_step": best_step, "steps": steps, "seconds": round(time.time() - t0, 1), "batch": bs, "lr": lr, "seed": seed}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    D = load_sets()
    print(f"read {len(D['train'][0])} training, {len(D['val'][0])} validation and {len(D['test'][0])} test pairs (CRC-32 checked)")
    m, info = fit(D, steps=a.steps, seed=a.seed)
    torch.save(m.state_dict(), HERE / "model_final.pt")
    tr, va, te = (errors(m, *D[s][:2]) for s in SETS)
    by_keep = {k: errors(m, *D["test"][:2], keep=k)["max"] for k in range(1, MODES + 1)}
    tall = errors(m, D["test"][0] * 3, D["test"][1] * 3)
    report = {"width": WIDTH, "modes": MODES, "n": N, "params": n_params(m), **info,
              "train_mse": tr["mse"], "val_mse": va["mse"], "test_mse": te["mse"], "test_rel": te["rel"], "test_max": te["max"],
              "test_max_by_keep": by_keep, "test3x_rel": tall["rel"], "test3x_max": tall["max"],
              "sizes": {s: len(D[s][0]) for s in SETS}, "data": {"seed": D["seed"], "opt": D["opt"]},
              "hardware": "RTX 5090 Laptop GPU (training); CPU float32, TF32 off (evaluation)"}
    with open(HERE / "train_metrics.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
