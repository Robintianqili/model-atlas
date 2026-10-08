"""ecg-autoencoder: train the autoencoder on synthetic normal beats only (GPU if available; fixed seed).

    npm run atlas -- run ecg-autoencoder data --opt '{"nTrain":4000,"nVal":400,"nTest":200,"nOdd":50}' --save   # first
    npm run atlas -- py pipeline/ecg-autoencoder/train.py [--code 4] [--hidden 32] [--epochs 150] [--seed 0] [--out model_final.pt]

Data: $ATLAS_SITE/.atlas/ecg-autoencoder/data.tape.json + .tape.bin (atlas-tape/1), written by the package's TS process
`data` (beats.ts): 4,000 training beats, all normal; the validation set (400 normal beats and 50 of each anomaly: small or missing P
wave, wide QRS, raised ST segment, noisy recording) and the test set (200 normal beats and 50 of each anomaly). Every chunk's CRC-32 of the beat bytes is
recomputed here and must equal the hash TS recorded, so the model trains on exactly the bytes the page computes.

Recipe: mean squared error over the 140 points, Adam lr 2e-3 with a cosine decay to 0, batch 128. The checkpoint with
the lowest error on the normal validation beats is kept, evaluated on the CPU in float32 (TF32 off) and saved to --out; its numbers go to
train_metrics.json (train_metrics_c<code>_h<hidden>.json for other sizes: the size screen, judged by the validation AUC).
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
from model import AE, CODE, H, KINDS, mse, n_params, score  # noqa: E402

SETS = ["train", "val", "test"]


def load_sets(check: bool = True) -> dict:
    """{'train': (X [n, 140], kinds [n]), 'val': ..., 'test': ...} from the TS tape; every chunk's CRC-32 checked."""
    h, blob = ak.tape.read_tape(ak.site_path(".atlas", "ecg-autoencoder", "data"))
    parts: dict = {s: ([], []) for s in SETS}
    for f in h["frames"]:
        st = {k: ak.tape.decode(v, blob) for k, v in f["state"].items()}
        x = np.ascontiguousarray(st["beats"], dtype="<f4")
        if check and zlib.crc32(x.tobytes()) != int(f["metrics"]["hash"]):
            raise SystemExit(f"chunk {f['step']}: CRC-32 of the bytes read differs from the TS hash")
        s = SETS[int(f["metrics"]["set"])]
        parts[s][0].append(x)
        parts[s][1].append(np.asarray(st["kinds"], dtype=np.int64))
    out = {s: (torch.tensor(np.concatenate(p[0])), torch.tensor(np.concatenate(p[1]))) for s, p in parts.items()}
    out["opt"], out["seed"] = h["opt"], h["seed"]
    return out


def auc(scores: np.ndarray, odd: np.ndarray) -> float:
    """Area under the ROC curve: the share of (odd, normal) pairs where the odd beat scores higher (ties count half)."""
    pos, neg = scores[odd == 1], scores[odd == 0]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


@torch.no_grad()
def scores_of(m, X) -> np.ndarray:
    m.eval()
    return score(X, m(X)).numpy().astype(np.float64)


def fit(D, *, code=CODE, hidden=H, epochs=150, seed=0, lr=2e-3, bs=128, log=print):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xg, Xva = D["train"][0].to(dev), D["val"][0][D["val"][1] == 0]   # early stopping: the normal validation beats
    m = AE(code=code, h=hidden).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    steps = epochs * math.ceil(len(Xg) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator(device=dev).manual_seed(seed)
    best, best_state, best_epoch = float("inf"), None, -1
    t0 = time.time()
    for epoch in range(epochs):
        m.train()
        perm = torch.randperm(len(Xg), device=dev, generator=gen)
        for i in range(0, len(perm), bs):
            x = Xg[perm[i:i + bs]]
            loss = mse(x, m(x)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            m.eval()
            v = float(mse(Xva.to(dev), m(Xva.to(dev))).mean())
        if v < best:
            best, best_state, best_epoch = v, copy.deepcopy(m.state_dict()), epoch + 1
        if (epoch + 1) % 50 == 0 or epoch == 0:
            log(f"epoch {epoch + 1}: val mse {v:.3e} {time.time() - t0:.0f}s")
    m.load_state_dict(best_state)
    return m.cpu().eval(), {"best_epoch": best_epoch, "epochs": epochs, "seconds": round(time.time() - t0, 1),
                            "steps": steps, "batch": bs, "lr": lr, "seed": seed}


def aucs(m, X, k) -> tuple[float, dict]:
    """AUC of all anomalies against the normal beats, and of each kind against the normal beats."""
    s, k = scores_of(m, X), k.numpy()
    return auc(s, (k > 0).astype(int)), {KINDS[j]: auc(s[(k == 0) | (k == j)], (k[(k == 0) | (k == j)] > 0).astype(int)) for j in range(1, len(KINDS))}


def evaluate(m, D) -> dict:
    """Mean squared error of the normal beats (training, validation); the AUC on the validation set (the size screen) and on the test set."""
    va, va_k = aucs(m, *D["val"])
    te, te_k = aucs(m, *D["test"])
    with torch.no_grad():
        tr, vn = D["train"][0], D["val"][0][D["val"][1] == 0]
        mses = float(mse(tr, m(tr)).mean()), float(mse(vn, m(vn)).mean())
    return {"train_mse": mses[0], "val_mse": mses[1],
            "val_auc": va, "val_auc_kind": va_k, "test_auc": te, "test_auc_kind": te_k}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code", type=int, default=CODE)
    ap.add_argument("--hidden", type=int, default=H)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D = load_sets()
    print(f"read {len(D['train'][0])} training, {len(D['val'][0])} validation and {len(D['test'][0])} test beats (CRC-32 checked)")
    m, info = fit(D, code=a.code, hidden=a.hidden, epochs=a.epochs, seed=a.seed)
    main = a.code == CODE and a.hidden == H
    out = HERE / (a.out or ("model_final.pt" if main else f"model_c{a.code}_h{a.hidden}.pt"))
    torch.save(m.state_dict(), out)
    report = {"code": a.code, "hidden": a.hidden, "params": n_params(m), **info, **evaluate(m, D),
              "n": {s: len(D[s][0]) for s in SETS}, "data": {"seed": D["seed"], "opt": D["opt"]},
              "hardware": "RTX 5090 Laptop GPU (training); CPU float32, TF32 off (evaluation)"}
    name = "train_metrics.json" if main else f"train_metrics_c{a.code}_h{a.hidden}.json"
    with open(HERE / name, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
