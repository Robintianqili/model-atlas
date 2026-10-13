"""contrastive-learning: re-run TinyCLIP's training recipe and keep checkpoints along the way (CPU, deterministic).

    npm run atlas -- run tinyclip synth --opt '{"n":24000}' --save              # first: TinyCLIP's TS process writes the pairs
    npm run atlas -- py pipeline/contrastive-learning/train.py [--run v1] [--epochs 60] [--seed 0]

The concept page's "training progress" slider shows the learner's batch through these checkpoints. The recipe is the one
of pipeline/tinyclip/train.py (copied, not imported: AdamW lr 2e-3, weight decay 0.01 on matrices, batch 256, cosine
decay, symmetric InfoNCE, learned temperature with s = e^t capped at 100, seed 0), on the same TS-rendered pairs
(scenes.ts, copied from tinyclip; every chunk's CRC-32 is checked against the hash TS recorded). Two differences: it
runs on the CPU (so a re-run gives the same bytes), and it keeps the weights after update k for every k in CKPTS
(runs/<run>/ck-<k>.pt) instead of the best validation epoch. At each checkpoint it measures, on the 2,000 test pairs in
consecutive batches of 6, the InfoNCE loss and how often a drawing's own caption wins its row.
Writes runs/<run>/metrics.json.
"""
from __future__ import annotations

import argparse
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import TinyCLIP, n_params  # noqa: E402

# updates after which the weights are kept (0 = the starting weights); 78 updates per epoch, 4,680 in 60 epochs
CKPTS = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 4680]


def load_pairs(check: bool = True):
    """(images [n, 1, 28, 28], ids [n, 8], scenes [n, 5], captions, hashes) from the TS tape; hashes checked."""
    h, blob = ak.tape.read_tape(ak.site_path(".atlas", "tinyclip", "synth"))
    imgs, ids, scenes, caps, hashes = [], [], [], [], []
    for f in h["frames"]:
        if f["phase"] != "chunk":
            continue
        st = {k: ak.tape.decode(v, blob) for k, v in f["state"].items()}
        x = np.ascontiguousarray(st["images"], dtype="<f4")
        if check:
            got = zlib.crc32(x.tobytes())
            if got != int(f["metrics"]["hash"]):
                raise SystemExit(f"chunk {f['step']}: CRC-32 of the bytes read {got} != TS hash {f['metrics']['hash']}")
        hashes.append(int(f["metrics"]["hash"]))
        imgs.append(x.reshape(-1, 1, 28, 28)); ids.append(st["ids"]); scenes.append(st["scenes"]); caps += st["captions"]
    return (torch.tensor(np.concatenate(imgs)), torch.tensor(np.concatenate(ids)).long(), torch.tensor(np.concatenate(scenes)),
            caps, hashes, h)


def load_fixed():
    """The fixed drawings of frame 0 (presets, partner pairs, edge cases): images [F, 1, 28, 28], ids [F, 3, 8], captions."""
    h, blob = ak.tape.read_tape(ak.site_path(".atlas", "tinyclip", "synth"))
    f = next(f for f in h["frames"] if f["phase"] == "fixed")
    st = {k: ak.tape.decode(v, blob) for k, v in f["state"].items()}
    x = np.ascontiguousarray(st["images"], dtype="<f4")
    if zlib.crc32(x.tobytes()) != int(f["metrics"]["hash"]):
        raise SystemExit("fixed drawings: CRC-32 of the bytes read differs from the TS hash")
    return torch.tensor(x.reshape(-1, 1, 28, 28)), torch.tensor(np.array(st["ids"])).long(), st["captions"], int(f["metrics"]["hash"])



def info_nce(m, x, ids):
    a, b = m.encode_image(x), m.encode_text(ids)
    L = m.scale() * a @ b.t()
    y = torch.arange(len(x), device=x.device)
    return (F.cross_entropy(L, y) + F.cross_entropy(L.t(), y)) / 2


@torch.no_grad()
def batches6(m, X, I) -> dict:
    """The test pairs in consecutive batches of 6: mean symmetric InfoNCE loss, the share of drawings whose own caption
    has the highest score in their row, and the mean cosine on and off the diagonal."""
    was = m.training
    m.eval()
    a, b = m.encode_image(X), m.encode_text(I)
    s = float(m.scale())
    rows = n = nb = 0
    loss = on = off = 0.0
    for k in range(0, len(X) - 5, 6):
        C = a[k:k + 6] @ b[k:k + 6].t()
        L = s * C
        y = torch.arange(6)
        rows += int((L.argmax(1) == y).sum()); n += 6; nb += 1
        loss += float((F.cross_entropy(L, y) + F.cross_entropy(L.t(), y)) / 2)
        on += float(C.diag().mean()); off += float((C.sum() - C.diag().sum()) / 30)
    m.train(was)
    return {"loss6": loss / nb, "rowAcc6": rows / n, "cosOn": on / nb, "cosOff": off / nb, "scale": s, "batches": nb}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    X, I, S, caps, hashes, h = load_pairs()
    print(f"read {len(X)} pairs from the TS tape, {len(hashes)} chunk hashes match ({time.time() - t0:.1f} s)")
    tr, te = slice(0, 20000), slice(22000, 24000)
    torch.manual_seed(a.seed)
    m = TinyCLIP()
    decay = [p for n, p in m.named_parameters() if p.dim() >= 2 and not n.startswith("emb")]
    rest = [p for n, p in m.named_parameters() if not (p.dim() >= 2 and not n.startswith("emb"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": rest, "weight_decay": 0.0}], lr=a.lr)
    Xtr, Itr = X[tr], I[tr]
    Xte, Ite = X[te], I[te]
    steps_per = len(Xtr) // a.bs
    total = steps_per * a.epochs
    if CKPTS[-1] != total:
        raise SystemExit(f"the last checkpoint must be the last update ({total})")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total) / total)))
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    cks, curve = [], []
    step = 0

    def keep():
        torch.save({k: v.detach().clone() for k, v in m.state_dict().items()}, run / f"ck-{step}.pt")
        r = {"step": step, **batches6(m, Xte, Ite)}
        cks.append(r)
        print(f"update {step:5d}  test loss (batches of 6) {r['loss6']:.4f}  own caption first {r['rowAcc6']:.3f}  "
              f"cos on {r['cosOn']:.3f} off {r['cosOff']:.3f}  s {r['scale']:.2f}  ({time.time() - t0:.0f} s)", flush=True)

    m.train()
    keep()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr), generator=g)
        for k in range(steps_per):
            j = perm[k * a.bs:(k + 1) * a.bs]
            loss = info_nce(m, Xtr[j], Itr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            curve.append(float(loss.detach()))
            if step in CKPTS:
                keep()
    out = {"run": a.run, "params": n_params(m), "epochs": a.epochs, "updates": total, "seed": a.seed, "lr": a.lr, "wd": a.wd,
           "bs": a.bs, "device": "cpu", "chunkHashes": hashes, "tape": {"seed": h["seed"], "opt": h["opt"]},
           "checkpoints": cks, "trainLoss": curve, "seconds": round(time.time() - t0, 1)}
    (run / "metrics.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"done: {total} updates in {out['seconds']} s; checkpoints {CKPTS}")


if __name__ == "__main__":
    main()
