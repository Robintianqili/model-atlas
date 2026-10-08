"""tinyclip: train TinyCLIP with the InfoNCE loss on the synthetic pairs the TS side rendered (GPU if available).

    npm run atlas -- run tinyclip synth --opt '{"n":24000}' --save      # first: the TS process writes the pairs
    npm run atlas -- py pipeline/tinyclip/train.py --run v1 [--epochs 60] [--seed 0]

Data: $ATLAS_SITE/.atlas/tinyclip/synth.tape.json + .tape.bin (atlas-tape/1), 24 chunks of 1,000 pairs: drawings
[1000, 784] float32 rendered by scenes.ts render() (TinyCNN's preprocessStrokes), word ids [1000, 8] and scene codes.
Every chunk's CRC-32 of the drawing bytes is recomputed here and must equal the hash TS recorded, so the
model trains on exactly the bytes the page computes. Chunks 0-19 train (20,000), 20-21 validation, 22-23 test.

Recipe: AdamW (lr 2e-3, weight decay 0.01 on matrices), batch 256, cosine decay over the epochs, symmetric InfoNCE
(CLIP), learned temperature (logit_scale, s = e^t capped at 100). Best validation loss -> runs/<run>/best.pt; then
model_final.pt and train_metrics.json (CPU float32, TF32 off); the starting weights go to model_init.pt.
"""
from __future__ import annotations

import argparse
import json
import zlib
import math
import sys
import time
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
def eval_loss(m, X, I, dev, bs=256):
    m.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X) - bs + 1, bs):
        tot += float(info_nce(m, X[i:i + bs].to(dev), I[i:i + bs].to(dev))) * bs
        n += bs
    return tot / n


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
    X, I, S, caps, hashes, h = load_pairs()
    print(f"read {len(X)} pairs from the TS tape, {len(hashes)} chunk hashes match ({time.time() - t0:.1f} s)")
    tr, va = slice(0, 20000), slice(20000, 22000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    m = TinyCLIP()
    torch.save(m.state_dict(), HERE / "model_init.pt")   # the weights before training (the lesson shows the batch with them too)
    m = m.to(dev)
    decay = [p for n, p in m.named_parameters() if p.dim() >= 2 and not n.startswith("emb")]
    rest = [p for n, p in m.named_parameters() if not (p.dim() >= 2 and not n.startswith("emb"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": rest, "weight_decay": 0.0}], lr=a.lr)
    Xtr, Itr = X[tr].to(dev), I[tr].to(dev)
    steps_per = len(Xtr) // a.bs
    total = steps_per * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total) / total)))
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    best, hist = math.inf, []
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(len(Xtr), generator=g).to(dev)
        tl = 0.0
        for k in range(steps_per):
            j = perm[k * a.bs:(k + 1) * a.bs]
            loss = info_nce(m, Xtr[j], Itr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss.detach())
        vl = eval_loss(m, X[va], I[va], dev)
        hist.append({"epoch": ep + 1, "train": tl / steps_per, "val": vl, "scale": float(m.scale())})
        if vl < best:
            best = vl
            torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, run / "best.pt")
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep + 1:3d}  train {tl / steps_per:.4f}  val {vl:.4f}  scale {float(m.scale()):.2f}  ({time.time() - t0:.0f} s)")
    m = TinyCLIP()
    m.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    torch.save(m.state_dict(), HERE / "model_final.pt")
    test = slice(22000, 24000)
    out = {"run": a.run, "params": n_params(m), "epochs": a.epochs, "bestVal": best, "testLoss256": eval_loss(m, X[test], I[test], "cpu"),
           "scale": float(m.scale()), "chunkHashes": hashes, "seed": a.seed, "lr": a.lr, "wd": a.wd, "bs": a.bs,
           "tape": {"seed": h["seed"], "opt": h["opt"]}, "history": hist, "seconds": round(time.time() - t0, 1)}
    (HERE / "train_metrics.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"best val {best:.4f}; test (batch 256) {out['testLoss256']:.4f}; scale {out['scale']:.2f}; params {out['params']}")


if __name__ == "__main__":
    main()
