"""crnn-ocr: train the tiny CRNN with the CTC loss on the words make_data.ts wrote (fixed seed).

    node pipeline/crnn-ocr/make_data.ts --site <site>              (first: the data, from the page's glyphs.ts)
    npm run atlas -- py pipeline/crnn-ocr/train.py [--epochs 30]

80,000 training words of 1-7 digits, batch 256, AdamW lr 3e-3 with a cosine decay to 0, gradient clip 1.0, CTC loss
(blank = class 0) on the log-softmax of the 32 columns' scores. The epoch with the best word accuracy on the 2,000
validation words is kept. GPU for training; the reported numbers are measured on the CPU in float32 with TF32 off
(atlas_kit switches it off on import). Writes model_final.pt and train_metrics.json next to this file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import atlas_kit as ak  # noqa: F401  (TF32 off)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak  # noqa: F401

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CRNN, greedy, load  # noqa: E402


def ctc(m: CRNN, x, y, n) -> torch.Tensor:
    lp = F.log_softmax(m(x), dim=-1)
    T, B = lp.shape[:2]
    return F.ctc_loss(lp, y, torch.full((B,), T, dtype=torch.long), n, blank=0, zero_infinity=True)


@torch.no_grad()
def evaluate(m: CRNN, x, y, n, texts, dev="cpu") -> dict:
    m = m.to(dev).eval()
    reads, loss = [], 0.0
    for s in range(0, len(x), 1000):
        xb = x[s:s + 1000].to(dev)
        reads += greedy(m(xb).cpu())
        loss += ctc(m, xb, y[s:s + 1000].to(dev), n[s:s + 1000].to(dev)).item() * len(xb)
    ok = [r == t for r, t in zip(reads, texts)]
    return {"acc": sum(ok) / len(ok), "loss": loss / len(x), "reads": reads, "ok": ok}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt, yt, nt, _ = load("train")
    xv, yv, nv, tv = load("val")
    m = CRNN().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(xt) // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + torch.cos(torch.tensor(s / steps * 3.141592653589793)).item()))
    xt, yt, nt = xt.to(dev), yt.to(dev), nt.to(dev)
    best, best_ep, t0 = -1.0, -1, time.time()
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(len(xt), generator=g).to(dev)
        tot = 0.0
        for s in range(0, len(xt) - a.batch + 1, a.batch):
            i = perm[s:s + a.batch]
            loss = ctc(m, xt[i], yt[i], nt[i])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        v = evaluate(m, xv, yv, nv, tv, dev)
        print(f"epoch {ep}: train ctc {tot / (len(xt) // a.batch):.4f}  val ctc {v['loss']:.4f}  val acc {v['acc']:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        if v["acc"] > best:
            best, best_ep = v["acc"], ep
            torch.save({k: t.detach().cpu() for k, t in m.state_dict().items()}, HERE / "model_final.pt")
    m = CRNN()
    m.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    xs, ys, ns, ts = load("test")
    te = evaluate(m, xs, ys, ns, ts)
    by_len = {}
    for t, ok in zip(ts, te["ok"]):
        by_len.setdefault(len(t), []).append(ok)
    va = evaluate(m, xv, yv, nv, tv)
    wi = evaluate(m, *load("wild"))
    metrics = {"params": sum(p.numel() for p in m.parameters()), "epochs": a.epochs, "best_epoch": best_ep, "batch": a.batch,
               "lr": a.lr, "optimizer": "AdamW (weight decay 1e-4), cosine decay, clip 1.0", "train_words": len(xt),
               "val_acc": va["acc"], "val_ctc": va["loss"], "test_acc": te["acc"], "test_ctc": te["loss"],
               "test_words": len(ts), "wild_acc": wi["acc"], "test_acc_by_length": {str(k): sum(v) / len(v) for k, v in sorted(by_len.items())},
               "seconds": round(time.time() - t0)}
    (HERE / "train_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps(metrics, indent=1))
