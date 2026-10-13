"""pointnet: train the tiny PointNet on make_data.ts's clouds (GPU if available; fixed seed; about a minute).

    node pipeline/pointnet/make_data.ts --site <site>        # first: the data, from the page's own shapes.ts
    npm run atlas -- py pipeline/pointnet/train.py [--epochs 60] [--seed 0]

Recipe: Adam (lr 3e-3, cosine decay to 0), batch 256, cross-entropy. Every cloud is padded to 256 points by repeating
its own points, which leaves its max (and so its answer) unchanged. The checkpoint with the best validation accuracy is
evaluated on the CPU in float32 (TF32 off) on the test set, saved as model_final.pt, and its numbers go to
train_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

try:
    import atlas_kit as ak  # noqa: F401  (switches TF32 off)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak  # noqa: F401

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import load  # noqa: E402
from model import CLASSES, PointNet, n_params  # noqa: E402


@torch.no_grad()
def predict(m, X, dev, bs=2048):
    m.eval()
    return torch.cat([m(X[i:i + bs].to(dev)).argmax(1).cpu() for i in range(0, len(X), bs)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"   # training may use the GPU; export is CPU only (AK102)
    torch.manual_seed(a.seed)
    Xtr, ytr, _ = load("train")
    Xva, yva, _ = load("val")
    Xd, yd = Xtr.to(dev), ytr.to(dev)
    m = PointNet().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    spe = math.ceil(len(Xtr) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / (a.epochs * spe)))))
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
    best, best_ep, hist = -1.0, -1, []
    (HERE / "runs").mkdir(exist_ok=True)
    print(f"PointNet {n_params(m)} params on {dev}; {len(Xtr)} train, {len(Xva)} val; {a.epochs} epochs x {spe} steps")
    t0 = time.time()
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(len(Xtr), device=dev, generator=gen)
        tot = 0.0
        for i in range(spe):
            idx = perm[i * a.bs:(i + 1) * a.bs]
            loss = F.cross_entropy(m(Xd[idx]), yd[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        va = float((predict(m, Xva, dev) == yva).float().mean())
        hist.append({"epoch": ep, "loss": tot / len(Xtr), "val_acc": va})
        if va > best:
            best, best_ep = va, ep
            torch.save(m.state_dict(), HERE / "runs" / "best.pt")
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"epoch {ep:3d} loss {tot / len(Xtr):.4f} val {va:.4f} ({time.time() - t0:.0f}s)")
    m = PointNet()
    m.load_state_dict(torch.load(HERE / "runs" / "best.pt", map_location="cpu"))
    m.eval()
    Xte, yte, _ = load("test")
    p = predict(m, Xte, "cpu")
    conf = [[int(((yte == i) & (p == j)).sum()) for j in range(len(CLASSES))] for i in range(len(CLASSES))]
    metrics = {"test_acc": float((p == yte).float().mean()), "test_n": len(yte), "val_acc": best, "best_epoch": best_ep,
               "epochs": a.epochs, "steps": a.epochs * spe, "batch": a.bs, "lr": a.lr, "seed": a.seed, "train_n": len(Xtr),
               "params": n_params(m), "per_class_test_acc": {c: conf[i][i] / max(1, sum(conf[i])) for i, c in enumerate(CLASSES)},
               "confusion": conf, "history": hist}
    torch.save(m.state_dict(), HERE / "model_final.pt")
    (HERE / "train_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"best epoch {best_ep}: val {best:.4f}, test {metrics['test_acc']:.4f}", metrics["per_class_test_acc"])


if __name__ == "__main__":
    main()
