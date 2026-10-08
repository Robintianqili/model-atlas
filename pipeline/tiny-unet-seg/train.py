"""tiny-unet-seg: train the tiny U-Net (or, with --noskip, the same network without skip connections) on the synthetic
scenes the page's own TS code rendered (GPU if available; fixed seed).

    node pipeline/tiny-unet-seg/make_data.ts --site <site>                 # first: the data, written by scenes.ts
    npm run atlas -- py pipeline/tiny-unet-seg/train.py [--epochs 40] [--seed 0]
    npm run atlas -- py pipeline/tiny-unet-seg/train.py --noskip           # the lesson's "upsampling alone" model

Data: runs/data/train.bin (16,000 scenes), val.bin (1,000, picks the checkpoint), test.bin (2,000), and three sets
outside the training range for the limits step: big (sizes 9-13 px), hollow (outlines), dark (dark shapes on a light
background); data.py checks every byte against the CRC-32 TS recorded. Loss: cross-entropy of every pixel, averaged.
Adam lr 3e-3 with a cosine decay to 0, batch 64. The checkpoint with the best validation mIoU is kept, evaluated on the
CPU in float32 (TF32 off) and saved to model_final.pt (model_noskip.pt) with train_metrics(_noskip).json.
"""
from __future__ import annotations

import argparse
import copy
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
from model import CLASSES, UNet, n_params  # noqa: E402

K = len(CLASSES)


def edge_mask(y: torch.Tensor) -> torch.Tensor:
    """True for pixels whose true class differs from one of their 4 neighbours (both sides of every border)."""
    e = torch.zeros_like(y, dtype=torch.bool)
    d = y[:, 1:, :] != y[:, :-1, :]
    e[:, 1:, :] |= d
    e[:, :-1, :] |= d
    d = y[:, :, 1:] != y[:, :, :-1]
    e[:, :, 1:] |= d
    e[:, :, :-1] |= d
    return e


@torch.no_grad()
def predict(m, X, dev="cpu", bs=500) -> torch.Tensor:
    m.eval()
    return torch.cat([m.logits(X[i:i + bs].to(dev)).argmax(1).cpu() for i in range(0, len(X), bs)])


def scores(p: torch.Tensor, y: torch.Tensor) -> dict:
    """IoU per class over the whole set (pixels in both / pixels in either), their mean, pixel accuracy, border accuracy."""
    iou = []
    for k in range(K):
        a, b = p == k, y == k
        u = int((a | b).sum())
        iou.append(int((a & b).sum()) / u if u else float("nan"))
    e = edge_mask(y)
    return {"iou": iou, "miou": sum(iou) / K, "acc": float((p == y).float().mean()), "edgeAcc": float((p == y)[e].float().mean()),
            "edgeShare": float(e.float().mean())}


def fit(*, skip=True, epochs=40, seed=0, lr=3e-3, bs=64, log=print):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr = load("train")
    Xva, Yva = load("val")
    Xg, Yg = Xtr.to(dev), Ytr.to(dev)
    m = UNet(skip=skip).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    steps = epochs * math.ceil(len(Xg) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator(device=dev).manual_seed(seed)
    best, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(epochs):
        m.train()
        perm = torch.randperm(len(Xg), device=dev, generator=gen)
        tot = 0.0
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(m.logits(Xg[j]), Yg[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(j)
        v = scores(predict(m, Xva, dev), Yva)
        if v["miou"] > best:
            best, best_state, best_epoch = v["miou"], copy.deepcopy(m.state_dict()), epoch + 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"epoch {epoch + 1}: train loss {tot / len(Xg):.4f}, val mIoU {v['miou']:.4f} (edge acc {v['edgeAcc']:.4f}) {time.time() - t0:.0f}s")
    m.load_state_dict(best_state)
    m = m.cpu().eval()
    return m, {"best_epoch": best_epoch, "epochs": epochs, "seconds": round(time.time() - t0, 1), "steps": steps, "batch": bs,
               "lr": lr, "seed": seed, "train_n": len(Xtr), "val_n": len(Xva)}


@torch.no_grad()
def test_loss(m, X, Y) -> float:
    return float(sum(F.cross_entropy(m.logits(X[i:i + 500]), Y[i:i + 500], reduction="sum") for i in range(0, len(X), 500)) / Y.numel())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noskip", action="store_true")
    a = ap.parse_args()
    m, info = fit(skip=not a.noskip, epochs=a.epochs, seed=a.seed)
    sets = {}
    for name in ("test", "big", "hollow", "dark"):
        X, Y = load(name)
        sets[name] = {**scores(predict(m, X), Y), "n": len(X), **({"loss": test_loss(m, X, Y)} if name == "test" else {})}
    tag = "" if not a.noskip else "_noskip"
    torch.save(m.state_dict(), HERE / f"model{'_final' if not a.noskip else '_noskip'}.pt")
    report = {"classes": CLASSES, "skip": not a.noskip, "params": n_params(m), **info, "sets": sets,
              "hardware": "RTX 5090 Laptop GPU (training); CPU float32, TF32 off (evaluation)"}
    with open(HERE / f"train_metrics{tag}.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sets"}))
    for k, v in sets.items():
        print(k, f"mIoU {v['miou']:.4f}", "IoU", [round(x, 4) for x in v["iou"]], f"acc {v['acc']:.4f} edge {v['edgeAcc']:.4f}")
