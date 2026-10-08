"""ddpm-unet: train the U-Net noise predictor on Quick, Draw! (GPU if available; fixed seed).

    npm run atlas -- py pipeline/ddpm-unet/train.py [--epochs 600] [--seed 0] [--classes star,umbrella,house]
                                                     [--ch 16,32,64] [--out model_final.pt]

Data: ak.data.quickdraw(classes, 'train'): TinyCNN's split (12,000 drawings per class) and its pixel/255 scaling, stretched
to -1..1 (2 x - 1, model.py); no augmentation, no download. Each step: a batch of 256 drawings x_0, a timestep t uniform in 1..T, noise eps ~ N(0, 1),
x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps (x_0 stretched), loss = mean squared error between UNet(x_t, t) and eps (DDPM's simple
loss). Adam lr 1e-3 with a cosine decay to 0; an exponential moving average (EMA, decay 0.999) of the weights is what we
keep (as DDPM does). The EMA weights are evaluated on the CPU in float32 (TF32 off): the loss on the validation split at
fixed (t, eps) (seed 0), then saved to --out with train_metrics.json (train_metrics_<stem>.json for another --out).
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
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CH, CLASSES, T, UNet, macs, n_params, schedule  # noqa: E402


def drawings(classes, split: str) -> torch.Tensor:
    X, _, _ = ak.data.quickdraw(classes, split)
    return torch.tensor(X).float().unsqueeze(1) / 255


@torch.no_grad()
def val_loss(m, X, bs=2048) -> float:
    """mean squared noise-prediction error on X with t and eps fixed by seed 0 (CPU)."""
    m.eval()
    g = torch.Generator().manual_seed(0)
    ab = torch.tensor(schedule()["abar"], dtype=torch.float32)
    tot = 0.0
    for i in range(0, len(X), bs):
        x0 = X[i:i + bs]
        t = torch.randint(1, T + 1, (len(x0),), generator=g)
        eps = torch.randn(x0.shape, generator=g)
        a = ab[t - 1][:, None, None, None]
        tot += float(F.mse_loss(m(a.sqrt() * x0 + (1 - a).sqrt() * eps, t.float()), eps, reduction="sum"))
    return tot / X.numel()


def fit(classes, *, ch=CH, epochs=200, seed=0, pm1=False, lr=1e-3, bs=256, ema=0.999, log=lambda m: print(m, flush=True)):
    torch.manual_seed(seed)
    # training only: the shipped numbers are evaluated (and exported) on the CPU in float32 with TF32 off
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xg = drawings(classes, "train").to(dev)
    if pm1:
        Xg = Xg * 2 - 1
    m = UNet(ch).to(dev)
    avg = copy.deepcopy(m).eval()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    steps = epochs * math.ceil(len(Xg) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    ab = torch.tensor(schedule()["abar"], dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    t0 = time.time()
    for epoch in range(epochs):
        m.train()
        perm = torch.randperm(len(Xg), device=dev, generator=gen)
        run = torch.zeros((), device=dev)
        for i in range(0, len(perm), bs):
            x0 = Xg[perm[i:i + bs]]
            t = torch.randint(1, T + 1, (len(x0),), device=dev, generator=gen)
            eps = torch.randn(x0.shape, device=dev, generator=gen)
            a = ab[t - 1][:, None, None, None]
            loss = F.mse_loss(m(a.sqrt() * x0 + (1 - a).sqrt() * eps, t.float()), eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for p, q in zip(avg.parameters(), m.parameters()):
                    p.mul_(ema).add_(q, alpha=1 - ema)
            run += loss.detach()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            log(f"epoch {epoch + 1}: train loss {float(run) / math.ceil(len(perm) / bs):.4f} {time.time() - t0:.0f}s")
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    return avg.cpu().eval(), {"epochs": epochs, "steps": steps, "batch": bs, "lr": lr, "ema": ema, "seed": seed,
                               "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--ch", default=",".join(map(str, CH)))
    ap.add_argument("--out", default="model_final.pt")
    ap.add_argument("--zero-one", action="store_true", help="experiment: train on pixels 0..1 (the shipped model uses -1..1)")
    a = ap.parse_args()
    classes = a.classes.split(",")
    ch = tuple(int(c) for c in a.ch.split(","))
    a.pm1 = not a.zero_one
    m, info = fit(classes, ch=ch, epochs=a.epochs, seed=a.seed, pm1=a.pm1)
    torch.save(m.state_dict(), HERE / a.out)
    report = {"classes": classes, "ch": list(ch), "T": T, "params": n_params(m), "macs": macs(ch), **info,
              "pm1": a.pm1, "val_loss": val_loss(m, drawings(classes, "val") * (2 if a.pm1 else 1) - (1 if a.pm1 else 0)), "val_n": 1500 * len(classes),
              "hardware": "RTX 5090 Laptop GPU (training); CPU float32, TF32 off (evaluation)"}
    name = "train_metrics.json" if a.out == "model_final.pt" else f"train_metrics_{Path(a.out).stem}.json"
    (HERE / name).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report))
