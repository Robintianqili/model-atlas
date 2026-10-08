"""vae-2d: train the 2-D VAE on Quick, Draw! (GPU if available; fixed seed).

    npm run atlas -- py pipeline/vae-2d/train.py [--epochs 80] [--seed 0] [--zdim 2] [--out model_final.pt]

Data: ak.data.quickdraw(CLASSES, split): TinyCNN's split (12,000 / 1,500 / 1,500 drawings per class, seed 20261005)
and its pixel/255 scaling; no augmentation and no download. Loss: the standard VAE loss (model.losses: BCE summed over
the 784 pixels + KL summed over the latent numbers, mean over the batch). Adam lr 1e-3 with a cosine decay to 0, batch
128. The checkpoint with the best validation loss is kept, evaluated on the CPU in float32 (TF32 off) on the test split
and saved to --out; its numbers go to train_metrics.json (train_metrics_z<k>.json for another --zdim, e.g. the 16-D
comparison the limits step quotes).
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CLASSES, VAE, losses, n_params  # noqa: E402


def tensors(classes, split: str):
    X, y, _ = ak.data.quickdraw(classes, split)
    return torch.tensor(X).float().unsqueeze(1) / 255, torch.tensor(y)


@torch.no_grad()
def evaluate(m, X, dev, bs=4096):
    """Mean reconstruction loss and KL per drawing, using z = mu (no noise) and z = mu + sigma*eps (seed 0)."""
    m.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    rec = kl = rec_s = 0.0
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(dev)
        mu, lv = m.encode(x)
        r, k = losses(m.decode_logits(mu), x, mu, lv)
        eps = torch.randn(mu.shape, generator=g).to(dev)
        rs, _ = losses(m.decode_logits(mu + torch.exp(0.5 * lv) * eps), x, mu, lv)
        rec += float(r.sum()); kl += float(k.sum()); rec_s += float(rs.sum())
    n = len(X)
    return {"rec_mu": rec / n, "kl": kl / n, "rec_sampled": rec_s / n, "loss": (rec_s + kl) / n}


def fit(classes, *, zdim=2, epochs=80, seed=0, lr=1e-3, bs=128, log=print):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, _ = tensors(classes, "train")
    Xva, _ = tensors(classes, "val")
    Xg = Xtr.to(dev)
    m = VAE(zdim=zdim).to(dev)
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
            mu, lv = m.encode(x)
            z = mu + torch.exp(0.5 * lv) * torch.randn(mu.shape, device=dev, generator=gen)
            r, k = losses(m.decode_logits(z), x, mu, lv)
            loss = (r + k).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        v = evaluate(m, Xva, dev)
        if v["loss"] < best:
            best, best_state, best_epoch = v["loss"], copy.deepcopy(m.state_dict()), epoch + 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log(f"epoch {epoch + 1}: val loss {v['loss']:.2f} (rec {v['rec_sampled']:.2f}, kl {v['kl']:.2f}) {time.time() - t0:.0f}s")
    m.load_state_dict(best_state)
    return m.cpu().eval(), {"best_epoch": best_epoch, "epochs": epochs, "seconds": round(time.time() - t0, 1),
                            "steps": steps, "batch": bs, "lr": lr, "seed": seed}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zdim", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m, info = fit(CLASSES, zdim=a.zdim, epochs=a.epochs, seed=a.seed)
    Xte, _ = tensors(CLASSES, "test")
    test = evaluate(m, Xte, "cpu")
    out = HERE / (a.out or ("model_final.pt" if a.zdim == 2 else f"model_z{a.zdim}.pt"))
    torch.save(m.state_dict(), out)
    report = {"classes": CLASSES, "zdim": a.zdim, "params": n_params(m), **info, "test": test, "test_n": len(Xte),
              "hardware": "RTX 5090 Laptop GPU (training); CPU float32, TF32 off (evaluation)"}
    name = "train_metrics.json" if a.zdim == 2 else f"train_metrics_z{a.zdim}.json"
    with open(HERE / name, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
