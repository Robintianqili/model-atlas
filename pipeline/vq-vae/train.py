"""vq-vae: train the tiny VQ-VAE on Quick, Draw! (GPU if available; fixed seed).

    npm run atlas -- py pipeline/vq-vae/train.py [--epochs 30] [--seed 0] [--out model_final.pt]
    npm run atlas -- py pipeline/vq-vae/train.py --codes 128 --out model_k128.pt  # the limits step's comparison (not shipped)
    npm run atlas -- py pipeline/vq-vae/train.py --plain --out model_plain.pt     # the textbook codebook loss (notes only)

Data: ak.data.quickdraw(CLASSES, split): TinyCNN's split (12,000 / 1,500 / 1,500 drawings per class, seed 20261005)
and its pixel/255 scaling; no augmentation and no download. Loss (model.losses): BCE averaged over the 784 pixels +
0.25 x commitment term (mean over the 128 latent numbers), mean over the batch; the straight-through estimator carries the
decoder's gradient past the lookup. The codebook starts as the z_e of 32 random training patches of the untrained
encoder and then follows the patches by an exponential moving average (decay 0.99, Laplace smoothing 1e-5): each code
moves towards the average z_e of the patches that snapped to it. Nothing restarts unused codes, so the usage the lesson
shows is what plain training gives. Adam lr 1e-3 (encoder and decoder) with a cosine decay to 0, batch 128. The checkpoint with the
best validation reconstruction is kept, evaluated on the CPU in float32 (TF32 off) on the test split, saved to --out; its numbers
go to train_metrics.json.
--plain trains the textbook variant for comparison: the codebook starts uniform in (-1/K, 1/K) and is a parameter moved by
the codebook loss ||sg[z_e] - e||^2 (mean over the 128 numbers) with Adam, no EMA; metrics in model_plain_metrics.json.
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
from model import BETA, CLASSES, DECAY, G, K, VQVAE, losses, n_params  # noqa: E402


def tensors(classes, split: str):
    X, y, _ = ak.data.quickdraw(classes, split)
    return torch.tensor(X).float().unsqueeze(1) / 255, torch.tensor(y)


@torch.no_grad()
def evaluate(m, X, dev, bs=4096):
    """Mean reconstruction BCE and commitment term per drawing, and how often each code is used."""
    m.eval()
    rec = cm = 0.0
    use = torch.zeros(m.codebook.shape[0], dtype=torch.long)
    for i in range(0, len(X), bs):
        r, k, _, idx = losses(m, X[i:i + bs].to(dev))
        rec += float(r.sum()); cm += float(k.sum())
        use += torch.bincount(idx.flatten().cpu(), minlength=len(use))
    n = len(X)
    return {"rec": rec / n, "commit": cm / n, "loss": (rec + BETA * cm) / n, "used": int((use > 0).sum()), "use": use.tolist()}


def fit(classes, *, epochs=30, seed=0, lr=1e-3, bs=128, plain=False, codes=K, log=print):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, _ = tensors(classes, "train")
    Xva, _ = tensors(classes, "val")
    Xg = Xtr.to(dev)
    m = VQVAE(k=codes).to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad():
        if plain:  # the textbook start: small uniform numbers
            m.codebook.uniform_(-1 / codes, 1 / codes, generator=gen)
        else:  # data-dependent start: the z_e of K random patches
            ze = m.encode(Xg[torch.randperm(len(Xg), device=dev, generator=gen)[:512]])
            flat = ze.permute(0, 2, 3, 1).reshape(-1, ze.shape[1])
            m.codebook.copy_(flat[torch.randperm(len(flat), device=dev, generator=gen)[:codes]])
    m.codebook.requires_grad_(plain)
    n_ema = torch.ones(codes, device=dev)
    s_ema = m.codebook.detach().clone()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    steps = epochs * math.ceil(len(Xg) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    best, best_state, best_epoch = float("inf"), None, -1
    t0 = time.time()
    for epoch in range(epochs):
        m.train()
        perm = torch.randperm(len(Xg), device=dev, generator=gen)
        for i in range(0, len(perm), bs):
            r, k, ze, idx = losses(m, Xg[perm[i:i + bs]])
            loss = (r + BETA * k).mean()
            if plain:  # the codebook loss: each picked code towards its patch (the commitment term, gradient on e)
                loss = loss + ((m.quantize(ze.detach())[0] - ze.detach()) ** 2).flatten(1).mean(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if plain:
                continue
            with torch.no_grad():  # EMA: each code towards the mean z_e of its patches
                flat = ze.detach().permute(0, 2, 3, 1).reshape(-1, ze.shape[1])
                onehot = torch.nn.functional.one_hot(idx.flatten(), codes).float()
                n_ema.mul_(DECAY).add_(onehot.sum(0), alpha=1 - DECAY)
                s_ema.mul_(DECAY).add_(onehot.t() @ flat, alpha=1 - DECAY)
                tot = n_ema.sum()
                m.codebook.copy_(s_ema / ((n_ema + 1e-5) / (tot + codes * 1e-5) * tot)[:, None])
        v = evaluate(m, Xva, dev)
        if v["rec"] < best:  # the best redrawings (the commitment term only measures how far z_e sits from its code)
            best, best_state, best_epoch = v["rec"], copy.deepcopy(m.state_dict()), epoch + 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"epoch {epoch + 1}: val loss {v['loss']:.4f} (rec {v['rec']:.4f}, commit {v['commit']:.4f}) codes used {v['used']} {time.time() - t0:.0f}s")
    m.load_state_dict(best_state)
    return m.cpu().eval(), {"best_epoch": best_epoch, "epochs": epochs, "seconds": round(time.time() - t0, 1),
                            "steps": steps, "batch": bs, "lr": lr, "seed": seed, "beta": BETA, "decay": None if plain else DECAY, "plain": plain}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="model_final.pt")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--codes", type=int, default=K)
    a = ap.parse_args()
    m, info = fit(CLASSES, epochs=a.epochs, seed=a.seed, plain=a.plain, codes=a.codes)
    Xte, _ = tensors(CLASSES, "test")
    test = evaluate(m, Xte, "cpu")
    torch.save(m.state_dict(), HERE / a.out)
    report = {"classes": CLASSES, "grid": G, "codes": a.codes, "params": n_params(m), **info, "test": test, "test_n": len(Xte),
              "hardware": "RTX 5090 Laptop GPU (training); CPU float32, TF32 off (evaluation)"}
    with open(HERE / ("train_metrics.json" if a.out == "model_final.pt" else Path(a.out).stem + "_metrics.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "test"}), json.dumps({k: v for k, v in test.items() if k != "use"}))
