"""tinymamba: train the tiny selective state-space model on the selective-copy task (data.py = task.ts).

    npm run atlas -- py pipeline/tinymamba/train.py                 # repeat until it prints "done" (resumable chunks)
    npm run atlas -- py pipeline/tinymamba/train.py --fresh         # start again

Data: the first 200,000 examples of seed 1 (training) and 5,000 of seed 2 (validation), exactly as task.ts batch()
draws them. Loss: cross-entropy at the '?' positions only. AdamW (weight decay 0.05 on the weight matrices), lr 3e-3 with a cosine decay to 3e-4, batch 256,
gradient clipping at 1; the GPU when there is one (TF32 off: atlas_kit switches it off on import). Batches are seeded
by (seed, step), so a resumed run sees the same batches. Writes runs/<run>/state.pt while it trains, then
model_final.pt and train_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
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
from data import batch  # noqa: E402
from model import CONFIG, TinyMamba  # noqa: E402

T_PAD = 20  # 16 body tokens + 4 '?'


def tensors(rows, T=T_PAD):
    X = torch.zeros(len(rows), T, dtype=torch.long)
    Y = torch.full((len(rows), T), -1, dtype=torch.long)
    for i, (x, y) in enumerate(rows):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X, Y


@torch.no_grad()
def evaluate(m, X, Y):
    lo = m(X)
    mk = Y >= 0
    pred = lo.argmax(-1)
    loss = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), Y.reshape(-1), ignore_index=-1).item()
    acc = (pred[mk] == Y[mk]).float().mean().item()
    exact = ((pred == Y) | ~mk).all(1).float().mean().item()
    return loss, acc, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wd", type=float, default=0.05, help="weight decay on the weight matrices (not A_log)")
    ap.add_argument("--chunk", type=float, default=480, help="seconds per call (resumable)")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--N", type=int, default=CONFIG["N"], help="state slots per channel")
    ap.add_argument("--D", type=int, default=CONFIG["D"], help="channels")
    ap.add_argument("--publish", action="store_true", help="also write model_final.pt and train_metrics.json next to this file")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, Y = tensors(batch(1, 200000))
    Xv, Yv = tensors(batch(2, 5000))
    X, Y, Xv, Yv = X.to(dev), Y.to(dev), Xv.to(dev), Yv.to(dev)
    print(f"data {time.time() - t0:.1f}s on {dev}", flush=True)
    torch.manual_seed(a.seed)
    cfg = {**CONFIG, "N": a.N, "D": a.D}
    m = TinyMamba(**cfg).to(dev)
    decay = [p for n, p in m.named_parameters() if p.dim() == 2 and "A_log" not in n]
    rest = [p for n, p in m.named_parameters() if not (p.dim() == 2 and "A_log" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": rest, "weight_decay": 0.0}], lr=a.lr)
    step, log = 0, []
    sp = run / "state.pt"
    if sp.is_file() and not a.fresh:
        st = torch.load(sp, map_location=dev)
        m.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, log = st["step"], st["log"]
        print(f"resumed at step {step}", flush=True)
    t0 = time.time()
    while step < a.steps and time.time() - t0 < a.chunk:
        lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        gen = torch.Generator(device="cpu").manual_seed(a.seed * 1_000_003 + step)
        idx = torch.randint(0, X.shape[0], (a.batch,), generator=gen).to(dev)
        lo = m(X[idx])
        loss = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), Y[idx].reshape(-1), ignore_index=-1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 1000 == 0 or step == a.steps:
            vl, va, ve = evaluate(m.eval(), Xv, Yv)
            m.train()
            log.append({"step": step, "loss": loss.item(), "valLoss": vl, "valAcc": va, "valExact": ve, "lr": lr})
            print(f"step {step} loss {loss.item():.4f} val {vl:.4f} acc {va:.4f} exact {ve:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "step": step, "log": log}, sp)
    if step < a.steps:
        print(f"chunk ended at step {step} of {a.steps}: run again to continue", flush=True)
        return
    m = m.cpu().eval()
    torch.save(m.state_dict(), run / "model_final.pt")
    vl, va, ve = evaluate(m, Xv.cpu(), Yv.cpu())
    metrics = {"config": cfg, "params": sum(p.numel() for p in m.parameters()), "steps": a.steps, "batch": a.batch,
               "lr": a.lr, "wd": a.wd, "seed": a.seed, "train": {"examples": int(X.shape[0]), "seed": 1}, "val": {"examples": int(Xv.shape[0]), "seed": 2},
               "valLoss": vl, "valAcc": va, "valExact": ve, "log": log}
    (run / "train_metrics.json").write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    if a.publish:
        torch.save(m.state_dict(), HERE / "model_final.pt")
        (HERE / "train_metrics.json").write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    print(f"done: val loss {vl:.4f} acc {va:.4f} exact {ve:.4f}", flush=True)


if __name__ == "__main__":
    main()
