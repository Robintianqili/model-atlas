"""tinyvit: train TinyViT on Quick, Draw! (GPU if available; fixed seed; resumable).

    npm run atlas -- py pipeline/tinyvit/train.py --run v1 --epochs 80 [--chunk 30] [--seed 0]

Data: ak.data.quickdraw(CLASSES, split): TinyCNN's 10 classes, its split (12,000 / 1,500 / 1,500 per class, seed
20261005) and its pixel/255 scaling. Augmentation: TinyCNN's own `augment` (pipeline/tinycnn/common.py: shift, scale,
rotation, stroke-thickness jitter), so both models saw the same kind of training drawings.

Recipe: AdamW (lr 2e-3, betas 0.9/0.999, weight decay 0.05 on matrices only), batch 512, 5 warm-up epochs then a
cosine decay to 0, label smoothing 0.1. Each call trains at most --chunk epochs and saves runs/<run>/last.pt, so a
long run is split into calls under 9 minutes (CONTRACTS rule 4); call it again to resume. The checkpoint with the best
validation accuracy is runs/<run>/best.pt. When the run is complete, the best checkpoint is evaluated on the CPU in
float32 (TF32 off) on the test split, copied to model_final.pt and its numbers are written to train_metrics.json.
"""
from __future__ import annotations

import argparse
import importlib.util
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

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CLASSES, TinyViT, n_params  # noqa: E402


def tinycnn_augment():
    """TinyCNN's GPU augmentation (pipeline/tinycnn/common.py, read only)."""
    p = ak.root_path("pipeline", "tinycnn", "common.py")
    spec = importlib.util.spec_from_file_location("tinyvit_tinycnn_common", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.augment


def tensors(split: str):
    X, y, _ = ak.data.quickdraw(CLASSES, split)
    return torch.tensor(X).float().unsqueeze(1) / 255, torch.tensor(y)


@torch.no_grad()
def accuracy(m, X, y, dev, bs=2048):
    m.eval()
    ok = 0
    for i in range(0, len(X), bs):
        ok += int((m(X[i:i + bs].to(dev)).argmax(1).cpu() == y[i:i + bs]).sum())
    return ok / len(X)


def param_groups(m, wd):
    decay, no = [], []
    for n, p in m.named_parameters():
        (decay if p.dim() >= 2 and n not in ("pos",) else no).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no, "weight_decay": 0.0}]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="v1")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--chunk", type=int, default=1000, help="train at most this many epochs in this call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--no-aug", action="store_true")
    a = ap.parse_args()

    out = HERE / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"   # training may use the GPU; export is CPU only (AK102)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    Xtr, ytr = tensors("train")
    Xva, yva = tensors("val")
    Xtr_d, ytr_d = Xtr.to(dev), ytr.to(dev)
    augment = tinycnn_augment()

    m = TinyViT().to(dev)
    opt = torch.optim.AdamW(param_groups(m, a.wd), lr=a.lr, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(Xtr) / a.bs)
    total = a.epochs * steps_per_epoch
    warm = a.warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    gen = torch.Generator(device=dev).manual_seed(a.seed + 1)
    state = {"epoch": 0, "best_val": -1.0, "best_epoch": -1, "history": []}
    last = out / "last.pt"
    if last.is_file():
        ck = torch.load(last, map_location=dev, weights_only=False)
        m.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu() if hasattr(ck["gen"], "cpu") else ck["gen"])
        torch.set_rng_state(ck["cpu_rng"])
        state = ck["state"]
        print(f"resumed {a.run} at epoch {state['epoch']}")
    print(f"TinyViT {n_params(m)} params on {dev}; {len(Xtr)} train, {len(Xva)} val; {a.epochs} epochs x {steps_per_epoch} steps")

    done_here = 0
    while state["epoch"] < a.epochs and done_here < a.chunk:
        t0 = time.time()
        m.train()
        perm = torch.randperm(len(Xtr), device=dev, generator=gen)
        tot, n = 0.0, 0
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            x = Xtr_d[idx]
            if not a.no_aug:
                x = augment(x, gen)
            loss = F.cross_entropy(m(x), ytr_d[idx], label_smoothing=a.smooth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
            n += len(idx)
        state["epoch"] += 1
        done_here += 1
        va = accuracy(m, Xva, yva, dev)
        state["history"].append({"epoch": state["epoch"], "loss": tot / n, "val_acc": va})
        if va > state["best_val"]:
            state["best_val"], state["best_epoch"] = va, state["epoch"]
            torch.save(m.state_dict(), out / "best.pt")
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "gen": gen.get_state(),
                    "cpu_rng": torch.get_rng_state(), "state": state, "args": vars(a)}, last)
        print(f"epoch {state['epoch']:3d}  loss {tot / n:.4f}  val {va:.4f}  best {state['best_val']:.4f}@{state['best_epoch']}"
              f"  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.1f}s", flush=True)

    if state["epoch"] < a.epochs:
        print(f"chunk done at epoch {state['epoch']} / {a.epochs}: call again to resume")
        return

    # final: the best-validation checkpoint, evaluated on the CPU in float32 (as the page runs it)
    best = TinyViT()
    best.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
    best.eval()
    Xte, yte = tensors("test")
    with torch.no_grad():
        pred = torch.cat([best(Xte[i:i + 2048]).argmax(1) for i in range(0, len(Xte), 2048)])
    test_acc = float((pred == yte).float().mean())
    val_acc = accuracy(best, Xva, yva, "cpu")
    conf = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    for t, p in zip(yte.tolist(), pred.tolist()):
        conf[t, p] += 1
    per = {c: round(float(conf[k, k] / conf[k].sum()), 4) for k, c in enumerate(CLASSES)}
    torch.save(best.state_dict(), HERE / "model_final.pt")
    report = {"run": a.run, "classes": CLASSES, "params": n_params(best), "epochs": a.epochs, "best_epoch": state["best_epoch"],
              "seed": a.seed, "steps": a.epochs * steps_per_epoch, "batch": a.bs, "lr": a.lr, "weight_decay": a.wd,
              "label_smoothing": a.smooth, "augment": not a.no_aug,
              "test_acc": round(test_acc, 4), "val_acc": round(val_acc, 4), "test_n": len(yte),
              "per_class_test_acc": per, "confusion": conf.tolist(), "evaluated": "CPU float32, TF32 off",
              "history": state["history"]}
    (out / "metrics.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    with open(HERE / "train_metrics.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("run", "params", "best_epoch", "test_acc", "val_acc", "test_n")}))


if __name__ == "__main__":
    main()
