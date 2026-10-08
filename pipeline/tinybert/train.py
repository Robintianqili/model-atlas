"""tinybert: train TinyBERT with masked-language modelling on TinyStories (GPU; resumable, in chunks under 8 minutes).

    npm run atlas -- py pipeline/tinybert/train.py --run v1 --steps 40000
    (run the same command again until it prints "done"; then model_final.pt and train_metrics.json are written)

Batches come from data.py (BERT layout and 80/10/10 masking) and are seeded by (seed, step), so a resumed run sees the
same batches as an uninterrupted one. AdamW, linear warm-up then cosine decay to 10% (TinyGPT's schedule), weight
decay on matrices only, gradient clipping at 1. TF32 is allowed for training; export and evaluation turn it off.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data
from model import TinyBERT, n_params

HERE = Path(__file__).resolve().parent


def lr_at(step: int, a) -> float:
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    prog = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


def to_dev(batch, dev):
    return [torch.from_numpy(x).to(dev, non_blocking=True) for x in batch]


def mlm_loss(model, batch):
    inputs, seg, pad, chosen, ids = batch
    h = model.hidden(inputs, seg, pad)
    logits = h[chosen] @ model.tok.weight.t()
    target = ids[chosen]
    return F.cross_entropy(logits, target), logits, target


@torch.no_grad()
def evaluate(model, batches, dev):
    model.eval()
    tot, n, right, top5 = 0.0, 0, 0, 0
    for b in batches:
        loss, logits, target = mlm_loss(model, to_dev(b, dev))
        k = target.numel()
        tot += loss.item() * k
        n += k
        right += (logits.argmax(-1) == target).sum().item()
        top5 += (logits.topk(5, -1).indices == target[:, None]).any(-1).sum().item()
    model.train()
    return {"loss": tot / n, "acc": right / n, "top5": top5 / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    train, val = data.Batches("train"), data.Batches("val")
    ev = [val.mlm(np.random.default_rng([1234, i]), a.batch) for i in range(a.eval_batches)]
    print(f"data: train {len(train.stream)} tokens, {len(train.starts)} sentence starts; val {len(val.stream)} ({time.time() - t0:.1f}s)")

    torch.manual_seed(a.seed)
    model = TinyBERT().to(dev)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not n.startswith(("pos", "seg"))]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and not n.startswith(("pos", "seg")))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-8, fused=True)
    step, curve = 0, []
    state = run / "state.pt"
    if state.exists():
        st = torch.load(state, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, curve = st["step"], st["curve"]
        print(f"resumed at step {step}")
    print(f"{n_params(model)} params; run {a.run}; steps {a.steps}")

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "curve": curve, "args": vars(a)}, state)
        (run / "curve.json").write_text(json.dumps(curve, indent=1), encoding="utf-8")

    start = time.time()
    ema = None
    while step < a.steps:
        if step % a.eval_every == 0 and (not curve or curve[-1]["step"] != step):
            e = evaluate(model, ev, dev)
            curve.append({"step": step, "val_loss": e["loss"], "val_acc": e["acc"], "val_top5": e["top5"], "train_loss": ema, "lr": lr_at(step, a)})
            print(f"step {step}: val loss {e['loss']:.4f} acc {e['acc']:.4f} top5 {e['top5']:.4f} (train ema {ema}) {time.time() - start:.0f}s", flush=True)
        if (time.time() - start) / 60 > a.max_minutes:
            save()
            print(f"stopped at step {step} (time limit); run the same command again to resume")
            return
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        b = to_dev(train.mlm(np.random.default_rng([a.seed, step]), a.batch), dev)
        loss, _, _ = mlm_loss(model, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        step += 1
    e = evaluate(model, ev, dev)
    curve.append({"step": step, "val_loss": e["loss"], "val_acc": e["acc"], "val_top5": e["top5"], "train_loss": ema, "lr": lr_at(step, a)})
    save()
    final = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(final, run / "final.pt")
    torch.save(final, HERE / "model_final.pt")
    metrics = {"run": a.run, "steps": step, "batch": a.batch, "lr": a.lr, "warmup": a.warmup, "weight_decay": a.wd,
               "seed": a.seed, "optimizer": "AdamW betas (0.9, 0.98), cosine to 10%, clip 1.0", "params": n_params(model),
               "val": e, "train_loss_ema": ema, "examples_seen": step * a.batch,
               "masking": {"rate": data.RATE, "mask": data.P_MASK, "random": data.P_RANDOM, "keep": 1 - data.P_MASK - data.P_RANDOM,
                           "atLeastOne": True}}
    (HERE / "train_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"done: step {step}, val loss {e['loss']:.4f}, acc {e['acc']:.4f}, top5 {e['top5']:.4f}")


if __name__ == "__main__":
    main()
