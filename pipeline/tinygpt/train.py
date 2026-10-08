"""Train TinyGPT on the encoded TinyStories tokens (resumable, chunked so each command stays short).

Usage:
  python train.py --name d64L4 --d 64 --layers 4 --steps 6000 --max-minutes 8
Run the same command again to resume a run that stopped at the time limit.
Outputs in pipeline/tinygpt/runs/<name>/: state.pt (resume), curve.json, replay/step_XXXXXX.pt, final.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "tinystories")


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def eval_steps(total, every):
    s = {0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000}
    s |= set(range(0, total + 1, every))
    s.add(total)
    return {x for x in s if x <= total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--replay-steps", type=str, default="")
    ap.add_argument("--max-minutes", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    run = os.path.join(HERE, "runs", args.name)
    os.makedirs(os.path.join(run, "replay"), exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train = torch.from_numpy(np.fromfile(os.path.join(DATA, "train.bin"), dtype=np.uint16).astype(np.int32)).to(dev)
    val = torch.from_numpy(np.fromfile(os.path.join(DATA, "val.bin"), dtype=np.uint16).astype(np.int32)).to(dev)
    T = args.context
    ar = torch.arange(T + 1, device=dev)

    def batch_from(data, starts):
        w = data[starts[:, None] + ar[None]].long()
        return w[:, :-1], w[:, 1:]

    # fixed evaluation windows (same for every eval and every run)
    g = torch.Generator(device="cpu").manual_seed(1234)
    ev_val = torch.randint(0, len(val) - T - 1, (args.eval_batches, args.batch), generator=g).to(dev)
    ev_tr = torch.randint(0, len(train) - T - 1, (args.eval_batches, args.batch), generator=g).to(dev)

    cfg = GPTConfig(vocab_size=args.vocab, context=T, n_layer=args.layers, n_head=args.heads, d_model=args.d)
    torch.manual_seed(args.seed)
    model = GPT(cfg).to(dev)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    step, curve, tokens_seen = 0, [], 0
    state_path = os.path.join(run, "state.pt")
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        step, curve, tokens_seen = st["step"], st["curve"], st["tokens_seen"]
        torch.set_rng_state(st["rng_cpu"].cpu())
        torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        print(f"resumed {args.name} at step {step}")
    else:
        print(f"new run {args.name}: {model.num_params()} params, {model.num_params() * 4} bytes fp32")
    replay_steps = {int(s) for s in args.replay_steps.split(",") if s.strip()}
    evals = eval_steps(args.steps, args.eval_every) | replay_steps

    @torch.no_grad()
    def evaluate(windows, data):
        model.eval()
        tot = 0.0
        for starts in windows:
            x, y = batch_from(data, starts)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            tot += loss.item()
        model.train()
        return tot / len(windows)

    def save_state():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "curve": curve,
                    "tokens_seen": tokens_seen, "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                    "args": vars(args), "cfg": cfg.to_dict()}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        with open(os.path.join(run, "curve.json"), "w") as f:
            json.dump({"args": vars(args), "cfg": cfg.to_dict(), "params": model.num_params(), "curve": curve}, f)

    t_start = time.time()
    step0 = step
    model.train()
    last_print = time.time()
    while step <= args.steps:
        if step in replay_steps and not os.path.exists(os.path.join(run, "replay", f"step_{step:06d}.pt")):
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "step": step},
                       os.path.join(run, "replay", f"step_{step:06d}.pt"))
        if step in evals and not any(c["step"] == step for c in curve):
            vl, tl = evaluate(ev_val, val), evaluate(ev_tr, train)
            curve.append({"step": step, "tokens": tokens_seen, "train_loss": tl, "val_loss": vl,
                          "lr": lr_at(step, args), "time": time.time() - t_start})
            print(f"step {step:6d} | train {tl:.4f} | val {vl:.4f} | lr {lr_at(step, args):.2e} | "
                  f"{time.time() - t_start:.0f}s", flush=True)
        if step == args.steps:
            break
        if (time.time() - t_start) / 60 > args.max_minutes:
            print(f"time limit reached at step {step}; run again to resume")
            save_state()
            return
        lr = lr_at(step, args)
        for gp in opt.param_groups:
            gp["lr"] = lr
        starts = torch.randint(0, len(train) - T - 1, (args.batch,), device=dev)
        x, y = batch_from(train, starts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tokens_seen += args.batch * T
        if time.time() - last_print > 30:
            print(f"  step {step} loss {loss.item():.4f} ({(time.time() - t_start) / max(1, step - step0) * 1000:.1f} ms/step this chunk)", flush=True)
            last_print = time.time()
    save_state()
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "step": step, "args": vars(args)},
               os.path.join(run, "final.pt"))
    print(f"done: {args.name} step {step}, final val {curve[-1]['val_loss']:.4f}, train {curve[-1]['train_loss']:.4f}")


if __name__ == "__main__":
    main()
