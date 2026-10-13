"""tinymoe: train the MoE model and its two comparisons on TinyStories characters (fixed seed, resumable).

    npm run atlas -- py pipeline/tinymoe/train.py --variant moe      # the page's model (load-balancing loss, alpha 0.01)
    npm run atlas -- py pipeline/tinymoe/train.py --variant noaux    # the same MoE trained without the balancing loss
    npm run atlas -- py pipeline/tinymoe/train.py --variant dense    # the dense twin: one MLP 32 -> 128 -> 32

Data: ak.data.tinystories_chars (the files downloaded for TinyGPT; no new download): the first --chars characters of
the train head for training and the first 2,000,000 characters of TinyStoriesV2-GPT4-valid.txt for validation, both
normalized by model.normalize. Random windows of 64 characters (+1 for the targets), batch 256, AdamW (lr 3e-3, weight
decay 0.01) with 300 warm-up steps and a cosine decay to 0, gradient clip 1.0; the same seed and the same windows for
every variant. Loss = cross-entropy + alpha * (sum over the two blocks of the load-balancing loss), alpha 0.01 for
'moe' (Switch Transformer's value) and 0 for 'noaux'. GPU for training; the validation loss is measured on the CPU in
float32 with TF32 off (atlas_kit switches it off on import). A call stops after --chunk steps and resumes from
runs/<variant>/last.pt (each call < 9 min). Writes runs/<variant>/final.pt and the variant's entry in
train_metrics.json; 'moe' also writes model_final.pt (what export.py exports).
"""
from __future__ import annotations

import argparse
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
from model import CTX, V, TinyMoE, active_params, balance_loss, encode, n_params  # noqa: E402

B = 256
ALPHA = {"moe": 0.01, "noaux": 0.0, "dense": 0.0}


def load(chars: int):
    tr = torch.tensor(encode(ak.data.tinystories_chars(n_chars=chars)), dtype=torch.long)
    va = torch.tensor(encode(ak.data.tinystories_chars("TinyStoriesV2-GPT4-valid.txt", n_chars=2_000_000)), dtype=torch.long)
    return tr, va


def windows(ids: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The first n non-overlapping windows of CTX characters and their targets, [n, CTX] each."""
    n = min(n, (len(ids) - 1) // CTX)
    return ids[: n * CTX].view(n, CTX), ids[1 : n * CTX + 1].view(n, CTX)


@torch.no_grad()
def evaluate(m: TinyMoE, ids: torch.Tensor, n_windows: int = 4000) -> float:
    """Mean cross-entropy (nats per character) over non-overlapping windows of CTX characters, CPU float32."""
    m = m.cpu().eval()
    x, y = windows(ids, n_windows)
    tot = 0.0
    for s in range(0, len(x), 500):
        tot += F.cross_entropy(m(x[s:s + 500]).reshape(-1, V), y[s:s + 500].reshape(-1), reduction="sum").item()
    return tot / y.numel()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=sorted(ALPHA), default="moe")
    ap.add_argument("--chars", type=int, default=40_000_000)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--chunk", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = HERE / "runs" / a.variant
    run.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tr, va = load(a.chars)
    m = TinyMoE(dense=a.variant == "dense").to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(a.seed)
    step, log = 0, []
    if (run / "last.pt").is_file():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); g.set_state(ck["gen"].cpu())
        step, log = ck["step"], ck["log"]
        print(f"resumed at step {step}")
    lr_at = lambda s: a.lr * min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    alpha = ALPHA[a.variant]
    t0 = time.time()
    stop = min(a.steps, step + a.chunk)
    m.train()
    while step < stop:
        starts = torch.randint(0, len(tr) - CTX - 1, (B,), generator=g)
        w = tr[starts[:, None] + torch.arange(CTX + 1)[None, :]].to(dev)        # [B, CTX + 1]
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step)
        z, routes = m.run(w[:, :-1])
        ce = F.cross_entropy(z.reshape(-1, V), w[:, 1:].reshape(-1))
        bal = sum(balance_loss(p, gt) for p, gt in routes) if a.variant != "dense" else torch.zeros((), device=dev)
        loss = ce + alpha * bal
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 500 == 0:
            log.append({"step": step, "loss": round(ce.item(), 4), "balance": round(float(bal), 4)})
            print(f"step {step}: ce {ce.item():.4f} balance {float(bal):.4f} ({time.time() - t0:.0f} s)", flush=True)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "gen": g.get_state(), "step": step, "log": log}, run / "last.pt")
    if step < a.steps:
        print(f"chunk done at step {step} of {a.steps}: run again to resume")
        sys.exit(0)
    m = m.cpu().eval()
    val = evaluate(m, va)
    torch.save(m.state_dict(), run / "final.pt")
    if a.variant == "moe":
        torch.save(m.state_dict(), HERE / "model_final.pt")
    mp = HERE / "train_metrics.json"
    allm = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
    allm[a.variant] = {"train_chars": len(tr), "val_chars": len(va), "params": n_params(m), "active": active_params(m),
                       "steps": a.steps, "batch": B, "window": CTX, "lr": a.lr, "alpha": alpha, "seed": a.seed,
                       "optimizer": "AdamW (weight decay 0.01), warm-up 300, cosine to 0, clip 1.0",
                       "val_loss": round(val, 5), "val_bpc": round(val / math.log(2), 5), "seconds": round(time.time() - t0), "log": log}
    with open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(allm, indent=1) + "\n")
    print(json.dumps({k: v for k, v in allm[a.variant].items() if k != "log"}))
