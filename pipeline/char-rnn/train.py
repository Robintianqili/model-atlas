"""char-rnn: train the character LSTM on TinyStories text (fixed seed, resumable).

    npm run atlas -- py pipeline/char-rnn/train.py [--run v1] [--steps 20000] [--chunk 6000]

Data: ak.data.tinystories_chars (the files downloaded for TinyGPT; no new download): the first --chars characters of
the train head for training and the first 2,000,000 characters of TinyStoriesV2-GPT4-valid.txt for validation, both
normalized by model.normalize (curly quotes -> plain). Random windows of 128 characters (+1 for the targets), batch 256,
AdamW lr 4e-3 with 300 warm-up steps and a cosine decay to 0, gradient clip 1.0. GPU for training; the reported
validation loss is measured on the CPU in float32 with TF32 off (atlas_kit switches it off on import).
A call stops after --chunk steps and resumes from runs/<run>/last.pt when called again (CONTRACTS rule 4: < 9 min).
Writes runs/<run>/last.pt, then model_final.pt and train_metrics.json next to this file when the run is complete.
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
from model import ALPHABET, CharLSTM, encode  # noqa: E402

T, B = 128, 256


def load(chars: int):
    tr = torch.tensor(encode(ak.data.tinystories_chars(n_chars=chars)), dtype=torch.long)
    va = torch.tensor(encode(ak.data.tinystories_chars("TinyStoriesV2-GPT4-valid.txt", n_chars=2_000_000)), dtype=torch.long)
    return tr, va


@torch.no_grad()
def evaluate(m: CharLSTM, ids: torch.Tensor, n_windows: int = 4000) -> float:
    """Mean cross-entropy (nats per character) over non-overlapping windows of T characters, CPU float32."""
    m = m.cpu().eval()
    n = min(n_windows, (len(ids) - 1) // T)
    x = ids[: n * T].view(n, T).T
    y = ids[1 : n * T + 1].view(n, T).T
    tot = 0.0
    for s in range(0, n, 500):
        z = m(x[:, s : s + 500])
        tot += F.cross_entropy(z.reshape(-1, len(ALPHABET)), y[:, s : s + 500].reshape(-1), reduction="sum").item()
    return tot / (n * T)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="v1")
    ap.add_argument("--chars", type=int, default=40_000_000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = HERE / "runs" / a.run
    run.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tr, va = load(a.chars)
    m = CharLSTM().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(a.seed)
    step, log = 0, []
    if (run / "last.pt").is_file():
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); g.set_state(ck["gen"].cpu())
        step, log = ck["step"], ck["log"]
        print(f"resumed at step {step}")
    lr_at = lambda s: a.lr * min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    t0 = time.time()
    stop = min(a.steps, step + a.chunk)
    m.train()
    while step < stop:
        starts = torch.randint(0, len(tr) - T - 1, (B,), generator=g)
        idx = starts[:, None] + torch.arange(T + 1)[None, :]
        w = tr[idx].T.to(dev)                                    # [T + 1, B]
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step)
        loss = F.cross_entropy(m(w[:-1]).reshape(-1, len(ALPHABET)), w[1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 500 == 0:
            log.append({"step": step, "loss": round(loss.item(), 4)})
            print(f"step {step}: loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "gen": g.get_state(), "step": step, "log": log}, run / "last.pt")
    if step < a.steps:
        print(f"chunk done at step {step} of {a.steps}: run again to resume")
        sys.exit(0)
    m = m.cpu().eval()
    val = evaluate(m, va)
    trn = evaluate(m, tr[-2_000_000:])
    torch.save(m.state_dict(), HERE / "model_final.pt")
    report = {"run": a.run, "train_chars": len(tr), "val_chars": len(va), "vocab": len(ALPHABET), "params": sum(p.numel() for p in m.parameters()),
              "steps": a.steps, "batch": B, "window": T, "lr": a.lr, "optimizer": "AdamW (wd 0), warm-up 300, cosine to 0, clip 1.0",
              "val_loss": round(val, 5), "val_bpc": round(val / math.log(2), 5), "train_tail_loss": round(trn, 5), "log": log}
    with open(HERE / "train_metrics.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "log"}))
