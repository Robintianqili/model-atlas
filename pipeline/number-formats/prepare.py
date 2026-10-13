"""number-formats: the measured facts of the concept page ({fact:<key>} in the narration), in PyTorch fp32.

1. Gradients (the mixed-precision steps): one backward pass of TinyGPT's cross-entropy on the first 64 windows of 65
   tokens of TinyStories' validation split (TinyGPT's tokenized copy, pipeline/data/tinystories/val.bin; 4,096 next
   tokens). Of the nonzero gradient entries of every parameter: the share that becomes 0 in fp16, bf16, e5m2 and e4m3,
   and in fp16 after loss scaling by GRAD_SCALE (the gradients multiplied by the scale before the cast).
2. The weight casts of the page's sweep (process.ts CASTS) on 1,000 non-overlapping 65-token windows of the same split
   (64,000 next tokens): perplexity and mean KL(fp32 || cast) per position.

Writes site/static/models/number-formats/facts.json (ATLAS_SITE: the lane's site when run through the CLI).

    npm run atlas -- py pipeline/number-formats/prepare.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flib import SITE, cast, forward, load_tinygpt, matrices, metrics, scaled, windows  # noqa: E402

GRAD_WINDOWS, BIG_N = 64, 1000
GRAD_SCALE = 1024
CASTS = [("bf16", "bf16", "none", 0), ("fp16", "fp16", "none", 0), ("e5m2", "e5m2", "none", 0), ("e4m3", "e4m3", "none", 0),
         ("e4m3-t", "e4m3", "tensor", 0), ("fp4", "fp4", "none", 0), ("fp4-b", "fp4", "block", 32), ("int4-b", "int4", "block", 64),
         ("nf4-b", "nf4", "block", 64)]


def key(cid: str) -> str:
    return "".join(p.capitalize() for p in cid.split("-"))


def main():
    torch.manual_seed(0)
    spec, W = load_tinygpt("cpu")
    today = datetime.date.today().isoformat()
    src = "pipeline/number-formats/prepare.py"
    facts = {}

    def put(k, v, unit):
        facts[k] = {"value": v, "source": src, "unit": unit, "measured": today}
        print(k, v)

    # 1. gradients
    Wg = {k: t.clone().requires_grad_(True) for k, t in W.items()}
    ids = windows(GRAD_WINDOWS)
    logits = forward(spec, Wg, ids[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    loss.backward()
    g = torch.cat([t.grad.flatten() for t in Wg.values()]).detach()
    g = g[g != 0]
    a = g.abs()
    put("gradTokens", GRAD_WINDOWS * 64, "next tokens in the batch of the backward pass")
    put("gradCount", int(g.numel()), "nonzero gradient entries (every parameter of TinyGPT)")
    put("gradMedian", float(f"{a.median().item():.4g}"), "median size |g| of those entries")
    put("gradMax", float(f"{a.max().item():.4g}"), "largest size |g|")
    for fmt in ("fp16", "bf16", "e5m2", "e4m3"):
        put(f"gradZero_{fmt}", round((cast(fmt, g) == 0).double().mean().item(), 5), f"share of nonzero gradient entries that become 0 in {fmt}")
    put("gradSub_fp16", round((a < 2 ** -14).double().mean().item(), 5), "share of nonzero gradient entries below fp16's smallest normal 2^-14 (subnormal or 0)")
    put("gradScale", GRAD_SCALE, "loss scale (the gradients are multiplied by it before the cast)")
    put("gradZeroScaled_fp16", round((cast("fp16", g * GRAD_SCALE) == 0).double().mean().item(), 5), "share that becomes 0 in fp16 after loss scaling")
    put("gradMaxScaled", float(f"{(a.max() * GRAD_SCALE).item():.4g}"), "largest scaled size (fp16 overflows above 65504)")

    # 2. the sweep's casts on 64,000 tokens
    big = windows(BIG_N)
    put("bigTokens", BIG_N * 64, "next tokens scored (1,000 windows of the validation split)")
    base = metrics(spec, W, W, big)
    put("bigPpl0", round(base["ppl0"], 4), "perplexity of the fp32 model on those tokens")
    names = matrices(spec)
    for cid, fmt, sc, block in CASTS:
        Wq = dict(W)
        for n in names:
            Wq[n] = scaled(W[n], fmt, sc, block or 32)
        m = metrics(spec, W, Wq, big)
        put("ppl" + key(cid), round(m["ppl"], 4), f"perplexity, every linear weight matrix cast to {cid}")
        put("kl" + key(cid), float(f"{m['kl']:.4g}"), f"mean KL(fp32 || {cid}) per position, nats")
    dst = SITE / "static" / "models" / "number-formats"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "facts.json").write_text(json.dumps({"format": "atlas-facts/1", "facts": facts}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", dst / "facts.json")


if __name__ == "__main__":
    main()
