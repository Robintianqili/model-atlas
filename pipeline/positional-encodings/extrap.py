"""positional-encodings: the extrapolation sweep (precomputed for the concept page /train/positional-encodings/).

Five TinyGPT-shaped decoders (d_model 76, 4 layers, 4 heads of 19, MLP 304, GELU tanh, tied head; TinyGPT's tokenizer
and its tokenized TinyStories) that differ only in how they get positions:
  none        no position information at all (NoPE): only the causal mask tells the order
  learned     a learned table of 64 rows added to the token embeddings (as TinyGPT)
  sinusoidal  the fixed sin/cos table of Vaswani et al. (2017), base 10000, added to the token embeddings, which are
              multiplied by sqrt(76) first as in that paper (else the table, of size about 6, drowns the tokens)
  rope        rotary embeddings (Su et al., 2021): q and k of every head rotated pair by pair, angle p * theta_i with
              theta_i = 10000^(-2i/18); the heads have 19 dims, so 18 of them (9 pairs) rotate and the last one does not
  alibi       a bias -m * (i - j) on every attention score (Press et al., 2021), slopes m = 2^(-2), 2^(-4), 2^(-6), 2^(-8)
Each is trained at context 64 with the same batches, then scored on windows of 256 tokens of the validation split: the
mean next-token loss (nats) at every position 0..255. The learned table has no row for position 64 or later, so that
model is scored only on positions 0..63.

Writes (paths from ATLAS_SITE):
  src/lib/explainers/positional-encodings/extrap.ts    the per-position losses in bins of 8 positions, per scheme
  static/models/positional-encodings/facts.json        the numbers the lesson narrates
Ideas for the GPT from nanoGPT / minGPT (Andrej Karpathy, MIT); code written fresh.
Run: npm run atlas -- py pipeline/positional-encodings/extrap.py [--steps 3000]
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("ATLAS_ROOT", Path(__file__).resolve().parents[2]))
SITE = Path(os.environ.get("ATLAS_SITE", ROOT / "site"))
DATA = ROOT / "pipeline" / "data" / "tinystories"
SCHEMES = ["none", "learned", "sinusoidal", "rope", "alibi"]
V, D, L, H, CTX, EVAL_CTX, BIN = 1024, 76, 4, 4, 64, 256, 8
HD = D // H
ROT = HD - HD % 2  # 18 rotated dims of 19


def sinusoidal_table(n: int, d: int) -> torch.Tensor:
    """PE[p, 2i] = sin(p w_i), PE[p, 2i+1] = cos(p w_i), w_i = 10000^(-2i/d)."""
    p = torch.arange(n, dtype=torch.float64)[:, None]
    i = torch.arange(d // 2, dtype=torch.float64)[None]
    ang = p * 10000.0 ** (-2 * i / d)
    pe = torch.zeros(n, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe.float()


def rope_angles(n: int) -> torch.Tensor:
    """angle[p, i] = p * theta_i, theta_i = 10000^(-2i/ROT), i = 0..ROT/2-1."""
    p = torch.arange(n, dtype=torch.float64)[:, None]
    i = torch.arange(ROT // 2, dtype=torch.float64)[None]
    return (p * 10000.0 ** (-2 * i / ROT)).float()


def rotate(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """x [..., T, HD]: rotate the pairs (x0, x1), (x2, x3), ... of the first ROT dims by ang [T, ROT/2]."""
    a, b = x[..., 0:ROT:2], x[..., 1:ROT:2]
    c, s = torch.cos(ang), torch.sin(ang)
    out = x.clone()
    out[..., 0:ROT:2] = a * c - b * s
    out[..., 1:ROT:2] = a * s + b * c
    return out


def alibi_slopes(h: int) -> torch.Tensor:
    return torch.tensor([2.0 ** (-8.0 * (k + 1) / h) for k in range(h)])


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.fc, self.out = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

    def forward(self, x, scheme, ang, bias):
        B, T, _ = x.shape
        q, k, v = (t.view(B, T, H, HD).transpose(1, 2) for t in self.qkv(self.ln1(x)).split(D, dim=2))
        if scheme == "rope":
            q, k = rotate(q, ang[:T]), rotate(k, ang[:T])
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        s = q @ k.transpose(-1, -2) / math.sqrt(HD)
        if scheme == "alibi":
            s = s + bias[:, :T, :T]
        w = torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1)
        y = (w @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.out(gelu_tanh(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, scheme: str):
        super().__init__()
        self.scheme = scheme
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(CTX, D) if scheme == "learned" else None
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * L))
            nn.init.normal_(b.out.weight, std=0.02 / math.sqrt(2 * L))
        self.register_buffer("sin", sinusoidal_table(EVAL_CTX, D), persistent=False)
        self.register_buffer("ang", rope_angles(EVAL_CTX), persistent=False)
        i = torch.arange(EVAL_CTX)
        dist = (i[:, None] - i[None]).clamp(min=0).float()
        self.register_buffer("bias", -alibi_slopes(H)[:, None, None] * dist[None], persistent=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx)
        if self.scheme == "learned":
            x = x + self.pos(torch.arange(T, device=idx.device))[None]
        elif self.scheme == "sinusoidal":
            x = x * math.sqrt(D) + self.sin[:T][None]
        for b in self.blocks:
            x = b(x, self.scheme, self.ang, self.bias)
        return self.lnf(x) @ self.tok.weight.t()


def train(scheme, train_ids, steps, batch, dev, lr=3e-3):
    torch.manual_seed(0)
    model = GPT(scheme).to(dev)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    rest = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": rest, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    rng = np.random.default_rng(0)  # the same batches for every scheme
    n = len(train_ids) - CTX - 1
    t0 = time.time()
    for step in range(steps):
        warm = min(1.0, (step + 1) / 200)
        cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr * warm * cos
        starts = rng.integers(0, n, size=batch)
        xb = torch.from_numpy(np.stack([train_ids[s: s + CTX + 1] for s in starts]).astype(np.int64)).to(dev)
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  {scheme} step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def per_position_loss(model, windows, dev):
    """Mean loss at every position (nats), over windows [N, EVAL_CTX + 1]."""
    T = CTX if model.scheme == "learned" else EVAL_CTX
    tot = torch.zeros(T, dtype=torch.float64)
    for i in range(0, len(windows), 50):
        w = torch.from_numpy(windows[i: i + 50].astype(np.int64)).to(dev)
        logits = model(w[:, :T])
        l = F.cross_entropy(logits.reshape(-1, V), w[:, 1: T + 1].reshape(-1), reduction="none").view(-1, T)
        tot += l.double().sum(0).cpu()
    out = np.full(EVAL_CTX, np.nan)
    out[:T] = (tot / len(windows)).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--windows", type=int, default=500)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    train_ids = np.fromfile(DATA / "train.bin", dtype="<u2")
    val = np.fromfile(DATA / "val.bin", dtype="<u2")
    windows = val[: a.windows * (EVAL_CTX + 1)].reshape(a.windows, EVAL_CTX + 1)
    curves = {}
    for s in SCHEMES:
        model = train(s, train_ids, a.steps, a.batch, dev).eval()
        curves[s] = per_position_loss(model, windows, dev)
        c = curves[s]
        print(s, "in", np.nanmean(c[:CTX]), "far", np.nanmean(c[192:]) if s != "learned" else "n/a", flush=True)

    bins = EVAL_CTX // BIN
    binned = {s: [None if np.isnan(c[b * BIN: (b + 1) * BIN]).any() else round(float(c[b * BIN: (b + 1) * BIN].mean()), 4) for b in range(bins)] for s, c in curves.items()}
    out = SITE / "src" / "lib" / "explainers" / "positional-encodings" / "extrap.ts"
    lines = [
        "// positional-encodings: GENERATED by pipeline/positional-encodings/extrap.py (do not edit). Five TinyGPT-shaped",
        f"// decoders that differ only in their positions, trained at context {CTX} ({a.steps} steps of {a.batch} windows) and",
        f"// scored on {a.windows} windows of {EVAL_CTX} tokens of TinyStories' validation split (TinyGPT's tokenized copy): the mean",
        f"// next-token loss in nats, averaged over bins of {BIN} positions (bin b holds positions {BIN}b to {BIN}b + {BIN - 1}). null: the",
        "// learned table has no row for those positions. TinyStories by Ronen Eldan and Yuanzhi Li (2023), CDLA-Sharing-1.0.",
        f"export const TRAIN_CTX = {CTX};",
        f"export const EVAL_CTX = {EVAL_CTX};",
        f"export const BIN = {BIN};",
        f"export const SCHEMES = {json.dumps(SCHEMES)} as const;",
        f"export const LOSS: Record<string, (number | null)[]> = {json.dumps(binned)};",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("wrote", out)

    today = datetime.date.today().isoformat()
    src = "pipeline/positional-encodings/extrap.py"
    facts = {}

    def put(k, v, unit):
        facts[k] = {"value": v, "source": src, "unit": unit, "measured": today}

    put("trainCtx", CTX, "tokens per training window")
    put("evalCtx", EVAL_CTX, "tokens per scoring window")
    put("steps", a.steps, "training steps per model")
    put("batch", a.batch, "windows per training step")
    put("windows", a.windows, "validation windows scored per model")
    tag = {"none": "None", "learned": "Learned", "sinusoidal": "Sin", "rope": "Rope", "alibi": "Alibi"}
    for s, c in curves.items():
        put("in" + tag[s], round(float(np.mean(c[:CTX])), 4), f"{s}: mean loss over positions 0-{CTX - 1}, nats")
        if s != "learned":
            put("mid" + tag[s], round(float(np.mean(c[CTX: 2 * CTX])), 4), f"{s}: mean loss over positions {CTX}-{2 * CTX - 1}, nats")
            put("far" + tag[s], round(float(np.mean(c[EVAL_CTX - CTX:])), 4), f"{s}: mean loss over positions {EVAL_CTX - CTX}-{EVAL_CTX - 1}, nats")
    # cross-checks of the page's formulas against the ones trained here
    put("chkSin", round(float(sinusoidal_table(EVAL_CTX, D)[10, 3]), 6), "sinusoidal PE[p = 10, dim 3] for d = 76")
    put("chkRope", round(float(rope_angles(EVAL_CTX)[100, 1]), 6), "RoPE angle of pair 1 at position 100 (radians, head of 19 dims)")
    dst = SITE / "static" / "models" / "positional-encodings"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "facts.json").write_text(json.dumps({"format": "atlas-facts/1", "facts": facts}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", dst / "facts.json")


if __name__ == "__main__":
    main()
