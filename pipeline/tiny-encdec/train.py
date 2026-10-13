"""tiny-encdec: train the encoder-decoder Transformer on tinyseq2seq's date task (fixed seed).

    node pipeline/tiny-encdec/make_data.ts                        # the pairs, from the package's TS generator
    npm run atlas -- py pipeline/tiny-encdec/train.py [--steps 6000] [--seed 0]

Data: pipeline/tiny-encdec/runs/data/{train,val,test,off}.json (written by make_data.ts from data.ts, the same
generator, seeds and pairs as tinyseq2seq; no download). Teacher forcing, cross-entropy over the 11 output symbols
(10 characters and the end symbol), Adam with a short warm-up and a cosine learning rate, batch 256. Writes
model_final.pt and train_metrics.json (exact-match accuracy of greedy decoding on the test set, per format, on test
dates never seen in training and on the formats of off.json, the validation accuracy every 250 steps, and where the
cross-attention of decoder step 0 looks).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import BOS, EOS, STEPS, T_MAX, EncDec  # noqa: E402

DATA = HERE / "runs" / "data"


def load(split: str) -> dict:
    p = DATA / f"{split}.json"
    if not p.is_file():
        sys.exit(f"{p} not found: run  node pipeline/tiny-encdec/make_data.ts  first")
    return json.loads(p.read_text(encoding="utf-8"))


def tensors(pairs, src_vocab, tgt_vocab, dev):
    si = {c: i for i, c in enumerate(src_vocab)}
    ti = {c: i for i, c in enumerate(tgt_vocab)}
    T = min(T_MAX, max(len(p[0]) for p in pairs))
    src = torch.zeros(len(pairs), T, dtype=torch.long)
    for b, p in enumerate(pairs):
        ids = [si[c] if c in si and si[c] > 1 else 1 for c in p[0]][-T_MAX:]
        src[b, :len(ids)] = torch.tensor(ids)
    lengths = torch.tensor([min(len(p[0]), T_MAX) for p in pairs])
    key_ok = torch.arange(T)[None, :] < lengths[:, None]
    tgt = torch.tensor([[ti[c] for c in p[1]] + [EOS] for p in pairs])          # [B, 11]: exactly 10 characters + end
    y_in = torch.cat([torch.full((len(pairs), 1), BOS), tgt[:, :-1]], dim=1)
    return src.to(dev), key_ok.to(dev), y_in.to(dev), tgt.to(dev)


def decode_text(ids, tgt_vocab):
    out = []
    for i in ids:
        if i == EOS:
            break
        out.append(tgt_vocab[i] if i > EOS else "?")
    return "".join(out)


@torch.no_grad()
def evaluate(m, pairs, src_vocab, tgt_vocab, dev, batch=1000):
    m.eval()
    right, texts = [], []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        src, key_ok, _, _ = tensors(chunk, src_vocab, tgt_vocab, dev)
        ids, _ = m.greedy(src, key_ok)
        for p, row in zip(chunk, ids.tolist()):
            texts.append(decode_text(row, tgt_vocab))
            right.append(texts[-1] == p[1])
    m.train()
    return right, texts


def train(a, data, dev):
    torch.manual_seed(a.seed)
    tr, va = data["train"], data["val"]
    m = EncDec(len(tr["src_vocab"]), len(tr["tgt_vocab"]), d=a.d, heads=a.heads, ff=a.ff, n_enc=a.n_enc, n_dec=a.n_dec).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    g = torch.Generator().manual_seed(a.seed)
    pairs = tr["pairs"]
    SRC, KOK, YIN, TGT = tensors(pairs, tr["src_vocab"], tr["tgt_vocab"], dev)
    t0 = time.time()
    curve = []
    for step in range(a.steps):
        idx = torch.randint(0, len(pairs), (a.batch,), generator=g).to(dev)
        logits = m(SRC[idx], KOK[idx], YIN[idx])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), TGT[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 250 == 0 or step == 0:
            r, _ = evaluate(m, va["pairs"], tr["src_vocab"], tr["tgt_vocab"], dev)
            curve.append([step + 1, sum(r) / len(r)])
            if (step + 1) % 1000 == 0 or step == 0:
                print(f"step {step + 1}: loss {loss.item():.4f}, val exact {sum(r) / len(r):.4f} ({time.time() - t0:.0f} s)", flush=True)
    return m.cpu().eval(), loss.item(), time.time() - t0, curve


@torch.no_grad()
def year_look(m, pairs, sv, tv):
    """Decoder step 0 writes the year's first digit: where its cross-attention puts the most weight, per head, on the
    test dates (the year's first digit, its second, elsewhere in the year, elsewhere)."""
    src, key_ok, _, _ = tensors(pairs, sv, tv, "cpu")
    mem, _ = m.encode(src, key_ok)
    y_in = torch.full((len(pairs), 1), BOS)
    _, _, wc = m.decode(y_in, mem, key_ok)
    w = wc[-1][:, :, 0]                                            # [B, H, T]: the last block, step 0
    pos = torch.tensor([re.search(r"\d{4}", p[0]).start() for p in pairs])
    out = {}
    for h in range(w.shape[1]):
        top = w[:, h].argmax(-1)
        out[f"head{h}"] = {"first": (top == pos).float().mean().item(), "second": (top == pos + 1).float().mean().item(),
                           "year": ((top >= pos) & (top < pos + 4)).float().mean().item()}
    both = w.sum(1).argmax(-1)
    out["sum"] = {"first": (both == pos).float().mean().item(), "second": (both == pos + 1).float().mean().item(),
                  "year": ((both >= pos) & (both < pos + 4)).float().mean().item()}
    return out


def report(m, data):
    tr = data["train"]
    sv, tv = tr["src_vocab"], tr["tgt_vocab"]
    test, off = data["test"]["pairs"], data["off"]["pairs"]
    seen = {p[0] for p in tr["pairs"]}
    r, _ = evaluate(m, test, sv, tv, "cpu")
    per = {}
    for p, ok in zip(test, r):
        per.setdefault(p[2], []).append(ok)
    unseen = [ok for p, ok in zip(test, r) if p[0] not in seen]
    ro, texts = evaluate(m, off, sv, tv, "cpu")
    per_off = {}
    for p, ok in zip(off, ro):
        per_off.setdefault(p[2], []).append(ok)
    out = {"test_exact": sum(r) / len(r), "test_n": len(r), "test_unseen_exact": sum(unseen) / len(unseen),
           "test_unseen_n": len(unseen), "per_format": {k: sum(v) / len(v) for k, v in per.items()},
           "off_formats": {k: sum(v) / len(v) for k, v in per_off.items()}, "year_look": year_look(m, test, sv, tv)}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--ff", type=int, default=64)
    ap.add_argument("--n-enc", dest="n_enc", type=int, default=1)
    ap.add_argument("--n-dec", dest="n_dec", type=int, default=1)
    ap.add_argument("--name", default="model", help="'model' writes model_final.pt; any other name runs/<name>.pt")
    a = ap.parse_args()
    data = {s: load(s) for s in ("train", "val", "test", "off")}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, loss, secs, curve = train(a, data, dev)
    (HERE / "runs").mkdir(exist_ok=True)
    torch.save(m.state_dict(), HERE / ("model_final.pt" if a.name == "model" else f"runs/{a.name}.pt"))
    metrics_path = HERE / "train_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    metrics["data"] = {s: {"n": data[s]["n"], "seed": data[s]["seed"], "sha256": data[s].get("sha256")} for s in data}
    metrics[a.name] = {"d": a.d, "heads": a.heads, "ff": a.ff, "n_enc": a.n_enc, "n_dec": a.n_dec, "steps": a.steps, "batch": a.batch,
                       "lr": a.lr, "seed": a.seed, "final_loss": loss, "seconds": round(secs, 1), "device": dev, "val_curve": curve,
                       "params": sum(p.numel() for p in m.parameters()), **report(m, data)}
    metrics_path.write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path}")
