"""tinyseq2seq: train the date converter (fixed seed), and the same model without attention (the bottleneck baseline).

    node pipeline/tinyseq2seq/make_data.ts                        # the pairs, from the package's TS generator
    npm run atlas -- py pipeline/tinyseq2seq/train.py [--steps 6000] [--seed 0]

Data: pipeline/tinyseq2seq/runs/data/{train,val,test,off}.json (written by make_data.ts from data.ts; no download).
Teacher forcing, cross-entropy over the 11 output symbols (10 characters and the end symbol), Adam with a cosine
learning rate, batch 256. Writes model_final.pt (with attention, H = 32), runs/noattn.pt (without) and train_metrics.json
(exact-match accuracy of greedy decoding on the test set, per format, on test dates never seen in training, and on the
formats of off.json the model never sees, and the validation accuracy every 250 steps).
--hidden 8 trains both with a state of 8 numbers (runs/attn_h8.pt, runs/noattn_h8.pt): the bottleneck experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import BOS, EOS, STEPS, Seq2Seq  # noqa: E402

DATA = HERE / "runs" / "data"


def load(split: str) -> dict:
    p = DATA / f"{split}.json"
    if not p.is_file():
        sys.exit(f"{p} not found: run  node pipeline/tinyseq2seq/make_data.ts  first")
    return json.loads(p.read_text(encoding="utf-8"))


def tensors(pairs, src_vocab, tgt_vocab, dev):
    si = {c: i for i, c in enumerate(src_vocab)}
    ti = {c: i for i, c in enumerate(tgt_vocab)}
    T = max(len(p[0]) for p in pairs)
    src = torch.zeros(len(pairs), T, dtype=torch.long)
    for b, p in enumerate(pairs):
        ids = [si.get(c, 1) for c in p[0]][:32]
        src[b, :len(ids)] = torch.tensor(ids)
    lengths = torch.tensor([min(len(p[0]), 32) for p in pairs])
    tgt = torch.tensor([[ti[c] for c in p[1]] + [EOS] for p in pairs])         # [B, 11] (exactly 10 characters + end)
    y_in = torch.cat([torch.full((len(pairs), 1), BOS), tgt[:, :-1]], dim=1)
    return src.to(dev), lengths, y_in.to(dev), tgt.to(dev)


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
    right, chars, nchar = [], 0, 0
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        src, lengths, _, tgt = tensors(chunk, src_vocab, tgt_vocab, dev)
        ids, _ = m.greedy(src, lengths, STEPS)
        for p, row in zip(chunk, ids.tolist()):
            text = decode_text(row, tgt_vocab)
            right.append(text == p[1])
            chars += sum(a == b for a, b in zip(text, p[1]))
            nchar += len(p[1])
    m.train()
    return right, chars / nchar


def train(attention: bool, a, data, dev, name):
    torch.manual_seed(a.seed)
    tr, va = data["train"], data["val"]
    m = Seq2Seq(len(tr["src_vocab"]), len(tr["tgt_vocab"]), hidden=a.hidden, attn=a.hidden, attention=attention).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    g = torch.Generator().manual_seed(a.seed)
    pairs = tr["pairs"]
    SRC, LEN, YIN, TGT = tensors(pairs, tr["src_vocab"], tr["tgt_vocab"], dev)   # the whole set once; a step indexes it
    t0 = time.time()
    curve = []
    for step in range(a.steps):
        idx = torch.randint(0, len(pairs), (a.batch,), generator=g)
        i = idx.to(dev)
        logits = m(SRC[i], LEN[idx], YIN[i])
        tgt = TGT[i]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 250 == 0 or step == 0:
            r, _ = evaluate(m, va["pairs"], tr["src_vocab"], tr["tgt_vocab"], dev)
            curve.append([step + 1, sum(r) / len(r)])
            if (step + 1) % 1000 == 0 or step == 0:
                print(f"{name} step {step + 1}: loss {loss.item():.4f}, val exact {sum(r) / len(r):.4f} ({time.time() - t0:.0f} s)")
    return m.cpu().eval(), loss.item(), time.time() - t0, curve


def report(m, data, name):
    tr = data["train"]
    sv, tv = tr["src_vocab"], tr["tgt_vocab"]
    test, off = data["test"]["pairs"], data["off"]["pairs"]
    seen = {p[0] for p in tr["pairs"]}
    r, char_acc = evaluate(m, test, sv, tv, "cpu")
    per = {}
    for p, ok in zip(test, r):
        per.setdefault(p[2], []).append(ok)
    unseen = [ok for p, ok in zip(test, r) if p[0] not in seen]
    ro, _ = evaluate(m, off, sv, tv, "cpu")
    per_off = {}
    for p, ok in zip(off, ro):
        per_off.setdefault(p[2], []).append(ok)
    out = {"test_exact": sum(r) / len(r), "test_n": len(r), "test_char": char_acc,
           "test_unseen_exact": sum(unseen) / len(unseen), "test_unseen_n": len(unseen),
           "per_format": {k: sum(v) / len(v) for k, v in per.items()},
           "off_formats": {k: sum(v) / len(v) for k, v in per_off.items()}}
    print(name, json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["attn", "noattn"], default=None)
    ap.add_argument("--hidden", type=int, default=32, help="state size H (and attention size); 32 for the page")
    a = ap.parse_args()
    data = {s: load(s) for s in ("train", "val", "test", "off")}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    metrics_path = HERE / "train_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    metrics["data"] = {s: {"n": data[s]["n"], "seed": data[s]["seed"], "sha256": data[s].get("sha256")} for s in data}
    for attention in (True, False):
        kind = "attn" if attention else "noattn"
        if a.only and a.only != kind:
            continue
        name = kind if a.hidden == 32 else f"{kind}_h{a.hidden}"
        m, loss, secs, curve = train(attention, a, data, dev, name)
        (HERE / "runs").mkdir(exist_ok=True)
        torch.save(m.state_dict(), HERE / ("model_final.pt" if name == "attn" else f"runs/{name}.pt"))
        metrics[name] = {"hidden": a.hidden, "steps": a.steps, "batch": a.batch, "lr": a.lr, "seed": a.seed, "final_loss": loss,
                         "seconds": round(secs, 1), "device": dev, "val_curve": curve,
                         "params": sum(p.numel() for p in m.parameters()), **report(m, data, name)}
        metrics_path.write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {metrics_path}")
