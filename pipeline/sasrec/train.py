"""sasrec: train the tiny SASRec on the TS logs (CPU, float32), then measure it against simple baselines.

    npm run atlas -- run sasrec logs --opt '{"n":10000}' --save     # first: the TS process writes the sessions
    npm run atlas -- py pipeline/sasrec/train.py [--epochs 30] [--seed 0]

Data: $ATLAS_SITE/.atlas/sasrec/logs.tape.json + .tape.bin (atlas-tape/1): 10 chunks of 1,000 sessions, product ids in
click order (−1 pads). Every chunk's sum is recomputed here and must equal the sum TS recorded. Sessions 0-7999 train,
8000-8999 validate, 9000-9999 test.
Loss: at every position t of a training session, cross-entropy of the click t + 1 over all 31 products (causal
attention, so position t sees clicks 0 … t only). Adam, lr 1e-2 with cosine decay, batches of 128 sessions, 60 epochs, no dropout (the tables start at std 0.1);
the epoch with the best validation hit rate @5 -> model_final.pt.
Evaluation (leave the last click out): the history is a session without its last click; the 5 best products the history
does not hold (ties: the smaller id) are the top 5; a hit when the held-out click is among them. Baselines on the same
test sessions: most clicked first, the most frequent click after the last click (a first-order Markov chain; ties by
clicks), SASRec on the history shuffled (order removed), SASRec on the last click alone.
Also: popularity bias (top-5 slots taken by the 5 most clicked products against their share of the held-out clicks;
products never in a top 5), the new arrival's place, and where the last position's attention goes.
Everything goes to train_metrics.json (export.py turns it into facts.json).
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

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import MAX_LEN, N_ITEMS, SASRec, n_params  # noqa: E402

TRAIN, VAL, K = 8000, 9000, 5
NEW_ITEM = N_ITEMS - 1


def load_logs():
    h, blob = ak.tape.read_tape(ak.site_path(".atlas", "sasrec", "logs"))
    seqs = []
    for f in h["frames"]:
        st = {k: np.asarray(ak.tape.decode(v, blob)) for k, v in f["state"].items()}
        a = st["fixed"] if f["phase"] == "fixed" else st["seqs"]
        if int(a.astype(np.int64).sum()) != int(f["metrics"]["sum"]):
            raise SystemExit(f"frame {f['phase']} {f.get('step')}: sum {int(a.sum())} != TS sum {f['metrics']['sum']}")
        if f["phase"] == "chunk":
            seqs.append(a.astype(np.int64))
    S = np.concatenate(seqs)
    return [list(r[r >= 0]) for r in S], h


def batch_xy(sess):
    """inputs [B, 12] (pads 0) and targets [B, 12] (−100 = no target)"""
    B = len(sess)
    x = torch.zeros(B, MAX_LEN, dtype=torch.long)
    y = torch.full((B, MAX_LEN), -100, dtype=torch.long)
    for b, s in enumerate(sess):
        n = len(s) - 1
        x[b, :n] = torch.tensor(s[:n])
        y[b, :n] = torch.tensor(s[1:])
    return x, y


def top_k(scores, hist, k=K):
    seen = set(hist)
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return [i for i in order if i not in seen][:k]


@torch.no_grad()
def last_scores(m, hists):
    """the scores at the last position of each history (each run on its own, as the page runs it)"""
    out = []
    for hst in hists:
        out.append(m(torch.tensor([hst]))[0, -1])
    return torch.stack(out)


def hit_rate(S, hists, targets):
    return float(np.mean([t in top_k(S[j], hists[j]) for j, t in enumerate(targets)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--init", type=float, default=0.1, help="std of the two tables at the start")
    ap.add_argument("--out", default="", help="suffix for an experiment (model_final<out>.pt, train_metrics<out>.json)")
    a = ap.parse_args()
    t0 = time.time()
    sess, h = load_logs()
    assert len(sess) == 10000, len(sess)
    tr = sess[:TRAIN]
    hold = lambda lo, hi: ([s[:-1] for s in sess[lo:hi]], [s[-1] for s in sess[lo:hi]])  # noqa: E731
    vh, vt = hold(TRAIN, VAL)
    th, tt = hold(VAL, len(sess))
    print(f"read {len(sess)} sessions from the TS tape (sums match, {time.time() - t0:.1f} s)")
    torch.manual_seed(a.seed)
    m = SASRec(dropout=a.dropout)
    with torch.no_grad():
        m.item.weight.normal_(0, a.init)
        m.pos.weight.normal_(0, a.init)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    steps_per = math.ceil(TRAIN / a.bs)
    total = steps_per * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total) / total)))
    X, Y = batch_xy(tr)
    g = torch.Generator().manual_seed(a.seed)
    best, log = -1.0, []
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(TRAIN, generator=g)
        tl = 0.0
        for k in range(steps_per):
            j = perm[k * a.bs:(k + 1) * a.bs]
            loss = F.cross_entropy(m(X[j]).reshape(-1, N_ITEMS), Y[j].reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss.detach())
        m.eval()
        vr = hit_rate(last_scores(m, vh), vh, vt)
        log.append({"epoch": ep + 1, "loss": tl / steps_per, "valHit5": vr})
        if vr > best:
            best = vr
            torch.save({k: v.detach().clone() for k, v in m.state_dict().items()}, HERE / f"model_final{a.out}.pt")
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep + 1:3d}  loss {tl / steps_per:.4f}  val hit@5 {vr:.4f}  ({time.time() - t0:.0f} s)")
    m = SASRec()
    m.load_state_dict(torch.load(HERE / f"model_final{a.out}.pt", map_location="cpu"))
    m.eval()

    clicks = [0] * N_ITEMS
    big = np.zeros((N_ITEMS, N_ITEMS), dtype=np.int64)
    for s in tr:
        for i in s:
            clicks[i] += 1
        for p, q in zip(s[:-1], s[1:]):
            big[p, q] += 1
    S = last_scores(m, th)
    hit = hit_rate(S, th, tt)
    pop_order = sorted(range(N_ITEMS), key=lambda i: (-clicks[i], i))
    hit_pop = float(np.mean([t in [i for i in pop_order if i not in set(hs)][:K] for hs, t in zip(th, tt)]))
    hit_markov = float(np.mean([t in top_k([big[hs[-1], i] + clicks[i] / 1e6 for i in range(N_ITEMS)], hs) for hs, t in zip(th, tt)]))
    rs = np.random.default_rng(a.seed)
    shuf = [list(rs.permutation(hs)) if len(hs) > 1 else list(hs) for hs in th]
    hit_shuf = hit_rate(last_scores(m, shuf), shuf, tt)
    last1 = [[hs[-1]] for hs in th]
    hit_last = hit_rate(last_scores(m, last1), th, tt)  # top 5 still leaves out the whole history

    # popularity bias: the 5 most clicked products, their share of the top-5 slots and of the held-out clicks
    top5pop = pop_order[:5]
    lists = [top_k(S[j], th[j]) for j in range(len(th))]
    slots = sum(len(x) for x in lists)
    share_rec = sum(i in top5pop for x in lists for i in x) / slots
    share_tgt = float(np.mean([t in top5pop for t in tt]))
    in_any = set(i for x in lists for i in x)
    never = [i for i in range(N_ITEMS) if i not in in_any and i != NEW_ITEM]
    never_but_target = [i for i in never if i in set(tt)]
    rec_counts = [sum(i in x for x in lists) for i in range(N_ITEMS)]
    tgt_counts = [tt.count(i) for i in range(N_ITEMS)]
    places = [1 + sum(1 for i in range(N_ITEMS) if i not in set(th[j]) and float(S[j, i]) > float(S[j, NEW_ITEM])) for j in range(len(th))]
    # where the last position's attention goes (histories of 2 clicks or more)
    att_last, att_w, att_max_first = [], [], 0
    with torch.no_grad():
        for hs in th:
            if len(hs) < 2:
                continue
            w = m.attention(torch.tensor([hs]))[2][0, -1]
            att_last.append(int(torch.argmax(w)) == len(hs) - 1)
            att_w.append(float(w[-1]))
    out = {
        "params": n_params(m), "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "seed": a.seed, "steps": total, "dropout": a.dropout, "init": a.init,
        "trainSessions": TRAIN, "valSessions": VAL - TRAIN, "testSessions": len(sess) - VAL,
        "trainTargets": sum(len(s) - 1 for s in tr), "trainClicks": sum(len(s) for s in tr),
        "bestValHit5": best, "bestEpoch": max(log, key=lambda r: r["valHit5"])["epoch"],
        "testHit5": hit, "testHit5Pop": hit_pop, "testHit5Markov": hit_markov, "testHit5Shuffled": hit_shuf, "testHit5LastOnly": hit_last,
        "popTop5": top5pop, "popShareRec": share_rec, "popShareTarget": share_tgt, "neverInTop5": never, "neverButTarget": never_but_target,
        "recCounts": rec_counts, "targetCounts": tgt_counts, "clicks": clicks,
        "newItemPlaceMin": min(places), "newItemPlaceMean": float(np.mean(places)), "newItemInTop5": sum(p <= K for p in places),
        "attLastIsMax": float(np.mean(att_last)), "attLastMean": float(np.mean(att_w)), "attSessions": len(att_last),
        "tape": {"seed": h["seed"], "opt": h["opt"]}, "history": log, "seconds": round(time.time() - t0, 1),
    }
    (HERE / f"train_metrics{a.out}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"best val hit@5 {best:.4f} (epoch {out['bestEpoch']}); test hit@5 {hit:.4f}: most clicked {hit_pop:.4f}, after-last-click {hit_markov:.4f}, "
          f"shuffled {hit_shuf:.4f}, last click only {hit_last:.4f}")
    print(f"popularity: top-5 most clicked {top5pop} take {share_rec:.3f} of top-5 slots, {share_tgt:.3f} of held-out clicks; never in a top 5: {never} "
          f"(of them held-out clicks: {never_but_target}); rec counts {rec_counts}")
    print(f"new arrival: place min {min(places)} mean {np.mean(places):.2f}, in top 5 {out['newItemInTop5']} times")
    print(f"attention: the last click gets the largest weight in {np.mean(att_last):.3f} of {len(att_last)} histories; mean weight {np.mean(att_w):.3f}")
    print(f"params {out['params']}; {out['seconds']} s")


if __name__ == "__main__":
    main()
