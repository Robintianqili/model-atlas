"""two-tower: train the two towers with in-batch negatives on the TS logs, then fit the tiny ranker (CPU, float32).

    npm run atlas -- run two-tower logs --opt '{"n":6000}' --save     # first: the TS process writes the logs
    npm run atlas -- py pipeline/two-tower/train.py [--epochs 40] [--seed 0]

Data: $ATLAS_SITE/.atlas/two-tower/logs.tape.json + .tape.bin (atlas-tape/1): the 36 movies' features (frame 0) and
6 chunks of 1,000 viewers (features [1000, 15], the movies each liked besides their history, the history). Every
chunk's sum of features is recomputed here and must equal the sum TS recorded. Viewers 0-4999 train, 5000-5499
validate (and fit the ranker), 5500-5999 test.

Loss (in-batch negatives): a batch of 64 (viewer, liked movie) pairs; each viewer's scores against the distinct movies
of the batch (a movie that occurs twice is one column; the movies of the viewer's own history are left out of the row,
since they are liked too), softmax along the row, cross-entropy of the viewer's own movie.
Adam, lr 3e-3, cosine decay; best validation recall@8 -> model_final.pt. Then:
  ranker   logistic regression p = sigmoid(w_s * score + w_p * pop + b) on the validation viewers' shortlists (the best
           8 movies they did not have in their history; label 1 = they liked it), pop = log(1 + likes in the training
           logs), standardized over the 36 movies. Fitted with LBFGS in float64.
  metrics  test recall@8 of the towers, of "most liked first" and of a random pick; test hits in the first 3 of the
           shortlist before and after the ranker; the twins' likes.
Everything goes to train_metrics.json (export.py turns it into facts.json and the spec extras).
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
from model import TwoTower, n_params  # noqa: E402

K = 8                      # the shortlist (recs.ts K)
TRAIN, VAL = 5000, 5500    # viewers [0, 5000) train, [5000, 5500) validate, [5500, 6000) test
TWINS = (28, 29)


def load_logs():
    """items [36, 8], fixed users [F, 15], fixed new items [F, 8], users [n, 15], (pos, owner) pairs, hist [n, 6]."""
    h, blob = ak.tape.read_tape(ak.site_path(".atlas", "two-tower", "logs"))
    users, hist, pos, owner, fixed = [], [], [], [], None
    for f in h["frames"]:
        st = {k: ak.tape.decode(v, blob) for k, v in f["state"].items()}
        if f["phase"] == "fixed":
            fixed = st
            got = float(np.asarray(st["items"], dtype=np.float64).sum() + np.asarray(st["fixedUsers"], dtype=np.float64).sum()
                        + np.asarray(st["fixedItems"], dtype=np.float64).sum())
        else:
            off = sum(len(u) for u in users)
            users.append(np.asarray(st["users"], dtype="<f4"))
            hist.append(np.asarray(st["hist"]))
            pos.append(np.asarray(st["pos"]))
            owner.append(np.asarray(st["owner"]) + off)
            got = float(np.asarray(st["users"], dtype=np.float64).sum())
        if abs(got - float(f["metrics"]["sum"])) > 1e-9 * max(1.0, abs(got)):
            raise SystemExit(f"frame {f['phase']} {f.get('step')}: sum of the numbers read {got} != TS sum {f['metrics']['sum']}")
    T = lambda a: torch.tensor(np.concatenate(a))  # noqa: E731
    return (torch.tensor(np.asarray(fixed["items"], dtype="<f4")), torch.tensor(np.asarray(fixed["fixedUsers"], dtype="<f4")),
            torch.tensor(np.asarray(fixed["fixedItems"], dtype="<f4")), T(users), T(pos).long(), T(owner).long(), T(hist).long(), h)


def liked_sets(pos, owner, n):
    out = [[] for _ in range(n)]
    for p, o in zip(pos.tolist(), owner.tolist()):
        out[o].append(p)
    return out


def shortlist(scores_row, hist_row, k=K):
    """indices of the best k movies not in the history (ties: the smaller index, as recs.ts)"""
    seen = {int(x) for x in hist_row if x >= 0}
    order = sorted(range(len(scores_row)), key=lambda i: (-float(scores_row[i]), i))
    return [i for i in order if i not in seen][:k]


@torch.no_grad()
def recall(m, items, users, hist, liked, lo, hi):
    S = m.scores(users[lo:hi], items)
    r = []
    for j, u in enumerate(range(lo, hi)):
        top = shortlist(S[j], hist[u])
        r.append(len(set(top) & set(liked[u])) / len(liked[u]))
    return float(np.mean(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    t0 = time.time()
    items, fixed_u, fixed_i, users, pos, owner, hist, h = load_logs()
    n = len(users)
    liked = liked_sets(pos, owner, n)
    tr = owner < TRAIN
    P_u, P_i = owner[tr], pos[tr]
    print(f"read {n} viewers, {int(tr.sum())} training pairs from the TS tape (sums match, {time.time() - t0:.1f} s)")
    torch.manual_seed(a.seed)
    m = TwoTower()
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    steps_per = len(P_u) // a.bs
    total = steps_per * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total) / total)))
    g = torch.Generator().manual_seed(a.seed)
    best, hist_log, uniq = -1.0, [], []
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(len(P_u), generator=g)
        tl = 0.0
        for k in range(steps_per):
            j = perm[k * a.bs:(k + 1) * a.bs]
            cols, target = torch.unique(P_i[j], return_inverse=True)
            S = m.user(users[P_u[j]]) @ m.item(items[cols]).t()
            seen = (hist[P_u[j]].unsqueeze(2) == cols.view(1, 1, -1)).any(1)   # a movie in the viewer's own history
            loss = F.cross_entropy(S.masked_fill(seen, float("-inf")), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss.detach())
            if ep == 0:
                uniq.append(len(cols))
        m.eval()
        vr = recall(m, items, users, hist, liked, TRAIN, VAL)
        hist_log.append({"epoch": ep + 1, "loss": tl / steps_per, "valRecall8": vr})
        if vr > best:
            best = vr
            torch.save({k: v.detach().clone() for k, v in m.state_dict().items()}, HERE / "model_final.pt")
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep + 1:3d}  loss {tl / steps_per:.4f}  val recall@8 {vr:.4f}  ({time.time() - t0:.0f} s)")
    m = TwoTower()
    m.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    m.eval()

    # popularity: likes in the training logs (history + targets of viewers 0-4999)
    likes = torch.zeros(36, dtype=torch.long)
    for u in range(TRAIN):
        for i in liked[u] + [int(x) for x in hist[u] if x >= 0]:
            likes[i] += 1
    lp = torch.log1p(likes.double())
    pop_mean, pop_std = float(lp.mean()), float(lp.std(unbiased=False))
    popz = (lp - pop_mean) / pop_std

    # the ranker on the validation viewers' shortlists
    with torch.no_grad():
        S_all = m.scores(users, items).double()

    def rows(lo, hi):
        X, Y, groups = [], [], []
        for u in range(lo, hi):
            top = shortlist(S_all[u], hist[u])
            groups.append(top)
            for i in top:
                X.append([float(S_all[u, i]), float(popz[i])])
                Y.append(1.0 if i in liked[u] else 0.0)
        return torch.tensor(X, dtype=torch.float64), torch.tensor(Y, dtype=torch.float64), groups

    Xv, Yv, _ = rows(TRAIN, VAL)
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    lbfgs = torch.optim.LBFGS([w], lr=1, max_iter=200, tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xv @ w[:2] + w[2], Yv)
        loss.backward()
        return loss
    lbfgs.step(closure)
    ws, wp, b0 = (float(x) for x in w.detach())

    # test metrics
    test = range(VAL, n)
    rec_t = recall(m, items, users, hist, liked, VAL, n)
    pop_order = sorted(range(36), key=lambda i: (-int(likes[i]), i))
    rec_pop = float(np.mean([len(set([i for i in pop_order if i not in {int(x) for x in hist[u] if x >= 0}][:K]) & set(liked[u])) / len(liked[u]) for u in test]))
    rec_rand = float(np.mean([K / (36 - int((hist[u] >= 0).sum())) for u in test]))
    before, after = [], []
    for u in test:
        top = shortlist(S_all[u], hist[u])
        z = [ws * float(S_all[u, i]) + wp * float(popz[i]) + b0 for i in top]
        rr = [top[k] for k in sorted(range(len(top)), key=lambda k: (-z[k], k))]
        before.append(len(set(top[:3]) & set(liked[u])))
        after.append(len(set(rr[:3]) & set(liked[u])))
    out = {
        "params": n_params(m), "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "seed": a.seed, "steps": total,
        "uniqueCols": float(np.mean(uniq)), "trainViewers": TRAIN, "trainPairs": int(tr.sum()), "valViewers": VAL - TRAIN,
        "testViewers": n - VAL, "bestValRecall8": best, "testRecall8": rec_t, "testRecall8Pop": rec_pop, "testRecall8Random": rec_rand,
        "hits3Before": float(np.mean(before)), "hits3After": float(np.mean(after)),
        "ranker": {"w": [ws, wp], "b": b0, "popMean": pop_mean, "popStd": pop_std, "fitRows": len(Yv), "fitPositives": int(Yv.sum())},
        "likes": likes.tolist(), "twinLikes": [int(likes[TWINS[0]]), int(likes[TWINS[1]])],
        "tape": {"seed": h["seed"], "opt": h["opt"]}, "history": hist_log, "seconds": round(time.time() - t0, 1),
    }
    (HERE / "train_metrics.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"best val recall@8 {best:.4f}; test recall@8 {rec_t:.4f} (most liked first {rec_pop:.4f}, random {rec_rand:.4f})")
    print(f"ranker w_s {ws:.4f} w_p {wp:.4f} b {b0:.4f}; test hits in the first 3: {np.mean(before):.4f} -> {np.mean(after):.4f}")
    print(f"twins' likes {out['twinLikes']}; params {out['params']}; {out['seconds']} s")


if __name__ == "__main__":
    main()
