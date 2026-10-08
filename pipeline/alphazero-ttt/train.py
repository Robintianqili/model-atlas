"""alphazero-ttt: train the policy/value network by self-play (AlphaZero's loop, tic-tac-toe size).

    npm run atlas -- py pipeline/alphazero-ttt/train.py                # all rounds (about 3 minutes on this machine)
    npm run atlas -- py pipeline/alphazero-ttt/train.py --rounds 10    # stop early; run again to resume

One round: the current network plays GAMES games against itself, each move chosen by a search of SIMS simulations (with
Dirichlet noise mixed into the root's priors, and for the first TEMP_MOVES moves a move drawn in proportion to the
visits, so the games vary). Every position of every game becomes a training example (board, pi, z): pi = the root's
visit shares, z = the final result for the player who was to move (+1 win, 0 draw, -1 loss); each example is used in
its 8 rotations and reflections. Then STEPS Adam steps on random batches from the last BUFFER rounds minimize
  (z - v)^2  -  sum_a pi_a * log p_a      (+ L2 weight decay)
After each round: the policy and value loss on that round's examples, and on the 4,520 positions where a move is still
to be made, how often the network's top prior is a perfect move (checked by the exact solver in game.py).
Writes runs/selfplay/{state.pt, best.pt, log.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import game as G  # noqa: E402
from mcts import NumpyNet, search  # noqa: E402
from model import PVNet  # noqa: E402

SEED, ROUNDS, GAMES, SIMS, TEMP_MOVES = 7, 40, 120, 50, 3
EPS, ALPHA = 0.25, 1.0
STEPS, BATCH, LR, WD, BUFFER = 150, 256, 2e-3, 1e-4, 4
RUN = HERE / "runs" / "selfplay"


def self_play(net, rng):
    b, hist = G.EMPTY, []
    while not G.terminal(b):
        t = search(b, net, SIMS, noise=(EPS, ALPHA, rng))
        visits = np.zeros(9)
        for k in t.kids[0]:
            visits[int(t.move[k])] = t.N[k]
        pi = visits / visits.sum()
        hist.append((b, pi))
        cell = int(rng.choice(9, p=pi)) if len(hist) <= TEMP_MOVES else int(np.argmax(visits))
        b = G.play(b, cell)
    w = G.winner(b)
    return [(b0, pi, 0.0 if w == 0 else (1.0 if w == G.to_move(b0) else -1.0)) for b0, pi in hist], w


def examples(games):
    xs, ms, ps, zs = [], [], [], []
    for b0, pi, z in games:
        x, m = G.view(b0), G.mask(b0)
        for perm in G.SYMS:
            xs.append([x[perm[i]] for i in range(9)])
            ms.append([m[perm[i]] for i in range(9)])
            ps.append([pi[perm[i]] for i in range(9)])
            zs.append(z)
    return torch.tensor(xs), torch.tensor(ms), torch.tensor(ps, dtype=torch.float32), torch.tensor(zs).unsqueeze(1)


def losses(model, x, m, p, z):
    logp, v = model.logits(x, m)
    lp = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(1).mean()
    lv = ((z - v) ** 2).mean()
    return lp, lv


POS = [b for b in G.all_positions() if not G.terminal(b)]
BEST = [set(G.best_moves(b)) for b in POS]
DECISIVE = [i for i, b in enumerate(POS) if len(BEST[i]) < len(G.legal(b))]
PX = torch.tensor([G.view(b) for b in POS])
PM = torch.tensor([G.mask(b) for b in POS])


def accuracy(model):
    with torch.no_grad():
        p, _ = model(PX, PM)
    top = p.argmax(1).tolist()
    ok = [top[i] in BEST[i] for i in range(len(POS))]
    return sum(ok) / len(ok), sum(ok[i] for i in DECISIVE) / len(DECISIVE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    a = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    model = PVNet()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.default_rng(SEED)
    log, buf, start = [], [], 0
    if (RUN / "state.pt").is_file():
        s = torch.load(RUN / "state.pt", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); rng.bit_generator.state = s["rng"]
        torch.set_rng_state(s["torch_rng"]); log, buf, start = s["log"], s["buf"], s["round"]
        print(f"resumed after round {start}")
    if start == 0:
        acc, dec = accuracy(model)
        log.append({"round": 0, "games": 0, "acc": acc, "accDecisive": dec, "lossP": None, "lossV": None})
    for r in range(start + 1, a.rounds + 1):
        t0 = time.time()
        net = NumpyNet(model.state_dict())
        games, res = [], {1: 0, -1: 0, 0: 0}
        for _ in range(GAMES):
            g, w = self_play(net, rng)
            games += g
            res[w] += 1
        buf = (buf + [games])[-BUFFER:]
        x, m, p, z = examples([e for gs in buf for e in gs])
        xr, mr, pr, zr = examples(games)
        with torch.no_grad():
            lp0, lv0 = losses(model, xr, mr, pr, zr)
        model.train()
        for _ in range(STEPS):
            idx = torch.randint(0, x.shape[0], (BATCH,))
            lp, lv = losses(model, x[idx], m[idx], p[idx], z[idx])
            opt.zero_grad()
            (lp + lv).backward()
            opt.step()
        model.eval()
        acc, dec = accuracy(model)
        log.append({"round": r, "games": r * GAMES, "acc": acc, "accDecisive": dec, "lossP": float(lp0), "lossV": float(lv0),
                    "xWins": res[1], "oWins": res[-1], "draws": res[0], "examples": len(games)})
        print(f"round {r}: {len(games)} positions, X {res[1]} O {res[-1]} draw {res[0]}, loss p {lp0:.3f} v {lv0:.3f} -> top prior perfect {acc:.3f} (decisive {dec:.3f}) [{time.time() - t0:.1f}s]", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "log": log, "buf": buf, "round": r}, RUN / "state.pt")
        torch.save(model.state_dict(), RUN / "best.pt")
        (RUN / "log.json").write_text(json.dumps({"seed": SEED, "rounds": r, "games": GAMES, "sims": SIMS, "log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
