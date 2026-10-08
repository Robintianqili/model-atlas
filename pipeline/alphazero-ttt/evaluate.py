"""alphazero-ttt: measure a checkpoint against the exact solver (game.py).

    npm run atlas -- py pipeline/alphazero-ttt/evaluate.py [checkpoint.pt] [--sims 50]

- top prior: on how many of the 4,520 positions where a move is still to be made the network's favourite move (its
  largest prior) is a perfect move (one that keeps the best result perfect play allows); "decisive": only the positions
  where some legal move is not perfect.
- search: the same for the move the search plays after `sims` simulations.
- unbeatable: the search plays every possible opponent, as X and as O: every line of opponent moves is tried (the search
  is deterministic), and each game's result is counted.
Prints a JSON dict (export.py imports measure()).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import game as G  # noqa: E402
from mcts import NumpyNet, most_visited, search  # noqa: E402

DEFAULT = HERE / "runs" / "selfplay" / "best.pt"


def search_move(net, b, sims):
    t = search(b, net, sims)
    return int(t.move[most_visited(t)])


def prior_move(net, b):
    p, _ = net(b)
    return int(max(G.legal(b), key=lambda c: (p[c], -c)))


def against_all(choose, ai):
    """Play the deterministic chooser `choose(b)` as `ai` (1 = X, -1 = O) against every opponent line: counts by result."""
    out = {"win": 0, "draw": 0, "loss": 0}
    cache = {}

    def rec(b):
        if G.terminal(b):
            w = G.winner(b)
            out["draw" if w == 0 else "win" if w == ai else "loss"] += 1
            return
        if G.to_move(b) == ai:
            if b not in cache:
                cache[b] = choose(b)
            rec(G.play(b, cache[b]))
        else:
            for c in G.legal(b):
                rec(G.play(b, c))

    rec(G.EMPTY)
    return out


def measure(state_dict, sims=50):
    net = NumpyNet(state_dict)
    pos = [b for b in G.all_positions() if not G.terminal(b)]
    best = [set(G.best_moves(b)) for b in pos]
    dec = [i for i, b in enumerate(pos) if len(best[i]) < len(G.legal(b))]
    pm = [prior_move(net, b) for b in pos]
    sm = [search_move(net, b, sims) for b in pos]
    okp = [pm[i] in best[i] for i in range(len(pos))]
    oks = [sm[i] in best[i] for i in range(len(pos))]
    r = {
        "positions": len(pos), "decisive": len(dec), "sims": sims,
        "priorPerfect": sum(okp) / len(pos), "priorPerfectDecisive": sum(okp[i] for i in dec) / len(dec),
        "searchPerfect": sum(oks) / len(pos), "searchPerfectDecisive": sum(oks[i] for i in dec) / len(dec),
        "fixedBySearch": sum(1 for i in range(len(pos)) if oks[i] and not okp[i]),
        "brokenBySearch": sum(1 for i in range(len(pos)) if okp[i] and not oks[i]),
    }
    for name, fn in (("search", lambda b: search_move(net, b, sims)), ("prior", lambda b: prior_move(net, b))):
        for ai, side in ((1, "X"), (-1, "O")):
            res = against_all(fn, ai)
            r[f"{name}As{side}"] = res
    return r


def sweep(state_dict, sims_list=(1, 5, 10, 25, 50, 100, 200), c=None):
    """Share of the 4,520 positions where the search's move is perfect, for several numbers of simulations."""
    import mcts
    net = NumpyNet(state_dict)
    pos = [b for b in G.all_positions() if not G.terminal(b)]
    best = [set(G.best_moves(b)) for b in pos]
    out = {}
    for s in sims_list:
        ok = 0
        for b, bs in zip(pos, best):
            t = search(b, net, s, c if c is not None else mcts.C_PUCT)
            ok += int(t.move[most_visited(t)]) in bs
        out[s] = ok / len(pos)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=str(DEFAULT))
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--write", action="store_true", help="also the sweep over 1, 10, 200 simulations, into runs/selfplay/stats.json")
    a = ap.parse_args()
    sd = torch.load(a.checkpoint, map_location="cpu")
    r = measure(sd, a.sims)
    if a.write:
        r["sweep"] = sweep(sd, (1, 10, 50, 200))
        (HERE / "runs" / "selfplay" / "stats.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(json.dumps(r, indent=1))
