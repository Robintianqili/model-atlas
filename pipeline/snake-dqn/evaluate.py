"""snake-dqn: measure the trained agents (the facts the page quotes), in float64 like the page.

    npm run atlas -- py pipeline/snake-dqn/evaluate.py --write     # runs/main/stats.json (and runs/hurry/ if trained)

The greedy agent (always the move with the highest Q) plays seeds 1..SEEDS from the start, as the page's `agent` process
does: a game stops after G.HUNGER moves without food or CAP moves. For each agent: scores, how games end, and
"avoidable crashes": crashes where at least one of the three moves would not have crashed on that move. Baselines: a
player that moves at random, and one that picks at random among the moves that do not crash at once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import game as G  # noqa: E402
from model import QNet  # noqa: E402

SEEDS, CAP = 1000, 1500


def run(policy, seeds, rewards=G.REWARDS["main"]) -> dict:
    """policy(list of games) -> list of actions. Plays every seed side by side."""
    games = [G.Game(s, rewards) for s in seeds]
    avoidable = 0
    alive = list(range(len(games)))
    while alive:
        acts = policy([games[i] for i in alive])
        nxt = []
        for i, a in zip(alive, acts):
            g = games[i]
            could = any(g.safe(b) for b in range(3))
            g.step(int(a))
            if g.over in ("wall", "self") and could:
                avoidable += 1
            if not g.over and g.hunger < G.HUNGER and g.moves < CAP:
                nxt.append(i)
        alive = nxt
    sc = np.array([g.score for g in games])
    ends = {k: sum(1 for g in games if (g.over or ("hungry" if g.hunger >= G.HUNGER else "cap")) == k) for k in ("wall", "self", "full", "hungry", "cap")}
    crashes = ends["wall"] + ends["self"]
    return {"games": len(games), "mean": float(sc.mean()), "median": float(np.median(sc)), "max": int(sc.max()), "min": int(sc.min()),
            "meanMoves": float(np.mean([g.moves for g in games])), "ends": ends, "crashes": crashes, "avoidable": avoidable,
            "avoidableShare": avoidable / crashes if crashes else 0.0, "scores": [int(x) for x in sc[:50]]}


def greedy(path: Path):
    ck = torch.load(path, map_location="cpu")
    net = QNet().double().eval()
    net.load_state_dict({k: v.double() for k, v in ck["online"].items()})

    def pol(games):
        x = torch.from_numpy(np.stack([g.encode() for g in games])).double()
        with torch.no_grad():
            return net(x).argmax(1).tolist()
    return pol, ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    seeds = range(1, SEEDS + 1)
    rng = np.random.default_rng(0)
    out = {"seeds": [1, SEEDS], "cap": CAP, "hunger": G.HUNGER}
    out["random"] = run(lambda gs: rng.integers(0, 3, len(gs)).tolist(), seeds)
    out["randomSafe"] = run(lambda gs: [int(rng.choice([b for b in range(3) if g.safe(b)] or [0])) for g in gs], seeds)
    for v in ("main", "hurry"):
        p = HERE / "runs" / v / "best.pt"
        if not p.exists():
            continue
        pol, ck = greedy(p)
        out[v] = run(pol, seeds)
        out[v]["checkpoint"] = {"moves": ck["moves"], "updates": ck["updates"], "sinceSync": ck["sinceSync"], "eval": ck["eval"]}
        print(v, {k: out[v][k] for k in ("mean", "median", "max", "meanMoves", "ends", "avoidable", "avoidableShare")})
    print("random", out["random"]["mean"], "random safe", out["randomSafe"]["mean"])
    if a.write:
        (HERE / "runs" / "main" / "stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("wrote runs/main/stats.json")


if __name__ == "__main__":
    main()
