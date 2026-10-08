"""snake-dqn: find the page's preset positions in the trained agent's own games (writes pipeline/snake-dqn/presets.json).

    npm run atlas -- py pipeline/snake-dqn/presets.py

Each preset is a food seed and the moves from the start (so it is reachable and the Capture can undo it). The agent
(float64, greedy) plays seeds 1, 2, ...; the first board on each seed that fits a description becomes that preset:
  start  move 2 of seed 7 (the lesson's start input)
  wall   the head on the edge, heading into the wall (straight on crashes), length >= 5
  food   the food right in front of the head
  long   length >= 15
  tight  only one of the three moves does not crash at once
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import game as G  # noqa: E402
from model import QNet  # noqa: E402

ck = torch.load(HERE / "runs" / "main" / "best.pt", map_location="cpu")
net = QNet().double().eval()
net.load_state_dict({k: v.double() for k, v in ck["online"].items()})


def q_of(g: G.Game) -> list:
    with torch.no_grad():
        return net(torch.from_numpy(g.encode()[None]).double())[0].tolist()


def boards(seed: int):
    g, moves = G.Game(seed), []
    while not g.over and g.hunger < G.HUNGER and g.moves < 1500:
        yield g, list(moves)
        q = q_of(g)
        a = int(np.argmax(q))
        moves.append(a)
        g.step(a)


TESTS = {
    "wall": lambda g: g.next_cell(0) < 0 and len(g.snake) >= 5,
    "food": lambda g: g.next_cell(0) == g.food,
    "long": lambda g: len(g.snake) >= 15,
    "tight": lambda g: sum(g.safe(a) for a in range(3)) == 1 and len(g.snake) >= 8,
}
NAMES = {"start": ("A game just started", "刚开局的一盘"), "wall": ("Next to the wall", "贴着墙"), "food": ("Food right ahead", "食物就在前面"),
         "long": ("A long snake", "一条长蛇"), "tight": ("Only one way out", "只有一条出路")}
found = {"start": (7, [1, 0])}
for seed in range(1, 200):
    for g, moves in boards(seed):
        for k, t in TESTS.items():
            if k not in found and len(moves) > 3 and t(g):
                found[k] = (seed, moves)
    if len(found) == len(NAMES):
        break
out = [{"id": k, "en": NAMES[k][0], "zh": NAMES[k][1], "seed": s, "moves": m} for k, (s, m) in found.items()]
for p in out:
    g = G.replay(p["seed"], p["moves"])
    print(p["id"], p["seed"], len(p["moves"]), "length", len(g.snake), "q", [round(v, 3) for v in q_of(g)], "safe", [g.safe(a) for a in range(3)])
(HERE / "presets.json").write_text(json.dumps(out), encoding="utf-8")
