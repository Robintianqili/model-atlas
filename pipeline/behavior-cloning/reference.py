"""behavior-cloning: Python reference for the process 'roll' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/behavior-cloning/{world,process}.ts in plain Python floats: the
same grid rules, the same expert (breadth-first distances to the goal, the first of up, right, down, left that gets one
move closer), the policy network's forward pass from the given weights (tanh hidden layer, four scores, softmax) and the
windy episodes. `npm run atlas -- parity behavior-cloning` records the TS tape of each roll source, runs this file on the
same input (the learner's input, the weights and the dataset's cells) and compares the tapes frame by frame: phase, step
and sub exactly, the visits and the cell exactly, every metric within 1e-12 (AK204).

The gusts are drawn first, one per move as the TS windOf does: rng.next() < wind for each of the episodes' T moves, from
ak.Rng(seed) (bit-identical to the TS rng for next()), so no draws need replaying.

    npm run atlas -- parity behavior-cloning                 # the usual way (it passes the arguments below)

CLI contract: --process roll --seed <n> --opt <json> --draws <draws.json> --input <json | @file> --out <stem>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np

MAPS = {
    "field": ["........G", ".........", ".........", "S........", ".........", "PPPPPPPPP"],
    "corner": ["S.......", "........", "........", ".......G", "........", "PPPPPPPP"],
}
OPEN, WALL, GOAL, PIT = 0, 1, 2, 3
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
GUST = 2  # a gust blows the agent one cell down


class World:
    def __init__(self, name: str):
        rows = MAPS.get(name, MAPS["field"])
        self.w, self.h = len(rows[0]), len(rows)
        self.cell, self.start = [], 0
        for y, r in enumerate(rows):
            for x, c in enumerate(r):
                if c == "S":
                    self.start = y * self.w + x
                self.cell.append({"#": WALL, "G": GOAL, "P": PIT}.get(c, OPEN))

    def end(self, s: int) -> bool:
        return self.cell[s] in (GOAL, PIT)

    def step(self, s: int, a: int) -> int:
        x, y = s % self.w, s // self.w
        nx, ny = x + DIRS[a][0], y + DIRS[a][1]
        if nx < 0 or ny < 0 or nx >= self.w or ny >= self.h or self.cell[ny * self.w + nx] == WALL:
            return s
        return ny * self.w + nx

    def distances(self) -> list[int]:
        d = [0 if c == GOAL else -1 for c in self.cell]
        q = deque(s for s, v in enumerate(d) if v == 0)
        while q:
            s = q.popleft()
            x, y = s % self.w, s // self.w
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    n = ny * self.w + nx
                    if self.cell[n] == OPEN and d[n] < 0:
                        d[n] = d[s] + 1
                        q.append(n)
        return d

    def expert(self, d: list[int], s: int) -> int:
        if self.cell[s] != OPEN or d[s] < 0:
            return -1
        for a in range(4):
            if d[self.step(s, a)] == d[s] - 1:
                return a
        return -1

    def features(self, s: int) -> tuple[float, float]:
        return (2 * (s % self.w)) / (self.w - 1) - 1, (2 * (s // self.w)) / (self.h - 1) - 1


def policy(P: dict, x: tuple[float, float]):
    """tanh(W0 x + b0), then W2 h + b2, softmax; the move with the highest score (ties to the smallest index)."""
    W0, b0, W2, b2 = (P[k]["data"] for k in ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"))
    H = P["net.0.weight"]["shape"][0]
    h = [math.tanh(W0[j * 2] * x[0] + W0[j * 2 + 1] * x[1] + b0[j]) for j in range(H)]
    z = []
    for k in range(4):
        v = b2[k]
        for j in range(H):
            v += W2[k * H + j] * h[j]
        z.append(v)
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    tot = e[0] + e[1] + e[2] + e[3]
    return [v / tot for v in e], z.index(mx)


def to_wind(v) -> float:
    return min(0.5, max(0.0, float(v))) if isinstance(v, (int, float)) and math.isfinite(v) else 0.2


def run_roll(seed: int, opt: dict, inp: dict):
    o = {"episodes": 20, "T": 30, **opt}
    raw, P, data = inp["raw"], inp["params"], set(inp["cells"])
    W = World(raw.get("map", "field"))
    wind_p = to_wind(raw.get("wind"))
    d = W.distances()
    rng = ak.Rng(seed)
    wind = [[rng.next() < wind_p for _ in range(o["T"])] for _ in range(o["episodes"])]
    schema = {"pos": {"kind": "int", "label": {"en": "pos", "zh": "pos"}, "tol": "exact"},
              "visits": {"kind": "tensor", "label": {"en": "visits", "zh": "visits"}, "track": "changes", "tol": "exact"}}
    phases = {"start": {"label": "start", "kind": "input"}, "move": {"label": "move", "kind": "act"},
              "end": {"label": "end", "kind": "observe"}, "done": {"label": "done", "kind": "aggregate"}}
    tape = ak.TapeWriter("roll", schema, seed=seed, opt=o, phases=phases, provenance="live")
    visits = np.zeros((W.h, W.w))
    goals = ex_goals = pits = 0
    n = {"onMoves": 0, "onGood": 0, "offMoves": 0, "offGood": 0}
    tape.frame("start", {"pos": W.start, "visits": visits.copy()},
               metrics={"episodes": o["episodes"], "T": o["T"], "wind": wind_p, "start": W.start, "cells": len(data)})
    for e in range(o["episodes"]):
        s, t, off, first_off, gusts = W.start, 0, 0, -1, 0
        while t < o["T"] and not W.end(s):
            p, a = policy(P, W.features(s))
            gust = wind[e][t]
            s2 = W.step(s, GUST if gust else a)
            seen = 1 if s in data else 0
            good = 1 if d[s] >= 0 and d[W.step(s, a)] == d[s] - 1 else 0
            if seen:
                n["onMoves"] += 1
                n["onGood"] += good
            else:
                n["offMoves"] += 1
                n["offGood"] += good
                off += 1
                if first_off < 0:
                    first_off = t
            gusts += 1 if gust else 0
            x2, y2 = s2 % W.w, s2 // W.w
            old = visits[y2, x2]
            visits[y2, x2] = old + 1
            tape.frame("move", {"pos": s2, "visits": visits.copy()}, step=t, sub=e,
                       changes=[{"field": "visits", "index": [y2, x2], "from": old, "to": old + 1}],
                       metrics={"e": e, "t": t, "s": s, "x": s % W.w, "y": s // W.w, "a": a, "gust": 1 if gust else 0, "s2": s2, "x2": x2, "y2": y2,
                                "seen": seen, "good": good, "ex": W.expert(d, s), "p0": p[0], "p1": p[1], "p2": p[2], "p3": p[3]})
            s = s2
            t += 1
        outcome = 1 if W.cell[s] == GOAL else -1 if W.cell[s] == PIT else 0
        # the expert on the same gusts
        c, k = W.start, 0
        while k < o["T"] and not W.end(c):
            c = W.step(c, GUST if wind[e][k] else W.expert(d, c))
            k += 1
        ex = 1 if W.cell[c] == GOAL else -1 if W.cell[c] == PIT else 0
        goals += outcome == 1
        pits += outcome == -1
        ex_goals += ex == 1
        tape.frame("end", {"pos": s}, step=e, sub=e,
                   metrics={"e": e, "outcome": outcome, "len": t, "off": off, "firstOff": first_off, "gusts": gusts, "last": s,
                            "exOutcome": ex, "goals": goals, "exGoals": ex_goals, "pits": pits})
    tape.frame("done", {"pos": W.start}, metrics={"episodes": o["episodes"], "goals": goals, "exGoals": ex_goals, "pits": pits,
                                                  "success": goals / o["episodes"], "exSuccess": ex_goals / o["episodes"], **n})
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="roll", choices=["roll"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran")
    ap.add_argument("--draws", default=None, help="draws.json written by the TS run (unused: no draws are recorded)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run: { raw, params, cells } (or @file)")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/behavior-cloning/roll.ref)")
    a = ap.parse_args()
    tape = run_roll(a.seed, json.loads(a.opt), ak.json_arg(a.input))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "behavior-cloning", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
