"""gridworld: Python reference for the processes 'bandit', 'race', 'q' and 'vi' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/gridworld/{world,process}.ts in plain Python floats: the same
maze rules, Q-learning, value iteration, slot machines and strategies. `npm run atlas -- parity gridworld` records the TS
tape of every process source (with its draws), runs this file and compares the two tapes frame by frame: phase, step and
sub exactly, every stored field within its tol (Q and V 1e-12, counts exact), every metric within 1e-12 (AK204).

Draws are replayed: every value the TS run drew (rng.next, rng.int, rng.normal, in call order across the forks) comes
from --draws. Each replayed draw is also checked against this file's own ak.Rng stream (bit-identical for next/int/fork;
normal within 1e-12, as libm may differ in the last ulp), so a reference that called the rng in another order fails here.

    npm run atlas -- parity gridworld                 # the usual way (it passes the arguments below)
    npm run atlas -- py pipeline/gridworld/reference.py --process q --seed 12 --input '{"rows": ["S.", ".G"]}'

CLI contract: --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np

PROBS = [0.3, 0.55, 0.45, 0.7]
K = len(PROBS)
BEST = PROBS.index(max(PROBS))
POLICIES = ["greedy", "eps", "ucb", "thompson"]
CLASSIC = [".....G", ".###..", "...#P.", ".#....", "S#.#.."]
DEFAULTS = {"rows": CLASSIC, "pulls": [], "gamma": 0.9, "epsilon": 0.1, "alpha": 0.5, "episodes": 50}
OPEN, WALL, GOAL, PIT = 0, 1, 2, 3
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
MAX_PULLS = 40


# ------------------------------------------------------------------ replayed draws
class Draws:
    def __init__(self, frames: list | None):
        self.flat = [d for f in (frames or []) for d in f]
        self.i = 0

    def take(self, own: float, exact: bool, what: str) -> float:
        if not self.flat:
            return own
        if self.i >= len(self.flat):
            sys.exit(f"reference: the TS run made fewer draws than this one ({what} #{self.i})")
        rec = self.flat[self.i]
        self.i += 1
        ok = rec == own if exact else abs(rec - own) <= 1e-12 * max(1.0, abs(rec))
        if not ok:
            sys.exit(f"reference: draw #{self.i - 1} ({what}) is {rec} in TS but {own} here: the rng is called in another order")
        return rec


class Stream:
    """An ak.Rng whose values are replaced by the replayed TS draws (and checked against them)."""

    def __init__(self, rng: "ak.Rng", draws: Draws):
        self.rng, self.draws = rng, draws

    def next(self) -> float:
        return self.draws.take(self.rng.next(), True, "next")

    def int(self, n: int) -> int:
        return int(self.draws.take(self.rng.int(n), True, "int"))

    def normal(self) -> float:
        return self.draws.take(self.rng.normal(), False, "normal")

    def fork(self, label: str) -> "Stream":
        return Stream(self.rng.fork(label), self.draws)


# ------------------------------------------------------------------ the learner's input and the maze
def clamp(v, lo, hi, d):
    return min(hi, max(lo, v)) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else d


def to_raw(raw) -> dict:
    r = raw if isinstance(raw, dict) else {}
    rows = [str(s)[:8] for s in r["rows"][:8]] if isinstance(r.get("rows"), list) and len(r["rows"]) >= 2 else CLASSIC
    w = max(2, *[len(s) for s in rows])
    seen, out = False, []
    for s in rows:
        row = ""
        for c in s.ljust(w, "."):
            if c == "S":
                row += "." if seen else "S"
                seen = True
            else:
                row += c if c in ".#GP" else "."
        out.append(row)
    if not seen:
        out[-1] = "S" + out[-1][1:]
    pulls = [a for a in (r.get("pulls") or []) if isinstance(a, int) and not isinstance(a, bool) and 0 <= a < K][:MAX_PULLS]
    return {"rows": out, "pulls": pulls, "gamma": clamp(r.get("gamma"), 0, 0.99, 0.9), "epsilon": clamp(r.get("epsilon"), 0, 1, 0.1),
            "alpha": clamp(r.get("alpha"), 0.05, 1, 0.5), "episodes": math.floor(clamp(r.get("episodes"), 1, 200, 50) + 0.5)}


class World:
    def __init__(self, raw):
        rows = to_raw(raw)["rows"]
        self.w, self.h = len(rows[0]), len(rows)
        self.cell = [WALL if c == "#" else GOAL if c == "G" else PIT if c == "P" else OPEN for s in rows for c in s]
        self.start = next(y * self.w + x for y, s in enumerate(rows) for x, c in enumerate(s) if c == "S")

    def end(self, s):
        return self.cell[s] in (GOAL, PIT)

    def step(self, s, a):
        x, y = s % self.w, s // self.w
        nx, ny = x + DIRS[a][0], y + DIRS[a][1]
        s2 = s if nx < 0 or ny < 0 or nx >= self.w or ny >= self.h or self.cell[ny * self.w + nx] == WALL else ny * self.w + nx
        r = 1 if self.cell[s2] == GOAL else -1 if self.cell[s2] == PIT else 0
        return s2, r, self.end(s2)

    def goal_distance(self):
        dist, queue = {self.start: 0}, [self.start]
        while queue:
            s = queue.pop(0)
            if self.cell[s] == GOAL:
                return dist[s]
            if self.end(s):
                continue
            for a in range(4):
                s2 = self.step(s, a)[0]
                if s2 not in dist:
                    dist[s2] = dist[s] + 1
                    queue.append(s2)
        return -1

    def backup(self, V, s, a, gamma):
        s2, r, done = self.step(s, a)
        return r + (0 if done else gamma * V[s2])

    def value_iteration(self, gamma, on_sweep=None):
        V = [0.0] * len(self.cell)
        for k in range(1000):
            nxt = [0 if self.cell[s] != OPEN else max(self.backup(V, s, a, gamma) for a in range(4)) for s in range(len(V))]
            delta = max(abs(v - V[s]) for s, v in enumerate(nxt))
            V = nxt
            if on_sweep:
                on_sweep(V, k, delta)
            if delta < 1e-10:
                break
        return V


def arrow(Q, s):
    v = list(Q[s])
    m = max(v)
    return v.index(m) if v.count(m) == 1 else -1


def greedy_path_len(W, Q):
    path, s = [W.start], W.start
    while len(path) <= len(W.cell):
        a = arrow(Q, s)
        if a < 0:
            return -1
        s = W.step(s, a)[0]
        if s in path or W.cell[s] == PIT:
            return -1
        path.append(s)
        if W.cell[s] == GOAL:
            return len(path) - 1
    return -1


def agreement(W, Q, V, gamma):
    decided = agree = 0
    for s, c in enumerate(W.cell):
        a = arrow(Q, s) if c == OPEN else -1
        if a < 0:
            continue
        decided += 1
        best = max(W.backup(V, s, b, gamma) for b in range(4))
        agree += W.backup(V, s, a, gamma) >= best - 1e-9
    return decided, agree


def label(en, zh):
    return (en, zh)


# ------------------------------------------------------------------ bandits
class Player:
    def __init__(self, policy, rng: Stream):
        self.policy, self.Q, self.N, self.R, self.regret = policy, [0.0] * K, [0.0] * K, [0.0] * K, 0.0
        self.rng = rng.fork(policy)
        self.arms = [rng.fork(f"arm{k}") for k in range(K)]


def gamma_draw(rng: Stream, a):
    d = a - 1 / 3
    c = 1 / math.sqrt(9 * d)
    while True:
        while True:
            x = rng.normal()
            v = 1 + c * x
            if v > 0:
                break
        v = v * v * v
        u = rng.next()
        if u < 1 - 0.0331 * x * x * x * x or math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
            return d * v


def decide(p: Player, t, opt, mine=None):
    m = max(p.Q)
    best = [i for i, q in enumerate(p.Q) if q == m]
    bonus = [opt["c"] * math.sqrt(math.log(t) / n) if n else -1 for n in p.N]
    u, explore = -1, 0

    def tie():
        return best[p.rng.int(len(best))] if len(best) > 1 else best[0]

    if p.policy == "you":
        arm = mine
    elif p.policy == "greedy":
        arm = tie()
    elif p.policy == "eps":
        u = p.rng.next()
        if u < opt["eps"]:
            arm, explore = p.rng.int(K), 1
        else:
            arm = tie()
    elif p.policy == "ucb":
        score = [q + bonus[k] for k, q in enumerate(p.Q)]
        arm = p.N.index(0) if 0 in p.N else score.index(max(score))
    else:
        theta = []
        for k in range(K):
            x = gamma_draw(p.rng, 1 + p.R[k])
            theta.append(x / (x + gamma_draw(p.rng, 1 + p.N[k] - p.R[k])))
        arm = theta.index(max(theta))
    return arm, u, explore, 1 if arm in best else 0, bonus


def pull(p: Player, arm):
    r = 1 if p.arms[arm].next() < PROBS[arm] else 0
    old = p.Q[arm]
    p.N[arm] += 1
    p.R[arm] += r
    p.Q[arm] = old + (r - old) / p.N[arm]
    p.regret += PROBS[BEST] - PROBS[arm]
    return r, old


def per(prefix, v):
    return {f"{prefix}{k}": x for k, x in enumerate(v)}


BANDIT_SCHEMA = {
    "Q": {"kind": "tensor", "label": label("Average payout of each machine", "每台机器的平均回报"), "shape": [K], "tol": 1e-12},
    "N": {"kind": "tensor", "label": label("Pulls of each machine", "每台机器被拉的次数"), "shape": [K], "tol": "exact"},
    "R": {"kind": "tensor", "label": label("Coins won on each machine", "每台机器赢到的金币"), "shape": [K], "tol": "exact"},
}
BANDIT_PHASES = {"init": {"label": "init", "kind": "other"}, "choose": {"label": "choose", "kind": "act"},
                 "score": {"label": "score", "kind": "act"}, "update": {"label": "update", "kind": "update"}}


def run_bandit(seed, opt, raw, draws):
    o = {"policy": "greedy", "steps": 200, "eps": 0.1, "c": 1, **opt}
    rng = Stream(ak.Rng(seed), draws)
    mine = to_raw(raw)["pulls"]
    p = Player(o["policy"], rng)
    steps = len(mine) if o["policy"] == "you" else o["steps"]
    tape = ak.TapeWriter("bandit", BANDIT_SCHEMA, seed=seed, opt=o, phases=BANDIT_PHASES, provenance="live")

    def snap():
        return {"Q": np.array(p.Q), "N": np.array(p.N), "R": np.array(p.R)}

    tape.frame("init", snap(), metrics={"steps": steps, "best": BEST, "pBest": PROBS[BEST], "eps": o["eps"], "c": o["c"]})
    for t in range(steps):
        arm, u, explore, is_best, bonus = decide(p, t, o, mine[t] if t < len(mine) else None)
        base = {"t": t, "n": t + 1, "arm": arm, **per("q", p.Q), **per("n", p.N)}
        tape.frame("score" if o["policy"] == "ucb" else "choose", {}, step=t,
                   metrics={**base, "u": u, "eps": o["eps"], "explore": explore, "isBest": is_best, **per("b", bonus),
                            "q": p.Q[arm], "c": o["c"], "cnt": p.N[arm], "score": p.Q[arm] + bonus[arm]})
        r, old = pull(p, arm)
        tape.frame("update", snap(), step=t, metrics={"t": t, "n": t + 1, "arm": arm, "r": r, "old": old, "new": p.Q[arm], "cnt": p.N[arm],
                                                       "regret": p.regret, "wins": sum(p.R), "nBest": p.N[BEST]})
    return tape


def run_race(seed, opt, draws):
    o = {"policy": "greedy", "steps": 200, "eps": 0.1, "c": 1, **opt}
    rng = Stream(ak.Rng(seed), draws)
    ps = [Player(pol, rng) for pol in POLICIES]
    schema = {"N": {"kind": "tensor", "label": label("Pulls per strategy and machine", "每种策略在每台机器上的拉动次数"), "shape": [4, K], "tol": "exact"},
              "R": {"kind": "tensor", "label": label("Coins per strategy and machine", "每种策略在每台机器上赢的金币"), "shape": [4, K], "tol": "exact"}}
    tape = ak.TapeWriter("race", schema, seed=seed, opt=o, phases={"pull": {"label": "pull", "kind": "act"}}, provenance="live")
    for t in range(o["steps"]):
        arms = [decide(p, t, o)[0] for p in ps]  # every player decides, then every player pulls (the TS order)
        for p, k in zip(ps, arms):
            pull(p, k)
        tape.frame("pull", {"N": np.array([p.N for p in ps]), "R": np.array([p.R for p in ps])}, step=t,
                   metrics={"t": t, "n": t + 1, **per("r", [p.regret for p in ps]), **per("a", arms)})
    return tape


# ------------------------------------------------------------------ the maze
def run_q(seed, opt, raw, draws):
    o = {"maxSteps": 80, **opt}
    rng = Stream(ak.Rng(seed), draws)
    g = to_raw(raw)
    W = World(g)
    gamma, eps, alpha, episodes, mx = g["gamma"], g["epsilon"], g["alpha"], g["episodes"], o["maxSteps"]
    n = len(W.cell)
    Q = np.zeros((n, 4))
    N = np.zeros(n)
    V = W.value_iteration(gamma)
    schema = {"board": {"kind": "tensor", "label": label("board", "board"), "tol": "exact"},
              "Q": {"kind": "tensor", "label": label("Q", "Q"), "track": "changes", "tol": 1e-12},
              "N": {"kind": "tensor", "label": label("N", "N"), "track": "changes", "tol": "exact"},
              "pos": {"kind": "int", "label": label("pos", "pos"), "tol": "exact"}}
    phases = {"init": {"label": "init", "kind": "other"}, "act": {"label": "act", "kind": "act"},
              "update": {"label": "update", "kind": "update"}, "end": {"label": "end", "kind": "observe"}}
    tape = ak.TapeWriter("q", schema, seed=seed, opt=o, phases=phases, provenance="live")
    n_open = W.cell.count(OPEN)
    tape.frame("init", {"board": np.array(W.cell, dtype=np.float64).reshape(W.h, W.w), "Q": Q.copy(), "N": N.copy(), "pos": W.start},
               metrics={"start": W.start, "open": n_open, "entries": n_open * 4, "gamma": gamma, "eps": eps, "alpha": alpha, "episodes": episodes,
                        "maxSteps": mx, "dist": W.goal_distance(), "vStar": V[W.start]})
    goals = 0
    for e in range(episodes):
        s, ret, how, moves = W.start, 0, 0, 0
        while moves < mx and not how:
            qs = list(Q[s])
            u = rng.next()
            explore, ties = 0, 1
            if u < eps:
                a, explore = rng.int(4), 1
            else:
                best = [i for i, v in enumerate(qs) if v == max(qs)]
                ties = len(best)
                a = best[rng.int(ties)] if ties > 1 else best[0]
            s2, r, done = W.step(s, a)
            nv = N[s]
            N[s] = nv + 1
            tape.frame("act", {"N": N.copy(), "pos": s2}, step=moves, sub=e, changes=[{"field": "N", "index": [s], "from": nv, "to": nv + 1}],
                       metrics={"s": s, "x": s % W.w, "y": s // W.w, "u": u, "eps": eps, "explore": explore, "ties": ties, "a": a, **per("q", qs),
                                "s2": s2, "x2": s2 % W.w, "y2": s2 // W.w, "r": r, "done": 1 if done else 0, "bump": 1 if s2 == s else 0})
            max_next = 0 if done else max(Q[s2])
            target = r + gamma * max_next
            old = Q[s, a]
            delta = target - old
            new = old + alpha * delta
            Q[s, a] = new
            tape.frame("update", {"Q": Q.copy()}, step=moves, sub=e, changes=[{"field": "Q", "index": [s, a], "from": old, "to": new}],
                       metrics={"s": s, "a": a, "s2": s2, "r": r, "done": 1 if done else 0, "gamma": gamma, "alpha": alpha,
                                "maxNext": max_next, "target": target, "old": old, "delta": delta, "new": new})
            ret += r
            moves += 1
            s = s2
            if done:
                how = 1 if r > 0 else -1
        goals += how == 1
        decided, agree = agreement(W, Q, V, gamma)
        tape.frame("end", {}, step=e, metrics={"ep": e, "moves": moves, "ret": ret, "how": how, "goals": goals, "vStart": max(Q[W.start]),
                                                "arrow": arrow(Q, W.start), "len": greedy_path_len(W, Q), "decided": decided, "agree": agree,
                                                "end": s if W.end(s) else -1})
    return tape


def run_vi(seed, opt, raw):
    W = World(raw)
    gamma = opt["gamma"] if opt.get("gamma", -1) >= 0 else to_raw(raw)["gamma"]
    schema = {"board": {"kind": "tensor", "label": label("board", "board"), "tol": "exact"},
              "V": {"kind": "tensor", "label": label("V", "V"), "tol": 1e-12}}
    tape = ak.TapeWriter("vi", schema, seed=seed, opt=opt, phases={"init": {"label": "init", "kind": "other"}, "sweep": {"label": "sweep", "kind": "update"}},
                         provenance="live")
    sweeps = []
    W.value_iteration(gamma, lambda V, k, delta: sweeps.append((V, k, delta)))
    tape.frame("init", {"board": np.array(W.cell, dtype=np.float64).reshape(W.h, W.w), "V": np.zeros((W.h, W.w))}, metrics={"start": W.start, "gamma": gamma, "dist": W.goal_distance()})
    for V, k, delta in sweeps:
        tape.frame("sweep", {"V": np.array(V).reshape(W.h, W.w)}, step=k, metrics={"k": k, "delta": delta, "gamma": gamma, "vStart": V[W.start]})
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="q", choices=["bandit", "race", "q", "vi"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran")
    ap.add_argument("--draws", default=None, help="draws.json written by the TS run (replayed and checked)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run (the learner's GameRaw)")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/gridworld/<process>.ref)")
    a = ap.parse_args()
    opt, raw = json.loads(a.opt), json.loads(a.input)
    draws = Draws(json.loads(Path(a.draws).read_text(encoding="utf-8"))["frames"] if a.draws else None)
    if a.process == "bandit":
        tape = run_bandit(a.seed, opt, raw, draws)
    elif a.process == "race":
        tape = run_race(a.seed, opt, draws)
    elif a.process == "q":
        tape = run_q(a.seed, opt, raw, draws)
    else:
        tape = run_vi(a.seed, opt, raw)
    if draws.flat and draws.i != len(draws.flat):
        sys.exit(f"reference: used {draws.i} of the {len(draws.flat)} TS draws")
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "gridworld", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames, {draws.i} draws replayed)")
