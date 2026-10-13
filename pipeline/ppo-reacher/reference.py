"""ppo-reacher: Python reference for the process 'episode' (kind C parity, design §6.2 item 8).

An independent implementation of site/src/lib/explainers/ppo-reacher/{sim,process}.ts in plain Python floats: the
two-link arm (motor torques with friction, Euler steps of 0.05 s, reward -distance - 0.05 * |a|^2), the fresh actor and
critic (their weights rebuilt from ak.Rng(seed).fork('init') exactly as the TS MLPs draw them: PyTorch's
U(-1/sqrt(in), 1/sqrt(in)), row-major, weight then bias, layer by layer, the actor before the critic), one episode
sampled from ak.Rng(seed).fork('batch') (two standard normal draws per step), and GAE backwards.
The normal draws are replayed from --draws (rng.normal is not bit-exact across engines); every replayed draw is also
checked against this file's own ak.Rng stream (next() exactly, normal() within 1e-12), so a reference that drew in
another order fails here. `npm run atlas -- parity ppo-reacher` records the TS tape and its draws, runs this file and
compares the tapes frame by frame: phase and step exactly, the state and the advantages within 1e-12 relative, every
metric within 1e-12 (AK204).

    npm run atlas -- parity ppo-reacher --fixtures      # also on every fixture target
    npm run atlas -- py pipeline/ppo-reacher/reference.py --input '{"target": [0, 0.9]}'

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

L1, L2, TAU, GAIN, DAMP, CTRL = 0.5, 0.5, 0.05, 20.0, 10.0, 0.05
START = [0.0, math.pi / 2, 0.0, 0.0]
LOG2PI = math.log(2 * math.pi)
DEFAULT_TARGET = [-0.3, 0.6]
DEFAULTS = {"T": 40, "gamma": 0.95, "lambda": 0.95, "hidden": 32, "logStd0": -0.5}

SCHEMA = {
    "s": {"kind": "tensor", "label": ("Arm [q1, q2, ω1, ω2]", "机械臂 [q1, q2, ω1, ω2]"), "shape": [4], "tol": 1e-12},
    "A": {"kind": "tensor", "label": ("Advantage of each step", "每一步的优势"), "track": "changes", "tol": 1e-12},
}
PHASES = {
    "start": {"label": ("The start", "起点"), "kind": "input"},
    "act": {"label": ("Sample the torques", "采样力矩"), "kind": "sample"},
    "move": {"label": ("One step of the arm", "机械臂走一步"), "kind": "other"},
    "end": {"label": ("The episode ends", "回合结束"), "kind": "observe"},
    "gae": {"label": ("Advantage (GAE)", "优势（GAE）"), "kind": "aggregate"},
}


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

    def normal(self) -> float:
        return self.draws.take(self.rng.normal(), False, "normal")

    def fork(self, label: str) -> "Stream":
        return Stream(self.rng.fork(label), self.draws)


# ------------------------------------------------------------------ the arm and the networks
def target_of(raw) -> list[float]:
    g = raw.get("target") if isinstance(raw, dict) else None
    try:
        x, y = float(g[0]), float(g[1])
    except (TypeError, ValueError, IndexError):
        return list(DEFAULT_TARGET)
    if not (math.isfinite(x) and math.isfinite(y)):
        return list(DEFAULT_TARGET)
    y = max(0.0, y)
    r = math.hypot(x, y)
    if r == 0:
        x, y, k = 0.0, 0.2, 1.0
    else:
        k = 1 / r if r > 1 else 0.2 / r if r < 0.2 else 1.0
    return [round_js(x * k * 100) / 100, round_js(y * k * 100) / 100]


def round_js(v: float) -> float:
    """JS Math.round: halves round up (toward +infinity)."""
    return float(math.floor(v + 0.5))


def tip(s: list[float]) -> list[float]:
    a = s[0] + s[1]
    return [L1 * math.cos(s[0]) + L2 * math.cos(a), L1 * math.sin(s[0]) + L2 * math.sin(a)]


def obs(s: list[float], g: list[float]) -> list[float]:
    p = tip(s)
    return [g[0] - p[0], g[1] - p[1], s[0], s[1], s[2], s[3]]


def step(s: list[float], a: list[float], g: list[float]) -> dict:
    w1 = s[2] + TAU * (GAIN * a[0] - DAMP * s[2])
    w2 = s[3] + TAU * (GAIN * a[1] - DAMP * s[3])
    s2 = [s[0] + TAU * w1, s[1] + TAU * w2, w1, w2]
    p = tip(s2)
    d = math.hypot(p[0] - g[0], p[1] - g[1])
    cost = CTRL * (a[0] * a[0] + a[1] * a[1])
    return {"s2": s2, "d": d, "cost": cost, "r": -d - cost}


def fresh(rng, hidden: int, logstd0: float) -> dict:
    """The actor's then the critic's Linear layers: each weight [out, in] then its bias, U(-b, b) with b = 1/sqrt(in)."""
    def linear(i: int, o: int):
        b = 1 / math.sqrt(i)
        w = [(2 * rng.next() - 1) * b for _ in range(o * i)]
        c = [(2 * rng.next() - 1) * b for _ in range(o)]
        return (w, c, o, i)
    pi = [linear(6, hidden), linear(hidden, 2)]
    v = [linear(6, hidden), linear(hidden, 1)]
    return {"pi": pi, "v": v, "logstd": [float(logstd0), float(logstd0)]}


def mlp(layers, x: list[float]) -> list[float]:
    h = x
    for li, (w, c, o, i) in enumerate(layers):
        y = []
        for k in range(o):
            acc = 0.0
            for j in range(i):
                acc += w[k * i + j] * h[j]
            y.append(acc + c[k])
        h = [math.tanh(v) for v in y] if li + 1 < len(layers) else y
    return h


def log_prob(a, mu, ls) -> float:
    s = 0.0
    for j in range(2):
        z = (a[j] - mu[j]) / math.exp(ls[j])
        s += -0.5 * z * z - ls[j] - 0.5 * LOG2PI
    return s


def named(s: list[float]) -> dict:
    return {"q1": s[0], "q2": s[1], "w1": s[2], "w2": s[3]}


def run(seed: int, opt: dict, raw, draws: Draws) -> "ak.TapeWriter":
    o = {**DEFAULTS, **opt}
    w = ak.TapeWriter("episode", SCHEMA, seed=seed, opt=o, phases=PHASES, provenance="live")
    g = target_of(raw)
    root = Stream(ak.Rng(seed), draws)
    net = fresh(root.fork("init"), int(o["hidden"]), float(o["logStd0"]))
    rng = root.fork("batch")
    T, gamma, lam = int(o["T"]), float(o["gamma"]), float(o["lambda"])
    s = list(START)
    p0 = tip(s)
    w.frame("start", {"s": np.array(s), "A": np.zeros(0)}, metrics={**named(s), "tx": g[0], "ty": g[1], "px": p0[0], "py": p0[1],
                                                                      "d": math.hypot(g[0] - p0[0], g[1] - p0[1]), "T": T})
    r, v = [], []
    ls = net["logstd"]
    sig = [math.exp(ls[0]), math.exp(ls[1])]
    for t in range(T):
        ob = obs(s, g)
        mu = mlp(net["pi"], ob)
        z1 = rng.normal()
        z2 = rng.normal()
        a = [mu[0] + sig[0] * z1, mu[1] + sig[1] * z2]
        lp = log_prob(a, mu, ls)
        V = mlp(net["v"], ob)[0]
        v.append(V)
        w.frame("act", {}, step=t, metrics={"t": t, "dx": ob[0], "dy": ob[1], **named(s), "mu1": mu[0], "mu2": mu[1], "s1": sig[0], "s2": sig[1],
                                             "ls1": ls[0], "ls2": ls[1], "z1": z1, "z2": z2, "a1": a[0], "a2": a[1], "logp": lp, "v": V})
        st = step(s, a, g)
        p2 = tip(st["s2"])
        r.append(st["r"])
        w.frame("move", {"s": np.array(st["s2"])}, step=t, metrics={"t": t, "a1": a[0], "a2": a[1], **named(st["s2"]), "px": p2[0], "py": p2[1],
                                                                   "d": st["d"], "cost": st["cost"], "r": st["r"]})
        s = st["s2"]
    v_last = mlp(net["v"], obs(s, g))[0]
    A = np.zeros(T)
    adv, delta = [0.0] * T, [0.0] * T
    nxt = 0.0
    for k in range(T - 1, -1, -1):
        v1 = v_last if k == T - 1 else v[k + 1]
        delta[k] = r[k] + gamma * v1 - v[k]
        adv[k] = nxt = delta[k] + gamma * lam * nxt
    ret = 0.0
    for x in r:
        ret += x
    pT = tip(s)
    w.frame("end", {"A": A}, step=T, metrics={"T": T, "ret": ret, "vLast": v_last, "v0": v[0], "d": math.hypot(g[0] - pT[0], g[1] - pT[1])})
    for k in range(T - 1, -1, -1):
        A[k] = adv[k]
        v1 = v_last if k == T - 1 else v[k + 1]
        w.frame("gae", {"A": A}, step=k, changes=[{"field": "A", "index": [k], "from": 0, "to": adv[k]}],
                metrics={"t": k, "t1": k + 1, "r": r[k], "v": v[k], "v1": v1, "gamma": gamma, "lambda": lam, "gl": gamma * lam,
                         "delta": delta[k], "next": 0 if k == T - 1 else adv[k + 1], "A": adv[k], "R": adv[k] + v[k]})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="episode")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options the TS source ran with")
    ap.add_argument("--draws", default=None, help="draws.json written by the TS run (replayed and checked)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run: {target: [x, y]}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/ppo-reacher/episode.ref)")
    a = ap.parse_args()
    if a.process != "episode":
        sys.exit(f"reference: no process {a.process!r} (only 'episode')")
    draws = Draws(json.loads(Path(a.draws).read_text(encoding="utf-8"))["frames"] if a.draws else None)
    tape = run(a.seed, json.loads(a.opt), json.loads(a.input), draws)
    if draws.flat and draws.i != len(draws.flat):
        sys.exit(f"reference: used {draws.i} of the {len(draws.flat)} TS draws")
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "ppo-reacher", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames, {draws.i} draws replayed)")
