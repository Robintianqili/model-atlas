"""cart-pole: Python reference for the processes 'episode' and 'seeds' (kind C parity, design §6.2 item 8).

An independent implementation of site/src/lib/explainers/cart-pole/{sim,process}.ts in plain Python floats: the classic
cart-pole physics, the fresh policy network (its weights rebuilt from ak.Rng(seed).fork('init') exactly as the TS MLP
draws them: PyTorch's U(-1/sqrt(in), 1/sqrt(in)), row-major, weight then bias, layer by layer), one episode sampled
from ak.Rng(seed).fork('batch') (bit-identical to the TS rng for next()), and the discounted returns, backwards.
`npm run atlas -- parity cart-pole` records the TS tape, runs this file and compares the tapes frame by frame: phase and
step exactly, the state and the returns within 1e-12 relative, every metric within 1e-12 (AK204).
'seeds' is the whole REINFORCE training repeated here for seeds 1 to 5: the episodes are played in plain Python floats,
the loss mean(A * -log pi) is differentiated by torch in float64 and stepped by torch.optim.Adam, and every round's mean
episode length must equal the TS run's (one flipped push anywhere would change it).

    npm run atlas -- parity cart-pole --fixtures      # also on every fixture start state
    npm run atlas -- py pipeline/cart-pole/reference.py --input '{"start": [0, 0, 0.1, 0]}'

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

GRAVITY, M_CART, M_POLE, HALF_LEN, FORCE, TAU, X_LIMIT = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02, 2.4
M_TOTAL = M_CART + M_POLE
PML = M_POLE * HALF_LEN
TH_LIMIT = (12 * 2 * math.pi) / 360
DEFAULT_START = [0.0, 0.0, 0.05, 0.0]
RANGES = [(-2, 2), (-1, 1), (-0.2, 0.2), (-1, 1)]
DEFAULTS = {"hidden": 16, "gamma": 0.95, "maxLen": 200, "lr": 0.02, "batch": 10, "steps": 100, "snapshotEvery": 2, "seeds": [1, 2, 3, 4, 5]}
START = 0.05

SCHEMA = {
    "s": {"kind": "tensor", "label": ("State [x, ẋ, θ, θ̇]", "状态 [x, ẋ, θ, θ̇]"), "shape": [4], "tol": 1e-12},
    "G": {"kind": "tensor", "label": ("Discounted return of each step", "每一步的折扣回报"), "track": "changes", "tol": 1e-12},
}
PHASES = {
    "start": {"label": ("The start state", "起始状态"), "kind": "input"},
    "act": {"label": ("Sample an action", "采样一个动作"), "kind": "sample"},
    "move": {"label": ("One physics step", "物理模拟一步"), "kind": "other"},
    "end": {"label": ("The episode ends", "回合结束"), "kind": "observe"},
    "return": {"label": ("Discounted return", "折扣回报"), "kind": "aggregate"},
}


def start_of(raw) -> list[float]:
    s = (raw or {}).get("start") if isinstance(raw, dict) else None
    out = []
    for i in range(4):
        try:
            v = float(s[i]) if isinstance(s, list) else DEFAULT_START[i]
        except (TypeError, ValueError, IndexError):
            v = DEFAULT_START[i]
        out.append(min(RANGES[i][1], max(RANGES[i][0], v)) if math.isfinite(v) else DEFAULT_START[i])
    return out


def fresh_policy(rng: "ak.Rng", hidden: int) -> dict:
    """Linear(4, hidden) then Linear(hidden, 1): each weight [out, in] then its bias, U(-b, b) with b = 1/sqrt(in)."""
    def uniform(n: int, bound: float) -> list[float]:
        return [(2 * rng.next() - 1) * bound for _ in range(n)]
    b1 = 1 / math.sqrt(4)
    w1 = uniform(hidden * 4, b1)
    c1 = uniform(hidden, b1)
    b2 = 1 / math.sqrt(hidden)
    return {"w1": w1, "c1": c1, "w2": uniform(hidden, b2), "c2": uniform(1, b2)[0], "hidden": hidden}


def p_left(pol: dict, s: list[float]) -> tuple[float, float]:
    h = []
    for j in range(pol["hidden"]):
        a = 0.0
        for i in range(4):
            a += pol["w1"][j * 4 + i] * s[i]
        h.append(math.tanh(a + pol["c1"][j]))
    z = 0.0
    for j in range(pol["hidden"]):
        z += pol["w2"][j] * h[j]
    z += pol["c2"]
    return z, 1 / (1 + math.exp(-z))


def step(s: list[float], left: int) -> dict:
    x, xd, th, thd = s
    force = -FORCE if left == 1 else FORCE
    cos, sin = math.cos(th), math.sin(th)
    temp = (force + PML * thd * thd * sin) / M_TOTAL
    th_acc = (GRAVITY * sin - cos * temp) / (HALF_LEN * (4.0 / 3.0 - (M_POLE * cos * cos) / M_TOTAL))
    x_acc = temp - (PML * th_acc * cos) / M_TOTAL
    s2 = [x + TAU * xd, xd + TAU * x_acc, th + TAU * thd, thd + TAU * th_acc]
    done = 1 if (s2[2] < -TH_LIMIT or s2[2] > TH_LIMIT) else 2 if (s2[0] < -X_LIMIT or s2[0] > X_LIMIT) else 0
    return {"force": force, "temp": temp, "thAcc": th_acc, "xAcc": x_acc, "s2": s2, "done": done}


def named(s: list[float]) -> dict:
    return {"x": s[0], "xd": s[1], "th": s[2], "thd": s[3]}


def run(seed: int, opt: dict, raw) -> "ak.TapeWriter":
    o = {**DEFAULTS, **opt}
    w = ak.TapeWriter("episode", SCHEMA, seed=seed, opt=o, phases=PHASES, provenance="live")
    root = ak.Rng(seed)
    pol = fresh_policy(root.fork("init"), int(o["hidden"]))
    rng = root.fork("batch")
    gamma, max_len = float(o["gamma"]), int(o["maxLen"])
    s = start_of(raw)
    w.frame("start", {"s": np.array(s), "G": np.zeros(0)}, metrics={**named(s), "maxLen": max_len, "gamma": gamma})
    t, end = 0, 3
    while t < max_len:
        z, p = p_left(pol, s)
        u = rng.next()
        left = 1 if u < p else 0
        w.frame("act", {}, step=t, metrics={"t": t, "z": z, "p": p, "u": u, "left": left, **named(s)})
        f = step(s, left)
        s2 = f["s2"]
        w.frame("move", {"s": np.array(s2)}, step=t, metrics={
            "t": t, "left": left, "force": f["force"], "temp": f["temp"], "thAcc": f["thAcc"], "xAcc": f["xAcc"], **named(s),
            "x2": s2[0], "xd2": s2[1], "th2": s2[2], "thd2": s2[3], "done": f["done"]})
        s = s2
        t += 1
        if f["done"]:
            end = f["done"]
            break
    G = np.zeros(t)
    w.frame("end", {"G": G}, step=t, metrics={"len": t, "end": end, **named(s)})
    nxt = 0.0
    for k in range(t - 1, -1, -1):
        g = 1 + gamma * nxt
        G[k] = g
        w.frame("return", {"G": G}, step=k, changes=[{"field": "G", "index": [k], "from": 0, "to": g}],
                metrics={"t": k, "t1": k + 1, "r": 1, "gamma": gamma, "next": nxt, "G": g, "len": t})
        nxt = g
    return w


# ------------------------------------------------------------------ seeds: the training, five times
def rollout(pol: dict, s: list[float], rng: "ak.Rng", max_len: int) -> tuple[list, list, int]:
    states, left = [], []
    while len(states) < max_len:
        p = p_left(pol, s)[1]
        a = 1 if rng.next() < p else 0
        states.append(s)
        left.append(a)
        f = step(s, a)
        s = f["s2"]
        if f["done"]:
            break
    return states, left, len(states)


def train(seed: int, o: dict, raw) -> list[tuple[str, int, float]]:
    """(phase, round, mean episode length) of every round, as trainingProcess(reinforce) records them (coarse)."""
    import torch
    torch.set_default_dtype(torch.float64)
    root = ak.Rng(seed)
    hidden = int(o["hidden"])
    fresh = fresh_policy(root.fork("init"), hidden)
    net = torch.nn.Sequential(torch.nn.Linear(4, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1)).double()
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(fresh["w1"]).reshape(hidden, 4))
        net[0].bias.copy_(torch.tensor(fresh["c1"]))
        net[2].weight.copy_(torch.tensor(fresh["w2"]).reshape(1, hidden))
        net[2].bias.copy_(torch.tensor([fresh["c2"]]))
    opt = torch.optim.Adam(net.parameters(), lr=float(o["lr"]), foreach=False)
    rng = root.fork("batch")
    start, gamma, max_len, steps = start_of(raw), float(o["gamma"]), int(o["maxLen"]), int(o["steps"])
    out = []
    for k in range(steps + 1):
        pol = {"w1": net[0].weight.detach().reshape(-1).tolist(), "c1": net[0].bias.detach().tolist(),
               "w2": net[2].weight.detach().reshape(-1).tolist(), "c2": float(net[2].bias.detach()[0]), "hidden": hidden}
        xs, ys, G, lens = [], [], [], []
        for e in range(max(1, int(o["batch"]))):
            s0 = start if e == 0 else [(2 * rng.next() - 1) * START for _ in range(4)]
            st, left, n = rollout(pol, s0, rng, max_len)
            xs += st
            ys += left
            g, ret = 0.0, [0.0] * n
            for t in range(n - 1, -1, -1):
                g = 1 + gamma * g
                ret[t] = g
            G += ret
            lens.append(n)
        mean = 0.0
        for v in G:
            mean += v
        mean /= len(G)
        q = 0.0
        for v in G:
            q += (v - mean) * (v - mean)
        std = math.sqrt(q / len(G))
        adv = [((v - mean) / std if std > 0 else 0.0) for v in G]
        total = 0
        for n in lens:
            total += n
        last = k >= steps
        out.append(("snapshot" if last or k % int(o["snapshotEvery"]) == 0 else "step", k, total / len(lens)))
        if not last:
            z = net[2](net[1](net[0](torch.tensor(xs))))
            y = torch.tensor(ys, dtype=torch.float64).reshape(-1, 1)
            nll = torch.relu(z) - z * y + torch.log(torch.exp(-torch.abs(z)) + 1)
            loss = torch.mean(nll * torch.tensor(adv).reshape(-1, 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return out


def run_seeds(seed: int, opt: dict, raw) -> "ak.TapeWriter":
    o = {**DEFAULTS, **opt}
    w = ak.TapeWriter("seeds", {"step": {"kind": "int", "label": ("Step", "步数")}}, seed=seed, opt=o, provenance="live",
                      phases={"init": {"label": ("Start", "开始"), "kind": "other"}, "step": {"label": ("Training step", "训练步"), "kind": "update"},
                              "snapshot": {"label": ("Snapshot", "快照"), "kind": "observe"}})
    runs = [train(int(s), o, raw) for s in o["seeds"]]
    w.frame("init", {"step": 0}, metrics={"step": 0})
    for r in range(len(runs[0])):
        phase, k, _ = runs[0][r]
        w.frame(phase, {"step": k}, step=k, metrics={"step": k, **{f"len{i + 1}": runs[i][r][2] for i in range(len(runs))}})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="episode")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options the TS source ran with")
    ap.add_argument("--draws", default=None, help="unused: the process draws only with next(), bit-identical in ak.Rng")
    ap.add_argument("--input", default="null", help="JSON input of the TS run: {start: [x, xd, th, thd]}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/cart-pole/episode.ref)")
    a = ap.parse_args()
    if a.process not in ("episode", "seeds"):
        sys.exit(f"reference: no process {a.process!r} (only 'episode' and 'seeds')")
    tape = (run if a.process == "episode" else run_seeds)(a.seed, json.loads(a.opt), json.loads(a.input))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "cart-pole", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
