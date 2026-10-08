"""snake-dqn: Python reference for the processes 'dqn', 'replay' and 'agent' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/snake-dqn/process.ts and qnet.ts: the Q-network in float64 from
the exported float32 weights (static/models/snake-dqn/weights.bin, and target/weights.bin for the target network), every
sum in the same order as the TS loops (so the numbers agree to the last bit), the game from game.py with the same seeds,
and the same frames, written with ak.TapeWriter. `atlas parity` compares them frame by frame (AK204 names the first
difference). The ε draws use ak.Rng, bit-identical to the TS rng for next(), int() and fork().

    npm run atlas -- parity snake-dqn --fixtures
    npm run atlas -- py pipeline/snake-dqn/reference.py --process agent --input "{\"seed\": 7}"

CLI contract (every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game as G  # noqa: E402

SLUG = "snake-dqn"
END = {"": 0, "wall": 1, "self": 2, "full": 3}


def load_weights(sub: str = "") -> dict:
    d = ak.site_path("static", "models", SLUG, *([sub] if sub else []))
    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    raw = np.frombuffer((d / "weights.bin").read_bytes(), dtype="<f4")
    return {t["name"]: raw[t["offset"]:t["offset"] + t["length"]].astype(np.float64).reshape(t["shape"]) for t in spec["tensors"]}


def conv(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3 x 3, padding 1, then ReLU; per output: the taps in (c, ky, kx) order from 0, then the bias (as qnet.ts)."""
    C = x.shape[0]
    xp = np.zeros((C, G.SIZE + 2, G.SIZE + 2))
    xp[:, 1:-1, 1:-1] = x
    acc = np.zeros((W.shape[0], G.SIZE, G.SIZE))
    for c in range(C):
        for ky in range(3):
            for kx in range(3):
                acc += W[:, c, ky, kx][:, None, None] * xp[c, ky:ky + G.SIZE, kx:kx + G.SIZE][None]
    acc += b[:, None, None]
    return np.where(acc < 0, 0.0, acc)


def dense(x: np.ndarray, W: np.ndarray, b: np.ndarray, relu: bool) -> np.ndarray:
    acc = np.zeros(W.shape[0])
    for j in range(W.shape[1]):
        acc += W[:, j] * x[j]
    acc += b
    return np.where(acc < 0, 0.0, acc) if relu else acc


def qvalues(w: dict, g: G.Game) -> list:
    x = g.encode().astype(np.float64)
    h = conv(conv(x, w["conv1.weight"], w["conv1.bias"]), w["conv2.weight"], w["conv2.bias"])
    h = dense(h.reshape(-1), w["fc1.weight"], w["fc1.bias"], True)
    return [float(v) for v in dense(h, w["out.weight"], w["out.bias"], False)]


def argmax(q) -> int:
    b = 0
    for i in range(1, len(q)):
        if q[i] > q[b]:
            b = i
    return b


def gj(g: G.Game) -> dict:
    return {"seed": g.seed, "snake": list(g.snake), "dir": g.dir, "food": g.food, "score": g.score, "moves": g.moves, "hunger": g.hunger, "over": g.over}


def step(g: G.Game, a: int) -> tuple:
    n = g.copy()
    r = int(n.step(a))
    return n, r, 1 if n.score > g.score else 0


def qm(q, p="q") -> dict:
    return {f"{p}0": q[0], f"{p}1": q[1], f"{p}2": q[2]}


def target(wt: dict, s2: G.Game, r: int, gamma: float) -> tuple:
    q = [0.0, 0.0, 0.0] if s2.over else qvalues(wt, s2)
    b = argmax(q)
    mx = 0.0 if s2.over else q[b]
    return q, b, mx, r + (0.0 if s2.over else gamma * mx)


L = lambda en, zh: (en, zh)  # noqa: E731
J = lambda en, zh: {"kind": "json", "label": L(en, zh), "tol": "exact"}  # noqa: E731
SCHEMAS = {
    "dqn": {"game": J("The board s", "棋盘 s"), "next": J("The board after the move, s′", "走完这一步后的棋盘 s′")},
    "replay": {"buf": J("The buffer: stored moves (s, a, r, s′)", "缓冲区：存下的步（s, a, r, s′）"), "game": J("The game being played", "正在下的这盘"),
               "pick": J("The slots drawn for the batch", "抽中的格位"), "rows": J("The batch: target and loss of each move", "这一批：每一步的目标值和损失")},
    "agent": {"game": J("The board", "棋盘")},
}
PHASES = {
    "dqn": {"observe": {"label": L("Look at the board", "看棋盘"), "kind": "observe"}, "act": {"label": L("Choose a move (ε-greedy)", "选一步（ε-贪心）"), "kind": "act"},
            "reward": {"label": L("Move and get the reward", "走一步，得到奖励"), "kind": "reward"}, "target": {"label": L("The target", "目标值"), "kind": "loss"},
            "loss": {"label": L("The loss", "损失"), "kind": "loss"}},
    "replay": {"store": {"label": L("Play a move and store it", "走一步并存起来"), "kind": "act"}, "sample": {"label": L("Draw a batch at random", "随机抽一批"), "kind": "sample"},
               "learn": {"label": L("Targets and losses of the batch", "这一批的目标值和损失"), "kind": "loss"}},
    "agent": {"start": {"label": L("A new game", "新的一盘"), "kind": "input"}, "move": {"label": L("A move", "一步"), "kind": "act"}, "end": {"label": L("Game over", "终局"), "kind": "output"}},
}
DEFAULTS = {"dqn": {"eps": 0.1, "gamma": 0.95}, "replay": {"eps": 0.1, "gamma": 0.95, "fill": 20, "cap": 12, "batch": 4}, "agent": {"cap": 1500}}


def run_dqn(inp: dict, opt: dict, seed: int, tw) -> None:
    w, wt = load_weights(), load_weights("target")
    eps, gamma = float(opt["eps"]), float(opt["gamma"])
    g = G.replay(int(inp["seed"]), [int(a) for a in inp.get("moves", [])])
    if g.over:
        return
    q = qvalues(w, g)
    best = argmax(q)
    tw.frame("observe", {"game": gj(g)}, step=0, metrics={**qm(q), "best": best, "len": len(g.snake), "score": g.score})
    rng = ak.Rng(seed).fork(f"draw{int(inp.get('draw', 0))}")
    u = rng.next()
    explore = 1 if u < eps else 0
    a = rng.int(3) if explore else best
    tw.frame("act", {}, step=0, metrics={**qm(q), "u": u, "eps": eps, "explore": explore, "a": a, "best": best, "qa": q[a]})
    g2, r, ate = step(g, a)
    tw.frame("reward", {"next": gj(g2)}, step=0, metrics={"a": a, "r": r, "ate": ate, "end": END[g2.over], "score": g2.score})
    tq, tb, tmax, y = target(wt, g2, r, gamma)
    o = [0.0, 0.0, 0.0] if g2.over else qvalues(w, g2)
    tw.frame("target", {}, step=0, metrics={"r": r, "gamma": gamma, "terminal": 1 if g2.over else 0, **qm(tq, "t"), "tmax": tmax, "tbest": tb, **qm(o, "o"), "omax": max(o), "y": y})
    d = y - q[a]
    tw.frame("loss", {}, step=0, metrics={"a": a, "qa": q[a], "y": y, "delta": d, "loss": d * d})


def run_replay(inp: dict, opt: dict, seed: int, tw) -> None:
    w, wt = load_weights(), load_weights("target")
    eps, gamma, fill, cap, batch = float(opt["eps"]), float(opt["gamma"]), int(opt["fill"]), int(opt["cap"]), int(opt["batch"])
    base = ak.Rng(seed)
    pr, sr = base.fork("play"), base.fork("sample")
    s0 = int(inp["seed"])
    g, ep, buf = G.Game(s0), 0, []
    for t in range(fill):
        q = qvalues(w, g)
        u = pr.next()
        explore = 1 if u < eps else 0
        a = pr.int(3) if explore else argmax(q)
        g2, r, _ = step(g, a)
        slot = t % cap
        item = {"t": t, "s": gj(g), "a": a, "r": r, "done": 1 if g2.over else 0, "s2": gj(g2)}
        if slot < len(buf):
            buf[slot] = item
        else:
            buf.append(item)
        tw.frame("store", {"buf": [dict(x) for x in buf], "game": gj(g2)}, step=t,
                 metrics={"t": t, "a": a, "r": r, "slot": slot, "size": len(buf), "done": item["done"], "explore": explore})
        if g2.over or g2.hunger >= G.HUNGER:
            ep += 1
            g = G.Game(s0 + 1000 * ep)
        else:
            g = g2
    pick = [sr.int(len(buf)) for _ in range(batch)]
    tw.frame("sample", {"pick": pick}, step=fill, metrics={"size": len(buf), "batch": batch, "oldest": min([fill] + [x["t"] for x in buf])})
    rows = []
    for slot in pick:
        x = buf[slot]
        sg, s2 = G.Game.__new__(G.Game), G.Game.__new__(G.Game)
        for gg, js in ((sg, x["s"]), (s2, x["s2"])):
            gg.seed, gg.snake, gg.dir, gg.food, gg.score = js["seed"], list(js["snake"]), js["dir"], js["food"], js["score"]
            gg.moves, gg.hunger, gg.over, gg.rew = js["moves"], js["hunger"], js["over"], G.REWARDS["main"]
        qa = qvalues(w, sg)[x["a"]]
        _, _, tmax, y = target(wt, s2, x["r"], gamma)
        d = y - qa
        rows.append({"slot": slot, "t": x["t"], "a": x["a"], "r": x["r"], "done": x["done"], "q": qa, "tmax": tmax, "y": y, "loss": d * d})
    mean = 0.0
    for x in rows:
        mean += x["loss"]
    tw.frame("learn", {"rows": rows}, step=fill, metrics={"mean": mean / len(rows), "batch": batch})


def run_agent(inp: dict, opt: dict, seed: int, tw) -> None:
    w, cap = load_weights(), int(opt["cap"])
    g = G.Game(int(inp["seed"]))
    q = qvalues(w, g)
    tw.frame("start", {"game": gj(g)}, step=0, metrics={**qm(q), "score": 0, "len": len(g.snake)})
    while True:
        a = argmax(q)
        g, r, ate = step(g, a)
        q = [0.0, 0.0, 0.0] if g.over else qvalues(w, g)
        tw.frame("move", {"game": gj(g)}, step=g.moves, metrics={"a": a, "r": r, "ate": ate, "score": g.score, "len": len(g.snake), **qm(q)})
        why = END[g.over] if g.over else 4 if g.hunger >= G.HUNGER else 5 if g.moves >= cap else 0
        if why:
            tw.frame("end", {}, step=g.moves, metrics={"score": g.score, "moves": g.moves, "why": why})
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="dqn", choices=sorted(DEFAULTS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options the TS source ran with")
    ap.add_argument("--draws", default=None, help="draws.json from the TS run (unused: the draws are made with ak.Rng)")
    ap.add_argument("--input", default="null", help='JSON input of the TS run: {"seed": 7, "moves": [...], "draw": 0}')
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/snake-dqn/<process>.ref)")
    a = ap.parse_args()
    opt = {**DEFAULTS[a.process], **json.loads(a.opt)}
    inp = ak.json_arg(a.input) or {}
    tw = ak.TapeWriter(a.process, SCHEMAS[a.process], seed=a.seed, opt=opt, phases=PHASES[a.process], provenance="live")
    {"dqn": run_dqn, "replay": run_replay, "agent": run_agent}[a.process](inp, opt, a.seed, tw)
    jpath, _ = tw.write(a.out or ak.site_path(".atlas", SLUG, f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tw.frames)} frames)")
