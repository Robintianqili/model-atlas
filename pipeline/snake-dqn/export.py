"""snake-dqn: export the DQN-trained Q-network to site/static/models/snake-dqn/ (kind A), and the target network of the
same checkpoint to site/static/models/snake-dqn/target/.

    npm run atlas -- py pipeline/snake-dqn/train.py                       # runs/main/{best.pt, log.json}
    npm run atlas -- py pipeline/snake-dqn/train.py --variant hurry       # runs/hurry/ (the limits step)
    npm run atlas -- py pipeline/snake-dqn/evaluate.py --write            # runs/main/stats.json
    npm run atlas -- export snake-dqn                                      # this file, then parity and the lesson draft
    npm run atlas -- parity snake-dqn --dir models/snake-dqn/target --write   # the target network's parity

describe() records the network's forward pass (model.py) once per case; export_model checks it against the module
(AK103) and writes spec.json, weights.bin, reference.json, facts.json and presets.json. The spec extras carry the
training log (the lesson's chart steps) and the names of the three moves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import game as G  # noqa: E402
from model import QNet  # noqa: E402
from train import EVAL_EVERY  # noqa: E402

RUN = HERE / "runs" / "main"
CK = torch.load(RUN / "best.pt", map_location="cpu")
MODEL = "pipeline/snake-dqn/model.py"


def net_of(key: str) -> QNet:
    n = QNet()
    n.load_state_dict(CK[key])
    return n.cpu().eval()


def describer(net: QNet):
    def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
        x = b.input("board", case.input, adapter="snake-board", kind="game", axes="c,y,x",
                    label=("The board as three planes: head, body, food", "棋盘的三层：蛇头、身体、食物"),
                    desc=("1 on the head; on the body (L − i) / L for the i-th cell behind the head; 1 on the food.", "蛇头处为 1；身体上从蛇头往后第 i 节为 (L − i) / L；食物处为 1。"))
        h = b.conv2d("conv1", x, net.conv1, label=("Convolution 1 (16 kernels)", "卷积 1（16 个卷积核）"), code=(MODEL, 27))
        h = b.relu("act1", h, label=("ReLU 1", "ReLU 1"), code=(MODEL, 27))
        h = b.conv2d("conv2", h, net.conv2, label=("Convolution 2 (32 kernels)", "卷积 2（32 个卷积核）"), code=(MODEL, 28))
        h = b.relu("act2", h, label=("ReLU 2", "ReLU 2"), code=(MODEL, 28))
        h = b.flatten("flat", h, label=("Flatten", "展平"), code=(MODEL, 29))
        h = b.linear("fc1", h, net.fc1, label=("Hidden layer (64)", "隐藏层（64）"), code=(MODEL, 29))
        h = b.relu("act3", h, label=("ReLU 3", "ReLU 3"), code=(MODEL, 29))
        q = b.linear("out", h, net.out, axes="k", label=("Q-values: straight on, left, right", "Q 值：直走、左转、右转"), code=(MODEL, 30))
        b.output(q)
    return describe


# (id, en, zh, seed, moves): positions reachable from the start (found with pipeline/snake-dqn/presets.py)
PRESETS = json.loads((HERE / "presets.json").read_text(encoding="utf-8"))


def case(cid: str, label: str, seed: int, moves: list, edge: bool = False) -> ak.Case:
    return ak.Case(cid, label, torch.from_numpy(G.replay(seed, moves).encode()), edge=edge)


CASES = [case(p["id"], p["en"], p["seed"], p["moves"]) for p in PRESETS] + [
    case("new-game", "the start of a game", 3, [], edge=True),
    case("wall-ahead", "the head next to the top wall", 3, [0, 0, 0, 0], edge=True),
]
ACTIONS = [("straight on", "直走"), ("turn left", "左转"), ("turn right", "右转")]


def facts() -> dict:
    st = json.loads((RUN / "stats.json").read_text(encoding="utf-8"))
    lg = json.loads((RUN / "log.json").read_text(encoding="utf-8"))
    hu = st.get("hurry")
    tr, ev = "pipeline/snake-dqn/train.py", "pipeline/snake-dqn/evaluate.py"
    f = lambda v, src, unit=None: {"value": v, "source": src, **({"unit": unit} if unit else {})}  # noqa: E731
    mn, ck, evals = st["main"], st["main"]["checkpoint"], lg["evals"]
    best = max(evals, key=lambda e: e["score"])
    half = next(e for e in evals if e["score"] >= best["score"] / 2)
    crashes = lambda s: s["ends"]["wall"] + s["ends"]["self"]  # noqa: E731
    out = {
        "params": f(sum(p.numel() for p in net_of("online").parameters()), "pipeline/snake-dqn/model.py"),
        "gamma": f(lg["gamma"], tr), "buffer": f(lg["buffer"], tr, "moves"), "batch": f(lg["batch"], tr, "moves"), "lr": f(lg["lr"], tr),
        "sync": f(lg["sync"], tr, "updates"), "epsEnd": f(lg["epsEnd"], tr), "epsMoves": f(round(lg["epsFrac"] * lg["moves"]), tr, "moves"),
        "trainMoves": f(lg["moves"], tr, "moves"), "minutes": f(round(lg["seconds"] / 60), tr, "minutes (RTX 5090 laptop GPU, shared)"),
        "ckMoves": f(ck["moves"], tr, "moves"), "ckUpdates": f(ck["updates"], tr, "updates"), "ckSinceSync": f(ck["sinceSync"], tr, "updates"),
        "evalEvery": f(EVAL_EVERY, tr, "moves"), "evalGames": f(lg["evalSeeds"][1] - lg["evalSeeds"][0], tr, "games"),
        "evalBest": f(best["score"], tr, "food per test game"), "evalHalf": f(round(best["score"] / 2, 1), tr, "food per test game"),
        "halfMoves": f(half["moves"], tr, "moves"),
        "agentGames": f(mn["games"], ev, "games (seeds 1 to 1,000)"), "agentMean": f(mn["mean"], ev, "food per game"), "agentMedian": f(mn["median"], ev),
        "agentMax": f(mn["max"], ev, "food"), "agentMoves": f(mn["meanMoves"], ev, "moves per game"), "agentCrashes": f(crashes(mn), ev, "games"),
        "agentHungry": f(mn["ends"]["hungry"], ev, "games"), "agentFull": f(mn["ends"]["full"], ev, "games"), "agentAvoidable": f(mn["avoidableShare"], ev, "share of crashes"),
        "randomMean": f(st["random"]["mean"], ev, "food per game"), "randomSafeMean": f(st["randomSafe"]["mean"], ev, "food per game"),
        "maxFood": f(G.CELLS - len(G.START), "pipeline/snake-dqn/game.py", "food to fill the board"), "hunger": f(G.HUNGER, "pipeline/snake-dqn/game.py", "moves"),
    }
    if hu:
        out.update({"hurryStep": f({"en": "−0.1", "zh": "−0.1"}, "pipeline/snake-dqn/game.py: REWARDS['hurry']", "reward per move"), "hurryMean": f(hu["mean"], ev, "food per game"), "hurryMoves": f(hu["meanMoves"], ev, "moves per game"),
                    "hurryAvoidable": f(hu["avoidableShare"], ev, "share of crashes"), "hurryCrashes": f(crashes(hu), ev, "games")})
    return out


def logs() -> dict:
    lg = json.loads((RUN / "log.json").read_text(encoding="utf-8"))
    rows = [r for r in lg["log"] if r["score"] is not None]
    # the probe board in a window of 8 copies of the target network, two thirds into training (as logged every 25 updates)
    pr = lg["probes"]
    mid = pr[len(pr) * 2 // 3][0]
    win = [p for p in pr if mid <= p[0] <= mid + 8 * lg["sync"]]
    a = max(range(3), key=lambda i: win[0][1 + i])  # the move the online network rates highest at the window's start
    out = {"dqnlog": {"moves": [e["moves"] for e in lg["evals"]], "score": [e["score"] for e in lg["evals"]],
                      "trainMoves": [r["moves"] for r in rows[4::5]], "train": [round(r["score"], 4) for r in rows[4::5]],
                      "probe": {"updates": [p[0] for p in win], "online": [p[1 + a] for p in win], "target": [p[4 + a] for p in win], "action": a}}}
    st = json.loads((RUN / "stats.json").read_text(encoding="utf-8"))
    out["agentCheck"] = {"seeds": st["seeds"], "scores": st["main"]["scores"][:20]}  # claims.test.ts replays these games in TS
    hp = HERE / "runs" / "hurry" / "log.json"
    if hp.exists():
        h = json.loads(hp.read_text(encoding="utf-8"))
        out["hurrylog"] = {"moves": [e["moves"] for e in h["evals"]], "score": [e["score"] for e in h["evals"]]}
    return out


if __name__ == "__main__":
    lg = json.loads((RUN / "log.json").read_text(encoding="utf-8"))
    online, target = net_of("online"), net_of("target")
    common = dict(module_input=lambda c: c.input.unsqueeze(0))
    report = ak.export_model(
        describer(online), CASES, slug="snake-dqn", name=("Snake DQN", "贪吃蛇 DQN"), kinds=("trace", "process"), module=online,
        facts=facts(), presets={"positions": PRESETS},
        training={"dataset": "its own games (no dataset)", "steps": CK["updates"], "metrics": {"testFood": CK["eval"]}},
        extras={"labels": [{"en": en, "zh": zh} for en, zh in ACTIONS], **logs()}, **common)
    print(report)
    report = ak.export_model(
        describer(target), CASES, slug="snake-dqn", name=("Snake DQN: the target network", "贪吃蛇 DQN：目标网络"), kinds=("trace",), module=target,
        out=str(ak.site_path("static", "models", "snake-dqn", "target")),
        training={"dataset": "its own games (no dataset)", "steps": CK["updates"] - CK["sinceSync"]},
        extras={"labels": [{"en": en, "zh": zh} for en, zh in ACTIONS], "copiedAtUpdate": CK["updates"] - CK["sinceSync"]}, **common)
    print(report)
