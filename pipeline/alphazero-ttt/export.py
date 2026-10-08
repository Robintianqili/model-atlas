"""alphazero-ttt: export the self-play-trained policy/value network to site/static/models/alphazero-ttt/ (kind A).

    npm run atlas -- py pipeline/alphazero-ttt/train.py              # self-play training (runs/selfplay/best.pt, log.json)
    npm run atlas -- py pipeline/alphazero-ttt/evaluate.py --write   # measured facts (runs/selfplay/stats.json)
    npm run atlas -- export alphazero-ttt                            # this file, then parity and the lesson draft

describe() records the network's forward pass (model.py) once per case; export_model checks it against the module
(AK103) and writes spec.json, weights.bin, reference.json, facts.json and presets.json. The spec extras carry the cell
names (the prior's classes) and the training log (the lesson's self-play step draws it).
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
from model import Packed, PVNet  # noqa: E402

RUN = HERE / "runs" / "selfplay"
net = PVNet()
net.load_state_dict(torch.load(RUN / "best.pt", map_location="cpu"))
net = net.cpu().eval()
MODEL = "pipeline/alphazero-ttt/model.py"


def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
    x = b.input("board", case.input[0], adapter="ttt-board", kind="game", axes="f",
                label=("The board, seen by the player to move", "走棋一方眼中的棋盘"),
                desc=("+1 for the mover's marks, −1 for the other player's, 0 for empty.", "走棋一方的棋子记 +1，对手的记 −1，空格记 0。"))
    m = b.input("mask", case.input[1], adapter="ttt-board", kind="game", axes="f", layer=False, label=("Mask", "掩码"))
    h = b.linear("fc1", x, net.fc1, label=("Hidden layer 1", "隐藏层 1"), code=(MODEL, 20))
    h = b.relu("act1", h, label=("ReLU 1", "ReLU 1"), code=(MODEL, 26))
    h = b.linear("fc2", h, net.fc2, label=("Hidden layer 2", "隐藏层 2"), code=(MODEL, 21))
    h = b.relu("act2", h, label=("ReLU 2", "ReLU 2"), code=(MODEL, 26))
    s = b.linear("pol", h, net.pol, label=("Move scores", "各步的分数"), axes="class", code=(MODEL, 22))
    s = b.add("masked", s, m, label=("Scores, taken cells at −∞", "分数（已占格子为 −∞）"), axes="class", code=(MODEL, 30))
    # axes "f", and names in extras["labels"] (not "classes"): the page then shows no "answer" beside the lesson, which
    # would give away step 2's guess; the Output section still names each bar by its cell
    p = b.softmax("prior", s, label=("Priors (policy head)", "先验（策略头）"), axes="f", code=(MODEL, 30))
    v = b.linear("val", h, net.val, label=("Value score", "价值分数"), code=(MODEL, 23))
    v = b.tanh("value", v, label=("Value (value head)", "价值（价值头）"), code=(MODEL, 31))
    b.output(p, v)


def case(cid: str, label: str, moves: list, edge: bool = False) -> ak.Case:
    bd = G.board_of(moves)
    return ak.Case(cid, label, (torch.tensor(G.view(bd)), torch.tensor(G.mask(bd))), edge=edge)


PRESETS = [
    ("start", "A game in progress", "一盘进行中的棋", [2, 3, 7]),
    ("chance", "A chance for O", "O 的机会", [5, 2, 8]),
    ("win-now", "X can win now", "X 现在就能赢", [0, 3, 1, 4]),
    ("empty", "Empty board", "空棋盘", []),
]
CASES = [case(i, en, mv) for i, en, _, mv in PRESETS] + [
    case("one-left", "one empty cell left", [0, 4, 8, 2, 6, 3, 5, 7], edge=True),
    case("full", "a full board (a draw)", [0, 4, 8, 2, 6, 3, 5, 7, 1], edge=True),
    case("x-won", "X has three in a row", [0, 3, 1, 4, 2], edge=True),
]
CELLS = [("top-left corner", "左上角"), ("top middle", "上边中间"), ("top-right corner", "右上角"), ("middle left", "左边中间"), ("centre", "正中间"),
         ("middle right", "右边中间"), ("bottom-left corner", "左下角"), ("bottom middle", "下边中间"), ("bottom-right corner", "右下角")]


def facts() -> dict:
    st = json.loads((RUN / "stats.json").read_text(encoding="utf-8"))
    lg = json.loads((RUN / "log.json").read_text(encoding="utf-8"))
    ev, tr, gm = "pipeline/alphazero-ttt/evaluate.py", "pipeline/alphazero-ttt/train.py", "pipeline/alphazero-ttt/game.py"
    rows = lg["log"]
    f = lambda v, src, unit=None: {"value": v, "source": src, **({"unit": unit} if unit else {})}
    paper = "Silver et al. 2018, Science 362:1140 (AlphaZero)"
    sx, so = st["searchAsX"], st["searchAsO"]
    return {
        "positions": f(len(G.all_positions()), gm, "positions reachable from the empty board"),
        "openPositions": f(st["positions"], ev, "positions with a move still to make"),
        "games": f(G.count_games(), gm, "lines of play from the empty board"),
        "treeNodes": f(G.count_nodes(), gm, "positions in the full game tree"),
        "priorPerfect": f(st["priorPerfect"], ev, "share"),
        "searchPerfect": f(st["searchPerfect"], ev, "share"),
        "search10": f(st["sweep"]["10"], ev, "share"),
        "search200": f(st["sweep"]["200"], ev, "share"),
        "fixedBySearch": f(st["fixedBySearch"], ev, "positions"),
        "brokenBySearch": f(st["brokenBySearch"], ev, "positions"),
        "searchLosses": f(sx["loss"] + so["loss"], ev, "games lost against every possible opponent"),
        "searchGames": f(sum(sx.values()) + sum(so.values()), ev, "games against every possible opponent (as X and as O)"),
        "rounds": f(lg["rounds"], tr), "gamesPerRound": f(lg["games"], tr), "selfplayGames": f(lg["rounds"] * lg["games"], tr),
        "trainSims": f(lg["sims"], tr, "simulations per move"),
        "examples": f(sum(r.get("examples", 0) for r in rows), tr, "positions"),
        "accStart": f(rows[0]["acc"], tr, "share"), "accEnd": f(rows[-1]["acc"], tr, "share"),
        "azSims": f(800, paper, "simulations per move"),
        "azGoGames": f({"en": "21 million", "zh": "2,100 万"}, paper + ", table S3"),
        "azTpus": f(5000, paper, "first-generation TPUs for self-play"),
        "goPositions": f({"en": "about 2 × 10¹⁷⁰", "zh": "大约 2 × 10¹⁷⁰"}, "Tromp 2016, The number of legal Go positions"),
    }


if __name__ == "__main__":
    lg = json.loads((RUN / "log.json").read_text(encoding="utf-8"))["log"]
    report = ak.export_model(
        describe, CASES, slug="alphazero-ttt", name=("AlphaZero-lite tic-tac-toe", "简化版 AlphaZero 井字棋"), kinds=("trace", "process"),
        module=Packed(net), module_input=lambda c: (c.input[0].unsqueeze(0), c.input[1].unsqueeze(0)),
        facts=facts(),
        presets={"positions": [{"id": i, "en": en, "zh": zh, "moves": mv} for i, en, zh, mv in PRESETS]},
        training={"dataset": "self-play games (no dataset)", "steps": len(lg) - 1, "metrics": {"topPriorPerfect": lg[-1]["acc"]}},
        extras={"labels": [{"en": en, "zh": zh} for en, zh in CELLS],
                "selfplay": {k: [r[k] if r[k] is not None else 0 for r in lg] for k in ("round", "games", "acc", "lossP", "lossV")}},
    )
    print(report)
