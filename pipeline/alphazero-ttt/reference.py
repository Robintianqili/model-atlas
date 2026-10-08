"""alphazero-ttt: Python reference for the processes 'search' and 'game' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/alphazero-ttt/process.ts and search.ts: the network in float64
from the exported float32 weights (static/models/alphazero-ttt/weights.bin, read through spec.json), Monte Carlo tree
search with PUCT, and the same frames, written with ak.TapeWriter. `atlas parity` compares them frame by frame
(discrete fields exact, W and P within 1e-12, metrics within 1e-12 relative; AK204 names the first difference).
Neither process draws random numbers, so --seed and --draws change nothing.

    npm run atlas -- parity alphazero-ttt --fixtures      # records the TS tapes, runs this file, compares
    npm run atlas -- py pipeline/alphazero-ttt/reference.py --process search --input "{\"moves\": [2, 3, 7]}"

CLI contract (every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game as G  # noqa: E402

SLUG = "alphazero-ttt"


def load_weights() -> dict:
    d = ak.site_path("static", "models", SLUG)
    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    raw = np.frombuffer((d / "weights.bin").read_bytes(), dtype="<f4")
    return {t["name"]: raw[t["offset"]:t["offset"] + t["length"]].astype(np.float64).reshape(t["shape"]) for t in spec["tensors"]}


def evaluate(w: dict, b) -> tuple:
    """Priors (0 on taken cells) and value, as search.ts evaluate()."""
    x = np.array(G.view(b), dtype=np.float64)
    h = np.maximum(w["fc1.weight"] @ x + w["fc1.bias"], 0.0)
    h = np.maximum(w["fc2.weight"] @ h + w["fc2.bias"], 0.0)
    s = w["pol.weight"] @ h + w["pol.bias"]
    leg = G.legal(b)
    m = max(s[i] for i in leg)
    p = np.zeros(9)
    tot = 0.0
    for i in leg:
        p[i] = math.exp(s[i] - m)
        tot += p[i]
    for i in leg:
        p[i] /= tot
    return p, math.tanh(float(w["val.weight"][0] @ h + w["val.bias"][0]))


def label(en: str, zh: str) -> tuple:
    return (en, zh)


TREE_FIELDS = ("N", "W", "P", "parent", "move")
SEARCH_SCHEMA = {
    "N": {"kind": "tensor", "label": label("Visits N of each node", "每个节点的访问次数 N"), "track": "changes", "tol": "exact"},
    "W": {"kind": "tensor", "label": label("Total value W of each node", "每个节点的总价值 W"), "track": "changes", "tol": 1e-12, "scale": "diverging"},
    "P": {"kind": "tensor", "label": label("Prior P of each node", "每个节点的先验 P"), "track": "changes", "tol": 1e-12, "scale": "sequential"},
    "parent": {"kind": "tensor", "label": label("Parent of each node", "每个节点的父节点"), "track": "changes", "tol": "exact"},
    "move": {"kind": "tensor", "label": label("The move into each node (cell 0 to 8)", "走到每个节点的那一步（格子 0 到 8）"), "track": "changes", "tol": "exact"},
    "path": {"kind": "json", "label": label("This simulation's path (node ids)", "这次模拟走过的路径（节点编号）"), "tol": "exact"},
    "count": {"kind": "int", "label": label("Nodes in the tree", "树里的节点数"), "tol": "exact"},
}
SEARCH_PHASES = {
    "start": {"label": label("The position", "局面"), "kind": "input"},
    "root": {"label": label("The network looks at the board", "网络看棋盘"), "kind": "forward"},
    "select": {"label": label("Select", "选择"), "kind": "act",
               "tex": "U = Q + c\\,P\\,\\frac{\\sqrt{N}}{1+n} = {{q}} + {{c}} \\times {{p}} \\times \\frac{\\sqrt{ {{N}} }}{1 + {{n}}} = {{u}}"},
    "expand": {"label": label("Expand and evaluate", "展开并评估"), "kind": "forward"},
    "backup": {"label": label("Back up", "回传"), "kind": "update"},
    "choose": {"label": label("Pick a move", "选一步棋"), "kind": "output"},
}


class Tree:
    def __init__(self, b, sims: int):
        cap = 1 + 9 * (sims + 1)
        self.N, self.W, self.P = np.zeros(cap), np.zeros(cap), np.zeros(cap)
        self.parent, self.move = np.full(cap, -1.0), np.full(cap, -1.0)
        self.kids: list = [[]]
        self.board: list = [b]
        self.count = 1

    def score(self, node: int, k: int, c: float) -> dict:
        n = self.N[k]
        q = self.W[k] / n if n > 0 else 0.0
        bonus = (c * self.P[k] * math.sqrt(self.N[node])) / (1 + n)
        return {"q": q, "bonus": bonus, "u": q + bonus, "n": n, "p": self.P[k]}

    def expand(self, node: int, w: dict) -> tuple:
        b = self.board[node]
        if G.terminal(b):
            return G.result_for_mover(b), True, []
        p, v = evaluate(w, b)
        for cell in G.legal(b):
            k = self.count
            self.count += 1
            self.P[k], self.parent[k], self.move[k] = p[cell], node, cell
            self.board.append(G.play(b, cell))
            self.kids.append([])
            self.kids[node].append(k)
        return v, False, list(self.kids[node])

    def backup(self, path: list, v: float) -> list:
        frm = [(self.N[x], self.W[x]) for x in path]
        s = -1.0
        for x in reversed(path):
            self.N[x] += 1
            self.W[x] += s * v
            s = -s
        return frm

    def most_visited(self, node: int = 0) -> int:
        best = -1
        for k in self.kids[node]:
            if best < 0 or self.N[k] > self.N[best]:
                best = k
        return best

    def top(self, kids: list) -> int:
        a = kids[0]
        for k in kids:
            if self.P[k] > self.P[a]:
                a = k
        return a

    def path_to(self, node: int) -> list:
        p = [node]
        while self.parent[p[0]] >= 0:
            p.insert(0, int(self.parent[p[0]]))
        return p


def best_move(w: dict, b, sims: int, c: float) -> tuple:
    t = Tree(b, sims)
    v, _, _ = t.expand(0, w)
    t.backup([0], v)
    for _ in range(sims):
        path = [0]
        while t.kids[path[-1]]:
            node = path[-1]
            pick, best = -1, -math.inf
            for k in t.kids[node]:
                u = t.score(node, k, c)["u"]
                if u > best:
                    pick, best = k, u
            path.append(pick)
        v, _, _ = t.expand(path[-1], w)
        t.backup(path, v)
    k = t.most_visited()
    return int(t.move[k]), t, k


def run_search(w: dict, inp: dict, opt: dict, seed: int) -> "ak.TapeWriter":
    sims, c = int(opt["sims"]), float(opt["c"])
    tw = ak.TapeWriter("search", SEARCH_SCHEMA, seed=seed, opt=opt, phases=SEARCH_PHASES, provenance="live")
    b0 = G.board_of(clean(inp.get("moves", [])))
    if G.terminal(b0):
        return tw
    t = Tree(b0, sims)
    tree = lambda: {k: getattr(t, k) for k in TREE_FIELDS}
    st = lambda path: {"path": [int(x) for x in path], "count": t.count}
    born = lambda kids: [ch for k in kids for ch in ({"field": "P", "index": [k], "from": 0, "to": t.P[k]},
                                                     {"field": "parent", "index": [k], "from": -1, "to": t.parent[k]},
                                                     {"field": "move", "index": [k], "from": -1, "to": t.move[k]})]
    tw.frame("start", {**tree(), **st([0])}, step=0, metrics={"sims": sims, "c": c, "mover": G.to_move(b0), "empties": 9 - sum(1 for x in b0 if x)})
    nets = 1
    v, _, kids = t.expand(0, w)
    t.backup([0], v)
    k = t.top(kids)
    tw.frame("root", st([0]), step=0, changes=[{"field": "N", "index": [0], "from": 0, "to": 1}, {"field": "W", "index": [0], "from": 0, "to": t.W[0]}] + born(kids),
             metrics={"v": v, "kids": len(kids), "top": t.move[k], "topP": t.P[k], "sims": sims, "c": c})
    for sim in range(1, sims + 1):
        path = [0]
        while t.kids[path[-1]]:
            node = path[-1]
            pick, best = -1, -math.inf
            for kk in t.kids[node]:
                u = t.score(node, kk, c)["u"]
                if u > best:
                    pick, best = kk, u
            s = t.score(node, pick, c)
            alt = -1
            for kk in t.kids[node]:
                if kk != pick and (alt < 0 or t.score(node, kk, c)["u"] > t.score(node, alt, c)["u"]):
                    alt = kk
            a = t.score(node, alt, c) if alt >= 0 else {"q": 0, "bonus": 0, "u": 0, "n": 0, "p": 0}
            tw.frame("select", st(t.path_to(pick)), step=sim, sub=len(path) - 1, metrics={
                "sim": sim, "depth": len(path) - 1, "node": node, "cell": t.move[pick], "q": s["q"], "p": s["p"], "n": s["n"], "N": t.N[node],
                "bonus": s["bonus"], "u": s["u"], "c": c, "kids": len(t.kids[node]), "alt": t.move[alt] if alt >= 0 else -1,
                "altQ": a["q"], "altP": a["p"], "altN": a["n"], "altBonus": a["bonus"], "altU": a["u"]})
            path.append(pick)
        leaf = path[-1]
        v, term, kids = t.expand(leaf, w)
        if not term:
            nets += 1
        k = t.top(kids) if kids else -1
        tw.frame("expand", st(t.path_to(leaf)), step=sim, changes=born(kids), metrics={
            "sim": sim, "node": leaf, "cell": t.move[leaf], "depth": len(path) - 1, "v": v, "terminal": 1 if term else 0, "kids": len(kids),
            "top": t.move[k] if k >= 0 else -1, "topP": t.P[k] if k >= 0 else 0, "mover": G.to_move(t.board[leaf])})
        frm = t.backup(path, v)
        f = path[1] if len(path) > 1 else 0
        n0, w0 = frm[1] if len(frm) > 1 else frm[0]
        ch = [c2 for i, x in enumerate(path) for c2 in ({"field": "N", "index": [x], "from": frm[i][0], "to": t.N[x]}, {"field": "W", "index": [x], "from": frm[i][1], "to": t.W[x]})]
        tw.frame("backup", st(path), step=sim, changes=ch, metrics={
            "sim": sim, "v": v, "depth": len(path) - 1, "leaf": leaf, "first": t.move[f], "firstN": t.N[f], "firstQ": t.W[f] / t.N[f],
            "firstQ0": w0 / n0 if n0 > 0 else 0, "rootN": t.N[0]})
    k = t.most_visited()
    tp = t.top(t.kids[0])
    tw.frame("choose", st([0, k]), step=sims, metrics={
        "cell": t.move[k], "n": t.N[k], "share": t.N[k] / (t.N[0] - 1), "q": t.W[k] / t.N[k], "p": t.P[k], "top": t.move[tp], "topP": t.P[tp],
        "topN": t.N[tp], "agree": 1 if k == tp else 0, "sims": sims, "nodes": t.count, "nets": nets})
    return tw


GAME_SCHEMA = {"board": {"kind": "tensor", "label": label("The board (1 = X, −1 = O)", "棋盘（1 = X，−1 = O）"), "shape": [9], "tol": "exact"}}
GAME_PHASES = {"start": {"label": label("New game", "新的一盘"), "kind": "input"}, "you": {"label": label("Your move", "你走一步"), "kind": "act"},
               "ai": {"label": label("The search's move", "搜索走一步"), "kind": "act"}, "end": {"label": label("Game over", "终局"), "kind": "output"}}


def run_game(w: dict, inp: dict, opt: dict, seed: int) -> "ak.TapeWriter":
    sims, c = int(opt["sims"]), float(opt["c"])
    tw = ak.TapeWriter("game", GAME_SCHEMA, seed=seed, opt=opt, phases=GAME_PHASES, provenance="live")
    you = -1 if inp.get("aiFirst") else 1
    mine = [x for x in inp.get("you", []) or []]
    b, i = G.EMPTY, 0
    snap = lambda bb: {"board": np.array(bb, dtype=np.float64)}
    tw.frame("start", snap(b), step=0, metrics={"you": you, "sims": sims})
    while not G.terminal(b):
        if G.to_move(b) == you:
            while i < len(mine) and not (isinstance(mine[i], int) and 0 <= mine[i] < 9 and b[mine[i]] == 0):
                i += 1
            if i >= len(mine):
                return tw
            cell = mine[i]
            i += 1
            b = G.play(b, cell)
            tw.frame("you", snap(b), step=sum(1 for x in b if x), metrics={"cell": cell, "you": you})
        else:
            cell, t, k = best_move(w, b, sims, c)
            b = G.play(b, cell)
            tw.frame("ai", snap(b), step=sum(1 for x in b if x), metrics={"cell": cell, "n": t.N[k], "sims": sims, "you": you})
    tw.frame("end", snap(b), step=sum(1 for x in b if x), metrics={"winner": G.winner(b), "you": you, "result": G.winner(b) * you})
    return tw


def clean(moves) -> list:
    """The legal prefix of a move list (rules.ts cleanMoves)."""
    out, b = [], G.EMPTY
    for c in moves if isinstance(moves, list) else []:
        if isinstance(c, int) and 0 <= c < 9 and b[c] == 0 and not G.terminal(b):
            out.append(c)
            b = G.play(b, c)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="search", choices=("search", "game"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options the TS source ran with")
    ap.add_argument("--draws", default=None, help="draws.json from the TS run (unused: nothing is drawn)")
    ap.add_argument("--input", default="null", help='JSON input of the TS run: {"moves": [...]} or {"you": [...], "aiFirst": false}')
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/alphazero-ttt/<process>.ref)")
    a = ap.parse_args()
    opt = {"sims": 50, "c": 1.5, **json.loads(a.opt)}
    inp = json.loads(a.input) or {}
    fn = run_search if a.process == "search" else run_game
    tape = fn(load_weights(), inp, opt, a.seed)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", SLUG, f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
