"""gnn-variants: Python references for the processes of site/src/lib/explainers/gnn-variants/process.ts (kind C parity).

    gat   GAT layer 1 into the member in focus. Its input is what the TS source hands the process: the adjacency, the
          traced messages z = X Wᵀ and the layer's learned vectors (a_src, a_dst, b). This file re-derives the events in
          float64: a score per sender (LeakyReLU(a_dst·zᵢ + a_src·zⱼ)), the softmax over the senders, one message α × zⱼ
          per sender with the running sums, then + b and tanh.
    drop  remove one friendship of the member in focus at a time and re-run the whole GAT network in float64 (an
          independent copy of graph.ts forwardGat: two GAT layers with tanh and the head); per friend the layer-1 and
          layer-2 α and the move of the member's score gap (the Officer's score − Mr. Hi's); then, over every member with
          2 or more friends, how often the friend with the largest α (layer 1, layer 2) is also the one whose removal
          moves the gap most.
    The loops run in the TS order, so the floats agree to the last bit or nearly (math.exp / tanh may differ by an ulp).

    npm run atlas -- parity gnn-variants --write --fixtures   # records the TS tapes, runs this file per process, compares

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

N, F = 34, 4
L = lambda en, zh: (en, zh)  # noqa: E731
GAT_SCHEMA = {
    "node": {"kind": "int", "label": L("Member in focus", "关注的成员"), "tol": "exact"},
    "w": {"kind": "tensor", "label": L("Score, then weight α, per member", "每个成员的分数，之后是权重 α"), "shape": [N], "tol": 1e-12},
    "sum": {"kind": "tensor", "label": L("Messages added so far", "目前加起来的消息"), "shape": [F], "tol": 1e-12},
}
GAT_PHASES = {
    "score": {"label": L("Score one sender", "给一个发送者打分"), "kind": "other"},
    "softmax": {"label": L("Softmax over the senders", "对发送者做 softmax"), "kind": "other"},
    "message": {"label": L("A weighted message", "一条加权消息"), "kind": "message"},
    "activate": {"label": L("Add the bias, then tanh", "加上偏置，再用 tanh"), "kind": "forward"},
}
DROP_SCHEMA = {
    "node": {"kind": "int", "label": L("Member in focus", "关注的成员"), "tol": "exact"},
    "alpha": {"kind": "tensor", "label": L("Attention α in layer 1", "第 1 层的注意力 α"), "shape": [N], "tol": 1e-12},
    "delta": {"kind": "tensor", "label": L("Move of the score gap without that friendship", "去掉这段朋友关系后分数差的变化"), "shape": [N], "tol": 1e-12},
}
DROP_PHASES = {
    "base": {"label": L("The answer with every friendship", "所有朋友关系都在时的答案"), "kind": "forward"},
    "remove": {"label": L("Remove one friendship and re-run", "去掉一段朋友关系，重新计算"), "kind": "other"},
    "verdict": {"label": L("Most attention vs largest effect", "注意力最大与影响最大"), "kind": "output"},
}


def vec(p: str, v) -> dict:
    return {f"{p}{k}": float(x) for k, x in enumerate(v)}


def leaky(x: float, slope: float) -> float:
    return x if x > 0 else slope * x


def senders(adj: list, i: int) -> list[int]:
    return [i] + [j for j in range(N) if j != i and adj[i * N + j]]


def gat(inp: dict | None, seed: int, opt: dict) -> "ak.TapeWriter":
    w = ak.TapeWriter("gat", GAT_SCHEMA, seed=seed, opt=opt, phases=GAT_PHASES, provenance="live")
    if not inp:
        return w
    i, z, slope, adj = inp["node"], inp["z"], inp["slope"], inp["adj"]
    S = senders(adj, i)

    def dot(v, j):
        s = 0.0
        for o in range(F):
            s += z[j * F + o] * v[o]
        return s

    def friends(j):
        return len(senders(adj, j)) - 1

    wt = [0.0] * N
    st = lambda s: {"node": i, "w": np.array(wt, dtype=np.float64), "sum": np.array(s, dtype=np.float64)}  # noqa: E731
    sd = dot(inp["ad"], i)
    e = []
    for k, j in enumerate(S):
        ss = dot(inp["as"], j)
        e.append(leaky(sd + ss, slope))
        wt[j] = e[k]
        w.frame("score", st([0.0] * F), step=k + 1, metrics={"node": i, "j": j, "k": k + 1, "of": len(S), "dj": friends(j), "sd": sd, "ss": ss,
                                                       "raw": sd + ss, "e": e[k], "slope": slope, **vec("x", z[j * F:j * F + F])})
    m = max(e)
    ex = [math.exp(v - m) for v in e]
    tot = 0.0
    for v in ex:
        tot += v
    for k, j in enumerate(S):
        wt[j] = ex[k] / tot
    top = S[1] if len(S) > 1 else i
    for j in S[1:]:
        if wt[j] > wt[top]:
            top = j
    w.frame("softmax", st([0.0] * F), step=0, metrics={"node": i, "of": len(S), "friends": len(S) - 1, "max": m, "tot": tot, "self": wt[i], "top": top,
                                                     "aTop": wt[top], "dTop": friends(top), "eTop": e[S.index(top)], "even": 1 / len(S)})
    s = [0.0] * F
    for k, j in enumerate(S):
        msg = [wt[j] * v for v in z[j * F:j * F + F]]
        s = [a + b for a, b in zip(s, msg)]
        w.frame("message", st(s), step=k + 1, metrics={"node": i, "j": j, "k": k + 1, "of": len(S), "a": wt[j], **vec("x", z[j * F:j * F + F]),
                                                  **vec("m", msg), **vec("s", s)})
    y = [v + b for v, b in zip(s, inp["b"])]
    w.frame("activate", st(s), step=0, metrics={"node": i, **vec("s", s), **vec("b", inp["b"]), **vec("y", y), **vec("h", [math.tanh(v) for v in y])})
    return w


def gat_layer(h, I: int, p: dict, pre: str, adj: list, slope: float):
    W, a_s, a_d, b = p[f"{pre}.weight"], p[f"{pre}.att_src"], p[f"{pre}.att_dst"], p[f"{pre}.bias"]
    O = len(b)
    z = [0.0] * (N * O)
    for j in range(N):
        for o in range(O):
            s = 0.0
            for k in range(I):
                s += (h[j * I + k] if h is not None else (1.0 if j == k else 0.0)) * W[o * I + k]
            z[j * O + o] = s

    def dot(v, j):
        s = 0.0
        for o in range(O):
            s += z[j * O + o] * v[o]
        return s

    alpha = [0.0] * (N * N)
    out = [0.0] * (N * O)
    for i in range(N):
        S = senders(adj, i)
        e = [leaky(dot(a_d, i) + dot(a_s, j), slope) for j in S]
        m = max(e)
        ex = [math.exp(v - m) for v in e]
        tot = 0.0
        for v in ex:
            tot += v
        for k, j in enumerate(S):
            alpha[i * N + j] = ex[k] / tot
        for o in range(O):
            s = 0.0
            for j in S:
                s += alpha[i * N + j] * z[j * O + o]
            out[i * O + o] = s + b[o]
    return alpha, out


def forward_gat(p: dict, adj: list, slope: float):
    alpha, o1 = gat_layer(None, N, p, "gat.gc1", adj, slope)
    h1 = [math.tanh(v) for v in o1]
    alpha2, o2 = gat_layer(h1, len(h1) // N, p, "gat.gc2", adj, slope)
    emb = [math.tanh(v) for v in o2]
    W, b = p["gat.head.weight"], p["gat.head.bias"]
    gap = [(W[2] - W[0]) * emb[r * 2] + (W[3] - W[1]) * emb[r * 2 + 1] + (b[1] - b[0]) for r in range(N)]
    return alpha, alpha2, gap


def adjacency(edges) -> list:
    a = [0.0] * (N * N)
    for i, j in edges:
        a[i * N + j] = a[j * N + i] = 1.0
    return a


def ablate(edges, i: int, p: dict, slope: float):
    adj = adjacency(edges)
    alpha, alpha2, gap = forward_gat(p, adj, slope)
    rows = []
    for j in senders(adj, i)[1:]:
        cut = [e for e in edges if not ((e[0] == i and e[1] == j) or (e[0] == j and e[1] == i))]
        rows.append({"j": j, "a": alpha[i * N + j], "a2": alpha2[i * N + j], "d": forward_gat(p, adjacency(cut), slope)[2][i] - gap[i]})
    top_a = top_a2 = top_e = rows[0] if rows else None
    for r in rows:
        if r["a"] > top_a["a"]:
            top_a = r
        if r["a2"] > top_a2["a2"]:
            top_a2 = r
        if abs(r["d"]) > abs(top_e["d"]):
            top_e = r
    return gap[i], rows, top_a, top_a2, top_e


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def drop(inp: dict | None, seed: int, opt: dict) -> "ak.TapeWriter":
    w = ak.TapeWriter("drop", DROP_SCHEMA, seed=seed, opt=opt, phases=DROP_PHASES, provenance="live")
    if not inp:
        return w
    i, p, slope, edges = inp["node"], inp["p"], inp["slope"], inp["edges"]
    gap, rows, A, A2, E = ablate(edges, i, p, slope)
    alpha, delta = [0.0] * N, [0.0] * N
    for r in rows:
        alpha[r["j"]] = r["a"]
    st = lambda: {"node": i, "alpha": np.array(alpha, dtype=np.float64), "delta": np.array(delta, dtype=np.float64)}  # noqa: E731
    w.frame("base", st(), step=0, metrics={"node": i, "gap": gap, "friends": len(rows)})
    for k, r in enumerate(rows):
        delta[r["j"]] = r["d"]
        w.frame("remove", st(), step=k, metrics={"node": i, "j": r["j"], "k": k + 1, "of": len(rows), "a": r["a"], "a2": r["a2"], "d": r["d"], "gap": gap,
                                                "after": gap + r["d"], "flip": 1 if sign(gap + r["d"]) != sign(gap) else 0})
    agree = agree2 = members = 0
    for m in range(N):
        _, q, qa, qa2, qe = ablate(edges, m, p, slope)
        if len(q) < 2:
            continue
        members += 1
        agree += 1 if qa["j"] == qe["j"] else 0
        agree2 += 1 if qa2["j"] == qe["j"] else 0
    pair = {"topA": A["j"], "aTopA": A["a"], "dTopA": A["d"], "topE": E["j"], "aTopE": E["a"], "a2TopE": E["a2"], "dTopE": E["d"], "same": 1 if A["j"] == E["j"] else 0,
            "topA2": A2["j"], "aTopA2": A2["a2"], "same2": 1 if A2["j"] == E["j"] else 0} if A else {}
    w.frame("verdict", st(), step=0, metrics={"node": i, "friends": len(rows), "gap": gap, **pair, "agree": agree, "agree2": agree2, "members": members})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", required=True, choices=["gat", "drop"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None, help="unused: these processes draw no random numbers")
    ap.add_argument("--input", default="null")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = a.input
    if raw.startswith("@"):  # a file path (long inputs)
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    tape = (gat if a.process == "gat" else drop)(json.loads(raw), a.seed, json.loads(a.opt))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "gnn-variants", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
