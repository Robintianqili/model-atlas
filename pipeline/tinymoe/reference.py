"""tinymoe: Python reference for the process 'dispatch' (kind C parity, DEVKIT section 7.2).

An independent implementation of site/src/lib/explainers/tinymoe/process.ts in plain Python floats: the router chances
of one block (the input, read from the TS trace) routed character by character to their top-k experts under the
capacity ceil(cf * k * T / E), then the load-balancing loss E * sum_e f_e * P_e. `npm run atlas -- parity tinymoe`
records the TS tape, runs this file and compares the tapes frame by frame (phase and step exactly, `load` exactly,
`kept` and every metric within 1e-12).

    npm run atlas -- parity tinymoe --fixtures

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
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np

SCHEMA = {"load": {"kind": "tensor", "label": ("slots taken per expert", "每个专家接下的位置数"), "shape": ["E"], "scale": "sequential", "tol": "exact"},
          "kept": {"kind": "tensor", "label": ("gates kept (characters × experts)", "保留的门控（字符 × 专家）"), "shape": ["T", "E"], "scale": "sequential", "tol": 1e-12}}
PHASES = {"start": {"label": ("Empty experts", "空的专家"), "kind": "input"},
          "route": {"label": ("Route one character", "分派一个字符"), "kind": "act"},
          "balance": {"label": ("Balancing loss", "负载均衡损失"), "kind": "loss"}}


def top_k(p: list[float], k: int) -> tuple[list[int], list[float]]:
    i = sorted(range(len(p)), key=lambda j: (-p[j], j))[:k]
    s = 0.0
    for j in i:
        s += p[j]
    return i, [p[j] / s for j in i]


def run(raw, seed: int = 1, opt: dict | None = None) -> "ak.TapeWriter":
    opt = {"cf": 1, **(opt or {})}
    w = ak.TapeWriter("dispatch", SCHEMA, seed=seed, opt=opt, phases=PHASES, provenance="live")
    ids, p, k = [int(i) for i in raw["ids"]], [[float(v) for v in r] for r in raw["p"]], int(raw["k"])
    T = len(ids)
    if not T or len(p) != T:
        return w
    E = len(p[0])
    cap = math.ceil(opt["cf"] * k * T / E - 1e-9)
    load, sent, kept = [0.0] * E, [0.0] * E, np.zeros((T, E))
    st = lambda: {"load": np.array(load), "kept": kept.copy()}
    w.frame("start", st(), metrics={"T": T, "E": E, "k": k, "cap": cap, "cf": opt["cf"], "block": int(raw["block"])})
    dropped = lost = 0
    for t in range(T):
        i, g = top_k(p[t], k)
        ok = []
        for n, j in enumerate(i):
            sent[j] += 1
            if load[j] >= cap:
                ok.append(0)
                continue
            load[j] += 1
            kept[t, j] = g[n]
            ok.append(1)
        miss = k - sum(ok)
        dropped += miss
        lost += miss == k
        b = i[1] if k > 1 else None
        w.frame("route", st(), step=t, metrics={"t": t, "id": ids[t], "a": i[0], "b": -1 if b is None else b, "ga": g[0], "gb": g[1] if k > 1 else 0,
                                                "pa": p[t][i[0]], "pb": 0 if b is None else p[t][b], "okA": ok[0], "okB": ok[1] if k > 1 else 0,
                                                "loadA": load[i[0]], "loadB": 0 if b is None else load[b], "cap": cap, "dropped": dropped, "lost": lost, "slots": k * (t + 1)})
    m = {"T": T, "E": E, "k": k, "cap": cap, "dropped": dropped, "lost": lost, "dropShare": dropped / (k * T), "lostShare": lost / T}
    aux, busiest = 0.0, 0
    for j in range(E):
        P = 0.0
        for t in range(T):
            P += p[t][j] / T
        f = sent[j] / (k * T)
        m.update({f"f{j}": f, f"P{j}": P, f"fp{j}": f * P, f"n{j}": sent[j], f"l{j}": load[j]})
        aux += E * f * P
        if f > m[f"f{busiest}"]:
            busiest = j
    w.frame("balance", st(), step=T, metrics={**m, "aux": aux, "busiest": busiest, "fBusiest": m[f"f{busiest}"]})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="dispatch")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None, help="not needed: dispatch draws nothing")
    ap.add_argument("--input", default="null", help="JSON input of the TS run ({ids, p, block, k}) or @file")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tape = run(ak.json_arg(a.input), seed=a.seed, opt=json.loads(a.opt))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "tinymoe", "dispatch.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
