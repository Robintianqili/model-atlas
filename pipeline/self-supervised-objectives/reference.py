"""self-supervised-objectives: Python reference for process 'hide' (kind C parity, design §4.11), copied from tinybert.

An independent implementation of site/src/lib/explainers/self-supervised-objectives/process.ts (BERT's training-time masking replayed on
the learner's text): ak.Rng is bit-identical to the TS rng for next() and int(), so the frames must match exactly.

    npm run atlas -- parity self-supervised-objectives             # records the TS tape, runs this file with the arguments below, compares

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

MASK, MAXC = 1024, 61
NONE, HIDE, RANDOM, KEEP = 0, 1, 2, 3
SCHEMA = {
    "fate": {"kind": "tensor", "label": ("What happens to each token", "每个词元的处理方式"), "track": "changes", "tol": "exact"},
    "shown": {"kind": "tensor", "label": ("The ids the model reads", "模型读到的 id"), "track": "changes", "tol": "exact"},
}
PHASES = {
    "start": {"label": ("Your text", "你的文字"), "kind": "input"},
    "draw": {"label": ("Draw for one token", "为一个词元抽签"), "kind": "sample"},
    "force": {"label": ("None chosen: pick one", "一个都没选中：随机挑一个"), "kind": "sample"},
    "done": {"label": ("The training example", "训练样本"), "kind": "output"},
}
DEFAULTS = {"rate": 0.15, "pMask": 0.8, "pRandom": 0.1, "vocab": 1023}


def run(raw, seed: int = 1, opt: dict | None = None) -> "ak.TapeWriter":
    o = {**DEFAULTS, **(opt or {})}
    w = ak.TapeWriter("hide", SCHEMA, seed=seed, opt=o, phases=PHASES, provenance="live")
    rng = ak.Rng(seed)
    c = [int(x) for x in (raw or {}).get("ids", [])][:MAXC] if isinstance(raw, dict) else []
    n = len(c)
    fate = np.zeros(n, dtype=np.float64)
    shown = np.array(c, dtype=np.float64)
    count = [0, 0, 0, 0]
    if n > 0:
        w.frame("start", {"fate": fate, "shown": shown}, metrics={"n": n})

    def treat(i: int, phase: str, u: float) -> None:
        v = rng.next()
        f = HIDE if v < o["pMask"] else RANDOM if v < o["pMask"] + o["pRandom"] else KEEP
        rid = rng.int(o["vocab"]) if f == RANDOM else -1
        to = MASK if f == HIDE else rid if f == RANDOM else c[i]
        fate[i] = f
        shown[i] = to
        count[f] += 1
        ch = [{"field": "fate", "index": [i], "from": 0, "to": f}]
        if to != c[i]:
            ch.append({"field": "shown", "index": [i], "from": c[i], "to": to})
        w.frame(phase, {"fate": fate, "shown": shown}, step=i, changes=ch,
                metrics={"i": i, "u": u, "chosen": 1, "v": v, "fate": f, "rid": rid, "rate": o["rate"]})

    for i in range(n):
        u = rng.next()
        if u < o["rate"]:
            treat(i, "draw", u)
        else:
            w.frame("draw", {"fate": fate, "shown": shown}, step=i,
                    metrics={"i": i, "u": u, "chosen": 0, "v": -1, "fate": 0, "rid": -1, "rate": o["rate"]})
    if n > 0 and sum(count[1:]) == 0:
        treat(rng.int(n), "force", -1)
    if n > 0:
        w.frame("done", {"fate": fate, "shown": shown},
                metrics={"n": n, "hidden": sum(count[1:]), "masked": count[1], "random": count[2], "kept": count[3]})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="hide")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None, help="draws.json from the TS run (not needed: next() and int() are bit-identical)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run ({text, ids, mask})")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = json.loads(a.input)
    tape = run(raw, seed=a.seed, opt=json.loads(a.opt))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "self-supervised-objectives", "hide.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
