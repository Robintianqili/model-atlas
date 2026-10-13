"""masked-lm: Python reference for process 'batch' (kind C parity), adapted from tinybert's reference for 'hide'.

An independent implementation of site/src/lib/explainers/masked-lm/process.ts (BERT's training-time masking drawn on a
small batch: the learner's text, then TinyBERT's example sentences from static/models/tinybert/presets.json, then the
same rule counted on a full training batch): ak.Rng is bit-identical to the TS rng for next() and int(), so the frames
must match exactly.

    npm run atlas -- parity masked-lm             # records the TS tape, runs this file with the arguments below, compares

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
    "start": {"label": ("The batch", "这个批次"), "kind": "input"},
    "draw": {"label": ("Draw for one token", "为一个词元抽签"), "kind": "sample"},
    "force": {"label": ("None chosen in a row: pick one", "一行里一个都没选中：随机挑一个"), "kind": "sample"},
    "done": {"label": ("The training batch", "训练批次"), "kind": "output"},
}
DEFAULTS = {"rate": 0.15, "pMask": 0.8, "pRandom": 0.1, "vocab": 1023, "big": 512, "bigLen": MAXC}


def stories() -> list[list[int]]:
    presets = Path(ak.site_path("static", "models", "tinybert", "presets.json"))
    return [[int(t) for t in p["ids"]] for p in json.loads(presets.read_text(encoding="utf-8"))["items"]]


def run(raw, seed: int = 1, opt: dict | None = None) -> "ak.TapeWriter":
    o = {**DEFAULTS, **(opt or {})}
    w = ak.TapeWriter("batch", SCHEMA, seed=seed, opt=o, phases=PHASES, provenance="live")
    rng = ak.Rng(seed)
    c = [int(x) for x in (raw or {}).get("ids", [])][:MAXC] if isinstance(raw, dict) else []
    rows = [c] + [s for s in stories() if s != c] if c else []
    flat = [t for r in rows for t in r]
    n = len(flat)
    if n == 0:
        return w
    fate = np.zeros(n, dtype=np.float64)
    shown = np.array(flat, dtype=np.float64)
    count = [0, 0, 0, 0]

    def tally() -> dict:
        return {"chosen": count[1] + count[2] + count[3], "masked": count[1], "random": count[2], "kept": count[3]}

    def fate_of() -> tuple[float, int, int]:
        v = rng.next()
        f = HIDE if v < o["pMask"] else RANDOM if v < o["pMask"] + o["pRandom"] else KEEP
        return v, f, (rng.int(o["vocab"]) if f == RANDOM else -1)

    w.frame("start", {"fate": fate, "shown": shown}, metrics={"rows": len(rows), "n": n})
    first = 0
    for r, ids in enumerate(rows):
        def treat(i: int, phase: str, u: float) -> None:
            v, f, rid = fate_of()
            g = first + i
            to = MASK if f == HIDE else rid if f == RANDOM else flat[g]
            fate[g] = f
            shown[g] = to
            count[f] += 1
            ch = [{"field": "fate", "index": [g], "from": 0, "to": f}]
            if to != flat[g]:
                ch.append({"field": "shown", "index": [g], "from": flat[g], "to": to})
            w.frame(phase, {"fate": fate, "shown": shown}, step=g, changes=ch,
                    metrics={"row": r, "i": i, "at": g, "u": u, "pick": 1, "v": v, "fate": f, "rid": rid, "rate": o["rate"], **tally()})

        in_row = 0
        for i in range(len(ids)):
            u = rng.next()
            if u < o["rate"]:
                in_row += 1
                treat(i, "draw", u)
            else:
                w.frame("draw", {"fate": fate, "shown": shown}, step=first + i,
                        metrics={"row": r, "i": i, "at": first + i, "u": u, "pick": 0, "v": -1, "fate": 0, "rid": -1, "rate": o["rate"], **tally()})
        if in_row == 0:
            treat(rng.int(len(ids)), "force", -1)
        first += len(ids)
    big = [0, 0, 0, 0]
    for _ in range(o["big"]):
        in_row = 0
        for _ in range(o["bigLen"]):
            if rng.next() < o["rate"]:
                in_row += 1
                big[fate_of()[1]] += 1
        if in_row == 0:
            rng.int(o["bigLen"])
            big[fate_of()[1]] += 1
    big_chosen = big[1] + big[2] + big[3]
    w.frame("done", {"fate": fate, "shown": shown},
            metrics={"rows": len(rows), "n": n, **tally(), "big": o["big"], "bigLen": o["bigLen"], "bigN": o["big"] * o["bigLen"],
                     "bigChosen": big_chosen, "bigMasked": big[1], "bigRandom": big[2], "bigKept": big[3]})
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="batch")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None, help="draws.json from the TS run (not needed: next() and int() are bit-identical)")
    ap.add_argument("--input", default="null", help="JSON input of the TS run ({text, ids, fate, ...})")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = json.loads(a.input)
    tape = run(raw, seed=a.seed, opt=json.loads(a.opt))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "masked-lm", "batch.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
