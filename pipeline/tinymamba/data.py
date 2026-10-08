"""tinymamba: the selective-copy task (the Python twin of site/src/lib/explainers/tinymamba/task.ts).

An example is a row of dots with a few letters among them, then one '?' per letter. At the k-th '?' the model must
give back the k-th letter (counting in reading order). Only the answers at the '?' count in the loss.

    alphabet  ".abcdefgh?"   id 0 '.', ids 1-8 the letters a-h, id 9 '?'
    body      6 to 16 tokens; 1 to 4 of them are letters (at distinct places, letters may repeat)

The generator draws with ak.Rng (bit-identical to the TS Rng), in the same order as task.ts, so both make the same
examples from the same seed (checked by claims.test.ts against data_check.json, which `python data.py` writes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

ALPHABET = ".abcdefgh?"
DOT, ASK = 0, 9
LETTERS = 8
BODY_MIN, BODY_MAX = 6, 16
K_MAX = 4


def example(rng: "ak.Rng", k_max: int = K_MAX, body_min: int = BODY_MIN, body_max: int = BODY_MAX):
    """(ids, targets): targets[i] is the letter to give back at a '?', else -1."""
    n = body_min + rng.int(body_max - body_min + 1)
    k = 1 + rng.int(k_max)
    pos = list(range(n))
    for i in range(k):  # partial Fisher-Yates: k distinct places
        j = i + rng.int(n - i)
        pos[i], pos[j] = pos[j], pos[i]
    places = sorted(pos[:k])
    letters = [1 + rng.int(LETTERS) for _ in range(k)]
    body = [DOT] * n
    for p, c in zip(places, letters):
        body[p] = c
    return body + [ASK] * k, [-1] * n + letters


def batch(seed: int, count: int, **kw):
    rng = ak.Rng(seed)
    return [example(rng, **kw) for _ in range(count)]


def text(ids) -> str:
    return "".join(ALPHABET[i] for i in ids)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "data_check.json"
    rows = [{"text": text(x), "targets": t} for x, t in batch(7, 200)]
    out.write_text(json.dumps({"seed": 7, "examples": rows}, indent=0) + "\n", encoding="utf-8")
    print("wrote", out, len(rows), rows[0])
