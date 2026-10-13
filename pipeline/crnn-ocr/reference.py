"""crnn-ocr: Python reference for the process 'decode' (kind C parity).

An independent float64 implementation of site/src/lib/explainers/crnn-ocr/process.ts on the same per-column chances
(the traced float32 values, passed as JSON): numpy's argmax per column (ties to the smallest class), CTC's collapse
(merge repeats, drop blanks), the log chance of that path, the total chance of the text over every path from PyTorch's
own torch.nn.functional.ctc_loss (float64), and the number of such paths counted with Python integers.
`npm run atlas -- parity crnn-ocr` records the TS tape, runs this file and compares the tapes frame by frame: phase and
step exactly, every field within its tol, every metric within 1e-12 (logText and logTarget within 1e-9).

    npm run atlas -- parity crnn-ocr --fixtures            # the usual way (every fixture input too)

CLI contract: --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>
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
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import SYMBOLS  # noqa: E402


def label(en: str, zh: str) -> dict:
    return {"en": en, "zh": zh}


def total(logp: np.ndarray, ids: list[int]) -> tuple[float, int]:
    """(log p(ids) from PyTorch's CTC loss in float64, the number of column paths that collapse to ids)."""
    T = logp.shape[0]
    lp = torch.tensor(logp, dtype=torch.float64)[:, None, :]
    loss = F.ctc_loss(lp, torch.tensor(ids, dtype=torch.long), torch.tensor([T]), torch.tensor([len(ids)]),
                      blank=0, reduction="sum", zero_infinity=False)
    ext = [0]
    for k in ids:
        ext += [k, 0]
    S = len(ext)
    n = [1 if s < 2 else 0 for s in range(S)]
    for _ in range(1, T):
        n = [n[s] + (n[s - 1] if s >= 1 else 0) + (n[s - 2] if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2] else 0)
             for s in range(S)]
    return -float(loss), n[S - 1] + (n[S - 2] if S >= 2 else 0)


def run_decode(inp: dict, seed: int, opt: dict) -> ak.TapeWriter:
    schema = {"path": {"kind": "tensor", "label": label("path", "path"), "tol": "exact"},
              "text": {"kind": "text", "label": label("text", "text"), "tol": "exact"}}
    phases = {"read": {"label": "read", "kind": "output"}, "done": {"label": "done", "kind": "output"}}
    w = ak.TapeWriter("decode", schema, seed=seed, opt=opt, phases=phases, provenance="live")
    T, V = int(inp.get("T", 0)), int(inp.get("V", 0))
    if T == 0:
        return w
    p = np.array(inp["probs"], dtype=np.float64).reshape(T, V)
    top = [int(k) for k in p.argmax(1)]
    path = np.full(T, -1.0)
    text = ""
    for t, k in enumerate(top):
        act = 0 if k == 0 else 1 if t > 0 and top[t - 1] == k else 2
        path[t] = k
        if act == 2:
            text += SYMBOLS[k]
        w.frame("read", {"path": path.copy(), "text": text}, step=t,
                metrics={"t": t, "k": k, "p": float(p[t, k]), "act": act, "n": len(text)})
    logp = np.log(p)
    log_text, paths = total(logp, [SYMBOLS.index(c) for c in text])
    metrics = {"T": T, "n": len(text), "logPath": float(sum(float(logp[t, k]) for t, k in enumerate(top))), "logText": log_text, "paths": paths}
    target = inp.get("target") or ""
    if target:
        lt, pt = total(logp, [SYMBOLS.index(c) for c in target])
        if pt > 0:
            metrics |= {"logTarget": lt, "pathsTarget": pt}
    w.frame("done", {"path": path.copy(), "text": text}, step=T, metrics=metrics)
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="decode", choices=["decode"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran")
    ap.add_argument("--draws", default=None, help="draws.json written by the TS run (decode draws nothing)")
    ap.add_argument("--input", default='{"T": 0}', help="JSON input of the TS run: {T, V, probs, target} (or @file)")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/crnn-ocr/<process>.ref)")
    a = ap.parse_args()
    tape = run_decode(ak.json_arg(a.input) or {}, a.seed, json.loads(a.opt))
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "crnn-ocr", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
