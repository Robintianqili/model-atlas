"""tiny-encdec: Python reference for process 'decode' (kind C parity, design §4.11).

An independent run of site/src/lib/explainers/tiny-encdec/process.ts in PyTorch float64: the same weights
(model_final.pt, the float32 values export.py wrote into weights.bin, widened to float64), model.py's encoder and
decoder in float64, greedy decoding (the decoder runs once per symbol on <s> and everything written so far) with ties
to the smaller id. `atlas parity` records the TS tape, runs this file and compares the frames one by one (floats
within 1e-12 relative, the rest exact).

    npm run atlas -- parity tiny-encdec          # passes --process decode --input '{"ids": [...]}' --out <stem>

Arguments (the CLI contract): --process <id> --seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>.
The process draws nothing, so --seed, --opt and --draws are ignored.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import BOS, EOS, STEPS, EncDec  # noqa: E402

SCHEMA = {
    "cross": {"kind": "tensor", "label": ("cross-attention weights [head, step, character]", "交叉注意力权重 [头, 步, 字符]"), "scale": "sequential", "tol": 1e-12},
    "out": {"kind": "tensor", "label": ("symbols written (−1: not yet)", "写下的符号（−1：还没写）"), "tol": "exact"},
}
PHASES = {
    "start": {"label": ("Start", "开始"), "kind": "input"},
    "attend": {"label": ("Look at the input", "看输入"), "kind": "forward"},
    "write": {"label": ("Write a symbol", "写一个符号"), "kind": "output"},
}


def load_model() -> EncDec:
    data = json.loads((HERE / "runs" / "data" / "train.json").read_text(encoding="utf-8"))
    c = json.loads((HERE / "train_metrics.json").read_text(encoding="utf-8"))["model"]
    m = EncDec(len(data["src_vocab"]), len(data["tgt_vocab"]), d=c["d"], heads=c["heads"], ff=c["ff"], n_enc=c["n_enc"], n_dec=c["n_dec"])
    m.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    return m.double().eval()


@torch.no_grad()
def run(ids: list[int]) -> "ak.TapeWriter":
    m = load_model()
    tape = ak.TapeWriter("decode", SCHEMA, seed=1, opt={}, phases=PHASES, provenance="live")
    T = len(ids)
    if T == 0:
        return tape
    mem, _ = m.encode(torch.tensor([ids]))                          # [1, T, D] float64
    H = m.dec[0].ca.h
    cross = np.zeros((H, STEPS, T))
    out = np.full(STEPS, -1.0)
    tape.frame("start", {"cross": cross.copy(), "out": out.copy()}, metrics={"T": T})
    y_in = [BOS]
    for t in range(STEPS):
        logits, _, wc = m.decode(torch.tensor([y_in]), mem)
        cross[:, t] = wc[-1][0, :, t].numpy()
        avg = cross[:, t].mean(0)
        j = int(np.argmax(avg))
        tape.frame("attend", {"cross": cross.copy()}, step=t, metrics={"t": t, "j": j, "w": float(avg[j])})
        p = torch.softmax(logits[0, t], dim=0)
        y = int(torch.argmax(p))                                    # ties: the smaller id (torch returns the first)
        out[t] = y
        tape.frame("write", {"out": out.copy()}, step=t, metrics={"t": t, "y": y, "p": float(p[y])})
        if y == EOS:
            break
        y_in.append(y)
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="decode")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default='{"ids": []}', help="JSON input of the TS run: {ids}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/tiny-encdec/decode.ref)")
    a = ap.parse_args()
    if a.process != "decode":
        sys.exit(f"reference.py knows process 'decode', not {a.process!r}")
    t = run([int(i) for i in json.loads(a.input).get("ids", [])])
    jpath, _ = t.write(a.out or ak.site_path(".atlas", "tiny-encdec", "decode.ref"))
    print(f"wrote {jpath} ({len(t.frames)} frames)")
