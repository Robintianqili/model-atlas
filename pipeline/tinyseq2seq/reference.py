"""tinyseq2seq: Python reference for process 'decode' (kind C parity, design §4.11).

An independent run of site/src/lib/explainers/tinyseq2seq/process.ts in PyTorch float64: the same weights
(model_final.pt, the float32 values export.py wrote into weights.bin, widened to float64), the GRU encoder (nn.GRU in
float64), Bahdanau attention and the decoder's GRUCell, greedy decoding with ties to the smaller id. `atlas parity`
records the TS tape, runs this file and compares the frames one by one (floats within 1e-12 relative, the rest exact).

    npm run atlas -- parity tinyseq2seq          # passes --process decode --input '{"ids": [...]}' --out <stem>

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
from model import BOS, EOS, STEPS, Seq2Seq  # noqa: E402

SCHEMA = {
    "align": {"kind": "tensor", "label": ("attention weights, one row per step", "注意力权重，每步一行"), "scale": "sequential", "tol": 1e-12},
    "out": {"kind": "tensor", "label": ("symbols written (−1: not yet)", "写下的符号（−1：还没写）"), "tol": "exact"},
}
PHASES = {
    "start": {"label": ("Start", "开始"), "kind": "input"},
    "attend": {"label": ("Look at the input", "看输入"), "kind": "forward"},
    "write": {"label": ("Write a symbol", "写一个符号"), "kind": "output"},
}


def load_model() -> Seq2Seq:
    data = json.loads((HERE / "runs" / "data" / "train.json").read_text(encoding="utf-8"))
    m = Seq2Seq(len(data["src_vocab"]), len(data["tgt_vocab"]))
    m.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
    return m.double().eval()


@torch.no_grad()
def run(ids: list[int]) -> "ak.TapeWriter":
    m = load_model()
    tape = ak.TapeWriter("decode", SCHEMA, seed=1, opt={}, phases=PHASES, provenance="live")
    T = len(ids)
    if T == 0:
        return tape
    states, _ = m.enc(m.src_emb(torch.tensor(ids))[None])        # [1, T, H] float64
    h = states[0]
    keys = m.att_k(h)                                             # [T, A]
    s, y = h[-1], BOS
    align = np.zeros((STEPS, T))
    out = np.full(STEPS, -1.0)
    tape.frame("start", {"align": align.copy(), "out": out.copy()}, metrics={"T": T})
    for t in range(STEPS):
        e = m.att_v(torch.tanh(m.att_q(s)[None, :] + keys))[:, 0]
        a = torch.softmax(e, dim=0)
        c = a @ h
        s = m.dec(torch.cat([m.tgt_emb.weight[y], c])[None], s[None])[0]
        logits = m.out(torch.cat([s, c]))
        y = int(torch.argmax(logits))                             # ties: the smaller id (torch returns the first)
        p = torch.softmax(logits, dim=0)
        align[t] = a.numpy()
        j = int(torch.argmax(a))
        tape.frame("attend", {"align": align.copy()}, step=t, metrics={"t": t, "j": j, "w": float(a[j])})
        out[t] = y
        tape.frame("write", {"out": out.copy()}, step=t, metrics={"t": t, "y": y, "p": float(p[y])})
        if y == EOS:
            break
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="decode")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default='{"ids": []}', help="JSON input of the TS run: {ids}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/tinyseq2seq/decode.ref)")
    a = ap.parse_args()
    if a.process != "decode":
        sys.exit(f"reference.py knows process 'decode', not {a.process!r}")
    t = run([int(i) for i in json.loads(a.input).get("ids", [])])
    jpath, _ = t.write(a.out or ak.site_path(".atlas", "tinyseq2seq", "decode.ref"))
    print(f"wrote {jpath} ({len(t.frames)} frames)")
