"""char-rnn: Python reference for the processes 'unroll' and 'sample' (kind C parity, design §4.11, §8.3 M4).

An independent float64 implementation in PyTorch of site/src/lib/explainers/char-rnn/{rnn,process}.ts: the trained
model (model_final.pt, the same float32 weights as weights.bin) cast to float64 (model.double()).
  unroll  the LSTM steps written out (z = W_ih x + W_hh h + b_ih + b_hh; i f g o; c; h) and softmax(head(h)),
          four frames per character (gates, cell, hidden, predict).
  sample  a plain Python loop with TinyGPT's sampling semantics: the model reads the last 128 characters from an empty
          memory, softmax(scores / temperature) in float64, u = ak.Rng(seed).next() (bit-identical to the TS rng), the
          first character whose running total passes u; temperature 0 takes the top character and draws nothing.
`npm run atlas -- parity char-rnn` records the TS tape of every process source (with its draws), runs this file and
compares the tapes frame by frame: phase and step exactly, every field within its tol, every metric within 1e-12.
Each draw this file makes is checked against the recorded TS draw.

    npm run atlas -- parity char-rnn --fixtures            # the usual way (every fixture input too)
    npm run atlas -- py pipeline/char-rnn/reference.py --process sample --opt "{\"temperature\": 0}" --input "{\"ids\": [3, 4]}"

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
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CharLSTM  # noqa: E402

CONTEXT = 128
m = CharLSTM()
m.load_state_dict(torch.load(HERE / "model_final.pt", map_location="cpu"))
m = m.double().eval()
H = m.lstm.hidden_size
W_ih, W_hh = m.lstm.weight_ih_l0.detach(), m.lstm.weight_hh_l0.detach()
B = (m.lstm.bias_ih_l0 + m.lstm.bias_hh_l0).detach()


def label(en: str, zh: str) -> dict:
    return {"en": en, "zh": zh}


def step(i: int, h: torch.Tensor, c: torch.Tensor):
    z = W_ih @ m.emb.weight[i].detach() + W_hh @ h + B
    ig, fg, gg, og = z[:H].sigmoid(), z[H:2 * H].sigmoid(), z[2 * H:3 * H].tanh(), z[3 * H:].sigmoid()
    c = fg * c + ig * gg
    return ig, fg, gg, og, c, og * c.tanh()


def scores(h: torch.Tensor) -> np.ndarray:
    return (m.head.weight.detach() @ h + m.head.bias.detach()).numpy()


def softmax(x: np.ndarray) -> np.ndarray:
    p = np.exp(x - x.max())
    return p / p.sum()


@torch.no_grad()
def run_unroll(ids: list[int], seed: int, opt: dict) -> ak.TapeWriter:
    schema = {k: {"kind": "tensor", "label": label(k, k), "tol": 1e-12} for k in ("i", "f", "g", "o", "c", "h", "probs")}
    phases = {"gates": {"label": "gates", "kind": "forward"}, "cell": {"label": "cell", "kind": "forward"},
              "hidden": {"label": "hidden", "kind": "forward"}, "predict": {"label": "predict", "kind": "output"}}
    w = ak.TapeWriter("unroll", schema, seed=seed, opt=opt, phases=phases, provenance="live")
    ids = ids[-CONTEXT:]
    h = torch.zeros(H, dtype=torch.float64)
    c = torch.zeros(H, dtype=torch.float64)
    hits = 0
    for t, ch in enumerate(ids):
        ig, fg, gg, og, c, h = step(ch, h, c)
        mt = {"t": t, "ch": ch}
        w.frame("gates", {"i": ig.numpy(), "f": fg.numpy(), "g": gg.numpy(), "o": og.numpy()}, step=t, metrics=mt)
        w.frame("cell", {"c": c.numpy()}, step=t, metrics=mt)
        w.frame("hidden", {"h": h.numpy()}, step=t, metrics=mt)
        p = softmax(scores(h))
        top = int(np.argmax(p))
        nxt = {"next": ids[t + 1], "pNext": float(p[ids[t + 1]])} if t + 1 < len(ids) else {}
        if t + 1 < len(ids) and top == ids[t + 1]:
            hits += 1
        w.frame("predict", {"probs": p}, step=t, metrics={**mt, "top": top, "pTop": float(p[top]), **nxt, "hits": hits,
                                                          "guesses": min(t + 1, len(ids) - 1)})
    return w


class Draws:
    """The TS run's recorded draws: each of this file's own draws must equal the next recorded one."""

    def __init__(self, frames: list | None):
        self.flat = [d for f in (frames or []) for d in f]
        self.i = 0

    def check(self, own: float) -> float:
        if self.flat:
            rec = self.flat[self.i] if self.i < len(self.flat) else None
            if rec != own:
                sys.exit(f"reference: draw #{self.i} is {own} here but {rec} in TS (sampling must use rng.next() only)")
            self.i += 1
        return own


def distribution(logits: np.ndarray, temperature: float) -> list[float]:
    """softmax(logits / temperature) in float64, summed in index order (top-k and top-p off, as the page runs it)."""
    x = [float(v) / temperature for v in logits]
    mx = max(x)
    p = [math.exp(v - mx) for v in x]
    s = 0.0
    for v in p:
        s += v
    return [v / s for v in p]


def pick(dist: list[float], u: float) -> int:
    token, cum = -1, 0.0
    for i, v in enumerate(dist):
        if v == 0:
            continue
        cum += v
        token = i
        if cum > u:
            break
    return token


def top_entries(p: list[float], n: int) -> list[dict]:
    idx = sorted(range(len(p)), key=lambda i: (-p[i], i))
    return [{"id": i, "p": p[i]} for i in idx[:n]]


@torch.no_grad()
def run_sample(ids: list[int], seed: int, opt: dict, draws: Draws) -> ak.TapeWriter:
    o = {"maxNew": 60, "temperature": 1, "topK": 0, "topP": 1, "stopId": -1, "topN": 10, **opt}
    if o["topK"] or o["topP"] < 1:
        sys.exit("reference: top-k and top-p are not used by the page")
    schema = {"ids": {"kind": "json", "label": label("ids", "ids"), "tol": "exact"},
              "token": {"kind": "int", "label": label("token", "token"), "tol": "exact"},
              "p": {"kind": "scalar", "label": label("p", "p"), "tol": 1e-12},
              "pModel": {"kind": "scalar", "label": label("pModel", "pModel"), "tol": 1e-12},
              "u": {"kind": "scalar", "label": label("u", "u"), "tol": "exact"},
              "top": {"kind": "json", "label": label("top", "top"), "tol": 1e-12},
              "contextStart": {"kind": "int", "label": label("contextStart", "contextStart"), "tol": "exact"}}
    phases = {"prompt": {"label": "prompt", "kind": "input"}, "logits": {"label": "logits", "kind": "forward"},
              "distribution": {"label": "distribution", "kind": "other"}, "sample": {"label": "sample", "kind": "sample"}}
    w = ak.TapeWriter("sample", schema, seed=seed, opt=o, phases=phases, provenance="live")
    rng = ak.Rng(seed)
    ids = list(ids[-CONTEXT:])
    w.frame("prompt", {"ids": list(ids)}, metrics={"length": len(ids)})
    T = o["temperature"]
    for s in range(o["maxNew"]):
        start = max(0, len(ids) - CONTEXT)
        h = torch.zeros(H, dtype=torch.float64)
        c = torch.zeros(H, dtype=torch.float64)
        for ch in ids[start:]:
            *_, c, h = step(ch, h, c)
        z = scores(h)
        p_model = distribution(z, 1)
        u = float("nan")
        if T <= 0:
            token = int(np.argmax(z))
            dist = [0.0] * len(z)
            dist[token] = 1.0
        else:
            dist = distribution(z, T)
            u = draws.check(rng.next())
            token = pick(dist, u)
        ids.append(token)
        shown = p_model if T <= 0 else dist
        w.frame("sample", {"ids": list(ids), "token": token, "p": dist[token], "pModel": p_model[token], "u": u,
                           "top": top_entries(shown, o["topN"]), "contextStart": start},
                step=s, metrics={"p": dist[token], "pModel": p_model[token], "u": u, "bits": -math.log2(p_model[token])})
        if token == o["stopId"]:
            break
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="sample", choices=["unroll", "sample"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran")
    ap.add_argument("--draws", default=None, help="draws.json written by the TS run (each draw is checked)")
    ap.add_argument("--input", default='{"ids": []}', help="JSON input of the TS run: {ids: [...]}")
    ap.add_argument("--out", default=None, help="output stem (default $ATLAS_SITE/.atlas/char-rnn/<process>.ref)")
    a = ap.parse_args()
    opt, inp = json.loads(a.opt), json.loads(a.input) or {}
    ids = [int(i) for i in inp.get("ids", [])]
    draws = Draws(json.loads(Path(a.draws).read_text(encoding="utf-8"))["frames"] if a.draws else None)
    tape = run_unroll(ids, a.seed, opt) if a.process == "unroll" else run_sample(ids, a.seed, opt, draws)
    if draws.flat and draws.i != len(draws.flat):
        sys.exit(f"reference: used {draws.i} of the {len(draws.flat)} TS draws")
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "char-rnn", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames, {draws.i} draws checked)")
