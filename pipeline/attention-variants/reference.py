"""attention-variants: Python reference for the processes 'cache' and 'share' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/attention-variants/process.ts (and data.ts):
  cache   the KV-cache arithmetic in plain Python integers, from the same clamped configuration
  share   TinyGPT in PyTorch fp32 (gqa.py) on the weights the TS run received, with TRUE grouped-query attention: the key
          and value projections pooled to kv heads (gqa.pool_kv, shape [kv*d, D]) and repeat_interleave'd to the query
          heads, where the browser copies the pooled rows to every head of the group. Scored on the same four windows
          and probe (evalset.ts, read from the TS file); metrics within the ProcessDef's metricTol (float32 passes).
`npm run atlas -- parity attention-variants` records the TS tapes, runs this file and compares them frame by frame.

Arguments (the CLI contract for every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json>
--input <json> --out <stem>. No process draws random numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repository's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gqa import SITE, forward, pool_kv  # noqa: E402

START = {"layers": 4, "heads": 4, "kvHeads": 4, "dHead": 19, "tokens": 64, "bytes": 4, "latent": 0}
LIMITS = {"layers": (1, 200), "heads": (1, 256), "dHead": (1, 512), "tokens": (1, 1_048_576), "latent": (0, 65_536)}
KVS = [4, 2, 1]


def clamp_int(v, lo, hi, d):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return d
    return min(hi, max(lo, int(math.floor(v + 0.5))))


def config_of(raw) -> dict:
    """data.ts configOf: whole numbers in range, kvHeads the divisor of heads nearest the asked value (ties: smaller)."""
    r = raw if isinstance(raw, dict) else {}
    heads = clamp_int(r.get("heads"), *LIMITS["heads"], START["heads"])
    want = clamp_int(r.get("kvHeads"), 1, heads, heads)
    kv = min((k for k in range(1, heads + 1) if heads % k == 0), key=lambda k: (abs(k - want), k))
    return {
        "layers": clamp_int(r.get("layers"), *LIMITS["layers"], START["layers"]),
        "heads": heads,
        "kvHeads": kv,
        "dHead": clamp_int(r.get("dHead"), *LIMITS["dHead"], START["dHead"]),
        "tokens": clamp_int(r.get("tokens"), *LIMITS["tokens"], START["tokens"]),
        "bytes": r.get("bytes") if r.get("bytes") in (4, 2, 1) else START["bytes"],
        "latent": clamp_int(r.get("latent"), *LIMITS["latent"], 0),
    }


def cache_proc(raw, tape):
    c = config_of(raw)
    per_block = lambda c: c["latent"] if c["latent"] > 0 else 2 * c["kvHeads"] * c["dHead"]
    per_token = lambda c: c["layers"] * per_block(c) * c["bytes"]
    mha = {**c, "kvHeads": c["heads"], "latent": 0}
    mqa = {**c, "kvHeads": 1, "latent": 0}
    kind = 3 if c["latent"] > 0 else 0 if c["kvHeads"] == c["heads"] else 2 if c["kvHeads"] == 1 else 1
    m = {**c, "kind": kind, "group": c["heads"] / c["kvHeads"], "perBlock": per_block(c), "perTokenNums": c["layers"] * per_block(c),
         "perToken": per_token(c), "total": per_token(c) * c["tokens"], "mhaPerToken": per_token(mha), "mhaTotal": per_token(mha) * c["tokens"],
         "mqaPerToken": per_token(mqa), "mqaTotal": per_token(mqa) * c["tokens"], "saving": per_token(mha) / per_token(c),
         "macs": c["heads"] * c["tokens"] * c["dHead"]}
    tape.frame("cache", {"per": np.array([per_token(mha), per_token(c), per_token(mqa)], dtype=np.float64)}, step=0, metrics=m)


def evalset():
    src = (SITE / "src" / "lib" / "explainers" / "attention-variants" / "evalset.ts").read_text(encoding="utf-8")
    ids = json.loads(re.search(r"EVAL_IDS: number\[\]\[\] = (\[.*?\]\]);", src).group(1))
    probe = json.loads(re.search(r"PROBE_IDS: number\[\] = (\[.*?\]);", src).group(1))
    return ids, probe


def share_proc(inp, opt, tape):
    if not isinstance(inp, dict) or not inp.get("weights"):
        return
    W = {k: torch.tensor(np.array(t["data"], dtype=np.float32)).view(*t["shape"]) for k, t in inp["weights"].items()}
    layers = (inp.get("spec") or {}).get("layers", [])
    H = next(l["params"]["nHead"] for l in layers if (l.get("params") or {}).get("nHead"))
    eps = next(l["params"]["eps"] for l in layers if l.get("op") == "layernorm")
    n_layer = len([k for k in W if re.match(r"^L\d+\.ln1\.weight$", k)])
    D = W["tok_emb.weight"].shape[1]
    d = D // H
    spec = {"dModel": D, "nHead": H, "nLayer": n_layer, "lnEps": eps}
    layer = int(opt.get("layer", 2))
    ids, probe_ids = evalset()
    T = len(probe_ids)
    pr = torch.tensor([probe_ids], dtype=torch.long)

    def run_probe(Wd):
        rec = {}
        with torch.no_grad():
            logits = forward(spec, Wd, pr, rec)
        P = f"L{layer}.attn."
        last = torch.log_softmax(logits[0, -1].double(), -1)
        return rec[P + "weights"][0, :, -1, :].double(), rec[P + "scores"][0, :, -1, :].double(), rec[P + "heads"][0, -1].double(), last

    def argmax(v):
        v = v.tolist()
        return max(range(len(v)), key=lambda j: (v[j], -j))

    def heads_of(w):
        m = {}
        for h in range(H):
            k = argmax(w[h])
            m[f"arg{h}"], m[f"top{h}"] = k, w[h][k].item()
        return m

    st = lambda w, s: {"w": w.numpy(), "s": s.numpy()}
    common = {"T": T, "layer": layer, "dHead": d, "nLayer": n_layer}
    w0, s0, mix0, last0 = run_probe(W)
    top0 = argmax(last0)
    tape.frame("mha", st(w0, s0), step=0, metrics={**common, "kv": H, "kvNums": 2 * H * d * T, **heads_of(w0), "next": top0, "nextP": math.exp(last0[top0].item())})
    for k, kv in enumerate(KVS[1:]):
        w, s, mix, last = run_probe(pool_kv(W, n_layer, H, kv, [layer]))
        nxt = argmax(last)
        tape.frame("shared", st(w, s), step=k, metrics={**common, "kv": kv, "kvNums": 2 * kv * d * T, **heads_of(w), "dwMax": (w - w0).abs().max().item(),
                                                         "outChange": math.sqrt(((mix - mix0) ** 2).sum().item() / (mix0 ** 2).sum().item()), "next": nxt,
                                                         "nextP": math.exp(last[nxt].item()), "next0P": math.exp(last[top0].item())})
    x = torch.tensor([w[:-1] for w in ids], dtype=torch.long)
    y = torch.tensor([w[1:] for w in ids], dtype=torch.long)
    tokens = int(y.numel())
    lp0 = None
    for k, kv in enumerate(KVS):
        Wd = W if kv == H else pool_kv(W, n_layer, H, kv)
        with torch.no_grad():
            lp = torch.log_softmax(forward(spec, Wd, x).double(), -1)
        if lp0 is None:
            lp0 = lp
        nll = -lp.gather(-1, y[..., None]).sum().item()
        kl = (lp0.exp() * (lp0 - lp)).sum().item()
        agree = (lp.argmax(-1) == lp0.argmax(-1)).sum().item()
        w, s, mix, last = run_probe(Wd)
        nxt = argmax(last)
        tape.frame("eval", st(w, s), step=k, metrics={**common, "kv": kv, "kvNums": 2 * kv * d * T, **heads_of(w), "ppl": math.exp(nll / tokens), "kl": kl / tokens,
                                                      "agree": agree / tokens, "tokens": tokens, "next": nxt, "nextP": math.exp(last[nxt].item()),
                                                      "next0P": math.exp(last[top0].item()), "cacheBytes": 4 * 2 * kv * d * n_layer})


SCHEMAS = {
    "cache": {"per": {"kind": "tensor", "label": ("Bytes per token: MHA, yours, MQA", "每个词元的字节数：MHA、你的配置、MQA"), "shape": [3], "role": "output", "tol": "exact"}},
    "share": {k: {"kind": "tensor", "label": (k, k), "shape": [4, "T"], "role": "output", "tol": 1e-4} for k in ("w", "s")},
}
PHASES = {
    "cache": {"cache": {"label": ("Count the cache", "计算缓存"), "kind": "output"}},
    "share": {"mha": {"label": ("mha", "mha"), "kind": "forward"}, "shared": {"label": ("shared", "shared"), "kind": "forward"}, "eval": {"label": ("eval", "eval"), "kind": "output"}},
}


def run(process: str, raw, opt: dict, seed: int) -> "ak.TapeWriter":
    if process not in SCHEMAS:
        raise SystemExit(f"reference.py: no reference for process '{process}'")
    tape = ak.TapeWriter(process, SCHEMAS[process], seed=seed, opt=opt, phases=PHASES[process], provenance="live")
    {"cache": lambda: cache_proc(raw, tape), "share": lambda: share_proc(raw, opt, tape)}[process]()
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="cache")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null", help="JSON input of the TS run, or @<file> (read it with ak.json_arg)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = ak.json_arg(a.input)
    tape = run(a.process, raw, json.loads(a.opt), a.seed)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "attention-variants", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
