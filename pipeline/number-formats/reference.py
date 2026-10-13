"""number-formats: Python reference for the processes 'formats', 'sweep' and 'sum' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/number-formats/process.ts (and fmt.ts), built on flib.py:
  formats  the code of x in every format: fp32 bits with torch's view, fp16 / bf16 / e4m3fn / e5m2 bits from PyTorch's
           own casts, fp4 from the nearest of its 8 sizes, int8 / int4 from round-half-even, nf4 from normal quantiles;
           every code must match exactly
  sweep    TinyGPT in PyTorch fp32 on the weights the TS run received, its 24 weight matrices cast by flib.scaled, scored
           on the same four windows and probe (evalset.ts); metrics within the ProcessDef's metricTol
  sum      one weight w0 with n updates of u: the update and every sum rounded to the format with PyTorch's dtypes
           (fp32, bf16, fp16), and a bf16 copy of the fp32 run; exact
`npm run atlas -- parity number-formats` records the TS tapes, runs this file and compares them frame by frame.

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
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from flib import BITS, FMT_IDS, LARGEST, cast, code, forward, scaled  # noqa: E402

START_X, X_MAX = 0.1, 3.4e38
CASTS = [("bf16", "bf16", "none", 0), ("fp16", "fp16", "none", 0), ("e5m2", "e5m2", "none", 0), ("e4m3", "e4m3", "none", 0),
         ("e4m3-t", "e4m3", "tensor", 0), ("fp4", "fp4", "none", 0), ("fp4-b", "fp4", "block", 32), ("int4-b", "int4", "block", 64),
         ("nf4-b", "nf4", "block", 64)]
FLOATS = ["fp32", "fp16", "bf16", "e4m3", "e5m2", "fp4"]
SMALLEST_NORMAL = {"fp32": 2.0 ** -126, "fp16": 2.0 ** -14, "bf16": 2.0 ** -126, "e4m3": 2.0 ** -6, "e5m2": 2.0 ** -14, "fp4": 1.0}
HIST_LO, HIST_HI = -26, 18
TOPK = 5
MATRIX = re.compile(r"^L\d+\.(attn\.(q|k|v|proj)|mlp\.(fc|proj))\.weight$")


def x_of(inp) -> float:
    """fmt.ts rawOf: a finite x within ±3.4e38, else the start number."""
    x = inp if isinstance(inp, (int, float)) and not isinstance(inp, bool) else (inp or {}).get("x") if isinstance(inp, dict) else None
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        return START_X
    return min(X_MAX, max(-X_MAX, float(x)))


def f32(v: float) -> float:
    return float(np.float32(v))


# ---------------------------------------------------------------- formats

def formats(inp, tape):
    x = x_of(inp)
    codes = np.array([code(f, x) for f in FMT_IDS], dtype=np.float64)
    tape.frame("store", {"codes": codes}, step=0, metrics={"x": x, "x32": f32(x)})


# ---------------------------------------------------------------- sweep

def evalset():
    """EVAL_IDS and PROBE_IDS of evalset.ts (JSON arrays on their own lines)."""
    src = (Path(ak.site_path("src", "lib", "explainers", "number-formats", "evalset.ts"))).read_text(encoding="utf-8")
    get = lambda name: json.loads(re.search(rf"export const {name}[^=]*= (.*);\n", src).group(1))
    return get("EVAL_IDS"), get("PROBE_IDS")


def sweep_proc(inp, tape):
    if not isinstance(inp, dict) or not inp.get("weights"):
        return
    W = {k: torch.tensor(np.array(t["data"], dtype=np.float32)).view(*t["shape"]) for k, t in inp["weights"].items()}
    layers = (inp.get("spec") or {}).get("layers", [])
    n_head = next(l["params"]["nHead"] for l in layers if (l.get("params") or {}).get("nHead"))
    eps = next(l["params"]["eps"] for l in layers if l.get("op") == "layernorm")
    spec = {"dModel": W["tok_emb.weight"].shape[1], "nHead": n_head, "nLayer": len([k for k in W if re.match(r"^L\d+\.ln1\.weight$", k)]), "lnEps": eps}
    names = [k for k in inp["weights"] if MATRIX.match(k)]
    params = sum(t.numel() for t in W.values())
    ids, probe_ids = evalset()
    x = torch.tensor([w[:-1] for w in ids], dtype=torch.long)
    y = torch.tensor([w[1:] for w in ids], dtype=torch.long)
    pr = torch.tensor([probe_ids], dtype=torch.long)
    tokens = int(y.numel())

    def logits(Wd):
        with torch.no_grad():
            return forward(spec, Wd, x).double(), forward(spec, Wd, pr)[0, -1].double()

    l0, p0l = logits(W)
    lp0 = torch.log_softmax(l0, -1)
    plp0 = torch.log_softmax(p0l, -1)
    order = sorted(range(plp0.numel()), key=lambda i: (-plp0[i].item(), i))[:TOPK]
    top0 = order[0]
    ppl0 = math.exp(-lp0.gather(-1, y[..., None]).sum().item() / tokens)

    allw = torch.cat([W[k].flatten() for k in names])
    n = allw.numel()
    a = np.sort(np.abs(allw.numpy().astype(np.float64)))
    nz = a[a > 0]
    hist = np.zeros(HIST_HI - HIST_LO)
    for k, c in zip(*np.unique(np.clip(np.floor(np.log2(nz)) - HIST_LO, 0, HIST_HI - HIST_LO - 1).astype(int), return_counts=True)):
        hist[k] = c
    stats = {"n": n, "nm": len(names), "params": params, "amax": float(a[-1]), "wMed": float((a[(n - 1) // 2] + a[n // 2]) / 2), "wMin": float(nz[0])}
    fates = {}
    wnz = allw[allw != 0]
    for f in FLOATS:
        c = cast(f, wnz)
        zero = c == 0
        over = ~zero & (~torch.isfinite(c) | (wnz.abs() > LARGEST[f]))
        sub = ~zero & ~over & (c.abs() < SMALLEST_NORMAL[f])
        m = wnz.numel()
        fates.update({f"sub_{f}": sub.sum().item() / m, f"zero_{f}": zero.sum().item() / m, f"over_{f}": over.sum().item() / m})

    def frame(phase, step, Wd, extra):
        l, pl = logits(Wd)
        lp = torch.log_softmax(l, -1)
        plp = torch.log_softmax(pl, -1)
        se = sum(((Wd[k].double() - W[k].double()) ** 2).sum().item() for k in names)
        ss = sum((W[k].double() ** 2).sum().item() for k in names)
        zeros = sum(((Wd[k] == 0) & (W[k] != 0)).sum().item() for k in names)
        st = {"top": np.array(order, dtype=np.float64), "p0": np.array([math.exp(plp0[i].item()) for i in order]),
              "p": np.array([math.exp(plp[i].item()) for i in order]), "hist": hist}
        m = {**stats, "ppl0": ppl0, "ppl": math.exp(-lp.gather(-1, y[..., None]).sum().item() / tokens),
             "kl": (lp0.exp() * (lp0 - lp)).sum(-1).sum().item() / tokens, "agree": (lp.argmax(-1) == lp0.argmax(-1)).sum().item() / tokens,
             "dmax": (l - l0).abs().max().item(), "tokens": tokens, "probeTop": int(plp.argmax().item()), "probeTop0": top0,
             "probeP": math.exp(plp[top0].item()), "probeP0": math.exp(plp0[top0].item()), "wRel": math.sqrt(se / ss), "zeroShare": zeros / n, **extra}
        tape.frame(phase, st, step=step, metrics=m)

    frame("fp32", 0, W, {**fates, "k": -1, "bits": 32, "bpw": 32, "bytes": 4 * params, "bytes0": 4 * params})
    for k, (cid, fmt, sc, block) in enumerate(CASTS):
        Wd = dict(W)
        groups = 0
        for name in names:
            Wd[name] = scaled(W[name], fmt, sc, block or 32)
            r, c = W[name].shape
            groups += 1 if sc == "tensor" else r * math.ceil(c / block) if sc == "block" else 0
        bpw = BITS[fmt] + (0 if sc == "none" else 32 * groups / n)
        frame("cast", k, Wd, {"k": k, "bits": BITS[fmt], "bpw": bpw, "bytes": 4 * (params - n) + n * bpw / 8, "bytes0": 4 * params})


# ---------------------------------------------------------------- sum

def sum_proc(opt, tape):
    o = {"n": 100, "u": 0.001, "w0": 1.0, **(opt or {})}
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    u = {k: torch.tensor([o["u"]], dtype=torch.float32).to(d) for k, d in dt.items()}
    w = {k: torch.tensor([o["w0"]], dtype=torch.float32).to(d) for k, d in dt.items()}
    for step in range(o["n"] + 1):
        if step > 0:
            w = {k: (w[k].float() + u[k].float()).to(dt[k]) for k in dt}
        v = {k: float(w[k].float().item()) for k in dt}
        mixed = float(w["fp32"].to(torch.bfloat16).float().item())
        tape.frame("update", {"w": np.array([v["fp32"], v["bf16"], v["fp16"], mixed])}, step=step,
                   metrics={"step": step, "fp32": v["fp32"], "bf16": v["bf16"], "fp16": v["fp16"], "mixed": mixed, "exact": o["w0"] + step * o["u"],
                            "d32": v["fp32"] - o["w0"], "dbf": v["bf16"] - o["w0"], "d16": v["fp16"] - o["w0"], "dmix": mixed - o["w0"]})


SCHEMAS = {
    "formats": {"codes": {"kind": "tensor", "label": ("The code of x in each format", "x 在每种格式里的编码"), "shape": [len(FMT_IDS)], "role": "output", "tol": "exact"}},
    "sweep": {
        **{k: {"kind": "tensor", "label": (k, k), "shape": [TOPK], "role": "output", "tol": "exact" if k == "top" else 1e-4} for k in ("top", "p0", "p")},
        "hist": {"kind": "tensor", "label": ("hist", "hist"), "shape": [HIST_HI - HIST_LO], "role": "data", "tol": "exact"},
    },
    "sum": {"w": {"kind": "tensor", "label": ("The weight", "权重"), "shape": [4], "role": "output", "tol": "exact"}},
}
PHASES = {
    "formats": {"store": {"label": ("Store x", "存下 x"), "kind": "forward"}},
    "sweep": {p: {"label": (p, p), "kind": "forward"} for p in ("fp32", "cast")},
    "sum": {"update": {"label": ("Add one update", "加一次更新"), "kind": "update"}},
}


def run(process: str, raw, opt: dict, seed: int) -> "ak.TapeWriter":
    if process not in SCHEMAS:
        raise SystemExit(f"reference.py: no reference for process '{process}'")
    tape = ak.TapeWriter(process, SCHEMAS[process], seed=seed, opt=opt, phases=PHASES[process], provenance="live")
    {"formats": lambda: formats(raw, tape), "sweep": lambda: sweep_proc(raw, tape), "sum": lambda: sum_proc(opt, tape)}[process]()
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="formats")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null", help="JSON input of the TS run, or @<file> (read it with ak.json_arg)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = ak.json_arg(a.input)
    tape = run(a.process, raw, json.loads(a.opt), a.seed)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "number-formats", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
