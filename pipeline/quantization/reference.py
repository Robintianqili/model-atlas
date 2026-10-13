"""quantization: Python reference for the processes 'formats', 'ledger' and 'sweep' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/quantization/process.ts (and quant.ts):
  formats  the fp32 bits of x with struct (IEEE 754 single precision)
  ledger   the rounding ledger in plain Python floats, round() (ties to even), the same scale, zero point, clipping and
           blocks; every float must match the TS tape to 1e-12 relative
  sweep    TinyGPT in PyTorch fp32 (qlib.py) on the weights the TS run received, quantized by qlib.quantize (the same
           float32 arithmetic as quant.ts quantizeMatrix), scored on the same four windows and probe (evalset.ts, read
           from the TS input's ids); metrics within the bars of the ProcessDef's metricTol (float32 forward passes)
`npm run atlas -- parity quantization` records the TS tapes, runs this file and compares them frame by frame.

Arguments (the CLI contract for every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json>
--input <json> --out <stem>. No process draws random numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:  # run without PYTHONPATH: use the repo's pipeline/ folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np

HERE = Path(__file__).resolve().parent
N = 8
START = {"x": 0.1, "vals": [0.31, -0.12, 0.05, -0.47, 0.22, 0.09, -0.26, 0.4]}
X_MAX, V_MAX = 3.4e38, 100.0


def fin(v, d):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else d


def raw_of(raw) -> dict:
    """quant.ts rawOf: x finite within ±3.4e38, exactly eight finite values within ±100."""
    r = raw if isinstance(raw, dict) else {}
    vals = r.get("vals") if isinstance(r.get("vals"), list) else START["vals"]
    x = min(X_MAX, max(-X_MAX, fin(r.get("x"), START["x"])))
    out = []
    for i in range(N):
        v = vals[i] if i < len(vals) else None
        out.append(min(V_MAX, max(-V_MAX, fin(v, START["vals"][i]))))
    return {"x": x, "vals": out}


def vals_of(inp) -> list[float]:
    return raw_of({"vals": inp})["vals"] if isinstance(inp, list) else raw_of(inp)["vals"]


# ---------------------------------------------------------------- formats

def formats(inp, tape):
    x = raw_of({"x": inp})["x"] if isinstance(inp, (int, float)) and not isinstance(inp, bool) else raw_of(inp)["x"]
    b = struct.unpack(">I", struct.pack(">f", x))[0]
    x32 = struct.unpack(">f", struct.pack(">f", x))[0]
    bits = np.array([(b >> (31 - i)) & 1 for i in range(32)], dtype=np.float64)
    tape.frame("formats", {"bits": bits}, step=0, metrics={"x": x, "x32": x32})


# ---------------------------------------------------------------- ledger

def block_of(x, o):
    levels = 2 ** o["bits"]
    if o["scheme"] == "affine":
        lo, hi = min(0.0, *x), max(0.0, *x)
        s = (hi - lo) / (levels - 1) if hi > lo else 1.0
        z = round(-lo / s) + 0
        return {"s": s, "z": z, "lo": lo, "hi": hi, "qlo": 0, "qhi": levels - 1}
    qmax = 2 ** (o["bits"] - 1) - 1
    mags = sorted((abs(v) for v in x), reverse=True)
    amax = mags[1] if o["clip"] == "second" and len(mags) > 1 else mags[0]
    if not amax > 0:  # a range of size 0 has one code, 0 (quant.ts)
        return {"s": 1.0, "z": 0, "lo": 0.0, "hi": 0.0, "qlo": 0, "qhi": 0}
    return {"s": amax / qmax, "z": 0, "lo": -amax, "hi": amax, "qlo": -qmax, "qhi": qmax}


def ledger(vals, o):
    size = max(1, min(len(vals), int(math.floor(o["block"] + 0.5))))
    blocks, bof = [], []
    for f in range(0, len(vals), size):
        t = min(len(vals), f + size)
        blocks.append(block_of(vals[f:t], o))
        bof += [len(blocks) - 1] * (t - f)
    r, q, xh, err, clipped, zeros = [], [], [], [], [], 0
    for i, v in enumerate(vals):
        b = blocks[bof[i]]
        ri = v / b["s"] + b["z"]
        qi = min(b["qhi"], max(b["qlo"], round(ri)))
        back = (qi - b["z"]) * b["s"]
        r.append(ri)
        q.append(qi)
        xh.append(back)
        err.append(back - v)
        if round(ri) != qi:
            clipped.append(i)
        if v != 0 and qi == b["z"]:
            zeros += 1
    ab = [abs(e) for e in err]
    extra = 2 if o["scheme"] == "affine" else 1
    s = 0.0
    for a in ab:
        s += a
    return {"x": vals, "r": r, "q": q, "xh": xh, "err": err, "blocks": blocks, "bof": bof, "maxErr": max(ab), "meanErr": s / len(ab),
            "clipped": clipped, "zeros": zeros, "bpv": o["bits"] + (32 * extra * len(blocks)) / len(vals)}


def biggest(vals):
    k = 0
    for i in range(1, len(vals)):
        if abs(vals[i]) > abs(vals[k]):
            k = i
    return k


def median_abs(vals):
    m = sorted(abs(v) for v in vals)
    return (m[(len(m) - 1) >> 1] + m[len(m) >> 1]) / 2


def mean(a):
    s = 0.0
    for v in a:
        s += v
    return s / len(a)


def ledger_proc(inp, opt, tape):
    o = {"bits": 8, "scheme": "absmax", "clip": "none", "block": 8, "alt": None, **opt}
    vals = vals_of(inp)
    L = ledger(vals, o)
    st = {k: np.array(L[n], dtype=np.float64) for k, n in (("x", "x"), ("q", "q"), ("xh", "xh"), ("err", "err"))}
    fo = {"bits": o["bits"], "levels": 2 ** o["bits"], "block": o["block"], "affine": 1 if o["scheme"] == "affine" else 0, "clip": 1 if o["clip"] == "second" else 0}
    b0 = L["blocks"][0]
    b1 = L["blocks"][1] if len(L["blocks"]) > 1 else b0
    k = biggest(vals)
    tape.frame("scale", st, step=0, metrics={**fo, "s": b0["s"], "z": b0["z"], "amax": max(abs(b0["lo"]), abs(b0["hi"])), "lo": b0["lo"], "hi": b0["hi"],
                                              "qmax": b0["qhi"], "qlo": b0["qlo"], "nBlocks": len(L["blocks"]), "s1": b1["s"]})
    for i in range(len(vals)):
        b = L["blocks"][L["bof"][i]]
        tape.frame("round", st, step=i, metrics={"i": i, "x": vals[i], "s": b["s"], "z": b["z"], "r": L["r"][i], "q": L["q"][i], "xh": L["xh"][i], "err": L["err"][i]})
    rest = [abs(e) for i, e in enumerate(L["err"]) if i != k]
    m = {**fo, "maxErr": L["maxErr"], "meanErr": L["meanErr"], "zeros": L["zeros"], "nClipped": len(L["clipped"]), "bitsPerValue": L["bpv"], "big": k,
         "bigX": vals[k], "bigErr": abs(L["err"][k]), "restMean": mean(rest), "restMax": max(rest), "med": median_abs(vals), "s": b0["s"], "s1": b1["s"],
         "nBlocks": len(L["blocks"])}
    if o.get("alt"):
        A = ledger(vals, {**o, **o["alt"]})
        ar = [abs(e) for i, e in enumerate(A["err"]) if i != k]
        m.update({"altMaxErr": A["maxErr"], "altMeanErr": A["meanErr"], "altZeros": A["zeros"], "altBits": A["bpv"], "altS": A["blocks"][0]["s"],
                  "altBigErr": abs(A["err"][k]), "altRestMean": mean(ar), "altRestMax": max(ar)})
    tape.frame("done", st, step=0, metrics=m)


# ---------------------------------------------------------------- sweep

LAYER = [(8, "tensor"), (4, "tensor"), (4, "channel"), (4, "block")]
ALL = [(8, "tensor"), (8, "channel"), (8, "block"), (4, "tensor"), (4, "channel"), (4, "block")]
GRAN = {"tensor": 0, "channel": 1, "block": 2}
MATRIX = re.compile(r"^L\d+\.(attn\.(q|k|v|proj)|mlp\.(fc|proj))\.weight$")


def evalset():
    """EVAL_IDS and PROBE_IDS of the generated evalset.ts (JSON arrays on their own lines)."""
    src = (Path(ak.site_path("src", "lib", "explainers", "quantization", "evalset.ts"))).read_text(encoding="utf-8")
    get = lambda name: json.loads(re.search(rf"export const {name}[^=]*= (.*);\n", src).group(1))
    return get("EVAL_IDS"), get("PROBE_IDS")


def sweep_proc(inp, opt, tape):
    import torch
    sys.path.insert(0, str(HERE))
    from qlib import forward, quantize

    if not isinstance(inp, dict) or not inp.get("weights"):
        return
    W = {k: torch.tensor(np.array(t["data"], dtype=np.float32)).view(*t["shape"]) for k, t in inp["weights"].items()}
    layers = (inp.get("spec") or {}).get("layers", [])
    n_head = next(l["params"]["nHead"] for l in layers if (l.get("params") or {}).get("nHead"))
    eps = next(l["params"]["eps"] for l in layers if l.get("op") == "layernorm")
    spec = {"dModel": W["tok_emb.weight"].shape[1], "nHead": n_head, "nLayer": len([k for k in W if re.match(r"^L\d+\.ln1\.weight$", k)]), "lnEps": eps}
    focus = opt.get("focus", "L0.mlp.fc.weight")
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
    order = sorted(range(plp0.numel()), key=lambda i: (-plp0[i].item(), i))[:5]
    top0 = order[0]
    ppl0 = math.exp(-lp0.gather(-1, y[..., None]).sum().item() / tokens)
    fw = W[focus]

    def frame(phase, step, Wd, bits, gran, nq, groups):
        l, pl = logits(Wd)
        lp = torch.log_softmax(l, -1)
        plp = torch.log_softmax(pl, -1)
        kl = (lp0.exp() * (lp0 - lp)).sum(-1).sum().item() / tokens
        agree = (lp.argmax(-1) == lp0.argmax(-1)).sum().item() / tokens
        dmax = (l - l0).abs().max().item()
        e = (Wd[focus].double() - fw.double()).flatten()
        st = {"top": np.array(order, dtype=np.float64), "p0": np.array([math.exp(plp0[i].item()) for i in order]), "p": np.array([math.exp(plp[i].item()) for i in order]),
              "l0": np.array([p0l[i].item() for i in order]), "l": np.array([pl[i].item() for i in order])}
        m = {"bits": bits, "gran": gran, "ppl0": ppl0, "ppl": math.exp(-lp.gather(-1, y[..., None]).sum().item() / tokens), "kl": kl, "agree": agree, "dmax": dmax,
             "tokens": tokens, "probeTop": int(plp.argmax().item()), "probeTop0": top0, "probeP": math.exp(plp[top0].item()), "probePq": math.exp(plp.max().item()), "probeP0": math.exp(plp0[top0].item()),
             "bytes": 4 * (params - nq) + nq * bits / 8 + 4 * groups, "bytes0": 4 * params, "params": params, "nq": nq, "groups": groups,
             "wRms": math.sqrt((e * e).sum().item() / e.numel()), "wMax": e.abs().max().item(), "fRows": fw.shape[0], "fCols": fw.shape[1]}
        tape.frame(phase, st, step=step, metrics=m)

    frame("fp32", 0, W, 32, -1, 0, 0)

    def groups_of(t, gran):
        r, c = t.shape
        return 1 if gran == "tensor" else r if gran == "channel" else r * math.ceil(c / 32)

    for phase, cfgs, targets in (("layer", LAYER, [focus]), ("all", ALL, names)):
        for k, (bits, gran) in enumerate(cfgs):
            Wd = dict(W)
            for n in targets:
                Wd[n] = quantize(W[n], bits, gran)
            frame(phase, k, Wd, bits, GRAN[gran], sum(W[n].numel() for n in targets), sum(groups_of(W[n], gran) for n in targets))


SCHEMAS = {
    "formats": {"bits": {"kind": "tensor", "label": ("The fp32 bits of x (sign first)", "x 的 fp32 各位（符号位在前）"), "shape": [32], "role": "data", "tol": "exact"}},
    "ledger": {
        "x": {"kind": "tensor", "label": ("The numbers x", "原始数值 x"), "shape": [N], "role": "data", "tol": 1e-12},
        "q": {"kind": "tensor", "label": ("Their codes q", "它们的编码 q"), "shape": [N], "role": "output", "tol": "exact"},
        "xh": {"kind": "tensor", "label": ("What the codes stand for", "编码代表的数值"), "shape": [N], "role": "output", "tol": 1e-12},
        "err": {"kind": "tensor", "label": ("Rounding errors", "舍入误差"), "shape": [N], "role": "output", "tol": 1e-12},
    },
    "sweep": {k: {"kind": "tensor", "label": (k, k), "shape": [5], "role": "output", "tol": "exact" if k == "top" else 1e-4} for k in ("top", "p0", "p", "l0", "l")},
}
PHASES = {
    "formats": {"formats": {"label": ("Store x", "存下 x"), "kind": "forward"}},
    "ledger": {"scale": {"label": ("Pick the scale", "确定缩放比例"), "kind": "input"}, "round": {"label": ("Round one value", "舍入一个数"), "kind": "forward"},
               "done": {"label": ("The errors", "误差"), "kind": "output"}},
    "sweep": {p: {"label": (p, p), "kind": "forward"} for p in ("fp32", "layer", "all")},
}


def run(process: str, raw, opt: dict, seed: int) -> "ak.TapeWriter":
    if process not in SCHEMAS:
        raise SystemExit(f"reference.py: no reference for process '{process}'")
    tape = ak.TapeWriter(process, SCHEMAS[process], seed=seed, opt=opt, phases=PHASES[process], provenance="live")
    {"formats": lambda: formats(raw, tape), "ledger": lambda: ledger_proc(raw, opt, tape), "sweep": lambda: sweep_proc(raw, opt, tape)}[process]()
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="ledger")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null", help="JSON input of the TS run, or @<file> (read it with ak.json_arg)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = ak.json_arg(a.input)
    tape = run(a.process, raw, json.loads(a.opt), a.seed)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "quantization", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
