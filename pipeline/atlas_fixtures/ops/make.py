"""Torch goldens for the DevKit CORE_OPS (design §8.1 W1-A).

For every op: 8 seeded cases with float32 inputs and weights; the reference output is computed by PyTorch in
float64 on the CPU from those same float32 values (TF32 off). The TS op tests
(site/src/lib/atlas/ops/goldens.test.ts) run each op's forward() on the same inputs and require
|a - b| <= 1e-6 + 1e-6 * |b| for every output value (and every sub-step of lstm).

Output: $ATLAS_SITE/tests/atlas/ops/<op>.json  (format 'atlas-golden/1', LF line endings, UTF-8 without BOM)

  python pipeline/atlas_fixtures/ops/make.py [--site PATH] [--op NAME]

Paths (design §3.1): ATLAS_SITE = --site, else $ATLAS_SITE, else the current folder when it is a site folder, else
ATLAS_ROOT/site. ATLAS_ROOT = $ATLAS_ROOT, else the nearest ancestor of ATLAS_SITE (or of this file) holding
CONTRACTS.md and pipeline/. This works from the main tree and from a lane copy (site/.work/<name>/site).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_dtype(torch.float32)

CASES_PER_OP = 8


# ------------------------------------------------------------------ paths
def find_root(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        if (p / "CONTRACTS.md").is_file() and (p / "pipeline").is_dir():
            return p
    return None


def resolve_paths(site_arg: str | None) -> tuple[Path, Path]:
    site: Path | None = None
    if site_arg:
        site = Path(site_arg).resolve()
    elif os.environ.get("ATLAS_SITE"):
        site = Path(os.environ["ATLAS_SITE"]).resolve()
    elif (Path.cwd() / "svelte.config.js").is_file():
        site = Path.cwd().resolve()
    root = Path(os.environ["ATLAS_ROOT"]).resolve() if os.environ.get("ATLAS_ROOT") else None
    if root is None:
        root = find_root(site if site else Path(__file__).resolve().parent)
    if root is None:
        sys.exit("make.py: cannot find ATLAS_ROOT (a folder with CONTRACTS.md and pipeline/); set ATLAS_ROOT")
    if site is None:
        site = root / "site"
    return root, site


# ------------------------------------------------------------------ encoding
def enc(t: torch.Tensor, dtype: str) -> dict:
    arr = t.detach().cpu().numpy()
    if dtype == "float32":
        b = arr.astype("<f4").tobytes()
    elif dtype == "float64":
        b = arr.astype("<f8").tobytes()
    elif dtype == "int32":
        b = arr.astype("<i4").tobytes()
    else:
        raise ValueError(dtype)
    return {"shape": list(arr.shape), "dtype": dtype, "b64": base64.b64encode(b).decode("ascii")}


class Gen:
    """Seeded float32 tensors (torch.Generator on the CPU)."""

    def __init__(self, seed: int):
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def randn(self, *shape, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, generator=self.g, dtype=torch.float32) * scale).to(torch.float32)

    def uniform(self, *shape, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
        return (torch.rand(*shape, generator=self.g, dtype=torch.float32) * (hi - lo) + lo).to(torch.float32)

    def ints(self, n: int, hi: int) -> torch.Tensor:
        return torch.randint(0, hi, (n,), generator=self.g, dtype=torch.int64).to(torch.int32)


def d(x: torch.Tensor) -> torch.Tensor:
    """float64 copy of float32 values (the reference sees exactly the inputs the TS op sees)."""
    return x.to(torch.float64)


def case(cid: str, op: str, params: dict, inputs: list, out: torch.Tensor, weights: dict | None = None, subs: dict | None = None) -> dict:
    c = {"id": cid, "op": op, "params": params, "inputs": inputs, "weights": {k: enc(v, "float32") for k, v in (weights or {}).items()}, "out": enc(out, "float64")}
    if subs:
        c["subs"] = {k: enc(v, "float64") for k, v in subs.items()}
    return c


def fin(x: torch.Tensor) -> dict:
    return enc(x, "float32")


def iin(x: torch.Tensor) -> dict:
    return enc(x, "int32")


# ------------------------------------------------------------------ ops
def g_input(G: Gen):
    shapes = [[1, 4, 5], [3, 2, 2], [7], [2, 3], [1, 28, 28], [1], [4, 1, 3], [2, 2, 2]]
    out = []
    for i, s in enumerate(shapes):
        x = G.uniform(*s, lo=0, hi=1) if i % 2 == 0 else G.randn(*s)
        out.append(case(f"input-{i + 1}", "input", {}, [fin(x)], d(x)))
    return out


def g_conv2d(G: Gen):
    cfgs = [
        # C, H, W, O, k, stride, pad, bias, accumulate
        (1, 6, 6, 2, 3, 1, 0, True, "bias-last"),
        (3, 7, 7, 4, 3, 1, 1, True, "bias-last"),
        (2, 9, 9, 3, 3, 2, 0, True, "bias-last"),
        (2, 8, 8, 2, 2, 2, 0, False, "bias-last"),
        (4, 5, 5, 3, 5, 1, 2, True, "bias-first"),
        (1, 28, 28, 8, 3, 1, 0, True, "bias-last"),
        (3, 6, 7, 2, 3, 1, 1, True, "bias-last"),
        (2, 7, 7, 2, 7, 7, 0, True, "bias-first"),
    ]
    out = []
    for i, (C, H, W, O, k, s, p, bias, acc) in enumerate(cfgs):
        x = G.uniform(C, H, W, lo=0, hi=1) if i % 2 == 0 else G.randn(C, H, W)
        w = G.randn(O, C, k, k, scale=1 / math.sqrt(C * k * k))
        b = G.randn(O, scale=0.1) if bias else None
        y = F.conv2d(d(x)[None], d(w), d(b) if b is not None else None, stride=s, padding=p)[0]
        params = {"kernel": k, "stride": s, "padding": p, "inChannels": C, "outChannels": O, "accumulate": acc}
        ws = {"weight": w}
        if b is not None:
            ws["bias"] = b
        out.append(case(f"conv2d-{i + 1}", "conv2d", params, [fin(x)], y, ws))
    return out


def g_linear(G: Gen):
    cfgs = [
        # op, in_shape, out, bias, params
        ("linear", [5], 3, True, {}),
        ("linear", [4, 6], 5, True, {"accumulate": "bias-first"}),
        ("linear", [8], 8, False, {}),
        ("linear_tied", [3, 10], 2, False, {"accumulate": "bias-first"}),
        ("dense", [256], 10, True, {"inFeatures": 256, "outFeatures": 10}),
        ("linear", [1, 7], 4, True, {}),
        ("linear", [6, 4], 12, True, {"nHead": 3, "dHead": 4, "accumulate": "bias-first"}),
        ("linear", [2, 3], 1, True, {}),
    ]
    out = []
    for i, (op, shp, O, bias, params) in enumerate(cfgs):
        x = G.randn(*shp)
        I = shp[-1]
        w = G.randn(O, I, scale=1 / math.sqrt(I))
        b = G.randn(O, scale=0.1) if bias else None
        y = F.linear(d(x), d(w), d(b) if b is not None else None)
        ws = {"weight": w}
        if b is not None:
            ws["bias"] = b
        out.append(case(f"{op}-{i + 1}", op, params, [fin(x)], y, ws))
    return out


def special_values(G: Gen, shape, scale):
    x = G.randn(*shape, scale=scale)
    flat = x.view(-1)
    n = flat.numel()
    if n >= 4:
        flat[0] = 0.0
        flat[1] = -0.0
        flat[2] = 30.0 if scale >= 3 else 2.0
        flat[3] = -30.0 if scale >= 3 else -2.0
    return x


def g_pointwise(name: str, fn, op_names=None, params_list=None):
    def gen(G: Gen):
        shapes = [[7], [2, 5], [3, 4, 4], [1, 9], [16], [2, 3, 2], [64], [4, 4]]
        scales = [1, 3, 0.5, 10, 1, 2, 4, 0.1]
        out = []
        for i, (s, sc) in enumerate(zip(shapes, scales)):
            x = special_values(G, s, sc)
            op = op_names[i] if op_names else name
            params = params_list[i] if params_list else {}
            out.append(case(f"{op}-{i + 1}", op, params, [fin(x)], fn(d(x), op, params)))
        return out

    return gen


def _gelu(x, op, params):
    approx = "tanh" if op == "gelu_tanh" or params.get("approx") == "tanh" else "none"
    return F.gelu(x, approximate=approx)


def g_maxpool2d(G: Gen):
    cfgs = [((2, 6, 6), 2, None), ((1, 9, 9), 3, None), ((3, 5, 5), 2, 1), ((2, 7, 7), 3, 2), ((2, 4, 4), 2, None), ((1, 6, 6), 2, None), ((8, 24, 24), 2, 2), ((1, 4, 6), 2, None)]
    out = []
    for i, (shape, k, s) in enumerate(cfgs):
        if i == 4:
            x = torch.randint(0, 3, shape, generator=G.g).to(torch.float32)  # many ties
        elif i == 5:
            x = -G.uniform(*shape, lo=0.1, hi=2.0)  # all negative
        else:
            x = G.randn(*shape)
        y = F.max_pool2d(d(x)[None], k, stride=s if s is not None else k)[0]
        params = {"kernel": k} if s is None else {"kernel": k, "stride": s}
        out.append(case(f"maxpool2d-{i + 1}", "maxpool2d", params, [fin(x)], y))
    return out


def g_flatten(G: Gen):
    shapes = [[2, 3, 4], [16, 4, 4], [5], [3, 2], [1, 1, 1], [8, 12, 12], [2, 2, 2, 2], [1, 7]]
    return [case(f"flatten-{i + 1}", "flatten", {}, [fin(x)], torch.flatten(d(x))) for i, x in enumerate(G.randn(*s) for s in shapes)]


def g_concat(G: Gen):
    cfgs = [([[2, 3], [1, 3]], 0), ([[4, 2], [4, 1], [4, 3]], -1), ([[2, 3, 4], [2, 1, 4]], 1), ([[1, 5], [2, 5]], 0), ([[3, 1], [3, 1]], 1), ([[2, 2, 2], [3, 2, 2]], 0), ([[6], [2]], 0), ([[2, 3, 1], [2, 3, 2]], -1)]
    out = []
    for i, (shapes, axis) in enumerate(cfgs):
        xs = [G.randn(*s) for s in shapes]
        y = torch.cat([d(x) for x in xs], dim=axis)
        out.append(case(f"concat-{i + 1}", "concat", {"axis": axis}, [fin(x) for x in xs], y))
    return out


def g_add(G: Gen):
    cfgs = [([3, 4], [3, 4]), ([9, 76], [9, 76]), ([2, 3, 4], [2, 3, 4]), ([5, 6], [6]), ([3, 1], [3, 4]), ([1], [1]), ([4, 2], [1, 2]), ([2, 2, 3], [3])]
    out = []
    for i, (a, b) in enumerate(cfgs):
        x, y = G.randn(*a), G.randn(*b, scale=0.5)
        out.append(case(f"add-{i + 1}", "add", {}, [fin(x), fin(y)], d(x) + d(y)))
    return out


def g_embedding(G: Gen):
    cfgs = [(10, 4, 6), (50, 8, 1), (1024, 16, 9), (5, 3, 5), (20, 2, 20), (7, 7, 3), (100, 6, 12), (2, 5, 4)]
    out = []
    for i, (V, D, T) in enumerate(cfgs):
        w = G.randn(V, D)
        ids = G.ints(T, V)
        if i == 3:
            ids = torch.tensor([1, 1, 4, 0, 1], dtype=torch.int32)  # repeated ids
        y = F.embedding(ids.long(), d(w))
        out.append(case(f"embedding-{i + 1}", "embedding", {}, [iin(ids)], y, {"weight": w}))
    return out


def g_position_embedding(G: Gen):
    cfgs = [(16, 8, 5), (16, 8, 1), (16, 8, 16), (64, 4, 9), (8, 3, 8), (10, 2, 3), (32, 6, 31), (4, 4, 2)]
    out = []
    for i, (P, D, T) in enumerate(cfgs):
        w = G.randn(P, D)
        ids = G.ints(T, 100)
        y = d(w)[:T]
        out.append(case(f"position_embedding-{i + 1}", "position_embedding", {}, [iin(ids)], y, {"weight": w}))
    return out


def g_layernorm(G: Gen):
    cfgs = [((4, 8), 1e-5, True, 1.0, 0.0), ((9, 76), 1e-5, True, 1.0, 0.0), ((1, 76), 1e-5, True, 2.0, 0.5), ((3, 5), 1e-3, False, 1.0, 0.0),
            ((2, 16), 1e-5, True, 1.0, 100.0), ((6,), 1e-5, True, 1.0, 0.0), ((2, 3, 4), 1e-5, True, 3.0, -1.0), ((5, 2), 1e-6, True, 0.01, 0.0)]
    out = []
    for i, (shape, eps, affine, scale, shift) in enumerate(cfgs):
        x = G.randn(*shape, scale=scale) + shift
        x = x.to(torch.float32)
        D = shape[-1]
        if affine:
            w = (1 + G.randn(D, scale=0.2)).to(torch.float32)
            b = G.randn(D, scale=0.2)
            y = F.layer_norm(d(x), (D,), d(w), d(b), eps)
            ws = {"weight": w, "bias": b}
        else:
            y = F.layer_norm(d(x), (D,), None, None, eps)
            ws = {}
        out.append(case(f"layernorm-{i + 1}", "layernorm", {"eps": eps}, [fin(x)], y, ws))
    return out


def heads(x: torch.Tensor, H: int, dH: int) -> torch.Tensor:
    """[T, H*dH] -> [H, T, dH]"""
    return x.view(x.shape[0], H, dH).transpose(0, 1)


def g_attention_scores(G: Gen):
    cfgs = [(2, 3, 4, 4, False, None), (4, 19, 9, 9, False, 1 / math.sqrt(19)), (2, 4, 5, 5, True, None), (1, 6, 3, 5, False, None),
            (2, 2, 1, 1, False, None), (3, 4, 6, 6, True, 0.5), (4, 19, 22, 22, False, None), (2, 8, 4, 7, False, None)]
    out = []
    for i, (H, dH, Tq, Tk, causal, scale) in enumerate(cfgs):
        q, k = G.randn(Tq, H * dH), G.randn(Tk, H * dH)
        sc = scale if scale is not None else 1 / math.sqrt(dH)
        s = heads(d(q), H, dH) @ heads(d(k), H, dH).transpose(-2, -1) * sc
        if causal:
            s = s.masked_fill(torch.ones(Tq, Tk, dtype=torch.bool).triu(1), float("-inf"))
        params = {"nHead": H, "dHead": dH, "causal": causal}
        if scale is not None:
            params["scale"] = scale
        out.append(case(f"attention_scores-{i + 1}", "attention_scores", params, [fin(q), fin(k)], s))
    return out


def g_causal_mask(G: Gen):
    shapes = [[2, 4, 4], [1, 1, 1], [4, 9, 9], [3, 5, 5], [2, 3, 3], [1, 6, 6], [4, 2, 2], [2, 7, 7]]
    out = []
    for i, s in enumerate(shapes):
        x = G.randn(*s)
        y = d(x).masked_fill(torch.ones(s[-2], s[-1], dtype=torch.bool).triu(1), float("-inf"))
        params = {} if i % 2 else {"fill": "-Infinity above the diagonal"}
        out.append(case(f"causal_mask-{i + 1}", "causal_mask", params, [fin(x)], y))
    return out


def g_softmax(G: Gen):
    out = []
    specs = [([10], -1, 1.0, False), ([3, 7], -1, 2.0, False), ([2, 4, 4], -1, 1.0, True), ([5, 3], 0, 1.0, False),
             ([2, 6], -1, 100.0, False), ([1, 1024], -1, 3.0, False), ([4], -1, 0.0, False), ([3, 4, 5], 1, 1.5, False)]
    for i, (shape, axis, scale, masked) in enumerate(specs):
        x = G.randn(*shape, scale=scale) if scale else torch.full(shape, 0.25, dtype=torch.float32)
        if masked:
            x = x.masked_fill(torch.ones(shape[-2], shape[-1], dtype=torch.bool).triu(1), float("-inf"))
        y = torch.softmax(d(x), dim=axis)
        params = {} if axis == -1 and i % 2 == 0 else {"axis": axis}
        out.append(case(f"softmax-{i + 1}", "softmax", params, [fin(x)], y))
    return out


def g_attention_mix(G: Gen):
    cfgs = [(2, 3, 4, 4, True), (4, 19, 9, 9, True), (1, 5, 3, 6, False), (2, 2, 1, 1, True), (3, 4, 5, 5, False), (2, 6, 7, 7, True), (4, 4, 2, 8, False), (1, 1, 3, 3, True)]
    out = []
    for i, (H, dH, Tq, Tk, causal) in enumerate(cfgs):
        s = G.randn(H, Tq, Tk, scale=2.0)
        if causal:
            s = s.masked_fill(torch.ones(Tq, Tk, dtype=torch.bool).triu(1), float("-inf"))
        w = torch.softmax(s, dim=-1).to(torch.float32)  # float32 weights, exactly as the trace stores them
        v = G.randn(Tk, H * dH)
        o = (d(w) @ heads(d(v), H, dH)).transpose(0, 1).reshape(Tq, H * dH)
        out.append(case(f"attention_mix-{i + 1}", "attention_mix", {"nHead": H, "dHead": dH}, [fin(w), fin(v)], o))
    return out


def lstm_reference(x, Wih, Whh, bih, bhh, H):
    """float64 loop with the PyTorch gate order i, f, g, o; returns h [T, H] and the sub-steps."""
    T = x.shape[0]
    h = torch.zeros(H, dtype=torch.float64)
    c = torch.zeros(H, dtype=torch.float64)
    subs = {k: [] for k in "ifgoch"}
    for t in range(T):
        z = Wih @ x[t] + Whh @ h
        if bih is not None:
            z = z + bih
        if bhh is not None:
            z = z + bhh
        i, f, g, o = z[:H].sigmoid(), z[H:2 * H].sigmoid(), z[2 * H:3 * H].tanh(), z[3 * H:].sigmoid()
        c = f * c + i * g
        h = o * c.tanh()
        for k, v in zip("ifgoch", (i, f, g, o, c, h)):
            subs[k].append(v)
    return torch.stack(subs["h"]), {k: torch.stack(v) for k, v in subs.items()}


def g_lstm(G: Gen):
    cfgs = [(3, 4, 5, True), (1, 2, 1, True), (6, 8, 12, True), (4, 3, 6, False), (24, 16, 8, True), (2, 5, 20, True), (5, 1, 4, True), (8, 6, 3, False)]
    out = []
    for i, (In, H, T, bias) in enumerate(cfgs):
        x = G.randn(T, In)
        sc = 1 / math.sqrt(H)
        Wih, Whh = G.uniform(4 * H, In, lo=-sc, hi=sc), G.uniform(4 * H, H, lo=-sc, hi=sc)
        bih = G.uniform(4 * H, lo=-sc, hi=sc) if bias else None
        bhh = G.uniform(4 * H, lo=-sc, hi=sc) if bias else None
        y, subs = lstm_reference(d(x), d(Wih), d(Whh), d(bih) if bias else None, d(bhh) if bias else None, H)
        # check the loop against torch.nn.LSTM in float64 (AK104 in miniature)
        m = torch.nn.LSTM(In, H, bias=bias).double()
        with torch.no_grad():
            m.weight_ih_l0.copy_(d(Wih))
            m.weight_hh_l0.copy_(d(Whh))
            if bias:
                m.bias_ih_l0.copy_(d(bih))
                m.bias_hh_l0.copy_(d(bhh))
            ym, _ = m(d(x))
        err = (ym - y).abs().max().item()
        if err > 1e-12:
            raise SystemExit(f"lstm-{i + 1}: reference loop differs from nn.LSTM by {err}")
        ws = {"weight_ih": Wih, "weight_hh": Whh}
        if bias:
            ws["bias_ih"], ws["bias_hh"] = bih, bhh
        out.append(case(f"lstm-{i + 1}", "lstm", {"hidden": H, "inputSize": In}, [fin(x)], y, ws, subs))
    return out


GENERATORS = {
    "input": g_input,
    "conv2d": g_conv2d,
    "linear": g_linear,
    "relu": g_pointwise("relu", lambda x, op, p: F.relu(x)),
    "tanh": g_pointwise("tanh", lambda x, op, p: torch.tanh(x)),
    "sigmoid": g_pointwise("sigmoid", lambda x, op, p: torch.sigmoid(x)),
    "silu": g_pointwise("silu", lambda x, op, p: F.silu(x)),
    "gelu": g_pointwise(
        "gelu",
        _gelu,
        op_names=["gelu", "gelu", "gelu", "gelu", "gelu", "gelu_tanh", "gelu_tanh", "gelu"],
        params_list=[{}, {"approx": "none"}, {}, {"approx": "tanh"}, {}, {}, {}, {"approx": "tanh"}],
    ),
    "maxpool2d": g_maxpool2d,
    "flatten": g_flatten,
    "concat": g_concat,
    "add": g_add,
    "embedding": g_embedding,
    "position_embedding": g_position_embedding,
    "layernorm": g_layernorm,
    "attention_scores": g_attention_scores,
    "causal_mask": g_causal_mask,
    "softmax": g_softmax,
    "attention_mix": g_attention_mix,
    "lstm": g_lstm,
}


def seed_of(op: str) -> int:
    """A stable seed per op (FNV-1a of the name), so adding an op never changes the others' goldens."""
    h = 0x811C9DC5
    for b in op.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", help="site folder (default: $ATLAS_SITE, the current folder, or ATLAS_ROOT/site)")
    ap.add_argument("--op", action="append", help="only these ops (repeatable)")
    a = ap.parse_args()
    root, site = resolve_paths(a.site)
    out_dir = site / "tests" / "atlas" / "ops"
    out_dir.mkdir(parents=True, exist_ok=True)
    ops = a.op or list(GENERATORS)
    total = 0
    for op in ops:
        cases = GENERATORS[op](Gen(seed_of(op)))
        if len(cases) != CASES_PER_OP:
            raise SystemExit(f"{op}: {len(cases)} cases, expected {CASES_PER_OP}")
        doc = {
            "format": "atlas-golden/1",
            "op": op,
            "generator": "pipeline/atlas_fixtures/ops/make.py",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "reference": "PyTorch float64 on the float32 inputs and weights; TF32 off",
            "tolerance": {"atol": 1e-6, "rtol": 1e-6},
            "cases": cases,
        }
        path = out_dir / f"{op}.json"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, separators=(",", ":"), allow_nan=False)
            f.write("\n")
        total += len(cases)
        print(f"{op:20s} {len(cases)} cases -> {path.relative_to(site)}")
    print(f"{total} cases in {len(ops)} files under {out_dir} (ATLAS_ROOT={root})")


if __name__ == "__main__":
    main()
