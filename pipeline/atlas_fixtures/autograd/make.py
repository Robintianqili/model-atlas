"""PyTorch float64 fixtures for the atlas autograd core (DevKit W1-B1, design §8.1).

Writes into $ATLAS_SITE/tests/atlas/autograd/:
  ops.json          every op: inputs, output, gradients of <r, y>, and double gradients d/dx sum_i <grad_i, v_i>
  train_torch.json  MLP 2-8-8-2 runs (SGD, Adam, momentum, AdamW; tanh and ReLU) from the TS-written train_inputs.json
  pinn_torch.json   tanh MLP u, u', u'' (create_graph), physics-loss gradients and 20 Adam steps from pinn_inputs.json

TS is the source of truth for initial weights and synthetic data: run `node tests/atlas/autograd/make-inputs.ts`
in the site folder first. Everything here runs on the CPU in float64 with TF32 off.

Usage (from the repo root, the site folder, or a lane's site folder):
  ATLAS_SITE=<site folder> python pipeline/atlas_fixtures/autograd/make.py
ATLAS_SITE defaults to the nearest folder at or above the working directory that has svelte.config.js,
else <ATLAS_ROOT>/site. ATLAS_ROOT is the nearest ancestor of ATLAS_SITE with CONTRACTS.md and pipeline/.
"""
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _VF

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_dtype(torch.float64)
DT = torch.float64


def resolve_paths():
    site = os.environ.get("ATLAS_SITE")
    if site:
        site = Path(site).resolve()
    else:
        cwd = Path.cwd().resolve()
        site = next((p for p in [cwd, *cwd.parents] if (p / "svelte.config.js").exists()), None)
        if site is None:
            root = next(p for p in Path(__file__).resolve().parents if (p / "CONTRACTS.md").exists() and (p / "pipeline").is_dir())
            site = root / "site"
    root = os.environ.get("ATLAS_ROOT")
    if root:
        root = Path(root).resolve()
    else:
        root = next((p for p in [site, *site.parents] if (p / "CONTRACTS.md").exists() and (p / "pipeline").is_dir()), None)
    return site, root


SITE, ROOT = resolve_paths()
OUT = SITE / "tests" / "atlas" / "autograd"

gen = torch.Generator().manual_seed(20261006)


def U(shape, lo=-2.0, hi=2.0):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, dtype=DT)


def AZ(shape, lo=0.15, hi=2.0):
    """values away from zero (kinks of relu/abs)"""
    mag = U(shape, lo, hi)
    sign = torch.where(torch.rand(shape, generator=gen, dtype=DT) < 0.5, -1.0, 1.0)
    return mag * sign


def T(t):
    t = t.detach().to(DT).clone(memory_format=torch.contiguous_format)
    return {"shape": list(t.shape), "data": t.reshape(-1).tolist()}


cases = []


def case(name, op, inputs, fn, attrs=None):
    xs = [t.clone().requires_grad_(True) for t in inputs]
    y = fn(*xs)
    r = U(tuple(y.shape), -1.0, 1.0)
    gs = torch.autograd.grad(y, xs, grad_outputs=r, create_graph=True, allow_unused=True)
    gs = [g if g is not None else torch.zeros_like(x) for g, x in zip(gs, xs)]
    vs = [U(tuple(x.shape), -1.0, 1.0) for x in xs]
    h = sum((g * v).sum() for g, v in zip(gs, vs))
    if isinstance(h, torch.Tensor) and h.requires_grad:
        ggs = torch.autograd.grad(h, xs, allow_unused=True)
        ggs = [gg if gg is not None else torch.zeros_like(x) for gg, x in zip(ggs, xs)]
    else:
        ggs = [torch.zeros_like(x) for x in xs]
    cases.append(
        {
            "name": name,
            "op": op,
            "attrs": attrs or {},
            "inputs": [T(x) for x in inputs],
            "y": T(y),
            "r": T(r),
            "grads": [T(g) for g in gs],
            "v": [T(v) for v in vs],
            "ggrads": [T(g) for g in ggs],
        }
    )


def norm_axis(a, r):
    return a + r if a < 0 else a


def make_ops():
    # binary, broadcasting
    case("add [3,4]+[4]", "add", [U((3, 4)), U((4,))], lambda a, b: a + b)
    case("add [2,1,3]+[4,1]", "add", [U((2, 1, 3)), U((4, 1))], lambda a, b: a + b)
    case("sub [3,4]-[3,1]", "sub", [U((3, 4)), U((3, 1))], lambda a, b: a - b)
    case("mul [3,4]*[3,4]", "mul", [U((3, 4)), U((3, 4))], lambda a, b: a * b)
    case("mul [5]*[] scalar", "mul", [U((5,)), U(())], lambda a, b: a * b)
    case("div [3,4]/[4]", "div", [U((3, 4)), AZ((4,), 0.5, 2.0)], lambda a, b: a / b)
    case("div []/[2,3]", "div", [U(()), AZ((2, 3), 0.5, 2.0)], lambda a, b: a / b)
    # unary
    case("neg", "neg", [U((2, 3))], lambda a: -a)
    case("scale -1.7", "scale", [U((2, 3))], lambda a: a * -1.7, {"k": -1.7})
    case("pow 2", "pow", [U((2, 3))], lambda a: torch.pow(a, 2), {"p": 2})
    case("pow 3", "pow", [U((2, 3))], lambda a: torch.pow(a, 3), {"p": 3})
    case("pow 0.5", "pow", [U((2, 3), 0.3, 2.0)], lambda a: torch.pow(a, 0.5), {"p": 0.5})
    case("pow -1.5", "pow", [U((2, 3), 0.3, 2.0)], lambda a: torch.pow(a, -1.5), {"p": -1.5})
    case("pow 1.3", "pow", [U((2, 3), 0.3, 2.0)], lambda a: torch.pow(a, 1.3), {"p": 1.3})
    case("square", "square", [U((4,))], lambda a: torch.square(a))
    case("sqrt", "sqrt", [U((2, 3), 0.3, 3.0)], torch.sqrt)
    case("exp", "exp", [U((2, 3))], torch.exp)
    case("log", "log", [U((2, 3), 0.3, 3.0)], torch.log)
    case("tanh", "tanh", [U((2, 3))], torch.tanh)
    case("sigmoid", "sigmoid", [U((2, 3), -4, 4)], torch.sigmoid)
    case("relu", "relu", [AZ((2, 4))], torch.relu)
    case("silu", "silu", [U((2, 4), -4, 4)], F.silu)
    case("gelu none", "gelu", [U((2, 4), -4, 4)], lambda a: F.gelu(a, approximate="none"), {"approx": "none"})
    case("gelu tanh", "gelu", [U((2, 4), -4, 4)], lambda a: F.gelu(a, approximate="tanh"), {"approx": "tanh"})
    case("gelu none wide", "gelu", [torch.tensor([-7.5, -5.0, -3.0, -2.5, -0.7, 0.0, 0.3, 2.5, 3.1, 6.5])], lambda a: F.gelu(a), {"approx": "none"})
    case("erf", "erf", [torch.tensor([-6.5, -3.2, -2.5, -2.4999, -1.0, -0.2, 0.0, 0.05, 0.9, 2.2, 2.5, 3.7, 5.9])], torch.erf)
    case("sin", "sin", [U((2, 3), -4, 4)], torch.sin)
    case("cos", "cos", [U((2, 3), -4, 4)], torch.cos)
    case("abs", "abs", [AZ((2, 4))], torch.abs)
    # linear algebra
    case("matmul [3,4]@[4,5]", "matmul", [U((3, 4)), U((4, 5))], lambda a, b: a @ b)
    case("matmul [4]@[4,2]", "matmul", [U((4,)), U((4, 2))], lambda a, b: a @ b)
    case("matmul [3,4]@[4]", "matmul", [U((3, 4)), U((4,))], lambda a, b: a @ b)
    case("linear [5,4] W[3,4] b", "linear", [U((5, 4)), U((3, 4)), U((3,))], F.linear)
    case("linear [4] W[3,4] b", "linear", [U((4,)), U((3, 4)), U((3,))], F.linear)
    case("linear [2,3,4] W[5,4]", "linear", [U((2, 3, 4)), U((5, 4))], lambda x, w: F.linear(x, w))
    # reductions and shapes
    case("sum all", "sum", [U((3, 4))], lambda a: a.sum(), {"axis": None, "keepdim": False})
    case("sum axis 1", "sum", [U((3, 4))], lambda a: a.sum(dim=1), {"axis": 1, "keepdim": False})
    case("sum axes [0,2] keepdim", "sum", [U((2, 3, 4))], lambda a: a.sum(dim=(0, 2), keepdim=True), {"axis": [0, 2], "keepdim": True})
    case("mean all", "mean", [U((3, 4))], lambda a: a.mean(), {"axis": None, "keepdim": False})
    case("mean axis -1 keepdim", "mean", [U((3, 4))], lambda a: a.mean(dim=-1, keepdim=True), {"axis": -1, "keepdim": True})
    case("reshape [3,4]->[2,6]", "reshape", [U((3, 4))], lambda a: a.reshape(2, 6), {"shape": [2, 6]})
    case("transpose [3,4]", "transpose", [U((3, 4))], lambda a: a.transpose(0, 1), {"dim0": -2, "dim1": -1})
    case("transpose [2,3,4] 0<->2", "transpose", [U((2, 3, 4))], lambda a: a.transpose(0, 2), {"dim0": 0, "dim1": 2})
    case("broadcastTo [1,4]->[3,4]", "broadcastTo", [U((1, 4))], lambda a: a.expand(3, 4), {"shape": [3, 4]})
    case("broadcastTo [3,1]->[2,3,4]", "broadcastTo", [U((3, 1))], lambda a: a.expand(2, 3, 4), {"shape": [2, 3, 4]})
    case("sumTo [2,3,4]->[3,1]", "sumTo", [U((2, 3, 4))], lambda a: a.sum_to_size(3, 1), {"shape": [3, 1]})
    case("concat axis 0", "concat", [U((2, 3)), U((1, 3))], lambda a, b: torch.cat([a, b], 0), {"axis": 0})
    case("concat axis 1", "concat", [U((2, 3)), U((2, 2)), U((2, 1))], lambda a, b, c: torch.cat([a, b, c], 1), {"axis": 1})
    case("stack axis 1", "stack", [U((2, 3)), U((2, 3))], lambda a, b: torch.stack([a, b], 1), {"axis": 1})
    case("slice axis 1 [1:3]", "slice", [U((3, 4))], lambda a: a[:, 1:3], {"axis": 1, "start": 1, "end": 3})
    case("slice axis 0 [-2:]", "slice", [U((4, 3))], lambda a: a[-2:], {"axis": 0, "start": -2})
    case("gather axis 0", "gather", [U((3, 4))], lambda a: torch.index_select(a, 0, torch.tensor([2, 0, 2, 1])), {"axis": 0, "index": [2, 0, 2, 1]})
    case("gather axis 1", "gather", [U((2, 3, 2))], lambda a: torch.index_select(a, 1, torch.tensor([1, 1, 0])), {"axis": 1, "index": [1, 1, 0]})
    case("scatterAdd axis 0", "scatterAdd", [U((4, 3))], lambda g: torch.zeros(3, 3, dtype=DT).index_add(0, torch.tensor([2, 0, 2, 1]), g), {"axis": 0, "index": [2, 0, 2, 1], "n": 3})
    case("gather then tanh (embedding-like)", "gatherTanh", [U((5, 3))], lambda w: torch.tanh(torch.index_select(w, 0, torch.tensor([4, 1, 4]))), {"axis": 0, "index": [4, 1, 4]})
    # softmax family
    case("logsumexp axis 1", "logsumexp", [U((3, 5), -3, 3)], lambda a: torch.logsumexp(a, 1), {"axis": 1, "keepdim": False})
    case("logsumexp axis 0 keepdim", "logsumexp", [U((3, 5), -3, 3)], lambda a: torch.logsumexp(a, 0, keepdim=True), {"axis": 0, "keepdim": True})
    case("softmax axis 1", "softmax", [U((3, 5), -3, 3)], lambda a: torch.softmax(a, 1), {"axis": 1})
    case("softmax axis 0", "softmax", [U((3, 5), -3, 3)], lambda a: torch.softmax(a, 0), {"axis": 0})
    case("logSoftmax axis -1", "logSoftmax", [U((3, 5), -3, 3)], lambda a: torch.log_softmax(a, -1), {"axis": -1})

    # conv1d
    def conv(stride=1, dilation=1, padding=0, causal=False):
        def f(x, w, b=None):
            k = w.shape[-1]
            if causal:
                x = F.pad(x, (dilation * (k - 1), 0))
                return F.conv1d(x, w, b, stride=stride, padding=0, dilation=dilation)
            return F.conv1d(x, w, b, stride=stride, padding=padding, dilation=dilation)

        return f

    case("conv1d causal d2", "conv1d", [U((2, 3, 9)), U((4, 3, 3)), U((4,))], conv(dilation=2, causal=True), {"dilation": 2, "causal": True})
    case("conv1d pad1 stride2", "conv1d", [U((2, 3, 9)), U((4, 3, 3)), U((4,))], conv(stride=2, padding=1), {"stride": 2, "padding": 1})
    case("conv1d unbatched no bias", "conv1d", [U((3, 7)), U((2, 3, 2))], conv(), {})
    case("conv1d causal d4 k2", "conv1d", [U((1, 2, 12)), U((3, 2, 2)), U((3,))], conv(dilation=4, causal=True), {"dilation": 4, "causal": True})
    # normalization
    case("layerNorm affine", "layerNorm", [U((3, 5)), U((5,)), U((5,))], lambda x, w, b: F.layer_norm(x, (5,), w, b, 1e-5), {"eps": 1e-5})
    case("layerNorm plain", "layerNorm", [U((2, 2, 6))], lambda x: F.layer_norm(x, (6,), None, None, 1e-5), {"eps": 1e-5})
    # losses
    case("mse", "mse", [U((4, 2)), U((4, 2))], F.mse_loss)
    case("crossEntropy", "crossEntropy", [U((5, 3), -3, 3)], lambda z: F.cross_entropy(z, torch.tensor([0, 2, 1, 2, 0])), {"targets": [0, 2, 1, 2, 0]})
    case("crossEntropy 1 row", "crossEntropy", [U((4,), -3, 3)], lambda z: F.cross_entropy(z.unsqueeze(0), torch.tensor([3])), {"targets": [3]})
    case("bceWithLogits", "bceWithLogits", [AZ((6,), 0.1, 4.0), U((6,), 0.0, 1.0)], F.binary_cross_entropy_with_logits)
    n = 4
    ar = torch.arange(n)
    case("infoNCE symmetric", "infoNCE", [U((n, n), -3, 3)], lambda L: (F.cross_entropy(L, ar) + F.cross_entropy(L.t(), ar)) / 2, {"symmetric": True})
    case("infoNCE rows", "infoNCE", [U((n, n), -3, 3)], lambda L: F.cross_entropy(L, ar), {"symmetric": False})
    # modules
    In, H = 3, 4
    lstm_inputs = [U((2, In)), U((2, H)), U((2, H)), U((4 * H, In), -0.5, 0.5), U((4 * H, H), -0.5, 0.5), U((4 * H,), -0.5, 0.5), U((4 * H,), -0.5, 0.5)]

    def lstm(x, h0, c0, wih, whh, bih, bhh):
        h, c = _VF.lstm_cell(x, (h0, c0), wih, whh, bih, bhh)
        return torch.cat([h, c], -1)

    case("LSTMCell (h, c)", "lstmcell", lstm_inputs, lstm, {"input": In, "hidden": H})


def build_mlp(sizes, act):
    acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU, "sigmoid": nn.Sigmoid}
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(acts[act]())

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return Net().to(DT)


def load_init(model, init):
    sd = {k: torch.tensor(v["data"], dtype=DT).reshape(v["shape"]) for k, v in init.items()}
    missing = set(model.state_dict().keys()) ^ set(sd.keys())
    if missing:
        raise SystemExit(f"state_dict names differ from the TS init: {sorted(missing)}")
    model.load_state_dict(sd)


def state(model):
    return {k: T(v) for k, v in model.state_dict().items()}


def make_opt(run, params):
    o = run["optimizer"]
    if o == "sgd":
        return torch.optim.SGD(params, lr=run["lr"], foreach=False)
    if o == "momentum":
        return torch.optim.SGD(params, lr=run["lr"], momentum=run["momentum"], foreach=False)
    if o == "adam":
        return torch.optim.Adam(params, lr=run["lr"], betas=tuple(run["betas"]), eps=run["eps"], foreach=False)
    if o == "adamw":
        return torch.optim.AdamW(params, lr=run["lr"], weight_decay=run["weightDecay"], foreach=False)
    raise SystemExit(f"unknown optimizer {o}")


def train_twin(inp):
    out = {"format": "atlas-autograd-train-torch/1", "torch": torch.__version__, "device": "cpu", "dtype": "float64", "runs": {}}
    batches = [(torch.tensor(b["x"], dtype=DT).reshape(-1, 2), torch.tensor(b["y"], dtype=torch.long)) for b in inp["batches"]]
    for run in inp["runs"]:
        model = build_mlp(inp["sizes"], run["act"])
        load_init(model, inp["init"])
        opt = make_opt(run, model.parameters())
        losses, grads0, params10 = [], None, None
        for k in range(run["steps"]):
            x, y = batches[k]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            if k == 0:
                grads0 = {n: T(p.grad) for n, p in model.named_parameters()}
            losses.append(loss.item())
            opt.step()
            if k == 9:
                params10 = state(model)
        with torch.no_grad():
            x, y = batches[run["steps"]]
            losses.append(F.cross_entropy(model(x), y).item())
        out["runs"][run["id"]] = {"losses": losses, "grads0": grads0, "params10": params10, "final": state(model)}
    return out


def pinn_twin(inp):
    model = build_mlp(inp["sizes"], "tanh")
    load_init(model, inp["init"])
    z, w, x0, v0 = inp["zeta"], inp["omega"], inp["x0"], inp["v0"]
    lp, li = inp["lambdaPhys"], inp["lambdaIc"]
    t_obs = torch.tensor(inp["tObs"], dtype=DT).reshape(-1, 1)
    u_obs = torch.tensor(inp["uObs"], dtype=DT).reshape(-1, 1)

    def parts(model):
        t = torch.tensor(inp["tColl"], dtype=DT).reshape(-1, 1).requires_grad_(True)
        u = model(t)
        du = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        ddu = torch.autograd.grad(du, t, torch.ones_like(du), create_graph=True)[0]
        r = (ddu + 2 * z * w * du + w * w * u) / (w * w)
        phys = (r * r).mean()
        data = F.mse_loss(model(t_obs), u_obs)
        t0 = torch.zeros(1, 1, dtype=DT, requires_grad=True)
        u0 = model(t0)
        du0 = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
        ic = ((u0 - x0) * (u0 - x0)).sum() + (((du0 - v0) / w) * ((du0 - v0) / w)).sum()
        total = data + lp * phys + li * ic
        return {"u": u, "du": du, "ddu": ddu, "r": r, "phys": phys, "data": data, "ic": ic, "total": total}

    p = parts(model)
    names = [n for n, _ in model.named_parameters()]
    params = [q for _, q in model.named_parameters()]
    g_total = torch.autograd.grad(p["total"], params, retain_graph=True)
    g_phys = torch.autograd.grad(p["phys"], params, retain_graph=True)
    out = {
        "format": "atlas-autograd-pinn-torch/1",
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "u": T(p["u"]),
        "du": T(p["du"]),
        "ddu": T(p["ddu"]),
        "r": T(p["r"]),
        "losses0": {k: p[k].item() for k in ("data", "phys", "ic", "total")},
        "gradTotal": {n: T(g) for n, g in zip(names, g_total)},
        "gradPhys": {n: T(g) for n, g in zip(names, g_phys)},
    }
    opt = torch.optim.Adam(model.parameters(), lr=inp["lr"], foreach=False)
    losses = []
    for _ in range(inp["steps"]):
        opt.zero_grad(set_to_none=True)
        p = parts(model)
        p["total"].backward()
        losses.append(p["total"].item())
        opt.step()
    losses.append(parts(model)["total"].item())
    out["adam"] = {"losses": losses, "final": state(model)}
    return out


def dump(name, obj):
    path = OUT / name
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


def main():
    print(f"ATLAS_SITE={SITE}\nATLAS_ROOT={ROOT}\ntorch {torch.__version__}, float64, CPU, TF32 off")
    OUT.mkdir(parents=True, exist_ok=True)
    make_ops()
    dump("ops.json", {"format": "atlas-autograd-ops/1", "torch": torch.__version__, "device": "cpu", "dtype": "float64", "cases": cases})
    for src, fn, dst in (("train_inputs.json", train_twin, "train_torch.json"), ("pinn_inputs.json", pinn_twin, "pinn_torch.json")):
        p = OUT / src
        if not p.exists():
            print(f"skip {dst}: {p} is missing (run node tests/atlas/autograd/make-inputs.ts first)", file=sys.stderr)
            continue
        dump(dst, fn(json.loads(p.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
