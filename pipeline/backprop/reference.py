"""backprop: PyTorch reference for the processes 'graph' and 'net' (kind C parity, design §4.11).

An independent implementation of site/src/lib/explainers/backprop/process.ts in PyTorch float64 on the CPU:
  graph  the toy graph L = (tanh(w·x + b) − y)² on the leaves; the local derivative of every edge written out by hand
         (e, e, 1, −1, 1 − a², 1, 1, x, w), the backward walk edge by edge (share = gradient of the child × local
         derivative, ADDED to the parent), and torch.autograd.grad as the referee: the walk must equal PyTorch's own
         gradients of every leaf (1e-12 relative) or this script fails. Then the numerical check (L(v ± h)).
  net    the 2-2-1 network (nn.Linear(2, 2), tanh, nn.Linear(2, 1), squared error) with the written-out weights:
         torch.autograd.grad for every node and parameter, the numerical check of the nine parameters and one SGD step.
`npm run atlas -- parity backprop` records the TS tapes, runs this file and compares them frame by frame (state within
1e-12 relative, the numerical estimates within 1e-8; AK204 names the first difference).

    npm run atlas -- parity backprop

Arguments (the CLI contract for every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json>
--input <json> --out <stem>. These processes draw no random numbers. The input of 'graph' is the leaves
{"x": .., "w": .., "b": .., "y": ..}; 'net' ignores its input.
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

torch.set_default_dtype(torch.float64)

# data.ts
LEAVES = [("x", (-2.0, 2.0)), ("w", (-3.0, 3.0)), ("b", (-2.0, 2.0)), ("y", (-1.0, 1.0))]
START = {"x": 1.5, "w": 0.8, "b": -0.5, "y": 0.9}
NODES = ["x", "w", "b", "y", "p", "z", "a", "e", "L"]
EDGES = [("e", "L", 0), ("e", "L", 1), ("a", "e", 0), ("y", "e", 1), ("z", "a", 0), ("p", "z", 0), ("b", "z", 1), ("w", "p", 0), ("x", "p", 1)]
NET_X = [1.0, -0.5]
NET_T = 0.8
NET_P0 = [0.7, -0.4, -0.5, 0.9, 0.1, -0.2, 0.8, -0.6, 0.05]   # W1 (row-major), b1, W2, b2


def leaves(raw) -> dict:
    """data.ts leavesOf: each leaf clamped into its range; a missing or non-numeric one takes START's value."""
    out = dict(START)
    if isinstance(raw, dict):
        for k, (lo, hi) in LEAVES:
            v = raw.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                out[k] = min(hi, max(lo, float(v)))
    return out


def graph_forward(lv: dict) -> dict:
    t = {k: torch.tensor([lv[k]], requires_grad=True) for k in ("x", "w", "b", "y")}
    t["p"] = t["w"] * t["x"]
    t["z"] = t["p"] + t["b"]
    t["a"] = torch.tanh(t["z"])
    t["e"] = t["a"] - t["y"]
    t["L"] = t["e"] * t["e"]
    return t


def graph_loss(lv: dict) -> float:
    with torch.no_grad():
        return float(graph_forward(lv)["L"])


def lab(en, zh):
    return (en, zh)


GRAPH_SCHEMA = {
    "vals": {"kind": "tensor", "label": lab("Value of each node", "每个节点的值"), "shape": [9], "role": "activation", "tol": 1e-12},
    "grads": {"kind": "tensor", "label": lab("Gradient of L at each node (0 until reached)", "每个节点上 L 的梯度（到达之前为 0）"), "shape": [9], "role": "grad", "tol": 1e-12},
    "known": {"kind": "tensor", "label": lab("Reached by the backward pass (1) or not yet (0)", "反向传播已到达（1）或尚未到达（0）"), "shape": [9], "role": "data", "tol": "exact"},
    "locals": {"kind": "tensor", "label": lab("Local derivative of each edge", "每条边的局部导数"), "shape": [9], "role": "activation", "tol": 1e-12},
}
GRAPH_PHASES = {
    "forward": {"label": lab("Forward pass", "前向传播"), "kind": "forward"},
    "seed": {"label": lab("Start at L", "从 L 开始"), "kind": "backward"},
    "edge": {"label": lab("One edge back", "倒退一条边"), "kind": "backward"},
    "check": {"label": lab("Numerical check", "数值检验"), "kind": "observe"},
}


def run_graph(tape, raw, opt):
    h = float(opt.get("h", 1e-5))
    lv = leaves(raw)
    t = graph_forward(lv)
    v = {k: float(t[k].detach()) for k in NODES}
    vals = np.array([v[k] for k in NODES])
    local = [v["e"], v["e"], 1.0, -1.0, 1.0 - v["a"] * v["a"], 1.0, 1.0, v["x"], v["w"]]
    grads = np.zeros(9)
    known = np.zeros(9)
    idx = {k: i for i, k in enumerate(NODES)}
    snap = lambda: {"vals": vals.copy(), "grads": grads.copy(), "known": known.copy(), "locals": np.array(local)}
    L = v["L"]
    tape.frame("forward", snap(), step=0, metrics={"L": L})
    grads[idx["L"]] = 1.0
    known[idx["L"]] = 1.0
    tape.frame("seed", snap(), step=0, metrics={"L": L})
    for k, (frm, to, _slot) in enumerate(EDGES):
        ci, pi = idx[to], idx[frm]
        up = grads[ci]
        share = up * local[k]
        acc = known[pi]
        before = grads[pi] if acc else 0.0
        grads[pi] = before + share
        known[pi] = 1.0
        tape.frame("edge", snap(), step=k, metrics={"L": L, "edge": k, "up": up, "local": local[k], "share": share, "before": before, "after": grads[pi], "acc": acc})
    # the referee: PyTorch's own gradients of the four leaves
    ref = torch.autograd.grad(t["L"], [t[k] for k in ("x", "w", "b", "y")])
    for k, g in zip(("x", "w", "b", "y"), ref):
        if abs(float(g) - grads[idx[k]]) > 1e-12 * max(1.0, abs(float(g))):
            raise SystemExit(f"reference.py: the edge-by-edge walk gives dL/d{k} = {grads[idx[k]]}, PyTorch {float(g)}")
    m = {"L": L, "h": h}
    worst = 0.0
    for k, _ in LEAVES:
        up = dict(lv)
        dn = dict(lv)
        up[k] += h
        dn[k] -= h
        num = (graph_loss(up) - graph_loss(dn)) / (2 * h)
        m["n" + k] = num
        worst = max(worst, abs(num - grads[idx[k]]))
    m["maxDiff"] = worst
    tape.frame("check", snap(), step=0, metrics=m)


NET_SCHEMA = {
    "vals": {"kind": "tensor", "label": lab("Value of each node", "每个节点的值"), "shape": [8], "role": "activation", "tol": 1e-12},
    "grads": {"kind": "tensor", "label": lab("Gradient of L at each node (0 until reached)", "每个节点上 L 的梯度（到达之前为 0）"), "shape": [8], "role": "grad", "tol": 1e-12},
    "known": {"kind": "tensor", "label": lab("Node reached (1) or not yet (0)", "节点已到达（1）或尚未到达（0）"), "shape": [8], "role": "data", "tol": "exact"},
    "params": {"kind": "tensor", "label": lab("The nine parameters", "九个参数"), "shape": [9], "role": "param", "tol": 1e-12},
    "pgrads": {"kind": "tensor", "label": lab("Their gradients (0 until reached)", "它们的梯度（到达之前为 0）"), "shape": [9], "role": "grad", "tol": 1e-12},
    "pknown": {"kind": "tensor", "label": lab("Parameter reached (1) or not yet (0)", "参数已到达（1）或尚未到达（0）"), "shape": [9], "role": "data", "tol": "exact"},
    "pnum": {"kind": "tensor", "label": lab("Numerical gradients (check frame)", "数值梯度（检验帧）"), "shape": [9], "role": "grad", "tol": 1e-8},
}
NET_PHASES = {
    "forward": {"label": lab("Forward pass", "前向传播"), "kind": "forward"},
    "out": {"label": lab("Output layer", "输出层"), "kind": "backward"},
    "hidden": {"label": lab("Hidden units", "隐藏单元"), "kind": "backward"},
    "first": {"label": lab("First layer", "第一层"), "kind": "backward"},
    "check": {"label": lab("Numerical check", "数值检验"), "kind": "observe"},
    "update": {"label": lab("One update", "一次更新"), "kind": "update"},
}


def net_forward(p, need_grad: bool):
    x = torch.tensor([NET_X], requires_grad=need_grad)
    W1 = torch.tensor([p[0:2], p[2:4]], requires_grad=need_grad)
    b1 = torch.tensor(p[4:6], requires_grad=need_grad)
    W2 = torch.tensor([p[6:8]], requires_grad=need_grad)
    b2 = torch.tensor(p[8:9], requires_grad=need_grad)
    z = torch.nn.functional.linear(x, W1, b1)
    hh = torch.tanh(z)
    yh = torch.nn.functional.linear(hh, W2, b2)
    e = yh - torch.tensor([[NET_T]])
    L = (e * e).sum()
    return dict(x=x, W1=W1, b1=b1, W2=W2, b2=b2, z=z, h=hh, yh=yh, L=L)


def net_loss(p) -> float:
    with torch.no_grad():
        return float(net_forward(p, False)["L"])


def run_net(tape, opt):
    h = float(opt.get("h", 1e-5))
    lr = float(opt.get("lr", 0.1))
    n = net_forward(NET_P0, True)
    gx, gz, gh, gy, gW1, gb1, gW2, gb2 = (g.detach().numpy().ravel() for g in torch.autograd.grad(n["L"], [n[k] for k in ("x", "z", "h", "yh", "W1", "b1", "W2", "b2")]))
    L = float(n["L"].detach())
    yh = float(n["yh"].detach())
    vals = np.concatenate([n["x"].detach().numpy().ravel(), n["z"].detach().numpy().ravel(), n["h"].detach().numpy().ravel(), [yh, L]])
    grads, known = np.zeros(8), np.zeros(8)
    params = np.array(NET_P0)
    pgrads, pknown, pnum = np.zeros(9), np.zeros(9), np.zeros(9)
    snap = lambda: {"vals": vals.copy(), "grads": grads.copy(), "known": known.copy(), "params": params.copy(), "pgrads": pgrads.copy(), "pknown": pknown.copy(), "pnum": pnum.copy()}

    def setn(nodes, gs):
        for k, g in zip(nodes, gs):
            grads[k] = g
            known[k] = 1.0

    def setp(start, gs):
        for i, g in enumerate(gs):
            pgrads[start + i] = g
            pknown[start + i] = 1.0

    base = {"L": L, "yh": yh, "t": NET_T}
    tape.frame("forward", snap(), step=0, metrics=dict(base))
    setn([7, 6], [1.0, gy[0]])
    setp(6, gW2)
    setp(8, gb2)
    tape.frame("out", snap(), step=1, metrics={**base, "gy": gy[0]})
    setn([4, 5, 2, 3], [gh[0], gh[1], gz[0], gz[1]])
    tape.frame("hidden", snap(), step=2, metrics={**base, "gy": gy[0], "gz1": gz[0], "gz2": gz[1]})
    setp(0, gW1)
    setp(4, gb1)
    setn([0, 1], gx)
    tape.frame("first", snap(), step=3, metrics={**base, "gy": gy[0], "gz1": gz[0], "gz2": gz[1]})
    worst = 0.0
    for i in range(9):
        up = list(NET_P0)
        dn = list(NET_P0)
        up[i] += h
        dn[i] -= h
        pnum[i] = (net_loss(up) - net_loss(dn)) / (2 * h)
        worst = max(worst, abs(pnum[i] - pgrads[i]))
    tape.frame("check", snap(), step=4, metrics={**base, "maxDiff": worst, "passes": 18, "params": 9, "h": h})
    nxt = [p - lr * g for p, g in zip(NET_P0, pgrads)]
    params[:] = nxt
    tape.frame("update", snap(), step=5, metrics={**base, "lr": lr, "L1": net_loss(nxt), "maxDiff": worst, "passes": 18, "params": 9})


def run(process: str, raw, opt: dict, seed: int) -> "ak.TapeWriter":
    if process == "graph":
        tape = ak.TapeWriter(process, GRAPH_SCHEMA, seed=seed, opt=opt, phases=GRAPH_PHASES, provenance="live")
        run_graph(tape, raw, opt)
    elif process == "net":
        tape = ak.TapeWriter(process, NET_SCHEMA, seed=seed, opt=opt, phases=NET_PHASES, provenance="live")
        run_net(tape, opt)
    else:
        raise SystemExit(f"reference.py: no reference for process '{process}' (graph and net only)")
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="graph")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}")
    ap.add_argument("--draws", default=None)
    ap.add_argument("--input", default="null", help="JSON input of the TS run, or @<file> (read it with ak.json_arg)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = ak.json_arg(a.input)
    tape = run(a.process, raw, json.loads(a.opt), a.seed)
    jpath, _ = tape.write(a.out or ak.site_path(".atlas", "backprop", f"{a.process}.ref"))
    print(f"wrote {jpath} ({len(tape.frames)} frames)")
