"""adversarial-tinycnn: Python reference for process 'attack' (kind C parity, design §4.11).

An independent run of site/src/lib/explainers/adversarial-tinycnn/process.ts in PyTorch float64 on the CPU: the same
TinyCNN weights (taken from the process input, exactly what the page used), the gradient of the loss with respect to the
input pixels from torch.autograd (the page's comes from the kit's autograd core, net.ts), the same FGSM, PGD and ε scan.
`npm run atlas -- parity adversarial-tinycnn` records the TS tape, runs this file and compares the two frame by frame
(the gradient within 1e-9 relative to max(1, |value|), drawings within 1e-12, signs and answers exactly; AK204 names the
first difference).

    npm run atlas -- parity adversarial-tinycnn [--fixtures]       # the usual way (it passes the arguments below)

Arguments (the CLI contract for every reference.py): --process <id> --seed <n> --opt <json> --draws <draws.json>
--input <json | @file> --out <stem>. The process draws no random numbers, so --seed and --draws are unused.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack import build, fgsm, input_grad, pgd_step  # noqa: E402

N = 784


def _schema() -> dict:
    def f(kind, en, zh, **kw):
        return {"kind": kind, "label": (en, zh), **kw}
    img = dict(shape=[1, 28, 28])
    return {
        "x": f("tensor", "Your drawing x", "你的画 x", **img, role="data", tol="exact"),
        "p0": f("tensor", "Class shares on your drawing", "你的画上的类别份额", shape=[10], role="output", scale="sequential", tol=1e-9),
        "g": f("tensor", "Gradient of the loss for every pixel", "损失对每个像素的梯度", **img, role="grad", scale="diverging", tol=1e-9),
        "s": f("tensor", "Its sign", "它的符号", **img, role="grad", scale="diverging", tol="exact"),
        "adv": f("tensor", "The changed drawing x′", "改过的画 x′", **img, role="data", tol=1e-12),
        "p": f("tensor", "Class shares on the changed drawing", "改过的画上的类别份额", shape=[10], role="output", scale="sequential", tol=1e-9),
        "scanP": f("tensor", "p(y) for every ε", "每个 ε 下的 p(y)", shape=["E"], role="output", tol=1e-9),
        "scanTop": f("tensor", "The answer for every ε", "每个 ε 下的答案", shape=["E"], role="output", tol="exact"),
    }


PHASES = {
    "clean": {"label": ("Your drawing and the loss", "你的画和损失"), "kind": "loss"},
    "grad": {"label": ("The gradient for every pixel", "每个像素的梯度"), "kind": "backward"},
    "sign": {"label": ("Keep only the sign", "只留下符号"), "kind": "other"},
    "fgsm": {"label": ("One step of size ε (FGSM)", "走一步，步长 ε（FGSM）"), "kind": "update"},
    "pgd": {"label": ("PGD: small steps, clipped", "PGD：小步走，再裁剪"), "kind": "update"},
    "scan": {"label": ("Every ε at once", "一次试遍所有 ε"), "kind": "observe"},
}


def _np(t: torch.Tensor, shape) -> np.ndarray:
    return t.detach().reshape(shape).numpy().astype(np.float64)


def judge(m, x: torch.Tensor, y: int) -> dict:
    with torch.no_grad():
        z = m(x)[0]
    p = torch.softmax(z, 0)
    top = int(torch.argmax(p))          # the first index on a tie, as the page's argmax
    loss = float(torch.logsumexp(z, 0) - z[y])
    return {"p": p, "top": top, "ptop": float(p[top]), "py": float(p[y]), "loss": loss, "flipped": 0 if top == y else 1}


def run(inp: dict, opt: dict, seed: int = 1) -> "ak.TapeWriter":
    tape = ak.TapeWriter("attack", _schema(), seed=seed, opt=opt, phases=PHASES, provenance="live")
    w = {k: np.asarray(v["data"], dtype=np.float64).reshape(v["shape"]) for k, v in inp["w"].items()}
    m = build(w, torch.float64)
    x = torch.tensor(np.asarray(inp["x"], dtype=np.float64)).reshape(1, 1, 28, 28)
    eps = float(inp["eps"])
    steps, alpha = int(opt["steps"]), float(opt["alphaFrac"]) * eps

    with torch.no_grad():
        z0 = m(x)[0]
    p0 = torch.softmax(z0, 0)
    y = int(torch.argmax(p0))
    r = int(torch.argmax(torch.where(torch.arange(10) == y, torch.tensor(-1.0, dtype=torch.float64), p0)))
    _, loss, g = input_grad(m, x, torch.tensor([y]))
    xf = x.flatten()
    ink = int((xf > 0).sum())
    tape.frame("clean", {"x": _np(x, [1, 28, 28]), "p0": _np(p0, [10]), "p": _np(p0, [10])}, step=0,
               metrics={"y": y, "py": float(p0[y]), "loss": float(loss[0]), "r": r, "pr": float(p0[r]), "eps": eps, "ink": ink})

    gf = g.flatten()
    a = gf.abs()
    at = int(torch.argmax(a))
    s_all = float(a.sum())
    on_ink = float(a[xf > 0].sum())
    tape.frame("grad", {"g": _np(g, [1, 28, 28])}, step=0,
               metrics={"gmax": float(a[at]), "row": at // 28, "col": at % 28, "gAt": float(gf[at]),
                        "onInk": on_ink / s_all if s_all > 0 else 0.0, "inkShare": ink / N})

    s = torch.sign(g)
    n_pos, n_neg = int((s > 0).sum()), int((s < 0).sum())
    tape.frame("sign", {"s": _np(s, [1, 28, 28])}, step=0, metrics={"nPos": n_pos, "nNeg": n_neg, "nZero": N - n_pos - n_neg})

    v = x + eps * s
    adv = fgsm(x, g, eps)
    f = judge(m, adv, y)
    tape.frame("fgsm", {"adv": _np(adv, [1, 28, 28]), "p": _np(f["p"], [10])}, step=0,
               metrics={"eps": eps, "y": y, "top": f["top"], "ptop": f["ptop"], "py": f["py"], "loss": f["loss"], "flipped": f["flipped"],
                        "clipped": int((adv != v).sum()), "meanMove": float((adv - x).abs().sum()) / N})

    xk, first = x, 0
    for k in range(1, steps + 1):
        gk = g if k == 1 else input_grad(m, xk, torch.tensor([y]))[2]
        xk = pgd_step(x, xk, gk, eps, alpha)
        j = judge(m, xk, y)
        if j["flipped"] and not first:
            first = k
        tape.frame("pgd", {"adv": _np(xk, [1, 28, 28]), "p": _np(j["p"], [10])}, step=k,
                   metrics={"k": k, "steps": steps, "eps": eps, "alpha": alpha, "y": y, "top": j["top"], "ptop": j["ptop"], "py": j["py"],
                            "loss": j["loss"], "flipped": j["flipped"], "firstFlip": first})

    step, top_e = float(opt["scanStep"]), float(opt["scanMax"])
    n = round(top_e / step)
    scan_p, scan_top, min_eps = np.zeros(n), np.zeros(n), -1.0
    for k in range(1, n + 1):
        e = k * step
        j = judge(m, torch.clamp(x + e * s, 0, 1), y)
        scan_p[k - 1], scan_top[k - 1] = j["py"], j["top"]
        if j["flipped"] and min_eps < 0:
            min_eps = e
    tape.frame("scan", {"scanP": scan_p, "scanTop": scan_top}, step=0,
               metrics={"eps": eps, "y": y, "minEps": min_eps, "found": 1 if min_eps > 0 else 0, "scanStep": step, "scanMax": top_e,
                        "flipped": f["flipped"], "pgdFlip": first})
    return tape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--process", default="attack")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--opt", default="{}", help="JSON options, as the TS source ran (steps, alphaFrac, scanStep, scanMax)")
    ap.add_argument("--draws", default=None, help="unused: the process draws no random numbers")
    ap.add_argument("--input", required=True, help="the TS run's input { x, eps, w } as JSON, or @file")
    ap.add_argument("--out", required=True, help="output stem")
    a = ap.parse_args()
    if a.process != "attack":
        sys.exit(f"reference.py knows process 'attack', not {a.process!r}")
    opt = {"steps": 10, "alphaFrac": 0.25, "scanStep": 0.005, "scanMax": 0.3, **ak.json_arg(a.opt, {})}
    t = run(ak.json_arg(a.input), opt, a.seed)
    jpath, _ = t.write(a.out)
    print(f"wrote {jpath} ({len(t.frames)} frames)")
