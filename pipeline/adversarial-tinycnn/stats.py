"""adversarial-tinycnn: the measured numbers behind the lesson (facts.json) and the preset drawings (presets.json).

    npm run atlas -- py pipeline/adversarial-tinycnn/stats.py            # writes both into $ATLAS_SITE/static/models/adversarial-tinycnn/
    npm run atlas -- py pipeline/adversarial-tinycnn/stats.py --every 10 # a quick look on every 10th test drawing (writes nothing)

TinyCNN is the shipped model (static/models/tinycnn/spec.json + weights.bin, not retrained), run in float64 on the CPU,
as the page runs it (attack.py; the page's net.ts equals it to ~1e-15, see reference.py). The attacks are attack.py's:
y = TinyCNN's own answer, FGSM x' = clip(x + eps*sign(g), 0, 1), PGD with K = 10 steps of alpha = eps/4, no random start.

Facts (over TinyCNN's 15,000 test drawings; the attack numbers over the ones it gets right):
- testN, cleanAcc, attackedN
- fgsm05 / fgsm10 / fgsm20 and pgd05 / pgd10 / pgd20: the share flipped at eps = 0.05 / 0.1 / 0.2
- minFgsmMedian: the median of each drawing's smallest eps on the page's grid (0.005, 0.010, ..., 0.3) at which FGSM
  flips it (drawings that never flip count as above 0.3); fgsmNever: the share that FGSM never flips on the grid
- minPgdMedian: the same for PGD, found by halving the grid range (PGD's success is taken as growing with eps)
- vitN, transferVit: PGD at eps = 0.1 made on TinyCNN, then shown to TinyViT (pipeline/tinyvit, float32): among the drawings
  both models get right and the attack flips on TinyCNN, the share that TinyViT also gets wrong; vitPgd10: the same attack
  made on TinyViT itself (white-box on TinyViT), for comparison
- gradF32Max: the largest |difference| between the input gradient in float32 and in float64 (PyTorch), over 1,000 drawings;
  signF32: the share of pixels whose sign differs between the two (a float32 softmax rounds p(y) to exactly 1 on a very
  sure drawing, so its gradient loses that part)
Presets: per class the first of TinyCNN's preset drawings that TinyCNN gets right. The start drawing (first in the file)
is a typical one: among those FGSM flips at the default eps, the one whose smallest flipping eps is closest to the median.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path

try:
    import atlas_kit as ak
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import atlas_kit as ak

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from attack import build, fgsm, input_grad, pgd, static_weights  # noqa: E402

EPS = 0.1          # the page's default (process.ts EPS)
STEPS = 10         # PGD steps (process.ts defaults.steps)
GRID = 0.005       # the page's eps grid (process.ts defaults.scanStep, scanMax)
GRID_N = 60
TABLE = (0.05, 0.1, 0.2)
B = 1500


def batches(n: int):
    for i in range(0, n, B):
        yield slice(i, min(n, i + B))


@torch.no_grad()
def predict(m, X: torch.Tensor) -> torch.Tensor:
    return torch.cat([m(X[s]).argmax(1) for s in batches(len(X))])


def flips_fgsm(m, X, y, eps) -> torch.Tensor:
    out = []
    for s in batches(len(X)):
        _, _, g = input_grad(m, X[s], y[s])
        with torch.no_grad():
            out.append(m(fgsm(X[s], g, eps)).argmax(1) != y[s])
    return torch.cat(out)


def flips_pgd(m, X, y, eps) -> torch.Tensor:
    out = []
    for s in batches(len(X)):
        xa = pgd(m, X[s], y[s], eps, STEPS, eps / 4)
        with torch.no_grad():
            out.append(m(xa).argmax(1) != y[s])
    return torch.cat(out)


def min_fgsm(m, X, y) -> torch.Tensor:
    """each drawing's smallest grid eps that flips it with FGSM (the same sign map for every eps), +inf if none"""
    best = torch.full((len(X),), float("inf"), dtype=torch.float64)
    for s in batches(len(X)):
        _, _, g = input_grad(m, X[s], y[s])
        sg = torch.sign(g)
        b = best[s]
        for k in range(1, GRID_N + 1):
            e = k * GRID
            with torch.no_grad():
                f = m(torch.clamp(X[s] + e * sg, 0, 1)).argmax(1) != y[s]
            b = torch.where(f & torch.isinf(b), torch.tensor(e, dtype=torch.float64), b)
        best[s] = b
    return best


def min_pgd(m, X, y) -> torch.Tensor:
    """each drawing's smallest grid eps that PGD flips, by halving the grid range [1, GRID_N], +inf if not even at the top"""
    lo = torch.zeros(len(X), dtype=torch.long)          # largest index known not to flip (0 = eps 0)
    hi = torch.full((len(X),), GRID_N + 1, dtype=torch.long)   # smallest index known to flip (GRID_N + 1 = none)
    top = flips_pgd_per(m, X, y, torch.full((len(X),), GRID_N, dtype=torch.long))
    hi[top] = GRID_N
    lo[~top] = GRID_N
    while bool((hi - lo > 1).any()):
        mid = (lo + hi) // 2
        act = hi - lo > 1
        f = flips_pgd_per(m, X[act], y[act], mid[act])
        idx = act.nonzero().squeeze(1)
        hi[idx[f]] = mid[idx[f]]
        lo[idx[~f]] = mid[idx[~f]]
    return torch.where(hi <= GRID_N, hi.double() * GRID, torch.tensor(float("inf"), dtype=torch.float64))


def flips_pgd_per(m, X, y, k: torch.Tensor) -> torch.Tensor:
    """PGD with a per-drawing eps = k * GRID"""
    out = []
    for s in batches(len(X)):
        eps = (k[s].double() * GRID).view(-1, 1, 1, 1)
        xa = pgd(m, X[s], y[s], eps, STEPS, eps / 4)
        with torch.no_grad():
            out.append(m(xa).argmax(1) != y[s])
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.bool)


def tinyvit():
    p = ak.root_path("pipeline", "tinyvit", "model.py")
    spec = importlib.util.spec_from_file_location("adv_tinyvit_model", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adv_tinyvit_model"] = mod
    spec.loader.exec_module(mod)
    v = mod.TinyViT()
    v.load_state_dict(torch.load(ak.root_path("pipeline", "tinyvit", "model_final.pt"), map_location="cpu"))
    return v.eval(), mod.CLASSES


def median_inf(a: torch.Tensor) -> float:
    s = torch.sort(a).values
    v = float(s[(len(s) - 1) // 2])
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--every", type=int, default=1, help="use every n-th test drawing (n > 1 writes nothing)")
    a = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    w, spec = static_weights(ak.atlas_site())
    classes = [c["id"] for c in spec["classes"]]
    m = build(w, torch.float64)
    Xu, y, _ = ak.data.quickdraw(classes, "test")
    X = torch.tensor(Xu[::a.every], dtype=torch.float64).unsqueeze(1) / 255
    y = torch.tensor(y[::a.every])
    pred = predict(m, X)
    ok = pred == y
    Xc, yc = X[ok], y[ok]
    facts: dict = {"testN": (len(X), "drawings"), "cleanAcc": (round(float(ok.double().mean()), 4), "share"), "attackedN": (int(ok.sum()), "drawings")}
    print(f"clean acc {facts['cleanAcc'][0]} on {len(X)}; attacking {len(Xc)}  ({time.time() - t0:.0f} s)", flush=True)
    for e in TABLE:
        k = f"{round(e * 100):02d}"
        facts[f"fgsm{k}"] = (round(float(flips_fgsm(m, Xc, yc, e).double().mean()), 4), "share")
        facts[f"pgd{k}"] = (round(float(flips_pgd(m, Xc, yc, e).double().mean()), 4), "share")
        print(f"eps {e}: FGSM {facts[f'fgsm{k}'][0]}  PGD {facts[f'pgd{k}'][0]}  ({time.time() - t0:.0f} s)", flush=True)
    mf = min_fgsm(m, Xc, yc)
    facts["minFgsmMedian"] = (median_inf(mf), "eps")
    facts["fgsmNever"] = (round(float(torch.isinf(mf).double().mean()), 4), "share")
    print(f"FGSM smallest eps: median {facts['minFgsmMedian'][0]}, never {facts['fgsmNever'][0]}  ({time.time() - t0:.0f} s)", flush=True)
    mp = min_pgd(m, Xc, yc)
    facts["minPgdMedian"] = (median_inf(mp), "eps")
    print(f"PGD smallest eps: median {facts['minPgdMedian'][0]}, never {float(torch.isinf(mp).double().mean())}  ({time.time() - t0:.0f} s)", flush=True)

    # transfer to TinyViT (float32, its own model): the same PGD drawings
    vit, vclasses = tinyvit()
    assert list(vclasses) == classes, (vclasses, classes)
    with torch.no_grad():
        vpred = torch.cat([vit(Xc[s].float()).argmax(1) for s in batches(len(Xc))])
    both = vpred == yc
    Xb, yb = Xc[both], yc[both]
    adv = torch.cat([pgd(m, Xb[s], yb[s], EPS, STEPS, EPS / 4) for s in batches(len(Xb))])
    with torch.no_grad():
        fooled_cnn = torch.cat([m(adv[s]).argmax(1) != yb[s] for s in batches(len(adv))])
        fooled_vit = torch.cat([vit(adv[s].float()).argmax(1) != yb[s] for s in batches(len(adv))])
    facts["vitN"] = (int(fooled_cnn.sum()), "drawings")
    facts["transferVit"] = (round(float(fooled_vit[fooled_cnn].double().mean()), 4), "share")
    vflip = []
    for s in batches(len(Xb)):
        xv = Xb[s].float()
        xa = xv
        for _ in range(STEPS):
            xg = xa.detach().clone().requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(vit(xg), yb[s], reduction="sum")
            (g,) = torch.autograd.grad(loss, xg)
            xa = torch.minimum(torch.clamp(xv + EPS, max=1), torch.maximum(torch.clamp(xv - EPS, min=0), xa + EPS / 4 * torch.sign(g)))
        with torch.no_grad():
            vflip.append(vit(xa).argmax(1) != yb[s])
    facts["vitPgd10"] = (round(float(torch.cat(vflip).double().mean()), 4), "share")
    print(f"transfer: {facts['transferVit'][0]} of {facts['vitN'][0]}; white-box on TinyViT {facts['vitPgd10'][0]}  ({time.time() - t0:.0f} s)", flush=True)

    # float32 vs float64 gradients
    m32 = build(w, torch.float32)
    sub = Xc[: min(1000, len(Xc))]
    ys = yc[: len(sub)]
    _, _, g64 = input_grad(m, sub, ys)
    _, _, g32 = input_grad(m32, sub.float(), ys)
    facts["gradF32Max"] = (float(f"{float((g32.double() - g64).abs().max()):.3g}"), "abs")
    facts["signF32"] = (round(float((torch.sign(g32.double()) != torch.sign(g64)).double().mean()), 4), "share")
    print(f"float32 vs float64: max |diff| {facts['gradF32Max'][0]}, sign differs on {facts['signF32'][0]}", flush=True)

    # presets: per class the first TinyCNN preset drawing it gets right; the start drawing first
    pre = json.loads((ak.atlas_site() / "static" / "models" / "tinycnn" / "presets.json").read_text(encoding="utf-8"))
    picks = []
    for k, c in enumerate(pre["classes"]):
        for it in c["items"]:
            x = torch.tensor(np.frombuffer(base64.b64decode(it["b64"]), dtype=np.uint8).astype(np.float64) / 255).view(1, 1, 28, 28)
            if int(predict(m, x)[0]) != k:
                continue
            mfx = float(min_fgsm(m, x, torch.tensor([k]))[0])
            picks.append({"id": c["id"], "en": c["en"], "zh": c["zh"], "b64": it["b64"], "minEps": mfx})
            break
    flips = [p for p in picks if p["minEps"] <= EPS]
    med = facts["minFgsmMedian"][0]
    first = min(flips, key=lambda p: abs(p["minEps"] - med)) if flips else picks[0]
    order = [first] + [p for p in picks if p is not first]
    print("presets:", [(p["id"], p["minEps"]) for p in order], flush=True)
    if a.every != 1:
        print("--every > 1: nothing written")
        return

    today = dt.date.today().isoformat()
    doc = {"format": "atlas-facts/1", "facts": {k: {"value": v, "unit": u, "source": "pipeline/adversarial-tinycnn/stats.py", "measured": today} for k, (v, u) in facts.items()}}
    for k, v in doc["facts"].items():
        if isinstance(v["value"], float) and not np.isfinite(v["value"]):
            sys.exit(f"fact {k} is not finite")
    out = ak.atlas_site() / "static" / "models" / "adversarial-tinycnn"
    out.mkdir(parents=True, exist_ok=True)
    ak.write_json(out / "facts.json", doc, indent=1)
    ak.write_json(out / "presets.json", {"note": "One Quick, Draw! test drawing per class that TinyCNN gets right (from static/models/tinycnn/presets.json); the first is the start drawing (pipeline/adversarial-tinycnn/stats.py).",
                                         "classes": [{"id": p["id"], "en": p["en"], "zh": p["zh"], "items": [{"b64": p["b64"]}]} for p in order]})
    print(f"wrote {out.as_posix()}/facts.json and presets.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
