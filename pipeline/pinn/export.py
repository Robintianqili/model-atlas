"""pinn: writes site/static/models/pinn/presets.json and facts.json, and checks the TS training against PyTorch.

    npm run atlas -- py pipeline/pinn/export.py

1. presets.json: the input chips (seed and noise of the kit's 'oscillator' series preset; TS makes the points).
2. ts_dump.ts (Node, the kit's autograd) writes the TS values into site/.atlas/pinn/.
3. Double-grad parity: on CPU float64, torch.autograd.grad(create_graph=True) recomputes u, u', u'', the residual, the
   physics loss and its gradient for every weight, for three weight sets (the shared start, the trained PINN, the
   trained plain network). Error per quantity: max |ts - torch| / max |torch|. Bar: 1e-10 (design §8.3 M9).
4. Twin parity per source: `atlas parity` runs the twin once per trainer (with the first source's options) and records
   that result for every source of the trainer. Here every train source runs with its own options (lam 0, 0.01, 100,
   c = 0, ...) through atlas_kit.train_reference: SGD 50 steps <= 1e-10, Adam 50 steps <= 1e-7 (relative).
5. facts.json: derivParity, derivParityD2 and twinParity (the worst errors), read by the lesson as {fact:...}.
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

import torch

import atlas_kit as ak
from atlas_kit._io import read_json, write_json
from atlas_kit.train_reference import train_reference

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import twin  # noqa: E402

BAR = 1e-10
PRESETS = [
    {"id": "oscillator", "en": "Eight points", "zh": "八个点", "seed": 1, "noise": 0.02},
    {"id": "seed2", "en": "Eight other points", "zh": "另外八个点", "seed": 2, "noise": 0.02},
    {"id": "noisy", "en": "Eight noisier points", "zh": "八个噪声更大的点", "seed": 3, "noise": 0.05},
]


def rel(ts, ref) -> float:
    ref = torch.as_tensor(ref, dtype=torch.float64).detach()
    a = torch.tensor(ts, dtype=torch.float64).reshape(ref.shape)
    return float((a - ref).abs().max() / ref.abs().max().clamp_min(1e-300))


def deriv_parity(d: dict) -> dict[str, float]:
    torch.set_default_dtype(torch.float64)
    t = torch.tensor(d["T"]).reshape(-1, 1)
    c, k = float(d["c"]), float(d["k"])
    worst: dict[str, float] = {}
    for name, s in d["sets"].items():
        model = twin.build({})
        with torch.no_grad():
            for key, p in model.named_parameters():
                p.copy_(torch.tensor(s["params"][key]["data"]).reshape(p.shape))
        u, du, d2u = twin.derivs(model, t)
        r = twin.residual(u, du, d2u, c, k)
        phys = torch.mean(r ** 2)
        named = list(model.named_parameters())
        gs = torch.autograd.grad(phys, [p for _, p in named])
        errs = {"u": rel(s["u"], u), "du": rel(s["du"], du), "d2u": rel(s["d2u"], d2u), "residual": rel(s["r"], r),
                "physics": abs(s["phys"] - phys.item()) / abs(phys.item()),
                "grads": max(rel(s["grads"][n], g) for (n, _), g in zip(named, gs))}
        print(f"  {name:6s} " + "  ".join(f"{q} {e:.2e}" for q, e in errs.items()) + f"   (physics loss {phys.item():.4g})")
        for q, e in errs.items():
            worst[q] = max(worst.get(q, 0.0), e)
    return worst


def twin_parity(dump_dir: Path) -> tuple[float, bool]:
    worst, ok = 0.0, True
    for fx_path in sorted(dump_dir.glob("variant_fixture.*.json")):
        fx = read_json(fx_path)
        ref = train_reference(str(fx_path), str(HERE / "twin.py"), str(fx_path.with_name(fx_path.name.replace("fixture", "reference"))))
        ts = fx["ts"]
        e = max(rel(ts["losses"], ref["losses"]),
                max(rel(ts["grads0"][n]["data"], torch.tensor(v["data"]).reshape(v["shape"])) for n, v in ref["grads0"].items()),
                max(rel(ts["final"][n]["data"], torch.tensor(v["data"]).reshape(v["shape"])) for n, v in ref["final"].items()))
        passed = e <= fx["bar"]
        ok &= passed
        if fx["opt"]["optimizer"] != "sgd":
            worst = max(worst, e)
        opt = fx["opt"]
        print(f"  {fx['source']:7s} {opt['optimizer']:5s} lam {opt['lam']:<6g} c {opt['c']:<5.3g}: {e:.2e} (bar {fx['bar']:g}) {'ok' if passed else 'FAIL'}")
    return worst, ok


def main() -> int:
    site = ak.atlas_site()
    out_dir = site / "static" / "models" / "pinn"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "presets.json", {"format": "pinn-presets/1", "presets": PRESETS})

    dump_dir = site / ".atlas" / "pinn"
    subprocess.run(["node", str(HERE / "ts_dump.ts"), str(dump_dir)], check=True, env=os.environ)
    print("double-grad parity (TS autograd vs torch.autograd.grad(create_graph=True), float64):")
    worst = deriv_parity(read_json(dump_dir / "deriv_dump.json"))
    top = max(worst.values())
    ok = top <= BAR
    print(f"  worst {top:.2e} relative; bar {BAR:g}: {'PASS' if ok else 'FAIL'}")
    print("twin parity, every train source with its own options (50 steps):")
    twin_top, twin_ok = twin_parity(dump_dir)

    today = datetime.date.today().isoformat()
    m1 = f"{today}, torch {torch.__version__} float64 CPU, 3 weight sets x 200 times"
    m2 = f"{today}, torch {torch.__version__} float64 CPU, every train source, Adam, 50 steps"
    src = "pipeline/pinn/export.py"
    write_json(out_dir / "facts.json", {"format": "atlas-facts/1", "facts": {
        "derivParity": {"value": float(f"{top:.3g}"), "unit": "relative", "source": src, "measured": m1},
        "derivParityD2": {"value": float(f"{worst['d2u']:.3g}"), "unit": "relative", "source": src, "measured": m1},
        "twinParity": {"value": float(f"{twin_top:.3g}"), "unit": "relative", "source": src, "measured": m2},
    }}, indent=1)
    print(f"wrote {out_dir / 'presets.json'} and {out_dir / 'facts.json'}")
    return 0 if ok and twin_ok else 1


if __name__ == "__main__":
    sys.exit(main())
