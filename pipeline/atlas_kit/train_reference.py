"""Kind-B parity: run a package's PyTorch float64 twin on the init params and batches the TS trainer wrote (design §4.11).

    python -m atlas_kit.train_reference --fixture $ATLAS_SITE/.atlas/<slug>/train_fixture.<trainer>.<opt>.json \\
        --twin pipeline/<slug>/twin.py --out $ATLAS_SITE/.atlas/<slug>/train_reference.<trainer>.<opt>.json [--steps K]

`npm run atlas -- parity <slug>` writes the fixture and runs this (TS is the source of truth for init and synthetic data).

The twin module (pipeline/<slug>/twin.py) provides
    build(cfg: dict) -> nn.Module                       same architecture and state_dict names as the TS Module.named()
    loss(model, batch: dict, cfg: dict) -> (loss, parts) batch values are float64 tensors with the TS shapes
                                                         (labels arrive as floats: call .long() where needed); loss may
                                                         use torch.autograd.grad(..., create_graph=True)
    optimizer(model, cfg) -> torch.optim.Optimizer       optional; default from cfg like the TS trainer (below)

Fixture (atlas-train-fixture/1): { slug, source, trainer, seed, opt, steps, init: {name: {shape, data}},
batches: [{step, batch: {key: {shape, data}}}], ts: {losses, grads0, final} }.
Output (atlas-train-reference/1): { losses: [K], grads0: {name: {shape, data}}, final: {name: {shape, data}}, steps, torch }.
Default optimizer from cfg (as site/src/lib/atlas/train/trainer.ts): 'sgd' (weightDecay), 'momentum' (momentum 0.9,
dampening 0, nesterov false), 'adam' (betas, eps 1e-8, weightDecay), 'adam' + decoupled: true = AdamW (weightDecay
default 0.01); all with foreach=False, float64, CPU. AK203 on a state_dict name mismatch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

from ._io import read_json, write_json
from .errors import fail


def _load_twin(path: str):
    p = Path(path).resolve()
    if not p.is_file():
        fail("AK203", f"no twin at {p.as_posix()}", fix="write pipeline/<slug>/twin.py with build(cfg) and loss(model, batch, cfg)")
    spec = importlib.util.spec_from_file_location(f"atlas_twin_{p.parent.name.replace('-', '_')}", p)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        sys.path.remove(str(p.parent))
    for name in ("build", "loss"):
        if not callable(getattr(mod, name, None)):
            fail("AK203", f"{p.name} has no {name}()", fix="the twin contract: build(cfg) -> nn.Module; loss(model, batch, cfg) -> (loss, parts)")
    return mod


def _t(v: dict) -> torch.Tensor:
    return torch.tensor(v["data"], dtype=torch.float64).reshape(v["shape"])


def _plain(t: torch.Tensor) -> dict:
    t = t.detach().to(torch.float64).cpu()
    return {"shape": list(t.shape), "data": t.reshape(-1).tolist()}


def default_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    kind = str(cfg.get("optimizer", "sgd"))
    lr = float(cfg.get("lr", 1e-3))
    wd = cfg.get("weightDecay")
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=float(wd or 0), foreach=False)
    if kind == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.9)), weight_decay=float(wd or 0), foreach=False)
    if kind in ("adam", "adamw"):
        betas = tuple(cfg.get("betas") or (0.9, 0.999))
        eps = float(cfg.get("eps", 1e-8))
        if kind == "adamw" or cfg.get("decoupled") is True:
            return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=float(0.01 if wd is None else wd), foreach=False)
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=float(wd or 0), foreach=False)
    fail("AK203", f"unknown optimizer {kind!r}", fix="sgd, momentum or adam (+ decoupled: true for AdamW), or define optimizer(model, cfg) in the twin")


def train_reference(fixture: str, twin: str, out: str, steps: int | None = None) -> dict:
    fx = read_json(fixture)
    if fx.get("format") != "atlas-train-fixture/1":
        fail("AK203", f"{fixture} is not an atlas-train-fixture/1 file")
    cfg: dict[str, Any] = dict(fx.get("opt") or {})
    mod = _load_twin(twin)
    torch.set_default_dtype(torch.float64)
    model = mod.build(cfg).to(torch.float64).cpu()
    init = {k: _t(v) for k, v in fx["init"].items()}
    names = set(model.state_dict().keys())
    if names != set(init):
        fail("AK203", "the twin's state_dict names differ from the TS parameter names",
             where=f"only in the twin: {sorted(names - set(init))}; only in TS: {sorted(set(init) - names)}",
             fix="name the twin's submodules like the TS Module (MLP('net', ...) -> net.0.weight, net.2.weight, ...)")
    with torch.no_grad():
        model.load_state_dict({k: v.reshape(model.state_dict()[k].shape) for k, v in init.items()})
    opt = mod.optimizer(model, cfg) if callable(getattr(mod, "optimizer", None)) else default_optimizer(model.parameters(), cfg)
    k_steps = steps if steps is not None else int(fx.get("steps") or len(fx["batches"]))
    batches = sorted(fx["batches"], key=lambda b: b["step"])[:k_steps]
    if len(batches) < k_steps:
        fail("AK203", f"the fixture has {len(batches)} batches, {k_steps} steps were asked for")
    losses: list[float] = []
    grads0: dict = {}
    named = dict(model.named_parameters())
    for b in batches:
        batch = {k: (_t(v) if isinstance(v, dict) and "data" in v else v) for k, v in b["batch"].items()}
        opt.zero_grad(set_to_none=False)
        loss, _parts = mod.loss(model, batch, cfg)
        loss.backward()
        if not grads0:
            grads0 = {k: _plain(p.grad if p.grad is not None else torch.zeros_like(p)) for k, p in named.items()}
        losses.append(float(loss.item()))
        opt.step()
    res = {"format": "atlas-train-reference/1", "steps": k_steps, "torch": torch.__version__, "losses": losses, "grads0": grads0,
           "final": {k: _plain(v) for k, v in model.state_dict().items()}}
    write_json(out, res)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m atlas_kit.train_reference", description=__doc__.splitlines()[0])
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--twin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args(argv)
    r = train_reference(a.fixture, a.twin, a.out, a.steps)
    print(f"train_reference: {r['steps']} float64 steps, loss {r['losses'][0]:.6g} -> {r['losses'][-1]:.6g} -> {a.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 - AtlasError prints its own block
        from .errors import AtlasError
        if isinstance(e, AtlasError):
            print(str(e), file=sys.stderr)
            sys.exit(1)
        raise
