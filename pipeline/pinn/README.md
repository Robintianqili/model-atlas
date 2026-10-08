# pipeline/pinn

Python side of the `pinn` package (kinds: train, process). The page code lives in `site/src/lib/explainers/pinn/`;
notes go in `notes/pinn.md`. Nothing is trained here: the page trains live in the browser, and these files check it.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`) and sets `PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch float64 twin of trainers `plain` and `pinn` (`build(cfg)`, `loss(model, batch, cfg)`, with `torch.autograd.grad(create_graph=True)` for u′ and u″) | `npm run atlas -- parity pinn --write` |
| `export.py` | writes `static/models/pinn/presets.json` and `facts.json`; checks u, u′, u″, the residual, the physics loss and its gradients against PyTorch (bar 1e-10 relative), and the twin on every train source with its own options | `npm run atlas -- py pipeline/pinn/export.py` |
| `ts_dump.ts` | the TS side of `export.py` (Node, the kit's autograd): the derivative dump and one twin fixture per source and optimizer | run by `export.py` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No downloads: the data is the kit's
analytic damped oscillator and its seeded noisy points.
