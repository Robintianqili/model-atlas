# pipeline/hello-xor-2

Python side of the `hello-xor-2` package (kinds: trace, train, process). Scaffolded by `atlas new model hello-xor-2` on 2026-10-06.
The page code lives in `site/src/lib/explainers/hello-xor-2/`; notes go in `notes/hello-xor-2.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `model.py` | the PyTorch model (train it with a fixed seed; save `runs/<run>/best.pt`) | `python model.py` (smoke test) |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json | `npm run atlas -- export hello-xor-2` |
| `twin.py` | the PyTorch fp64 twin of the TS trainer (`build(cfg)`, `loss(model, batch, cfg)`) | `npm run atlas -- parity hello-xor-2` |
| `reference.py` | an independent Python run of the TS process (same rng, same frames) | `npm run atlas -- run hello-xor-2 run --save`, then `python reference.py` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads without
the user's OK. Each file <= 5 MB, each model <= 15 MB (AK105). Every dataset or ported code goes into `credits` in
`explainer.ts`.
