# pipeline/mlp-playground

Python side of the `mlp-playground` package (kinds: train). Scaffolded by `atlas new model mlp-playground` on 2026-10-06.
The page code lives in `site/src/lib/explainers/mlp-playground/`; notes go in `notes/mlp-playground.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch fp64 twin of the TS trainer (`build(cfg)`, `loss(model, batch, cfg)`) | `npm run atlas -- parity mlp-playground` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads without
the user's OK. Each file <= 5 MB, each model <= 15 MB (AK105). Every dataset or ported code goes into `credits` in
`explainer.ts`.
