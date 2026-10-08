# pipeline/tinyvit

Python side of the `tinyvit` package (kind A: a trained TinyViT traced in the browser). Scaffolded by
`atlas new model tinyvit --template drawing-classifier` on 2026-10-06. The page code lives in
`site/src/lib/explainers/tinyvit/`; notes, decisions and measured numbers are in `notes/tinyvit.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `model.py` | TinyViT in PyTorch: patch 7 (conv, stride 7), [CLS] row, position table, 4 pre-norm blocks (d 48, 4 heads, MLP 192, GELU), class head on row 0; 116,938 params | `npm run atlas -- py pipeline/tinyvit/model.py` (smoke test) |
| `train.py` | trains on TinyCNN's 10 Quick, Draw! classes and split with TinyCNN's augmentation (GPU; resumable with `--chunk`); writes `runs/<run>/`, `model_final.pt`, `train_metrics.json` | `npm run atlas -- py pipeline/tinyvit/train.py --run v1 --epochs 100` |
| `stats.py` | measured facts on the 15,000 test drawings (CPU float32): accuracy of TinyViT and TinyCNN, agreement, where the [CLS] row looks, the lesson head | `npm run atlas -- py pipeline/tinyvit/stats.py` |
| `ops_tokens.py` | Python mirrors of the local ops `to_tokens`, `cls_token`, `select_token` (`explainers/tinyvit/ops/tokens.ts`) | used by `export.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json (5 drawings + blank + dot), facts.json, presets.json (TinyCNN's) | `npm run atlas -- export tinyvit` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads without
the user's OK (the Quick, Draw! files in `pipeline/data/quickdraw/` were downloaded for TinyCNN). Each file <= 5 MB,
each model <= 15 MB (AK105).
