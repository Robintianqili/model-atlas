# pipeline/tinymamba

Python side of the `tinymamba` package (kinds: trace, process): a tiny selective state-space model (Mamba-style) on the
selective-copy task. The page code lives in `site/src/lib/explainers/tinymamba/`; notes are in `notes/tinymamba.md`.

Run everything through the CLI from `site/` (or the lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| `data.py` | the task (the twin of `task.ts`: same draws with `ak.Rng`); `python data.py` writes `data_check.json` (200 rows of seed 7, compared with task.ts by claims.test.ts) | `npm run atlas -- py pipeline/tinymamba/data.py` |
| `model.py` | the model: embedding, 2 blocks (LayerNorm, linear + SiLU, Δ/B/C, selective scan, linear, residual), LayerNorm, head | (imported) |
| `ops_ssm_scan.py` | the Python mirror of the local op `ssm_scan` (`ops/scan.ts`) | (imported by export.py) |
| `train.py` | trains on 200,000 rows of seed 1 in resumable chunks (GPU if present); `--publish` writes `model_final.pt` and `train_metrics.json` | `npm run atlas -- py pipeline/tinymamba/train.py --run n4 --N 4 --publish` (repeat until "done") |
| `stats.py` | measured numbers on new rows (more letters, long gaps, Δ by token kind) -> `stats.json` | `npm run atlas -- py pipeline/tinymamba/stats.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json, presets.json | `npm run atlas -- export tinymamba` |
| `reference.py` | the float64 Python twin of the process `scan` (atlas parity compares the frames) | `npm run atlas -- parity tinymamba --write --fixtures` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No downloads: the data is generated.
