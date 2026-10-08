# pipeline/tinyforecaster

Python side of the `tinyforecaster` package (kinds: train, process, trace on the live weights). The page code lives in
`site/src/lib/explainers/tinyforecaster/`; notes go in `notes/tinyforecaster.md`. Nothing is trained here and no weights
ship: the forecaster trains live in the browser from a seeded start (TS is the source of truth for data and init).

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `export.py` | writes `static/models/tinyforecaster/presets.json` (the five series presets of the input widget) | `npm run atlas -- py pipeline/tinyforecaster/export.py` |
| `twin.py` | the PyTorch float64 twin of trainer `lstm` (`nn.LSTMCell(1, 16)` + `nn.Linear(16, 1)`, input minus its last value, forecast = last value + head) | `npm run atlas -- parity tinyforecaster` |
| `reference.py` | an independent Python run of the three processes `windows`, `forecast` (training with the twin and the same `ak.Rng` streams, rollout, seasonal-naive baseline) and `walks` | `npm run atlas -- parity tinyforecaster` (`--fixtures` on every fixture) |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No new downloads: every series is
synthetic (`site/src/lib/atlas/data/series.ts`).
