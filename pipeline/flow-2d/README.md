# pipeline/flow-2d

Python side of the `flow-2d` package (kinds: train, process; kind A on the live weights). The page code lives in
`site/src/lib/explainers/flow-2d/`; notes go in `notes/flow-2d.md`. Nothing here is shipped: the network trains in the
browser, and the only assets are the presets list and the phone tape (`atlas run flow-2d flow --precompute`).

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch float64 twin of trainer `fit`: `Sequential(Linear(3, 64), SiLU, Linear(64, 64), SiLU, Linear(64, 64), SiLU, Linear(64, 2))`, mean squared error | `npm run atlas -- parity flow-2d --steps 20` (SGD and Adam) |
| `reference.py` | the Euler check (process `eulerCheck`): replays the TS draws, trains `twin.build()` with torch.optim.Adam in float64, then samples with Euler (K = 50 and K = 4); frames must equal the TS tape | `npm run atlas -- parity flow-2d` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads without
the user's OK (the data are the kit's seeded smiley, moons and ring, or the learner's painted points).
