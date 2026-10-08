# pipeline/score-2d

Python side of the `score-2d` package (kinds: train, process). The page code lives in
`site/src/lib/explainers/score-2d/`; notes go in `notes/score-2d.md`. Nothing here is shipped: the network trains in the
browser (the package worker), and the page loads no model asset.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch float64 twin of trainer `fit`: `Sequential(Linear(3, 64), SiLU, Linear(64, 64), SiLU, Linear(64, 2))` on (x̃, ỹ, log σ), mean squared error against −ε | `npm run atlas -- parity score-2d` (SGD and Adam, 50 steps) |
| `reference.py` | the Langevin check (process `langevin`): loads the trained weights into `twin.build()`, then runs the chains in a plain Python loop with `ak.Rng` and the same seed (no replayed draws); frames must equal the TS tape at 1e-9 | `npm run atlas -- parity score-2d` (sources walk, plain, annealed, still) |

The fifth process, `limits` (the learned score against the true one, one frame), has no Python reference (`reference: false`).

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads without
the user's OK (the data are the kit's seeded smiley, ring, moons and two blobs, or the learner's painted points).
