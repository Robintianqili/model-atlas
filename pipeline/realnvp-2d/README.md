# pipeline/realnvp-2d

Python side of the `realnvp-2d` package (kinds: train, process). The page code lives in
`site/src/lib/explainers/realnvp-2d/`; notes go in `notes/realnvp-2d.md`. Nothing here is shipped: the flow trains in the
browser (the package worker), and the page loads no model asset.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch float64 twin of trainer `fit`: 8 affine coupling layers, each with a small network `Sequential(Linear(1, 24), Tanh, Linear(24, 24), Tanh, Linear(24, 2))` of the kept coordinate (s = tanh of its first output, t its second); loss the mean of 0.5·\|z\|² + log 2π − Σs | `npm run atlas -- parity realnvp-2d` (SGD and Adam, 50 steps, for the sources train, sgd1 and rings) |
| `reference.py` | the process checks: loads the trained weights into `twin.build()`, then repeats process `flow` (your points and a grid through the layers, log p on a grid) and process `sample` (the replayed TS normal draws pulled back through the layers) in torch float64; frames must equal the TS tape at 1e-9 | `npm run atlas -- parity realnvp-2d` (sources flow, sample, ringSample) |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads without
the user's OK (the data are the kit's seeded smiley, ring, moons, two blobs and circles, or the learner's painted points).
