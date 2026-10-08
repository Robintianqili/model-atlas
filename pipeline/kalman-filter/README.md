# pipeline/kalman-filter

Python side of the `kalman-filter` package (kind C: one process, `kf`). The page code lives in
`site/src/lib/explainers/kalman-filter/`; notes go in `notes/kalman-filter.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `reference.py` | an independent Python run (plain floats) of the TS process `kf`: a 2-D (position, velocity) Kalman filter with a constant-velocity motion model; frames start, then per second predict, measure, gain, update, then done (the run scored against the truth and the best q, r on the sliders' 7 × 7 grid) | `npm run atlas -- parity kalman-filter` (default inputs), `npm run atlas -- parity kalman-filter --fixtures --write` (every fixture; writes the badge record) |

The process reads plain numbers (`track.ts` `dataOf()`: the true positions, the sensor readings, q, r and the moving
average's window): every source passes them as its `input`, so the reference gets the exact TS readings through
`--input` and never regenerates the noise (the seeded normal draws are made in TS). The process itself draws nothing.

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No downloads: every track is a TS preset
or the learner's drawing.
