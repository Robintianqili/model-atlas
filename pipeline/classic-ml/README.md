# pipeline/classic-ml

Python side of the `classic-ml` package (kind C: processes only). The page code lives in
`site/src/lib/explainers/classic-ml/`; notes go in `notes/classic-ml.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `reference.py` | an independent Python run (plain floats, no sklearn) of the four TS processes: `knn` (one new point: distances, the k nearest, the vote), `knnMap` (kNN's answer on a 40 × 40 grid for k = 1, 3, 5, 9, 15, 25), `tree` (CART with Gini: hold-out split, every node breadth first, the depth sweep 1..8) and `kmeans` (Lloyd's algorithm, plus the k = 1..6 sweep) | `npm run atlas -- parity classic-ml` (default inputs), `npm run atlas -- parity classic-ml --fixtures` (every fixture) |

The processes read plain numbers (`ml.ts` `dataOf()`: points, classes, the new point, k, depth, k-means' k): every source
passes them as its `input`, so the reference gets the exact TS coordinates through `--input` and never regenerates a
preset. The tree's shuffle and k-means' starting points use `ak.Rng`, bit-identical to the TS rng for `int()`.

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No downloads: every point set is the
learner's or a seeded TS generator (`atlas/data/points2d.ts`).
