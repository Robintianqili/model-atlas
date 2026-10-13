# pipeline/pointnet

Python side of the `pointnet` package (kind A: a tiny PointNet that says which of five shapes, circle, square,
triangle, star or ring, a set of loose 2-D points forms). The page code lives in `site/src/lib/explainers/pointnet/`;
notes go in `notes/pointnet.md`.

The data is made by the page's own TypeScript (`shapes.ts` `sample()` and `normalize()`, and the page's adapter for the
reference cases), so PyTorch trains on exactly the bytes the page computes. Run the Python scripts through the CLI from
`site/` (or a lane's `site/`): it uses the configured Python and sets `PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and
`ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `make_data.ts` | samples the clouds (seeded `createRng` forks; 16 to 200 points, any rotation, stretch 0.85-1.15, jitter, uneven spacing, small gaps), normalizes them as the page does and writes `runs/data/<set>.bin` (float32 clouds repeated up to 256 points, uint8 classes, uint16 counts) and `meta.json` (counts, CRC-32 of each part): train 30,000, val 2,000, test 5,000, stray (the test clouds plus one stray point, `withStray`), half (a seeded half of each test cloud's points); `cases.json`: the reference cases (the page's presets and edge fixtures through its adapter, and a 1,024-point circle) | `node pipeline/pointnet/make_data.ts --site site/.work/pointnet/site` from the repo root (about 2 s; default `--site`: `$ATLAS_SITE`, else `site`) |
| `data.py` | reads a set and checks its CRC-32 against what TS recorded; reads the cases | used by train.py, stats.py and export.py |
| `model.py` | the PointNet: shared MLP 2 → 32 → 32 (ReLU after each) on every point, max over the points, classifier 32 → 32 → 5 | `npm run atlas -- py pipeline/pointnet/model.py` (smoke test) |
| `train.py` | trains on the GPU (seed 0, Adam 3e-3 cosine, batch 256, 60 epochs, cross-entropy); keeps the best validation accuracy; evaluates on CPU float32 on test; `model_final.pt` + `train_metrics.json` | `npm run atlas -- py pipeline/pointnet/train.py` (about 1 min) |
| `stats.py` | the measured facts: test accuracy, critical points per cloud, dead features, the effect of removing half of the points and of one stray point | `npm run atlas -- py pipeline/pointnet/stats.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec, weights, reference on the 8 cases, facts | `npm run atlas -- export pointnet` |
| `ops_max_points.py` | Python mirror of the local op `max_points` (`explainers/pointnet/ops/max_points.ts`): `x.max(dim=0).values` | used by export.py |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No downloads. Each file
<= 5 MB, each model <= 15 MB (AK105). `runs/` is git-ignored (the data is re-made by `make_data.ts` in seconds).
