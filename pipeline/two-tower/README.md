# pipeline/two-tower

Python side of the `two-tower` package (kinds: trace, process). The page code lives in
`site/src/lib/explainers/two-tower/`; notes in `notes/two-tower.md`. Run everything from `site/` (or the lane's `site/`)
through the CLI, which uses the configured Python and sets `PYTHONPATH`, `ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| (TS) `explainers/two-tower/process.ts` `logs` | writes the synthetic logs made by `world.ts`: the 36 movies' features and the fixed inputs (frame 0), then 6 chunks of 1,000 viewers (features, liked movies, history) | `npm run atlas -- run two-tower logs --opt '{"n":6000}' --save` (writes `.atlas/two-tower/logs.tape.*`, about 0.7 MB) |
| `model.py` | the two towers (8 → 16 → 8 and 15 → 16 → 8, ReLU), 672 parameters | imported |
| `train.py` | reads the TS tape (checks every frame's sum), trains with in-batch negatives on the CPU (40 epochs, best validation recall@8 kept), fits the ranker on the validation viewers, measures the test numbers; writes `model_final.pt` and `train_metrics.json` | `npm run atlas -- py pipeline/two-tower/train.py` (about 1 to 5 min, CPU) |
| `export.py` | `describe()` + `export_model`: both towers and the scoring layer whose weight is the item table (the item tower on the 36 movies, computed here); facts, likes, ranker and the ranker reference | `npm run atlas -- export two-tower` |
| `reference.py` | the `inbatch` process (6×6 scores, row softmax, loss, gradient) in torch float64 | called by `npm run atlas -- parity two-tower` |

Run `logs --save` again before `train.py` or `export.py` if `.atlas/two-tower/` was cleaned. CPU only and TF32 off for
references (importing `atlas_kit` does it). No downloads: all data is synthetic.
