# pipeline/sasrec

Python side of the `sasrec` package (kinds: trace, process). The page code lives in
`site/src/lib/explainers/sasrec/`; notes in `notes/sasrec.md`. Run everything from `site/` (or the lane's `site/`)
through the CLI, which uses the configured Python and sets `PYTHONPATH`, `ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| (TS) `explainers/sasrec/process.ts` `logs` | writes the synthetic sessions made by `world.ts`: the fixed inputs' clicks (frame 0), then 10 chunks of 1,000 sessions (product ids in click order, −1 pads) | `npm run atlas -- run sasrec logs --opt '{"n":10000}' --save` (writes `.atlas/sasrec/logs.tape.*`, about 0.5 MB) |
| `model.py` | the tiny SASRec: 31 × 16 item table, 12 × 16 position table, one pre-norm block (one causal head, FFN 16 → 16 → 16), last LayerNorm, scores against the same item table; 2,144 parameters | imported |
| `train.py` | reads the TS tape (checks every frame's sum), trains on the CPU (next click at every position, full softmax, Adam 1e-2 cosine, 60 epochs, best validation hit rate @5 kept), measures the test numbers and the baselines; writes `model_final.pt` and `train_metrics.json` | `npm run atlas -- py pipeline/sasrec/train.py` (about 1.5 min, CPU) |
| `export.py` | `describe()` + `export_model`: the 20 layers on the learner's ids (T = 1 … 12); facts, the product names (read from `world.ts` with Node), clicks per product | `npm run atlas -- export sasrec` |

Run `logs --save` again before `train.py` or `export.py` if `.atlas/sasrec/` was cleaned. CPU only and TF32 off for
references (importing `atlas_kit` does it). No downloads: all data is synthetic.
