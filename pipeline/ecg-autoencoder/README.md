# pipeline/ecg-autoencoder

Python side of the `ecg-autoencoder` package (kind A: a small dense autoencoder on synthetic heartbeats; kind C: the
ROC sweep's Python reference). The page code lives in `site/src/lib/explainers/ecg-autoencoder/`; notes go in
`notes/ecg-autoencoder.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| (TS) `explainers/ecg-autoencoder/process.ts` `data` | makes the beats with `beats.ts`: 4,000 normal training beats, a validation set (400 normal + 50 of each anomaly) and a test set (200 normal + 50 of each anomaly) | `npm run atlas -- run ecg-autoencoder data --opt '{"nTrain":4000,"nVal":400,"nTest":200,"nOdd":50}' --save` (writes `.atlas/ecg-autoencoder/data.tape.*`) |
| `model.py` | the autoencoder (140 → 32 → ReLU → 4 → 32 → ReLU → 140), the loss (mean squared error) and the anomaly score (its square root) | `npm run atlas -- py pipeline/ecg-autoencoder/model.py` (smoke test) |
| `train.py` | reads the TS tape (CRC-32 checked), trains on the GPU (seed 0, Adam 2e-3 cosine, batch 128, 150 epochs, about 40 s), keeps the best checkpoint on the normal validation beats; `model_final.pt`, `train_metrics.json`; `--code k` for the size screen (`model_c<k>_h32.pt`, `train_metrics_c<k>_h32.json`) | `npm run atlas -- py pipeline/ecg-autoencoder/train.py`, then `--code 2`, `--code 8`, `--code 16` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json, testset.json (PyTorch scores of the 400 test beats; the input widget's presets are made in TS, `presetsFromAdapter`) | `npm run atlas -- export ecg-autoencoder` |
| `reference.py` | the `sweep` process in plain Python (threshold slides down, ROC points, trapezoid area) for `atlas parity` | run by `npm run atlas -- parity ecg-autoencoder` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No downloads: every beat
is synthetic and made in TS. Each file <= 5 MB, each model <= 15 MB (AK105).
