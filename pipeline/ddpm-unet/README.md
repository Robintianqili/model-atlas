# pipeline/ddpm-unet

Python side of the `ddpm-unet` package (kinds A + C: a tiny DDPM, a small U-Net noise predictor trained on three Quick,
Draw! classes, and its sampling loop). The page code lives in `site/src/lib/explainers/ddpm-unet/`; notes go in
`notes/ddpm-unet.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `model.py` | the schedule (T = 100, linear β scaled by 1000 / T), `time_embed`, the U-Net (16 / 32 / 64 channels at 28 / 14 / 7 pixels, two skip concatenations), `Noised` (x0, ε, t → x_t → ε̂) and `sample()` (DDPM ancestral sampling, also on a subset of steps) | `npm run atlas -- py pipeline/ddpm-unet/model.py` (smoke test) |
| `train.py` | trains on the GPU: pixels stretched to −1..1, t uniform, MSE on the noise, Adam 1e-3 cosine, batch 256, 600 epochs, EMA 0.999 (about 16 min on an idle RTX 5090 Laptop); `model_final.pt`, `train_metrics.json` | `npm run atlas -- py pipeline/ddpm-unet/train.py` |
| `screen.py` | judges a checkpoint: samples, TinyCNN's share of the trained classes, a PNG grid in `screen/` (used to choose the classes, the width and the −1..1 scaling) | `npm run atlas -- py pipeline/ddpm-unet/screen.py model_final.pt --classes star,umbrella,house --pm1` |
| `stats.py` | the measured facts (`stats.json`): TinyCNN's shares for 1,000 samples with 100 and 10 steps and for real drawings, grey and ink shares | `npm run atlas -- py pipeline/ddpm-unet/stats.py` (before export) |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json, presets.json | `npm run atlas -- export ddpm-unet` |
| `reference.py` | the Python reference of the processes `sample` (the U-Net in PyTorch, the TS draws replayed) and `forward` | run by `npm run atlas -- parity ddpm-unet` |
| `ops_q_sample.py`, `ops_time_embed.py`, `ops_upsample.py` | Python mirrors of the three local ops (`explainers/ddpm-unet/ops/`) | used by export.py |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads. Each file
<= 5 MB, each model <= 15 MB (AK105).
