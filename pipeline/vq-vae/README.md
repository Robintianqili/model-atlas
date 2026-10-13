# pipeline/vq-vae

Python side of the `vq-vae` package (kind A: a tiny VQ-VAE on five Quick, Draw! classes, traced live on the learner's
drawing). The page code lives in `site/src/lib/explainers/vq-vae/`; notes go in `notes/vq-vae.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `model.py` | the VQ-VAE: a 7×7 stride-7 conv (1 → 64, ReLU) and a 1×1 conv (→ 8) per patch of a 4×4 grid; the nearest of 32 codes; a 3×3 conv (8 → 64, ReLU), 1×1 convs (→ 64, ReLU, → 49), pixel_shuffle(7), sigmoid; the loss with the straight-through estimator | `npm run atlas -- py pipeline/vq-vae/model.py` (smoke test) |
| `train.py` | trains on the GPU (seed 0, Adam 1e-3 cosine, batch 128, 30 epochs, about 1 min); codebook from 32 random patches, then a moving average (decay 0.99); keeps the best validation reconstruction; `model_final.pt`, `train_metrics.json`. `--codes 128` and `--plain --codes 128` train the limits step's comparison models (`model_k128.pt`, `model_plain_k128.pt`, not shipped) | `npm run atlas -- py pipeline/vq-vae/train.py` |
| `stats.py` | facts (test error, distances, code usage, the block-edge error share, the comparison models), `usage.json`, the preset picks | run by export.py; alone: `npm run atlas -- py pipeline/vq-vae/stats.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json, presets.json, usage.json | `npm run atlas -- export vq-vae` |
| `ops_vq.py`, `ops_pixel_shuffle.py` | Python mirrors of the two local ops (`explainers/vq-vae/ops/`) | used by export.py |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads. Each file
<= 5 MB, each model <= 15 MB (AK105). GPU training is not bit-reproducible: a re-run gives slightly different weights
(the second run of the same command moved the validation BCE per pixel by 3e-4), so re-export after any re-training.
