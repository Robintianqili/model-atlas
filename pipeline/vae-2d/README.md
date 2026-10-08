# pipeline/vae-2d

Python side of the `vae-2d` package (kind A: a 2-D VAE on five Quick, Draw! classes). The page code lives in
`site/src/lib/explainers/vae-2d/`; notes go in `notes/vae-2d.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `model.py` | the VAE (784 → 256 → 64 → μ, log σ² → z = μ + σ·ε → 64 → 256 → 784 → sigmoid), the loss, the classes | `npm run atlas -- py pipeline/vae-2d/model.py` (smoke test) |
| `screen.py` | which five classes: a 20-epoch VAE per candidate set, 10-nearest-neighbour agreement on the 2-D map | `npm run atlas -- py pipeline/vae-2d/screen.py --epochs 20` (about 6 min) |
| `train.py` | trains on the GPU (seed 0, Adam 1e-3 cosine, batch 128, 80 epochs, about 3 min); keeps the best validation checkpoint; `model_final.pt`, `train_metrics.json` | `npm run atlas -- py pipeline/vae-2d/train.py` and, for the limits step, `... train.py --zdim 16` (`model_z16.pt`, not shipped) |
| `stats.py` | facts (test losses, map agreement, σ, grey share, the 16-D comparison), the map (`latent.json`), the preset picks | run by export.py; alone: `npm run atlas -- py pipeline/vae-2d/stats.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json, presets.json, latent.json | `npm run atlas -- export vae-2d` |
| `ops_reparam.py`, `ops_to_image.py` | Python mirrors of the two local ops (`explainers/vae-2d/ops/`) | used by export.py |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads. Each file
<= 5 MB, each model <= 15 MB (AK105).
