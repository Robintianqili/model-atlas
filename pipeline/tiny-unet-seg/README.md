# pipeline/tiny-unet-seg

Python side of the `tiny-unet-seg` package (kind A: a tiny U-Net that labels every pixel of a 32×32 synthetic scene as
background, circle, square or triangle). The page code lives in `site/src/lib/explainers/tiny-unet-seg/`; notes go in
`notes/tiny-unet-seg.md`.

The data is made by the page's own TypeScript (`scenes.ts` `render()`), so PyTorch trains on exactly the bytes the page
computes. Run the Python scripts through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and
sets `PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `make_data.ts` | draws the scenes (seeded `createRng` forks) and writes `runs/data/<set>.bin` (float32 pictures, then uint8 classes) and `meta.json` (counts, CRC-32 of both parts, class pixel counts, case ids): train 16,000, val 1,000, test 2,000, big 500 (sizes 9-13), hollow 500 (outlines), dark 500 (dark on light), cases (the page's presets and edge fixtures) | `node pipeline/tiny-unet-seg/make_data.ts --site site/.work/tiny-unet-seg/site` from the repo root (about 11 s; default `--site`: `$ATLAS_SITE`, else `site`) |
| `data.py` | reads a set and checks its CRC-32 against what TS recorded | used by train.py and export.py |
| `model.py` | the U-Net (down 1→8→8, pool, 8→16→16, pool, bottleneck 16→32→32, bilinear ×2 + concat skip, 48→16→16, ×2 + concat, 24→8→8, 1×1 head → 4, softmax over the classes); `UNet(skip=False)` is the same network without the two concatenations | `npm run atlas -- py pipeline/tiny-unet-seg/model.py` (smoke test) |
| `train.py` | trains on the GPU (seed 0, Adam 3e-3 cosine, batch 64, 40 epochs, per-pixel cross-entropy); keeps the best validation mIoU; evaluates on CPU float32 on test, big, hollow and dark; `model_final.pt` + `train_metrics.json` (`--noskip`: `model_noskip.pt` + `train_metrics_noskip.json`) | `npm run atlas -- py pipeline/tiny-unet-seg/train.py` and `... train.py --noskip` (about 5-8 min each) |
| `export.py` | `describe()` + `ak.export_model(...)` twice: the U-Net to `static/models/tiny-unet-seg/` (spec, weights, reference on 4 cases, facts) and the no-skip network to `.../noskip/` (3 cases) | `npm run atlas -- export tiny-unet-seg`, then `npm run atlas -- parity tiny-unet-seg --dir models/tiny-unet-seg/noskip --write` |
| `ops_upsample.py` | Python mirror of the local op `upsample` (`explainers/tiny-unet-seg/ops/upsample.ts`): `F.interpolate(..., mode="bilinear", align_corners=False)` | used by export.py |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No downloads. Each file
<= 5 MB, each model <= 15 MB (AK105). `runs/` is git-ignored (the data is re-made by `make_data.ts` in seconds).
