# pipeline/adversarial-tinycnn

Python side of the `adversarial-tinycnn` package (kinds A + C: FGSM and PGD on the learner's drawing against TinyCNN, the
shipped model in `site/static/models/tinycnn/`, not retrained). The page code lives in
`site/src/lib/explainers/adversarial-tinycnn/`; notes go in `notes/adversarial-tinycnn.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `attack.py` | TinyCNN in PyTorch (loaded from the page's own `spec.json` + `weights.bin`), the input gradient (torch.autograd), FGSM and PGD with the page's exact arithmetic | imported by the other two |
| `reference.py` | the kit's process reference: the same frames as `process.ts` ('attack'), float64 on the CPU, weights taken from the process input | `npm run atlas -- parity adversarial-tinycnn [--fixtures]` |
| `stats.py` | the measured facts over TinyCNN's 15,000 test drawings (FGSM / PGD flip rates, the smallest flipping ε, transfer to TinyViT, float32 vs float64 gradients) and the preset drawings; writes `facts.json` and `presets.json` | `npm run atlas -- py pipeline/adversarial-tinycnn/stats.py` (about 15 min on the CPU; `--every 10` for a quick look that writes nothing) |

Rules: CPU and float64 for the reference (TF32 off: importing `atlas_kit` does it). No new downloads: the test drawings
are TinyCNN's Quick, Draw! split (`ak.data.quickdraw`), TinyViT is `pipeline/tinyvit/model_final.pt`.
