# pipeline/llava-toy

Python side of the `llava-toy` package (kind A: a LLaVA-style connector toy traced in the browser). Scaffolded by
`atlas new model llava-toy --like tinyvit` on 2026-10-13. The page code lives in `site/src/lib/explainers/llava-toy/`;
notes, decisions and measured numbers are in `notes/llava-toy.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `model.py` | the frozen TinyViT (read only from `pipeline/tinyvit/`: model.py and model_final.pt), blocks 1-3, [CLS] dropped: 16 image features of 48; the projector (48 -> 64, GELU, 64 -> 64); the tiny word-level decoder LM (20 ids, d 64, 2 causal blocks of 4 heads, MLP 128); 78,484 trainable params | `npm run atlas -- py pipeline/llava-toy/model.py` (smoke test) |
| `train.py` | trains the projector and the LM (TinyViT frozen) on generated questions about TinyCNN's 10 Quick, Draw! classes and split, with TinyCNN's augmentation (GPU, ~2 min); writes `runs/<run>/`, `model_final.pt`, `train_metrics.json` | `npm run atlas -- py pipeline/llava-toy/train.py --run v1` |
| `stats.py` | measured facts on the 15,000 test drawings (CPU float32): answer accuracies, TinyViT's own head, where the answer row looks, held-out ladders and envelopes | `npm run atlas -- py pipeline/llava-toy/stats.py` |
| `ops_tokens.py` | Python mirrors of the local ops `to_tokens`, `cls_token` (from tinyvit), `drop_cls`, `merge_image` | used by `export.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json (TinyCNN preset drawings with questions of 1 to 8 words + blank), facts.json, presets.json (TinyCNN's) | `npm run atlas -- export llava-toy` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No new downloads (the Quick, Draw! files in
`pipeline/data/quickdraw/` were downloaded for TinyCNN). Each file <= 5 MB, each model <= 15 MB (AK105).
