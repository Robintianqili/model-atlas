# pipeline/tiny-encdec

Python side of the `tiny-encdec` package (kinds: trace, process): a tiny T5-style encoder-decoder Transformer (one
encoder block, one decoder block, d = 32, two heads, 25,344 parameters) that turns a date written in one of eight
formats into ISO 8601 (2021-03-05): tinyseq2seq's task, with the same generator, seeds and training pairs, so the two
pages can be compared. The page code lives in `site/src/lib/explainers/tiny-encdec/`; notes are in
`notes/tiny-encdec.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| `make_data.ts` | writes the training, validation, test and off-format pairs from the package's TS generator (`data.ts`, a copy of tinyseq2seq's) into `runs/data/` | `node pipeline/tiny-encdec/make_data.ts` (`ATLAS_SITE` picks a lane) |
| `model.py` | `EncDec` (embeddings and learned positions, a pre-norm encoder block, a pre-norm decoder block with causal self-attention and cross-attention, the head), `GreedyProbs` for the export check | |
| `train.py` | trains with a fixed seed (`model_final.pt` with `--name model`, else `runs/<name>.pt`); writes `train_metrics.json` (test accuracy per format, on unseen texts and on the formats of off.json, the validation curve, where step 0 looks) | `npm run atlas -- py pipeline/tiny-encdec/train.py` |
| `ops_greedy.py` | PyTorch mirror of the package's local op `greedy` (greedy decoding, written from the weight slots) | used by `export.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json (also the GRU's numbers from `pipeline/tinyseq2seq/train_metrics.json`, read only), presets.json | `npm run atlas -- export tiny-encdec` |
| `reference.py` | the independent PyTorch float64 run of process `decode` (`atlas parity` calls it) | `npm run atlas -- parity tiny-encdec --write --fixtures` |

`runs/` (git-ignored) holds the generated pairs; `model_final.pt` and `train_metrics.json` are the shipped model's
checkpoint and measurements (trained on the GPU in 44 s; a CUDA re-run can differ in the last bits: re-export after).
