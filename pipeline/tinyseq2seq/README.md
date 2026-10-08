# pipeline/tinyseq2seq

Python side of the `tinyseq2seq` package (kinds: trace, process): a GRU encoder-decoder with Bahdanau attention that
turns a date written in one of eight formats into ISO 8601 (2021-03-05). The page code lives in
`site/src/lib/explainers/tinyseq2seq/`; notes are in `notes/tinyseq2seq.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| `make_data.ts` | writes the training, validation, test and off-format pairs from the package's TS generator (`data.ts`, the source of truth) into `runs/data/` | `node pipeline/tinyseq2seq/make_data.ts` (`ATLAS_SITE` picks a lane) |
| `model.py` | `Seq2Seq` (embedding, `nn.GRU` encoder, additive attention, `nn.GRUCell` decoder, output layer), `GreedyProbs` for the export check | |
| `train.py` | trains with attention (`model_final.pt`) and without (`runs/noattn.pt`); `--hidden 8` trains both with a state of 8 numbers (the bottleneck experiment); writes `train_metrics.json` | `npm run atlas -- py pipeline/tinyseq2seq/train.py [--hidden 8]` |
| `ops_gru.py`, `ops_attn_decoder.py` | PyTorch mirrors of the package's local ops `gru` and `attn_decoder` | used by `export.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json, presets.json | `npm run atlas -- export tinyseq2seq` |
| `reference.py` | the independent PyTorch float64 run of process `decode` (`atlas parity` calls it) | `npm run atlas -- parity tinyseq2seq --write --fixtures` |

`runs/` (git-ignored) holds the generated pairs and the baseline checkpoints; `model_final.pt` and
`train_metrics.json` are the shipped model's checkpoint and measurements. Training uses the GPU when there is one;
GRU training on CUDA is not bit-deterministic, so a re-run gives slightly different weights: re-export afterwards.
