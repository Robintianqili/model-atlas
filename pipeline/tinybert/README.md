# pipeline/tinybert

Python side of the `tinybert` package (kinds: trace, process): a small BERT-style encoder trained with masked-language
modelling on TinyStories, with TinyGPT's tokenizer, size and blocks. The page code lives in
`site/src/lib/explainers/tinybert/`; notes, decisions and measured numbers are in `notes/tinybert.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `model.py` | TinyBERT in PyTorch: token, position and segment tables, 4 pre-norm blocks without a causal mask (d 76, 4 heads, MLP 304, GELU tanh), tied head; 364,420 params | `npm run atlas -- py pipeline/tinybert/model.py` (smoke test) |
| `data.py` | examples from TinyGPT's `train.bin` / `val.bin` in BERT's layout ([CLS] A [SEP] B [SEP]) and BERT's 15% / 80-10-10 masking; imports `pipeline/tinygpt/bpe.py` | used by the scripts below |
| `train.py` | trains on the GPU, resumable in chunks under 8 minutes; writes `runs/<run>/`, `model_final.pt`, `train_metrics.json` | `npm run atlas -- py pipeline/tinybert/train.py --run v1 --steps 40000` (repeat until "done") |
| `stats.py` | measured facts on the validation split: MLM accuracy, and one gap read with both sides / the left side only / by TinyGPT; writes `stats.json` | `npm run atlas -- py pipeline/tinybert/stats.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json (with `vocab` and `tokenFlags`), weights.bin, reference.json (T 12, 16, 1, 64), facts.json, presets.json; copies TinyGPT's tokenizer.json | `npm run atlas -- export tinybert` |
| `reference.py` | the Python twin of process `hide` (same rng, same frames) | run by `npm run atlas -- parity tinybert --write --fixtures` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads (the
TinyStories files in `pipeline/data/tinystories/` were downloaded for TinyGPT). Each file <= 5 MB, each model <= 15 MB
(AK105).
