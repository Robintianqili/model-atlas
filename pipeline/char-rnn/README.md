# pipeline/char-rnn

Python side of the `char-rnn` package (kinds: trace, process): a character-level LSTM (74 characters, 24-number
embeddings, 64 units, 29,626 parameters) trained on TinyStories. The page code lives in
`site/src/lib/explainers/char-rnn/`; notes are in `notes/char-rnn.md`.

Run everything through the CLI from `site/` (or the lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `model.py` | the PyTorch model, the 74-character alphabet and the text normalization (curly quotes to plain ones) | `npm run atlas -- py pipeline/char-rnn/model.py` (smoke test) |
| `train.py` | trains on the TinyStories train head (GPU, seed 0, resumable chunks); writes `model_final.pt`, `train_metrics.json` | `npm run atlas -- py pipeline/char-rnn/train.py --run v1 --steps 40000 --chunk 22000` (twice) |
| `units.py` | what the units do on held-out text: the letter-counter unit, forget-gate statistics, the quote check, the top contexts of every unit | `npm run atlas -- py pipeline/char-rnn/units.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json (T = 1 to 128), facts.json, presets.json, unitTop.json | `npm run atlas -- export char-rnn` |
| `reference.py` | the processes `unroll` and `sample` in PyTorch float64, for kind-C parity | `npm run atlas -- parity char-rnn --write --fixtures` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads (the
TinyStories files were downloaded for TinyGPT). Each file <= 5 MB, each model <= 15 MB (AK105).
