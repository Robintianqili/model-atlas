# pipeline/crnn-ocr

Python side of the `crnn-ocr` package (kinds: trace, process): a tiny CRNN (two 3x3 convolutions with 8 and 16
channels, one 2x2 max pool, 32 columns of 128 features, a bidirectional LSTM of 2 x 32 units, 11 chances per column;
43,435 parameters) trained with the CTC loss on synthetic handwriting-like numbers. The page code lives in
`site/src/lib/explainers/crnn-ocr/`; notes are in `notes/crnn-ocr.md`.

Run everything through the CLI from `site/` (or the lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane. The
data folder is `runs/data/` next to this file unless `CRNN_OCR_DATA` names another one (make_data.ts `--out`).

| File | What it does | Run |
|---|---|---|
| `make_data.ts` | writes the training, validation, test and "wild" (messier) words with the page's own `glyphs.ts` (strokes, then `render()`), plus `cases.json` (the page's presets and edge fixtures through the page's adapter) | `node pipeline/crnn-ocr/make_data.ts --site <site> [--out <folder>]` (about 15 s) |
| `model.py` | the PyTorch model (`fwd` and `bwd` LSTMs written out as a bidirectional pair), greedy CTC decoding, the data reader (CRC-32 checked against the TS record) | `npm run atlas -- py pipeline/crnn-ocr/model.py` (smoke test) |
| `train.py` | CTC training (GPU, seed 0, 40 epochs, best validation epoch kept); writes `model_final.pt`, `train_metrics.json` | `npm run atlas -- py pipeline/crnn-ocr/train.py` (about 100 s) |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json (with `measure()` on the test words) | `npm run atlas -- export crnn-ocr` |
| `ops_columns.py`, `ops_flip.py` | the PyTorch mirrors of the local ops `columns` and `flip` | used by export.py |
| `reference.py` | the process `decode` with numpy and PyTorch's `ctc_loss` (float64), for kind-C parity | `npm run atlas -- parity crnn-ocr --write --fixtures` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No downloads: every
word is synthetic. Each file <= 5 MB, each model <= 15 MB (AK105).
