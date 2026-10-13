# pipeline/tinymoe

Python side of the `tinymoe` package (kinds: trace, process): a character-level Transformer (74 characters, 32 numbers
per character, 2 pre-norm causal blocks of 2 heads, at most 64 characters) whose MLPs are mixtures of 4 experts
(32 -> 64 -> 32, ReLU) with a top-2 router; 49,418 parameters, 32,650 used per character. Trained on TinyStories
characters, with a dense twin (one MLP 32 -> 128 -> 32) and a copy trained without the balancing loss for comparison.
The page code lives in `site/src/lib/explainers/tinymoe/`; notes are in `notes/tinymoe.md`.

Run everything through the CLI from `site/` (or the lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `model.py` | the model, its dense twin, the router (`route`), the balancing loss, the alphabet and normalization | `npm run atlas -- py pipeline/tinymoe/model.py` (smoke test) |
| `train.py` | trains one variant (`moe`, `noaux`, `dense`) on the TinyStories train head (GPU, seed 0, resumable chunks); writes `runs/<variant>/final.pt`, its entry of `train_metrics.json`, and `model_final.pt` for `moe` | `npm run atlas -- py pipeline/tinymoe/train.py --variant moe` (twice: chunks of 6,000 steps) |
| `stats.py` | routing on 256,000 held-out characters: expert shares, balancing loss, slots dropped at capacity factors 1 to 2, the characters each expert takes; writes `stats.json` | `npm run atlas -- py pipeline/tinymoe/stats.py` |
| `ops_moe.py` | the Python mirror of the local op `moe` (top-k gates, only the chosen experts run, the gate-weighted sum) | used by `export.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json (T = 1 to 64), facts.json, presets.json | `npm run atlas -- export tinymoe` |
| `reference.py` | the process `dispatch` in plain Python floats, for kind-C parity | `npm run atlas -- parity tinymoe --write --fixtures` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads (the
TinyStories files were downloaded for TinyGPT). Each file <= 5 MB, each model <= 15 MB (AK105).
