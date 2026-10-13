# pipeline/contrastive-learning

Python side of the `contrastive-learning` concept package. The page code lives in
`site/src/lib/explainers/contrastive-learning/`; notes in `notes/contrastive-learning.md`. Run everything from `site/`
(or the lane's `site/`) through the CLI, which uses the configured Python and sets `PYTHONPATH`, `ATLAS_SITE` and
`ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| (TS) tinyclip's process `synth` | renders TinyCLIP's 24,000 training pairs (the tinyclip package's own process) | `npm run atlas -- run tinyclip synth --opt '{"n":24000}' --save` (`.atlas/tinyclip/synth.tape.*`, about 90 MB) |
| `model.py` | TinyCLIP (copied from pipeline/tinyclip): conv image tower, embedding + mean text tower, l2norm, learned scale; 15,905 parameters | imported |
| `train.py` | re-runs TinyCLIP's recipe on the CPU (deterministic, seed 0), checks every chunk's CRC-32, keeps 15 checkpoints and measures each on the 2,000 test pairs in batches of 6 | `npm run atlas -- py pipeline/contrastive-learning/train.py --run v1` (about 12.5 min) |
| `export.py` | exports the 8 checkpoints the page shows to `static/models/contrastive-learning/ck/<update>/` (spec, weights, 3-case reference) and writes `facts.json` | `npm run atlas -- py pipeline/contrastive-learning/export.py --run v1` |
| `ops_clip.py` | PyTorch mirrors of the local ops `mean_pool`, `l2norm`, `similarity` (copied from pipeline/tinyclip) | imported by `export.py` |
| `reference.py` | the `nce` process (cosines, cos / τ, row and column softmax, InfoNCE, gradient) in torch float64 | called by `npm run atlas -- parity contrastive-learning` |

`runs/` is git-ignored. No downloads: all data is synthetic.
