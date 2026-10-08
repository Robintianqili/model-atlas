# pipeline/tinyclip

Python side of the `tinyclip` package (kinds: trace, process). The page code lives in
`site/src/lib/explainers/tinyclip/`; notes in `notes/tinyclip.md`. Run everything from `site/` (or the lane's `site/`)
through the CLI, which uses the configured Python and sets `PYTHONPATH`, `ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| (TS) `explainers/tinyclip/process.ts` `synth` | renders the 24,000 training pairs and the 14 fixed drawings with TinyCNN's `preprocessStrokes` | `npm run atlas -- run tinyclip synth --opt '{"n":24000}' --save` (writes `.atlas/tinyclip/synth.tape.*`, about 90 MB) |
| `model.py` | TinyCLIP: conv image tower, embedding + mean text tower, l2norm, learned scale; 15,905 parameters | imported |
| `train.py` | reads the TS tape (checks the CRC-32 of every chunk), trains with symmetric InfoNCE on the GPU, saves `model_init.pt` (the starting weights) and `model_final.pt` (best validation loss) | `npm run atlas -- py pipeline/tinyclip/train.py --run v1` (about 30 s) |
| `stats.py` | test numbers on the 2,000 test pairs (batches of 6, zero-shot shape/size/place, order ties) | `npm run atlas -- py pipeline/tinyclip/stats.py` |
| `ops_clip.py` | PyTorch mirrors of the local ops `mean_pool`, `l2norm`, `similarity` | imported by `export.py` |
| `export.py` | `describe()` + `export_model` for the trained model (`static/models/tinyclip/`) and the starting weights (`static/models/tinyclip/init/`); facts from `stats.py` | `npm run atlas -- export tinyclip` |
| `reference.py` | the `nce` process (batch scores, row and column softmax, InfoNCE, gradient) in torch float64 | called by `npm run atlas -- parity tinyclip` |

Run `synth --save` again before `train.py` if `.atlas/tinyclip/` was cleaned. CPU only and TF32 off for references
(importing `atlas_kit` does it). No downloads: all data is synthetic.
