# pipeline/hello-xor

Python side of the dev-only `hello-xor` package (kinds: trace, train, process), the DevKit's own worked example.
The page code lives in `site/src/lib/explainers/hello-xor/`; the guide is `site/DEVKIT.md` (section 4 walks through
this package). Notes: `notes/devkit-tools.md` (W1-E2).

| File | What it does | Run |
|---|---|---|
| `model.py` | `XorNet`: 2 inputs -> 2 tanh hidden units -> 1 output score | `python pipeline/hello-xor/model.py` |
| `train.py` | full-batch Adam on the four XOR points (torch seed 0, 1,500 steps); writes `model_final.pt` and `train_metrics.json` | `python pipeline/hello-xor/train.py` |
| `export.py` | kind A: `describe()` + `ak.export_model(...)` -> `site/static/models/hello-xor/` | `npm run atlas -- export hello-xor` |
| `twin.py` | kind B: the PyTorch float64 twin of trainer `xor` | `npm run atlas -- parity hello-xor` |
| `reference.py` | kind C: an independent Python run of process `sgdStep` (same frames) | `npm run atlas -- parity hello-xor` |

Run the CLI from `site/` (or a lane's `site/`): it uses the configured Python (`site/scripts/atlas/atlas.config.local.json`,
else `python` on PATH) with `PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`.
No third-party data: the XOR truth table is written here.
