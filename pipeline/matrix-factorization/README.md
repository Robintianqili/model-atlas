# pipeline/matrix-factorization

Python side of the `matrix-factorization` package (kinds: train, process). There is nothing to train or export here:
the page trains live in the browser on synthetic ratings that TS generates (`site/src/lib/explainers/matrix-factorization/world.ts`).
These two files only prove that the TS code computes what PyTorch / plain Python compute.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch float64 twin of trainer `fit`: `nn.Embedding` tables `user` [25, 2] and `item` [20, 2], prediction = dot product, mean squared error over the batch (one rating) | `npm run atlas -- parity matrix-factorization` |
| `reference.py` | an independent Python run of process `nudge` (one SGD step on one rating: start, error, gradient, update), with `ak.Rng` for the starting numbers | `npm run atlas -- parity matrix-factorization --fixtures` |

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`. CPU only, TF32 off (importing `atlas_kit` does it).
No data is downloaded or stored here.
