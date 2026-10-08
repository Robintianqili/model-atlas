# pipeline/node2vec

Python side of the `node2vec` package (kinds: train, process). The page code lives in
`site/src/lib/explainers/node2vec/`; notes go in `notes/node2vec.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `reference.py` | the Python reference of process `walks`: the biased random walks (p, q) again with `ak.Rng`, frame by frame | `npm run atlas -- parity node2vec` |
| `twin.py` | the PyTorch fp64 twin of the TS trainer `sg` (skip-gram over the walks; `build(cfg)`, `loss(model, batch, cfg)`) | `npm run atlas -- parity node2vec` |

Data: Zachary's karate club (`ak.data.karate()`, from networkx; the same graph as `atlas/data/karate.ts`). No
downloads, no assets: the page trains live and ships no weights.
