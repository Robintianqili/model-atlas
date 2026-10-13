# pipeline/transe

Python side of the `transe` package (kinds: train, process). The page code lives in
`site/src/lib/explainers/transe/`; notes go in `notes/transe.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch fp64 twin of the TS trainer `transe` (TransE with the margin ranking loss; `build(cfg)`, `loss(model, batch, cfg)`) | `npm run atlas -- parity transe` |
| `reference.py` | the Python reference of process `rank`: every candidate tail scored by its distance from h + r, then the ranking, frame by frame | `npm run atlas -- parity transe` |

Data: a small knowledge graph of 34 entities and 33 start facts, generated in TypeScript
(`site/src/lib/explainers/transe/kg.ts`); the Python side receives the facts only through the TS fixtures and process
inputs. No downloads, no assets: the page trains live and ships no weights.
