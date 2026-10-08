# pipeline/gcn

Python side of the `gcn` package (kinds: trace, train, process). Scaffolded by `atlas new model gcn` on 2026-10-06.
The page code lives in `site/src/lib/explainers/gcn/`; notes go in `notes/gcn.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `model.py` | the GCN (34 -> 4 -> 2 graph layers, 2 -> 2 head; Â computed in forward) and the 8-layer `Deep` variant | |
| `train.py` | trains on Zachary's club from the labels of members 0 and 33 (CPU, < 10 s); measures 2 vs 8 layers over 10 seeds | `npm run atlas -- py pipeline/gcn/train.py` |
| `ops_gcn.py` | Python mirrors of the local ops `gcn_norm` and `gcn_conv` | |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json, karate.json | `npm run atlas -- export gcn` |
| `twin.py` | the PyTorch fp64 twin of the TS trainer `gcn` | `npm run atlas -- parity gcn` |
| `reference.py` | independent Python runs of the processes `mp` and `smooth` | `npm run atlas -- parity gcn --fixtures` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads without
the user's OK. Each file <= 5 MB, each model <= 15 MB (AK105). Every dataset or ported code goes into `credits` in
`explainer.ts`.
