# pipeline/egnn

Python side of the `egnn` package (kinds: trace, process). The page code lives in `site/src/lib/explainers/egnn/`;
notes go in `notes/egnn.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `make_data.ts` | writes the toy molecules with the page's own `mol.ts` (40,000 train, 2,000 val, 5,000 test; 2-10 atoms) and the export's reference cases (the page's presets and edge fixtures through its adapter) | `node pipeline/egnn/make_data.ts [--site <site>] [--out <folder>]` |
| `data.py` | reads those files (folder: `EGNN_DATA`, else `runs/data`) and checks their CRC-32 against the TS record | |
| `model.py` | the EGNN: embedding, 2 EGNN layers (messages from h and d², pushes along x_i - x_j, node update), energy head; force = sum of the pushes | `npm run atlas -- py pipeline/egnn/model.py` (an equivariance self-test) |
| `train.py` | energy + force regression, Adam, cosine, 60 epochs, seed 0 (GPU about 2.5 min) | `npm run atlas -- py pipeline/egnn/train.py` |
| `ops_egnn.py` | Python mirrors of the local ops `pair_dist2`, `edge_concat`, `vec_sum`, `nb_sum`, `sum_atoms` | |
| `stats.py` | test-set errors and the largest turn and mirror differences, for facts.json | |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json | `npm run atlas -- export egnn` |

The `spin` process has no Python reference (`reference: false`): it reruns the exported model, whose layers parity
already checks against PyTorch. No downloads: every molecule is generated in TS.
