# pipeline/gnn-variants

Python side of the `gnn-variants` package (kinds: trace, process): GCN, GraphSAGE (mean), GIN and GAT on Zachary's
karate club (networkx, unweighted; no download), each 34 -> 4 -> 2 graph layers with tanh and a 2 -> 2 head, trained
the same way on the labels of members 0 and 33 only. The page code lives in `site/src/lib/explainers/gnn-variants/`;
notes are in `notes/gnn-variants.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| `model.py` | the four networks (`Net(kind)`) and `Variants` (all four on one input) | `npm run atlas -- py pipeline/gnn-variants/model.py` (parameter counts) |
| `train.py` | trains each network (CPU float32, Adam lr 0.01, 200 steps, seed 0) and the same recipe over seeds 0-9 | `npm run atlas -- py pipeline/gnn-variants/train.py` (about 75 s): `model_final.pt`, `train_metrics.json` |
| `ops_gnn.py` | PyTorch mirrors of the local ops `gcn_conv`, `sage_conv`, `gin_agg`, `gat_conv` (with their sub-steps) | used by `export.py` |
| `export.py` | `describe()` (one spec, 28 layers, all four networks side by side) + `ak.export_model(...)`; also `karate.json` | `npm run atlas -- export gnn-variants` |
| `reference.py` | independent float64 runs of the processes `gat` (attention into one member) and `drop` (remove one friendship at a time) | `npm run atlas -- parity gnn-variants --write --fixtures` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No new downloads. Each file <= 5 MB,
each model <= 15 MB. Credits are in `explainer.ts`.
