# pipeline/behavior-cloning

Python side of the `behavior-cloning` package (kinds B and C: a trainer and a process; nothing is exported, the
network trains live in the browser). The page code lives in `site/src/lib/explainers/behavior-cloning/`; notes go in
`notes/behavior-cloning.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch float64 twin of trainer `bc` (MLP 2 -> 16 -> 4, tanh, cross-entropy): trains on the batches the TS run recorded | `npm run atlas -- parity behavior-cloning` |
| `reference.py` | an independent Python implementation of process `roll`: the grid, the expert, the policy's forward pass from the given weights and the windy episodes (gusts from `ak.Rng`) | `npm run atlas -- parity behavior-cloning` (default input); `node --test src/lib/explainers/behavior-cloning/reference.test.ts` (every fixture) |

CLI contract of reference.py: `--process roll --seed --opt --draws --input --out`; the input is `{ raw, params, cells }`
(the learner's input, the weights, the dataset's cells), long inputs arrive as `@file`.

No data is downloaded: the maps and the demonstrations are made in TypeScript (`world.ts`, the adapter's presets).
