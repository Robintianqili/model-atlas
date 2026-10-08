# pipeline/cart-pole

Python side of the `cart-pole` package (kinds B and C: REINFORCE on the cart-pole, trained live in the browser). The page
code lives in `site/src/lib/explainers/cart-pole/`; notes go in `notes/cart-pole.md`. Nothing here is exported or shipped:
there are no weights (the policy starts from seeded weights and trains in the browser) and no data (the simulator is the data).

Run everything through the CLI from `site/` (or the lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch float64 twin of the TS trainer `reinforce`: the same 4 → 16 → 1 policy and the loss mean(A · −log π(push)), trained on the rounds TS played (its states, pushes and advantages) | `npm run atlas -- parity cart-pole` |
| `reference.py` | an independent Python run of the TS processes: `episode` (the simulator, the fresh policy and the returns, frame by frame) and `seeds` (the whole training, 5 seeds × 50 rounds: plain-Python episodes, torch float64 gradients, torch Adam) | `npm run atlas -- parity cart-pole --fixtures` |

By hand: `npm run atlas -- py pipeline/cart-pole/reference.py --process episode --input '{"start": [0, 0, 0.1, 0]}'`
writes `$ATLAS_SITE/.atlas/cart-pole/episode.ref.tape.json`.

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No downloads.
