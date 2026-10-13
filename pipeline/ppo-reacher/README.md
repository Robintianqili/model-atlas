# pipeline/ppo-reacher

Python side of the `ppo-reacher` package (kinds B and C: PPO with GAE on a two-link arm, trained live in the browser).
The page code lives in `site/src/lib/explainers/ppo-reacher/`; notes go in `notes/ppo-reacher.md`. Nothing here is
exported or shipped: there are no weights (the networks start from seeded weights and train in the browser) and no data
(the arm simulator is the data).

Run everything through the CLI from `site/` (or the lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch float64 twin of the TS trainer `ppo`: the same actor (6 → 32 → 2 and a learned log σ) and critic (6 → 32 → 1), the clipped PPO loss plus the critic's squared error, trained on the minibatches TS played | `npm run atlas -- parity ppo-reacher` |
| `reference.py` | an independent Python run of the TS process `episode`: the arm, the fresh actor and critic, one sampled episode (the normal draws replayed from TS and checked) and GAE, frame by frame | `npm run atlas -- parity ppo-reacher --fixtures` |

By hand: `npm run atlas -- py pipeline/ppo-reacher/reference.py --input '{"target": [0, 0.9]}'` writes
`$ATLAS_SITE/.atlas/ppo-reacher/episode.ref.tape.json` (without `--draws` it draws its own normals).

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No downloads.
