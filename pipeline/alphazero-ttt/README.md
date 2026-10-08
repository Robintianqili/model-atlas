# pipeline/alphazero-ttt

Python side of the `alphazero-ttt` package (kinds: trace, process). Scaffolded by `atlas new model alphazero-ttt` on 2026-10-07.
The page code lives in `site/src/lib/explainers/alphazero-ttt/`; notes go in `notes/alphazero-ttt.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `game.py` | tic-tac-toe rules, the board as the network reads it, an exact solver (memoized minimax), counts | `npm run atlas -- py pipeline/alphazero-ttt/game.py` |
| `model.py` | the policy/value network PVNet (9 → 96 → 96 → 9 priors + 1 value, 11,242 parameters) | smoke test as above |
| `mcts.py` | the network in float64 numpy and the PUCT search (as the page's `search.ts`) | imported |
| `train.py` | self-play training: 40 rounds × 120 games, 50 simulations per move; `runs/selfplay/{best.pt,log.json}` (resumable) | `npm run atlas -- py pipeline/alphazero-ttt/train.py` (about 7 min, CPU) |
| `evaluate.py` | the network and the search against the exact solver; `--write` saves `runs/selfplay/stats.json` | `npm run atlas -- py pipeline/alphazero-ttt/evaluate.py --write` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json, facts.json, presets.json | `npm run atlas -- export alphazero-ttt` |
| `reference.py` | an independent Python run of the TS processes `search` and `game` (same frames) | `npm run atlas -- parity alphazero-ttt --write --fixtures` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads without
the user's OK. Each file <= 5 MB, each model <= 15 MB (AK105). Every dataset or ported code goes into `credits` in
`explainer.ts`.
