# pipeline/snake-dqn

Python side of the `snake-dqn` package (kinds: trace, process). Scaffolded by `atlas new model snake-dqn --like
alphazero-ttt` on 2026-10-07. The page code lives in `site/src/lib/explainers/snake-dqn/`; notes go in
`notes/snake-dqn.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `game.py` | the snake game (9 × 9, three relative moves, +1 food, −1 crash), the same as the page's `rules.ts`; the board as three planes | `npm run atlas -- py pipeline/snake-dqn/game.py` (smoke test) |
| `model.py` | the Q-network QNet (conv 3 → 16 → 32, linear 2,592 → 64 → 3; 171,235 parameters) | smoke test as above |
| `train.py` | DQN: 64 games side by side, ε-greedy, replay buffer 100,000, target copy every 500 updates; `runs/<variant>/{best.pt, log.json}` | `npm run atlas -- py pipeline/snake-dqn/train.py [--variant hurry]` (about 18 min on a shared GPU) |
| `evaluate.py` | the greedy agents in float64 on seeds 1 to 1,000, and two random baselines; `--write` saves `runs/main/stats.json` | `npm run atlas -- py pipeline/snake-dqn/evaluate.py --write` |
| `presets.py` | the page's five positions, found in the agent's own games; writes `presets.json` | `npm run atlas -- py pipeline/snake-dqn/presets.py` |
| `export.py` | `describe()` + `ak.export_model(...)` twice: the online network (spec, weights, reference, facts, presets) and the target network (`static/models/snake-dqn/target/`) | `npm run atlas -- export snake-dqn`, then `npm run atlas -- parity snake-dqn --dir models/snake-dqn/target --write` and `npm run atlas -- parity snake-dqn --write --fixtures` |
| `reference.py` | an independent Python run of the TS processes `dqn`, `replay` and `agent` (same frames, same bits) | run by `atlas parity` |

`runs/` is git-ignored: `export.py` needs `runs/main/{best.pt, log.json, stats.json}` and `runs/hurry/log.json`
(commit them with `git add -f`, as alphazero-ttt did, or re-train; GPU training is not bit-reproducible).

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No new downloads without
the user's OK. Each file <= 5 MB, each model <= 15 MB (AK105). Every dataset or ported code goes into `credits` in
`explainer.ts`.
