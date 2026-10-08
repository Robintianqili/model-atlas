# pipeline/gridworld

Python side of the `gridworld` package (kind C: processes only; no model is trained or exported). The page code lives in
`site/src/lib/explainers/gridworld/`; notes go in `notes/gridworld.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `reference.py` | an independent Python implementation of the four processes (`bandit`, `race`, `q`, `vi`): the maze rules, Q-learning, value iteration, the slot machines and the strategies. It replays the TS draws and checks each against its own `ak.Rng` stream | `npm run atlas -- parity gridworld` (default input); `node --test src/lib/explainers/gridworld/reference.test.ts` (every fixture) |
| `export.py` | writes `static/models/gridworld/presets.json`, the maze presets (classic, cliff, maze, walled) | `npm run atlas -- py pipeline/gridworld/export.py` |

By hand: `npm run atlas -- py pipeline/gridworld/reference.py --process q --seed 12 --input '{"rows": ["S.", ".G"]}'`
writes `.atlas/gridworld/q.ref.tape.json` (CLI contract: `--process --seed --opt --draws --input --out`).

Rules: no new downloads without the user's OK. Each file <= 5 MB (AK105). The cliff board follows Sutton & Barto's
example 6.6 (credited in `explainer.ts`; ideas only).
