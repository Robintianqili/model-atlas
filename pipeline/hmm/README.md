# pipeline/hmm

Python side of the `hmm` package (kind C: four processes, `forward`, `viterbi`, `sample`, `spells`). The page code
lives in `site/src/lib/explainers/hmm/`; notes go in `notes/hmm.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `reference.py` | an independent numpy run of the four TS processes of a two-state hidden Markov model (forward lattice, Viterbi with back-pointers and trace back, sampling with ak.Rng, the spell counts of the limits step), frame by frame | `npm run atlas -- parity hmm` |
| `facts.py` | copies the four char-RNN numbers the comparison step quotes (LSTM width, parameters, the letter-counter cell and its correlation) from `site/static/models/char-rnn/` into `site/static/models/hmm/facts.json` | `npm run atlas -- py pipeline/hmm/facts.py` |

The processes read plain numbers (`hmm.ts` `dataOf()`: the activities and the tables as probabilities, percent / 100,
plus the starting tables): every source passes them as its `input`, so the reference gets the exact TS numbers through
`--input`. The sampling process draws with ak.Rng, bit-identical to the TS rng for `next()`, so no draws are replayed.

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No downloads: the days, the tables and
the made-up stretch of weather are TS data.
