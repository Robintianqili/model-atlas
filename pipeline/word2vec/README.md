# pipeline/word2vec

Python side of the `word2vec` package (kinds: train, process). The page code lives in
`site/src/lib/explainers/word2vec/`; notes go in `notes/word2vec.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `presets.py` | chooses the preset mini-corpora from TinyStories by fixed rules; writes `static/models/word2vec/presets.json` | `npm run atlas -- py pipeline/word2vec/presets.py` |
| `twin.py` | the PyTorch fp64 twin of the TS trainer `sg` (`build(cfg)`, `loss(model, batch, cfg)`) | `npm run atlas -- parity word2vec` |

There is no `reference.py`: `atlas parity` would run it for every process source, and three of them (`one`, `run`,
`run2`) are the trainer itself wrapped as a process (see `notes/word2vec.md`, kit issues); the trainer is checked by the
twin instead.

Data: `pipeline/data/tinystories/TinyStoriesV2-GPT4-valid.txt` (already on disk; CDLA-Sharing-1.0). No downloads.
