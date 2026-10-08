# pipeline/gan-2d

Python side of the `gan-2d` package (kind B: a GAN trained live in the browser). The page code lives in
`site/src/lib/explainers/gan-2d/`; notes are in `notes/gan-2d.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch fp64 twin of the TS trainer `gan` (`build(cfg)`, `loss(model, batch, cfg)`, `optimizer(model, cfg)`: two optimizers, one per net, stepping the net whose turn it is) | `npm run atlas -- parity gan-2d` |
| `export.py` | writes `static/models/gan-2d/presets.json`, the preset list of the input box (the points are made in TS) | `npm run atlas -- py pipeline/gan-2d/export.py` |

No weights ship: every run starts from the seeded init in the browser. No data is downloaded; every point set is a
seeded generator (`site/src/lib/explainers/gan-2d/data.ts`) or the kit's ring and smiley.
