# pipeline/tiny-tcn

Python side of the `tiny-tcn` package (kinds: train, process). The page code lives in `site/src/lib/explainers/tiny-tcn/`;
notes go in `notes/tiny-tcn.md`. Nothing is trained here and no weights ship: the TCN trains live in the browser from a
seeded start (TS is the source of truth for data and init), and ARIMA-lite is fitted in the browser too.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `twin.py` | the PyTorch float64 twin of trainer `tcn` (four `nn.Conv1d` with kernel 2 and dilations 1, 2, 4, 8 after a left-only `F.pad`, ReLU, residual blocks from layer 2, a 1x1 head) | `npm run atlas -- parity tiny-tcn` |
| `reference.py` | an independent Python run of process `arima` (the series, the ARIMA(p, 1, q)-lite fit, the forecast) in plain Python floats with the TS loop order | `npm run atlas -- parity tiny-tcn` (`--fixtures` on every fixture) |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No new downloads: every series is
synthetic (`site/src/lib/atlas/data/series.ts`).
