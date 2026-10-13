# pipeline/fno

Python side of the `fno` package (kind A: a tiny Fourier neural operator for the heat equation on a ring of 64 points;
kind C: the numerical solver's Python reference). The page code lives in `site/src/lib/explainers/fno/`; notes go in
`notes/fno.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python and sets
`PYTHONPATH=<repo>/pipeline`, `ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| (TS) `explainers/fno/process.ts` `data` | makes the training pairs with `heat.ts`: a random starting curve (waves, pulses, bumps or lines) and the solver's curve 50 steps later; 4,000 training, 400 validation and 400 test pairs | `npm run atlas -- run fno data --opt '{"nTrain":4000,"nVal":400,"nTest":400}' --save` (writes `.atlas/fno/data.tape.*`) |
| `model.py` | the FNO: lift (1 -> 4), two Fourier layers (FFT, the lowest 16 modes, complex weights `wr` + i `wi`, inverse FFT, plus a pointwise linear path; GELU after the first), projection (4 -> 1); `keep` cuts the modes at run time | `npm run atlas -- py pipeline/fno/model.py` (smoke test) |
| `train.py` | reads the TS tape (CRC-32 of u0 and uT checked), trains on the GPU (seed 0, Adam 1e-2 cosine, batch 64, 6,000 steps, about 1 minute), keeps the best validation checkpoint; `model_final.pt`, `train_metrics.json` (errors by kept modes, on 3 times taller curves) | `npm run atlas -- py pipeline/fno/train.py` |
| `ops_fno.py` | Python mirrors of the local ops rfft, keep_modes, spectral_mix, irfft | used by `export.py` |
| `export.py` | `describe()` + `ak.export_model(...)`: spec.json, weights.bin, reference.json (test pairs of every family, cut to 1, 4 and 8 modes, flat, top, spike), facts.json | `npm run atlas -- export fno` |
| `reference.py` | the `solve` process in numpy float64 (the same finite-difference arithmetic) for `atlas parity` | run by `npm run atlas -- parity fno` |

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it; AK102 otherwise). No downloads: every curve
is synthetic and made in TS. Each file <= 5 MB, each model <= 15 MB (AK105).
