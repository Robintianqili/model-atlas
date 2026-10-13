# pipeline/gaussian-process

Python side of the `gaussian-process` package (kind C: one process, `gp`). The page code lives in
`site/src/lib/explainers/gaussian-process/`; notes go in `notes/gaussian-process.md`.

Run everything through the CLI from `site/` (or a lane's `site/`): it uses the configured Python
(`site/scripts/atlas/atlas.config.local.json`, else `python` on PATH) and sets `PYTHONPATH=<repo>/pipeline`,
`ATLAS_SITE` and `ATLAS_ROOT`, so files land in the right site even inside a lane.

| File | What it does | Run |
|---|---|---|
| `reference.py` | an independent numpy run of the TS process `gp`: Gaussian process regression with an RBF kernel, using LAPACK's Cholesky and solves (`np.linalg`) instead of the hand-written TS loops; frames prior, kernel, one add per point, sample, evidence, done | `npm run atlas -- parity gaussian-process` (default inputs), `npm run atlas -- parity gaussian-process --fixtures --write` (every fixture; writes the badge record) |

The process reads plain numbers (`gp.ts` `dataOf()`: the points, ℓ, σf², σn² and the standard-normal draws of the sample
functions): every source passes them as its `input`, so the reference gets the exact TS draws through `--input` (a file,
`@<path>`, because the draws make the input longer than 8,000 characters) and never draws itself.

Tolerances: 1e-9 for the posterior, the weights, the marginal likelihoods and every metric (numpy's `exp` and LAPACK's
summation order differ from V8's in the last bits); 1e-6 for the sample functions, whose 81 x 81 grid covariance is
nearly singular (a jitter of 1e-6 σf² keeps it positive definite). Measured: max difference 2.05e-10 over 10 fixtures.

Rules: CPU only and TF32 off for references (importing `atlas_kit` does it). No downloads: every point set is a TS
preset or the learner's clicks.
