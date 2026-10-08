# Afternoon test: hello-xor-2 (newcomer, DEVKIT.md only)

The record of the DevKit's afternoon test (2026-10-06, DevKit wave 1b): a fresh agent that had only the guide built a
small lesson package from the scaffold to a draft that passes every automated gate. The guide and the kit were fixed
after the test (the findings at the end were addressed before the next wave), so some details describe the DevKit as
it was then. The /devkit/ start page reads its summary line from this file.

Task: build `hello-xor-2` as a dev draft: a 2-2-1 MLP on XOR with a SIGMOID output (hello-xor uses a tanh output),
following `site/DEVKIT.md` step by step, until `atlas check hello-xor-2` passes in draft/dev mode.
Only documentation used: `site/DEVKIT.md` and the files it points to (the hello-xor package, KIT.md if needed).

Durations are wall-clock times measured with `date` while the test ran. All commands run from `site/` unless noted.

## Step 0: read DEVKIT.md (11 s)

- Read sections 1-17 (1084 lines). For a human newcomer that is about 25-30 minutes of reading.
- The scaffold example in section 4 uses `my-xor`; I substitute `hello-xor-2`.

## Step 1: set up / doctor (9 s)

- `npm run atlas -- doctor` -> `doctor: OK` in 3.5 s. All 10 checks ok (python from atlas.config.local.json, torch 2.11).
- Could not write this log file in place with the agent's editor tool (a permission quirk of the test computer).
  DEVKIT.md section 17 does mention this ("write into a scratch folder and copy with cp"), but only as the very last
  bullet of the last section; a newcomer who reads sections 1-4 first (as section 0 recommends) hits it before
  reading about it. Workaround: write in a scratch folder, `cp` into the repo. Cost: ~1 min.
  **Doc gap:** move that note up into section 1 (Set up).

## Step 2: scaffold (1 s)

- `npm run atlas -- new model hello-xor-2 --dev --kind trace,train,process --input xor-point` -> wrote 11 files, 1 s.
  Printed `note: adapter 'xor-point' is not a kit adapter: add it as a local adapter in adapters/xor-point/` (matches
  DEVKIT 4.1, good).
- Small oddities in the printed "Next:" list: every item of the first block is numbered `1.`; it suggests port 5210
  (I was given 5221; DEVKIT 11 says "anything else: 5210 and up (atlas new picks a free one)", so fine but note
  the scaffold's notes file also hard-codes 5210).
- The scaffold is NOT a hello-xor copy: the trainer is a 2-4-1 tanh MLP (hidden: 4, lr 0.1, snapshotEvery 10), the
  process is a coin-flip bias estimator, the model.py is a 2-4-2 MLP, export.py points at `runs/TODO/best.pt`
  (hello-xor uses `model_final.pt` and a `train.py` the scaffold does not create). DEVKIT 4.1 says "Each kind gets a
  working toy to replace", so this is as documented; but for a "hello-xor variant" it would be faster to have
  `--template` or `--like hello-xor`. Not a blocker.
- Read the worked answer (hello-xor: explainer.ts, lesson.ts, trainer.ts, process.ts, claims.test.ts, adapter, the 5
  pipeline files). ~1 min for me; for a human ~20 min.

## Step 3: copy the local adapter (40 s)

- `mkdir -p hello-xor-2/adapters/xor-point && cp hello-xor/adapters/xor-point/{adapter.ts,Capture.svelte} ...`
  (DEVKIT 4.1 says to copy them). Edited the header comments only.

## Step 4: kind A, train and export (1 min 25 s)

Design decision: sigmoid output = `sigmoid(fc2(tanh(fc1(x))))`, trained with binary cross-entropy (the natural pair
for a sigmoid output; computed from the logits with `binary_cross_entropy_with_logits`), Adam lr 0.05, 1,500 steps,
like hello-xor.
- Wrote `pipeline/hello-xor-2/model.py` (XorNet with `logits()` and a sigmoid `forward()`), `train.py`, `export.py`
  (`describe()`: input, fc1, tanh act1, fc2, `b.sigmoid("out", ...)`, as the DEVKIT 4.2 comment says).
- Ran train.py directly with the configured Python (`<python> pipeline/hello-xor-2/train.py --seed s`)
  for s = 0..3. Seeds 0, 2, 3: 4/4 correct, BCE ~3e-4. Seed 1: stuck (2/4, loss 0.347). Kept seed 0.
  **Doc gap (minor):** DEVKIT 4.2 says `python pipeline/hello-xor/train.py`, but on the test computer `python` on PATH
  was too old; there is no `npm run atlas -- py <script>` to run an arbitrary script with the configured Python. I used the
  configured Python's path directly (known from section 1's JSON).
- `npm run atlas -- export hello-xor-2` (10 s): PASS first try. 5 layers, 9 params, 6 cases,
  describe() vs module(x) max |diff| 0; parity kind A max |diff| 9.54e-7 at fc2. It also ran kind B parity and
  kind C parity against the scaffold's (still unchanged) twin.py and reference.py: both pass out of the box. Then the
  lesson draft: 15 auto steps. Very smooth.

## Step 5: kind B trainer and kind C process (2 min 15 s)

- Looked at `src/lib/atlas/autograd/nn.ts` and `loss.ts` (DEVKIT 6.1 names them) to see whether `MLP` can end in a
  sigmoid: it cannot ("No activation after the last layer"). DEVKIT 6.1 says the model may be "your own class with
  `params()` and `named()`", so I wrote `SigmoidMLP` (wraps `MLP('net', [2, h, 1], 'tanh')`, adds `op.sigmoid`,
  records `'out'`, and has a `toSpec()` built with `stackToSpec` so a `live-model` source can trace it).
  **Doc gap:** DEVKIT does not say that `MLP` has no output activation, nor how to add one; I had to read nn.ts.
  Also `toSpec()` (needed for `live-model`) is only discoverable from `trainer.ts`'s error text / tospec.ts; section
  6.2 says nothing about a custom module needing `toSpec()` for `live-model`.
- Loss: `bceWithLogits(m.logits(x), y)` (stable form), metric `correct`.
- process.ts: rewrote hello-xor's `sgdStep` as `sigmoidStep`: forward adds `z2` and `yhat = sigmoid(z2)`, BCE loss,
  backward starts from `(yhat - y) / 4`. Renamed ids (`xorSigmoid`, `sigmoidStep`) because I did not know whether
  process / trainer ids are global across packages. **Doc gap:** DEVKIT does not say whether ids are per-package.
- First `npm run atlas -- run hello-xor-2 train --frames 0..20`: AK301 "source 'train' runs process
  'train:helloXor2', which the package does not register / fix registered: sigmoidStep, train:xorSigmoid". Clear and
  correct (I had renamed the trainer but not yet the manifest). Fixed in explainer.ts in 1 min.
- Wrote the full explainer.ts (goals, prequestion, misconception, lessonNote, sources fwd/train/stuck/step1/live,
  presets, start point) modelled on hello-xor's.
- `npm run atlas -- run hello-xor-2 train --frames 0..10`: 302 frames in 14 ms; seed 1 ends at loss 0.00785,
  4/4 correct.
- Seed scan `--seed 2..8 --frames 301..301`: seeds 2, 4, 5, 8 get stuck (loss ~0.348, 2/4 right). Seed 2 with
  `--opt '{"steps":3000}'` still 0.3466; with `--opt '{"hidden":3}'` it reaches 0.0031 and 4/4. Same story as
  hello-xor, so I kept the "stuck" limits step (seed 2). `--opt` is only documented in the CLI notes of section 12,
  not in `atlas run` usage line; worked as expected.
- `npm run atlas -- run hello-xor-2 step1`: 9 frames, loss 0.6991 -> 0.6987 with lr 0.5.

## Step 6: twin and reference, parity (51 s)

- twin.py: `Sequential(Linear, Tanh, Linear)` + `logits()` + sigmoid forward; loss
  `F.binary_cross_entropy_with_logits(model.logits(x), y)`. reference.py: port of hello-xor's with sigmoid + BCE.
- `npm run atlas -- parity hello-xor-2` (8 s): PASS first try. A 9.54e-7; B sgd 2.2e-16, adam 6.7e-16; C max diff 1.1e-16.

## Step 7: lesson, claims (1 min 55 s)

- `npm run atlas -- lesson hello-xor-2 draft`: 16 auto steps. It says "17 steps in total" although the
  scaffold's author step `estimate-wobbles` still pointed at the removed source `run`; draft did not flag it (lint
  would have). Fine.
- Wrote lesson.ts (10 steps: 4 auto with title overrides incl. the new `op:out` sigmoid step, `answer` with a predict,
  `sgd-forward`, `sgd-update`, `train`, `live` with a task, `stuck` kindOf 'limits'), modelled on hello-xor.
- `npm run atlas -- lesson hello-xor-2 lint`: 0 errors, 0 warnings on the first try.
- `npm run atlas -- lesson hello-xor-2 print --fixture preset:p11`: read every narration. Found a teaching problem lint
  cannot see: "the loss falls to 0.699" when it started at 0.699 (0.6991 -> 0.6987; 3 significant digits hide the drop).
  There is no `n4` format (section 8.3 lists none, n3, int, pct, signed). Fixed by narrating the drop itself:
  `drop: loss - lossAfter` in `vars()` -> "falls by 3.76e-4 to 0.699". (A format with more digits, or a warning when two
  placeholders in one sentence render identically, would help.)
- claims.test.ts: 5 tests (stuck seed 2 / 3,000 steps / 3 hidden units; train seed 1 4/4; SGD rule + loss drop + the
  output gradient = sum of (p - y)/4, checked against a finite difference; training recipe matches spec.training;
  every preset on the right side of one half and out = sigmoid(fc2)). `node --test .../claims.test.ts`: 5/5 pass.

## Step 8: check --fast (10 s)

- `npm run atlas -- check hello-xor-2 --fast`: PASS in 9.4 s. schema, parity, op tests, lesson (claims 5 passed),
  determinism, honesty, budgets ok; paperwork warns AK501 x4 (notes TODOs; a warning for dev packages, as documented).

## Step 9: look at it in the Lesson Lab (1 min 30 s)

- `npm run dev -- --port 5221 --strictPort` (background), ready in 1.2 s; opened
  `http://localhost:5221/dev/atlas/lab/hello-xor-2/`. Lint panel 0/0, per-layer parity fc2 9.5e-7, out 2.2e-10, the
  overview shows the 5 layers incl. "Sigmoid: the probability", Code panel shows `out = nn.Sigmoid()`. No console errors.
- Deep link `#lesson-sgd-update`: narration and the formula render (the long formula scrolls inside its own box).
- **Kit issue found:** opening the deep link `#lesson-live` directly shows "This step could not be shown. source 'live'
  has no frames (status error: source 'train' is not a train source with frames)" and the narration shows "…".
  hello-xor's own `#lesson-live` deep link shows exactly the same error, so it is the kit (a live-model source entered
  before its train source has run), not my package. Stepping into it with Next from step 8 works (probability 0.991 for
  (1, 0)). Lint in Node passes because it enters every step. Not fixed (kit); noted under Kit issues.

## Step 10: notes, stop the server, full check (3 min 16 s)

- Filled `notes/hello-xor-2.md` (What, Decisions, Final numbers, How to re-run, Kit issues).
- Stopped the dev server before the build (DEVKIT 11: "stop your dev server before building (exFAT locks files)").
  **Machine gotcha:** stopping the background `npm run dev` killed npm but left the `node .../vite.js dev --port 5221`
  child listening (found with `netstat -ano`; stopped by hand, after checking its command line). DEVKIT does
  not say how to stop a dev server on Windows or that the vite child can outlive npm. Cost: ~1 min.
- `npm run atlas -- check hello-xor-2` (2 min 24 s): **PASS**, every gate:
  schema, parity (A 9.54e-7; B sgd 2.22e-16, adam 6.66e-16; C 1.11e-16), op tests, lesson (0/0; claims 5/5),
  determinism, honesty, budgets, site (svelte-check 18 s, check:catalog, build 62 s, check:build 4 s),
  e2e (4 passes x 10 steps, 0 errors, 20 screenshots in notes/shots/hello-xor-2/), paperwork "notes complete";
  ownership skipped (main tree, no lane).
- Looked at two screenshots (1280 light step 9, 360 dark step 6): the live step shows the probability and the task; the
  sigmoid formula wraps cleanly at 360 px.
- Small observation: the live step gave 0.991 for (1, 0) in my browser and 0.982 in the e2e screenshot: a
  `live-model` traces the train source's weights *as far as it has run*, so the number depends on timing. True to
  "the weights just trained in your browser", but a newcomer could be surprised; DEVKIT 6.2 says "the current weights"
  without saying they may be mid-run.
- Unclear but harmless: DEVKIT 12 says e2e "starts npm run dev on the package's port", but hello-xor-2 has no row in
  the port table and nothing tells me which port check/e2e picked (the scaffold suggested 5210). It worked; I did not
  have to know. `check` has no documented `--port` option (e2e does).
- Did not commit. `git status`: only this log, `notes/hello-xor-2.md` (the builder's notes), `pipeline/hello-xor-2/`,
  `site/src/lib/explainers/hello-xor-2/`, `site/static/models/hello-xor-2/` are new (plus git-ignored `.atlas/` and
  `notes/shots/hello-xor-2/`).

## Summary

- From the start of the reading to draft gates green: **14 min 12 s** wall clock (an agent; a human would spend most of an
  afternoon on the reading, the Python training and the lesson writing, but would not get stuck anywhere).
- Stuck: never more than ~1 minute at a time. Errors seen: 1 AK301 (clear, gave the fix) and 4 AK501 warnings
  (expected, clear). Zero failed gates in either check run.
- Files written: pipeline/hello-xor-2/{model.py, train.py, export.py, twin.py, reference.py, model_final.pt,
  train_metrics.json}, site/src/lib/explainers/hello-xor-2/{explainer.ts, lesson.ts, trainer.ts, process.ts,
  claims.test.ts, adapters/xor-point/*} (worker.ts and README.md from the scaffold, unchanged),
  site/static/models/hello-xor-2/* (by export), notes/hello-xor-2.md (the builder's notes).

### Where DEVKIT.md was unclear, wrong or missing (for the doc owner)

1. Section 17's note that an editor tool can fail to write a file in place (write in a scratch folder and `cp`) is the
   last bullet of the last section; it bites at minute 1. Put it in section 1.
2. Section 4.2 says `python pipeline/hello-xor/train.py`; on the test computer `python` on PATH was too old. Say "the
   configured Python (section 1)" or offer `npm run atlas -- py <script>`.
3. Section 6.1 does not say that `MLP` has no activation after the last layer, nor how to add an output activation
   (sigmoid / softmax) for a trainer. Needed reading `atlas/autograd/nn.ts`.
4. Section 6.2's `live-model` needs the model to have `toSpec()`; a custom module must write one (via `stackToSpec` in
   `atlas/autograd/tospec.ts`). Not mentioned; found only by reading the code.
5. Not said whether trainer / process ids are global or per package (I renamed mine to be safe).
6. Section 8.3 formats: nothing between 3 significant digits and a hand-computed difference; two different values can
   read the same ("falls to 0.699" from 0.699). Lint cannot catch it; only `lesson print` reading did.
7. The scaffold for `--kind trace,train,process --input xor-point` produces generic toys (2-4-2 classifier, coin flip,
   `runs/TODO/best.pt`, no train.py) rather than hello-xor's structure, while section 4 says "like hello-xor". Expected
   per "a working toy to replace", but a `--like hello-xor` / template would save most of the edits.
8. `atlas new`'s "Next:" list numbers four items all as `1.`; it and the notes file suggest port 5210 even when the
   builder has another port.
9. The port check/e2e uses for a package not in the table is not stated; `check` has no `--port`.
10. Windows: stopping `npm run dev` can leave the vite child on the port; worth one line in section 17.
11. `live-model` values depend on how far the train source has run (timing), not stated.

### Kit issues seen (not fixed, not mine)

- Deep link straight to a `live-model` step (`#lesson-live`) fails with "source 'live' has no frames (status error:
  source 'train' is not a train source with frames)" in both hello-xor and hello-xor-2; stepping into it works and the
  e2e gate passes.
