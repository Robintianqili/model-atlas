# Procedure: commands, checks, browser test, lanes

All commands run from `site/` (or a lane's `site/`) as `npm run atlas -- <command>`; `npm run atlas` lists them.
DEVKIT §12 has the generated command table; this page is the working subset.

## Contents

1. Set up and scaffold
2. Make the numbers
3. Write and lint the lesson
4. Check modes
5. The browser test (e2e)
6. The dev server
7. Lanes and merging
8. AK quick fixes

## 1. Set up and scaffold

```
npm run atlas -- doctor                                   # Node, packages, Python, Playwright, ports, git
npm run atlas -- new model <slug> --like <package> [--dev] [--port n]
npm run atlas -- new model <slug> --kind trace|train|process[,...] --input <adapter> [--template <name>]
npm run atlas -- new concept <slug> [--kind ...] [--input <adapter>] | new concept <slug> --like <package>
npm run atlas -- new view <key> --local <slug>            # a local view (meta.ts + View.svelte)
npm run atlas -- new op <name> --local <slug> --like <op> # a local op (TS + test + PyTorch mirror)
```

- The slug must be in the catalog (`src/lib/catalog/models.ts`, or `concepts.ts` for a concept; AK701). `--dev` makes a
  lab-only package for experiments.
- `new` prints the package's dev-server port (also in `notes/<slug>.md`) and a numbered "Next:" list. It never
  overwrites a file.
- `--like` renames the source package everywhere (`hello-xor`, `helloXor`, `hello_xor`) and leaves out checkpoints,
  metrics and data. Rewrite every claim of the copied `claims.test.ts`: they still state the old package's facts.
- Templates: `drawing-classifier` (A), `char-seq` (A + C), `points2d-train` (B), `series-train` (B + C), `graph-node`
  (B), `process-blank` (C).

## 2. Make the numbers

| Kind | You write | Commands |
|---|---|---|
| A, a trained model | `pipeline/<slug>/model.py`, `train.py`, `export.py` (`describe()` with builder calls) | `py pipeline/<slug>/train.py`, then `export <slug>` (export, `parity --write`, `lesson draft`) |
| B, live training | `trainer.ts` (a `TrainerDef` on the kit's autograd); optional `twin.py` | `run <slug> <source> --frames 0..20`, `parity <slug> --write` |
| C, a process | `process.ts` (a `ProcessDef` generator; numbers in each frame's `metrics`); optional `reference.py` | `run <slug> <source> --frames 0..40`, `parity <slug> --write` |

- Training runs with a fixed seed and the configured Python. Put measured numbers in `facts` (`facts.json`); narration
  reads them as `{fact:<key>}`.
- No downloads: Quick, Draw!, TinyStories, the karate club and the tokenizer files are local. A new dataset needs the
  user's OK first.

## 3. Write and lint the lesson

```
npm run atlas -- lesson <slug> draft            # the auto plan and its step ids
npm run atlas -- lesson <slug> lint --cap 200   # every step x every fixture x en/zh; --cap only while iterating
npm run atlas -- lesson <slug> print [--lang zh] [--fixture id]   # every narration, resolved, as tables
```

Lint errors: AK401 AK403 AK404 AK405 AK406 AK409 AK411 AK501 AK502; warnings: AK402 AK407 AK410 AK412. Lint cannot
judge meaning: read the print as a learner would.

## 4. Check modes

| Command | Runs | When |
|---|---|---|
| `check <slug> --quick` | schema, parity, op tests, lesson + claims, determinism, honesty, budgets, types (svelte-check of the package, 0/0), paperwork, ownership; parity and op tests from `.atlas/<slug>/gate-cache.json` while their inputs are unchanged | while iterating |
| `check <slug>` | the same without the cache, + site (`check:catalog`, a package-only build into `.atlas/build-<slug>/`, `check:build`) + e2e (default passes) | once, at the end |
| `check <slug> --fast` | the quick gates without types and without the cache | to distrust the cache cheaply |
| `check a b` | the packages' gates, one package-only build, one e2e sequence | a builder with two packages |
| `check --all [<slugs>] [--e2e a,b\|all]` | every package's fast gates, the kit gates (honesty, docs, all Node tests, legacy parity), `npm run check`, one full build, one e2e sequence (`--passes all`) | integration, once |
| `build <slug>[,<slug>]` | the package-only build alone (`ATLAS_ONLY`) | to look at built pages |

- The cache does not count `lesson.ts`, `claims.test.ts`, `.svelte` or `.md` files (unless a package test imports
  them). A lesson fixture you add is tested by parity and the op tests only in the full check.
- `--only gate,gate` picks gates; `--json` prints the report (also `.atlas/<slug>/check.json`). A failed site step
  keeps its output in `.atlas/build-<slug>.<step>.log`.
- Typical times (busy machine): `--quick` after a lesson edit 10-25 s, cold 50-220 s; e2e with 2 passes 60-120 s on a
  reused server; the full check 3-9 min (the package-only build is most of it); `check --all` about an hour. Every
  shell command must stay under 9 minutes (CONTRACTS rule 4), so run a full check or `check --all` in the background
  with its output in a log file, and wait for it to exit.

## 5. The browser test (e2e)

```
npm run atlas -- e2e <slug>                     # default passes: desktop-en-light, phone-zh-dark (+ a 360 px scroll probe)
npm run atlas -- e2e <slug> --axe warn          # while iterating: axe reports without failing
npm run atlas -- e2e <slug> --passes all        # 5 passes: + narrow-en-dark, desktop-zh-light, desktop-en-light-reduced
npm run atlas -- e2e a,b                        # several packages in one sequence
npm run atlas -- e2e --stop-server              # stop the kept dev server
```

- Pass names: `<desktop|phone|narrow|px>-<en|zh>-<light|dark>[-reduced]`; presets `default`, `all`, `full`.
- It opens the Lesson Lab (dev packages), `/m/<slug>/`, or `/train/<slug>/` (concept packages); `--url <path>` opens
  another page.
- Gates: AK601 (console error), AK602 (horizontal scroll at 360 px), AK603 (a long task: over 100 ms fails, over 50 ms
  warns), AK604 (axe serious or critical), AK502 (a stage without a badge).
- Output, replaced on each run: `notes/shots/<slug>/e2e-<pass>-<nn>.png` and `notes/shots/<slug>/contact/`:
  `<pass>.png` (every step on one labelled sheet), `lesson-print.txt`, `axe.txt`, `axe-<pass>.json`.
- Training is capped at 200 steps (`?atlas-e2e=1`), so a `live-model` step can show other numbers than your browser;
  `--cap off` runs the full length.
- AK603 is load-sensitive: the same page shows 2-3 times longer tasks while other processes keep the CPU busy. Re-run on
  a quieter machine before you chase a 50-100 ms warning; move real work to the worker.

## 6. The dev server

- `e2e` and `check` reuse a dev server of this site: the one they kept (`.atlas/dev-server.json`), or one on the
  package's port that answers the identity probe (`.svelte-kit/atlas-dev-probe.txt`). Else they start one and keep it.
- A kept server stops itself after 20 idle minutes; `e2e --stop-server` stops it now; `--no-keep-server` stops it when
  the command ends. A server that is not kept exits when the atlas process dies.
- Builds may run beside it. On a locked file the site gate stops a server it kept and retries once.
- Your own `npm run dev -- --port <port>` is fine for looking at the page; stop it (and on Windows its `vite.js dev`
  child) before you finish.

## 7. Lanes and merging

```
npm run atlas -- worktree <name> --from-git HEAD   # a lane in site/.work/<name>/site from the last commit
npm run atlas -- worktree <name> --sync [--from-git <rev>]   # refresh the non-owned files (AK703 if you edited one)
npm run atlas -- merge <name> --dry-run            # AK702 for any change outside the owned paths
npm run atlas -- merge <name>                      # main session only
```

- A model lane owns `pipeline/<slug>/**`, `site/src/lib/explainers/<slug>/**`, `site/static/models/<slug>/**`,
  `notes/<slug>.md`, `notes/shots/<slug>/**` (`--owns a,b` for anything else).
- `--from-git` takes the files from a commit, so another workflow editing the main tree cannot leak half-done work into
  the lane. The local CLI config (`scripts/atlas/atlas.config.local.json`) is copied, never synced or merged.
- Python writes `pipeline/<slug>/` in the main tree directly; its assets go to the lane's `static/` (`ATLAS_SITE`).
- Delete a lane right after it is merged (each costs 1-4 GB with its builds).

## 8. AK quick fixes

| Code | Usual cause | Fix |
|---|---|---|
| AK201 | a missing attribute in `describe()` (mask, scale, eps, padding) or a different layout | compare layer by layer; fix `describe()` or teach the op |
| AK203 / AK204 | the twin or reference differs from the TS (activation, reduction, tie-break, draw order) | `run <slug> <source> --frames a..b` on both sides; break ties to the smallest index; `recordDraws` |
| AK401 | a placeholder has no value at that frame, or the step ran on the empty input | put the number in that frame's `metrics`, bind it in `vars()`, write the idea in sentences without numbers |
| AK402 | a number typed into narration | a placeholder or `{fact:<key>}` |
| AK404 | a fixed cursor index that a short input does not reach | a cursor query `{ phase, step, last }` |
| AK406 | a label without Chinese | add `zh` |
| AK407 | lesson shape | 5-12 steps, a predict, a task, a limits step, 2-4 sentences, ≤ 90 words |
| AK410 | two different numbers print the same | narrate the difference, or `{name\|n4}` |
| AK412 | a predict step gives the answer away | move the answer to `predict.explain` |
| AK503 | a `$lib` import, a missing `.ts` extension, an `enum`, `Math.random` in a pure file | relative `.ts` imports from `src/lib/atlas/**`, erasable TypeScript, the kit's `rng` |
| AK602 | a wide element at 360 px | `grid-template-columns: minmax(0, 1fr)`; let the wide view scroll in its own box with `use:keyScroll` |
| AK603 | heavy work on the main thread | leave `run` unset (the worker); yield every ≤ 8 ms; re-run on a quiet machine first |
| AK604 | contrast, an unreachable scroll box, an svg with buttons, an ARIA table, a `<dl>` | [charts-and-a11y.md](charts-and-a11y.md) |
| AK702 | a change outside the owned paths | undo it; a kit bug goes under "Kit issues" |
