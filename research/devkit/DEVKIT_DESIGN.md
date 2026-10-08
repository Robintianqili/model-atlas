# Model Atlas DevKit ("atlas kit"): final design

This is the final design, revised after review. It replaces design v1. Section 10 lists what changed and which review points were rejected.

I read the code at commit `23ee7c6` ("Round 3: kit Walkthrough, TinyCNN guided lesson, site fixes") while round-3 stage 2 was still running in `site/.work/{tinygpt,tokenization,next-token-prediction,transfer-learning}`. Nothing under `site/` was changed.

Facts this design relies on (each one re-checked against the code or the machine):

- **Runtime code:**
  - `tinycnn.ts` is a 7-op spec interpreter. Its loops accumulate in f64 and store f32. Conv and dense start the sum at 0 and add the bias **last**.
  - `tinygpt.ts` runs a hand-coded forward pass. Its `linear` starts the f64 sum **at the bias**.
  - The TinyGPT `spec.json` names 11 op types. It has no `input` layer (`tok_emb.in = ['input']`, `input.shape = ['T']`) and uses the top-level keys `context: 64` and `lnEps`. Its `linear_tied` layer carries `{ weight: 'tok_emb.weight', transposed: true }`, although the stored `[1024, 76]` matrix is already in `[out, in]` layout.
  - Layer ids already contain dots (`L0.attn.q`).
  - `kit/Walkthrough.svelte` takes:
    - `steps: WalkStep[]`;
    - `stage?: Snippet<[WalkStep, WalkContext]>`;
    - `aside`;
    - `ready`.

    Each step can have `waitForDone` and can call `ctx.done()`.
  - `ModelPageShell` has **15 snippet props** (`facts` … `credits`, `moreLinks`) and **13 section keys**.
  - `kit/tex.ts` fills `{{name}}` placeholders (`fillTex`). `sym(role, tex)` emits `\htmlClass{sym-role}{…}`.
  - `kit/spec.ts` `ModelSpec` types `outShape` as `number[]` and `parity.max_abs_diff` as `number`. `ProvenanceBadge` prints a tolerance whenever it is not `undefined`.
  - `I18n.zh` is optional.
  - Four separate worker stacks exist (tinygpt, tokenization, next-token, transfer), all built as `new Worker(new URL(…), { type: 'module' })`. Vite's default worker format is `iife`, which inlines dynamic imports.
- **Routes and build:**
  - Root `prerender = true`.
  - `/m/[slug]` generates entries only for non-live slugs and returns 404 for live slugs that have no static folder.
  - `server/assets.ts` excludes only `reference.json` from the measured download size.
  - `svelte.config.js` and `scripts/check-build.ts` hard-code `build`.
  - `check:catalog` fails on any `runtime`/`params`/`size` string that has no `TECH_ZH` entry.
  - The build has 192 pages, 101 of them planned model pages.
  - The `/m/tinycnn/` page shell is 297.9 KB gzip against a 300 KB budget (`notes/page-tinycnn.md`).
- **Pipeline:**
  - `pipeline/parity/tinycnn.parity.ts` and `tinygpt.parity.ts` **rewrite `spec.parity`, including today's date, in `site/static/models/*/spec.json` on every run**. The root layout embeds `parity.checked` into every page.
  - All pipeline scripts import `../../site/...` relative to their own file.
- **Reference data:**
  - TinyCNN: 3 reference inputs; `presets.json` has 10 × 6 = 60 class presets plus 2 × 6 held-out presets.
  - TinyGPT: the shipped reference covers T = 9, 22 and 37, with one `{shape, dtype, b64}` entry per tensor. The T = 1 and 64 references exist only in the git-ignored `verify_ref.json`.
  - TinyGPT's `mulberry32` already matches the Python sampler (4 of 4 `generate()` runs identical).
- **Environment:**
  - Python 3.10.11 with torch 2.11 cu128. `torch.backends.cudnn.allow_tf32` is `True` by default.
  - networkx 3.4.2. Its karate-club edges carry `weight` attributes.
  - No `jsonschema` in the venv and no `ajv` in the site. npm packages are pre-approved (LOG "Open items"), and CONTRACTS allows pip installs when needed.
- **Catalog:**
  - Every next-batch slug exists as `planned`.
  - Inputs: `gridworld` `['game']`, `pinn` `['series']`, `tinyclip` `['drawing', 'text']`.
  - `tinyforecaster` lists the planned `scalar-lstm` as a prerequisite.
  - The `mlp-playground` runtime says "port of TF Playground".
  - The `gcn` hook says "Edit a molecule".

### What came from the three candidate designs

| Source design | Kept | Changed or left out |
|---|---|---|
| Contributor-first | <ul><li>The PyTorch-student persona and the "afternoon test"</li><li>One CLI with coded errors (AKxxx)</li><li>Lanes with an ownership check</li><li>`promote` as the only writer of shared files</li><li>The rule of two for local ops</li><li>Per-slug ports</li><li>`doctor`</li></ul> | <ul><li>torch.fx auto-capture is deferred (wrong teaching granularity; fails on data-dependent code)</li><li>No top-level `devkit/` folder (exFAT: no symlinks; Node resolves packages upward)</li><li>The CLI shrank to 12 commands (§4.12)</li></ul> |
| Spec-driven auto-explainer | <ul><li>Op registry with `explain(unit)`</li><li>Zoom views chosen by kind of arithmetic</li><li>Displays chosen by axes</li><li>A Python `SpecBuilder`, so the graph is written once</li><li>NaN-safe compare at the extreme values of each symbolic dim</li><li>`hello-xor`</li><li>Exporting a live model to a spec</li></ul> | <ul><li>Spec renames became an additive spec 1.1, trimmed to what the batch needs</li><li>The planner lost pattern collapsing and budget folding</li><li>The binary reference format was dropped for TinyGPT's existing base64 layout</li></ul> |
| Process / lesson-first | <ul><li>The lesson is the primary artifact</li><li>`ProcessDef` generators record into a `Tape`</li><li>Inference is a special case of a process</li><li>`trainingProcess` has coarse and fine phases</li><li>Lessons and runtimes are pure TS, so Node can dry-run them</li><li>`facts.json`</li><li>FocusBus</li><li>`Rng.fork`</li><li>Synthetic data flows TS → Python</li></ul> | <ul><li>Claims became ordinary per-package tests</li><li>No recipe enum</li><li>No `facts.ts`</li><li>One package folder per model, found by glob</li></ul> |

---

## 1. Goals and non-goals

### Goals

1. **Teaching first.** Every illustration is a guided lesson that runs on the learner's own input and shows real numbers. The kit turns the brief's rules into code and lint:
   - one idea per step;
   - every number has provenance;
   - predict-first questions and "your turn" tasks;
   - honest limits.

   The Walkthrough stays the primary path through a page (KIT rule 15).
2. **Fast to build the next models (dogfooding).**
   - Each model in the next batch is built with the kit and needs **≤ 1,000 lines of page-specific TS/Svelte**, not counting Python or tests.
   - Today's hand-built page folders are much larger: about 7.3k lines for TinyCNN and 4.8k for TinyGPT.
   - A new kind-A model whose ops already exist needs **no Svelte at all**.
3. **One lesson format for three kinds:**
   - (A) inference traces of a trained model;
   - (B) live in-browser training;
   - (C) algorithmic step traces.

   All three share one tape model, one player, one view contract, one narration language and one provenance system.
4. **Honest by construction:**
   - Narration numbers come from placeholders bound to live state or to an exporter-written `facts.json`.
   - Factual sentences are backed by per-package claim tests.
   - Parity values reach `spec.parity` only through `atlas parity`.
   - `Math.random` is banned outside `atlas/core/rng.ts`.
5. **Open to other contributors.** The user's goal is that others can join. So the kit ships:
   - `site/DEVKIT.md` (how to build an illustration);
   - `CONTRIBUTING.md` (the external path through git branches and PRs);
   - `pipeline/requirements.txt`;
   - machine-neutral defaults (Python from `PATH`, plus a git-ignored local override);
   - a Node-only path for kind B and kind C.

   A LICENSE file waits for the user's choice (decision D9). Nothing is pushed until the user asks.
6. **Fits the project's constraints:**
   - fully static, with no third-party requests;
   - bilingual `{ en, zh }`;
   - Node 24 type stripping (erasable syntax only; explicit `.ts` imports);
   - exFAT quirks;
   - the CONTRACTS asset and trace formats, which are extended but never broken;
   - size budgets;
   - `licenses.json`.
7. **No breakage:**
   - `/m/tinycnn/`, `/m/tinygpt/`, the three concept pages and all 192 prerendered pages keep working.
   - Their source files are not changed by the kit, and their parity scripts stay green.
   - The kit never rewrites their shipped assets.

### Who the contributors are

| Contributor | Writes | Needs |
|---|---|---|
| Our build agents (waves 1–3), the first and main users | everything | the full kit; must stay within owned paths |
| PyTorch student ("Mei"), kind A | `pipeline/<slug>/*.py`, `explainer.ts` and `lesson.ts` (typed data) | Python ≥ 3.10 with CPU torch and Node 24; no Svelte |
| JS or teaching contributor, kind B or C | `trainer.ts` or `process.ts`, `lesson.ts`, perhaps one view | Node 24 only. A Python twin is optional and gives a stronger badge |
| External contributor (after the repo is public) | one package on a git branch | `CONTRIBUTING.md`, `npm ci`, `pip install -r pipeline/requirements.txt` |
| Maintainer (main session) | `src/lib/atlas/**`, shared ops and views, merges, `promote` | runs `check --all` |

### Afternoon test

A contributor can follow `site/DEVKIT.md` from `atlas new` to a draft that passes every automated gate in **≤ 4 hours**. We test this at the wave-1 exit: a fresh agent that has only the docs builds `hello-xor-2`, an XOR variant with a sigmoid output.

### Non-goals for v1

- Hub/ONNX runtimes. The `GraphModel` interface leaves room for an ORT backend.
- torch.fx auto-capture, `sequential()` and `transformer_block()` helpers. Models write their blocks with plain builder calls in a loop.
- **Concept packages.** There is no `AutoConceptPage`, no `atlas new concept` and no change to `/train/[slug]`. None of the next batch is a concept, and that route serves 69 planned pages. The manifest's `type` is `'model'` only; concepts come after wave 2.
- GPU/WebGPU kernels, a visual lesson editor, and npm or PyPI publishing. `atlas/` imports no `$app` and no kit outside `ui/`, so it can be extracted later.
- Migrating the hand-built TinyCNN/TinyGPT pages. That is optional, parity-gated, and comes after wave 3.
- Any GitHub push or CI. CI is designed (§6.5) but switched on only when the user agrees to push.

---

## 2. Three kinds of illustration, one lesson format

### 2.1 The six nouns

```
learner input ──adapter.preprocess──▶ Preprocessed (layer 0, shown as numbers)
        │
        ▼
SOURCES (declared in explainer.ts)        kinds: trace | scan | train | process | tape | live-model
   │ produce
   ▼
TAPE  = recorded Frames {phase, step, sub, state, changes, focus, metrics, caption, draws}
        keyframes + changes; derived fields recomputed on demand; state(i) is pure in i
   ▲ trace/scan: main thread (ms). train/process: a SourceRunner — in-process (Node, tests)
   │ or the package's own worker (browser), streaming frames into a TapeStore on the main thread
   │
LESSON (lesson.ts, pure TS) = ordered STEPS
   step = { cursor into a source, view key, narration template, focus, provenance, [predict] [task] [compare] }
   │ compileLesson()
   ▼
LessonPlayer.svelte ──▶ kit Walkthrough (unchanged) ──▶ StageHost ──▶ VIEW (registry key, lazy)
                                                        ▲ the lesson's own Selection  ▲ FocusBus (linked highlight)
```

A package arrives at the runtime as one `PackageModules` object (§4.7). In the browser it is built by a Vite glob loader; in Node, by a path-based loader. That object is the only way package code reaches the runtime.

How each kind maps onto the tape:

| | A: inference trace | B: live training toy | C: algorithmic process |
|---|---|---|---|
| Producer | `GraphModel.forward` (spec + op registry) → `traceToTape` (one frame per layer). An op unit scan (`scanTape`) gives one frame per output element | `trainingProcess(TrainerDef)` on the autograd core | a hand-written `ProcessDef` generator |
| Frames | layer frames; scan frames per cell, head, time step or edge | coarse: a snapshot every N steps. Fine: one step as `forward → loss → backward → update` | one per event: an update, merge, message, Euler step or pull |
| Runs in | main thread | the package's worker; in-process in Node; precomputed tapes for phones | main thread or worker; Node for precompute |
| Parity | every traced id and sub-step ≤ 1e-4 against PyTorch fp32 on CPU; `explain()` equals the trace | against a PyTorch fp64 twin: <ul><li>SGD for 50 steps, ≤ 1e-10 relative</li><li>the page's own optimizer for 50 steps, ≤ 1e-7 relative</li><li>step-1 gradients (and double gradients), ≤ 1e-10 relative</li></ul> | against a Python reference tape: discrete fields exact, floats within the field's `tol`. Random draws are replayed from the TS tape |
| Batch examples | TinyViT, char-RNN, GCN inference, TinyCLIP towers | MLP playground, flow-2d, forecaster, PINN, GCN retrain | gridworld Q-learning, bandits, Euler sampling, rollout, message-passing scan |

**Relative** means |a − b| ≤ bar · max(1, |b|).

Combinations are normal: `explainer.ts` declares several sources.

- flow-2d = B + C (training + Euler sampling) + A on the live weights;
- char-rnn = A + C (unroll and sampling);
- gcn = A + B.

### 2.2 The same `StepDef` for all three kinds

These examples type-check against §4.

**Kind A.** The kernel slides over the drawing. The view is the generic weighted-sum zoom, played over a scan source.

```ts
{ id: 'slide', source: 'scan1', view: 'zoom.weighted-sum', provenance: 'live',
  title: { en: 'One kernel slides over your drawing', zh: '一个卷积核在你的画上滑动' },
  select: (rt) => ({ layer: 'conv1_1', ch: rt.num('layer.conv1_1.strongestChannel') }),
  play: { from: 'first', to: 'last', interval: 14 },
  narration: {
    en: 'A kernel is {n|int} learned numbers. At each stop the window is multiplied by it, the {n|int} products are added, and a bias is added: {s} + {b} = {y}.',
    zh: '卷积核就是 {n|int} 个学来的数。每到一处，窗口和它逐个相乘，{n|int} 个乘积相加，再加上偏置：{s} + {b} = {y}。',
    vars: (rt, at) => ({ n: at.unit!.terms!.length, s: at.unit!.values.s, b: at.unit!.bias!, y: at.unit!.result }) } }
```

**Kind B.** One training step, slowed down. The `sgd1` source is the same trainer with `optimizer: 'sgd'`, so the arithmetic in the narration is the arithmetic that ran.

```ts
{ id: 'update', source: 'sgd1', at: { phase: 'update', step: 0 }, view: 'train-step', provenance: 'live',
  title: { en: 'Nudge every weight a little', zh: '把每个权重推动一点点' },
  focus: () => [{ field: 'params', key: 'net.0.weight', index: [0, 1], role: 'param' }],
  narration: {
    en: 'This weight was {old}. Its gradient is {g}, so with learning rate {lr} it becomes {old} − {lr} × {g} = {new}.',
    zh: '这个权重原来是 {old}。它的梯度是 {g}，学习率是 {lr}，所以它变成 {old} − {lr} × {g} = {new}。',
    vars: (rt, at) => updateOf(at, 'net.0.weight', [0, 1]) } }   // train/helpers.ts: { old, g, lr, new }
```

**Kind C.** One Q-learning update, with live numbers.

```ts
{ id: 'target', source: 'q', at: { phase: 'update', where: (f) => f.metrics!.r !== 0 }, view: 'zoom.formula', provenance: 'live',
  title: { en: 'The target: reward now plus the best guess for later', zh: '目标：眼前的奖励加上对以后的最好估计' },
  narration: {
    en: 'The agent got {r}. The target is {r} + {gamma} × {next} = {target}.',
    zh: '智能体得到奖励 {r}。目标是 {r} + {gamma} × {next} = {target}。',
    vars: (rt, at) => ({ r: at.frame!.metrics!.r, gamma: rt.num('opt.gamma'), next: at.frame!.metrics!.next, target: at.frame!.metrics!.target }) } }
```

What the kit adds to every step automatically:

- the provenance badge;
- deep links (`#lesson-<id>`);
- keyboard control, reduced motion and live-region announcements (from Walkthrough);
- the clamped cursor (a new, shorter input never crashes a step);
- "Computing… k / n" while a live source catches up;
- the view's `describe()` text for screen readers.

---

## 3. Architecture and package layout

### 3.1 Folders (all new unless marked)

```
model-atlas/
  CONTRACTS.md                       + additive section "DevKit additions" (wave 0)
  CONTRIBUTING.md                    external-contributor path (W1-E2; inactive until the repo is pushed)
  LICENSE                            only after the user picks one (D9)
  .gitignore                         (wave 0) + site/.atlas/, site/scripts/atlas/atlas.config.local.json
  research/devkit/DEVKIT_DESIGN.md   this document
  pipeline/
    requirements.txt                 torch (a CPU wheel is enough), numpy, networkx; Python >= 3.10
    atlas_kit/                       Python SDK (PYTHONPATH=pipeline; templates add a sys.path fallback)
      __init__.py paths.py numerics.py spec.py export.py refs.py rng.py tape.py
      train_reference.py validate.py errors.py
      ops/            Python mirrors of CORE_OPS, checked against site/src/lib/atlas/schema/ops.json
      data/           quickdraw.py (wraps pipeline/tinycnn/common.py loaders), tinystories.py, karate.py
    atlas_fixtures/
      ops/make.py       torch op goldens   -> $ATLAS_SITE/tests/atlas/ops/
      autograd/make.py  torch fp64 cases   -> $ATLAS_SITE/tests/atlas/autograd/
    parity/*.ts                      legacy scripts; wave 0 adds a --no-write flag, nothing else changes
    <slug>/                          model.py train.py export.py [twin.py] [reference.py] [ops_<op>.py] README.md
  site/
    package.json                     + "atlas": "node scripts/atlas/cli.ts", "check:atlas" (no new dependencies)
    svelte.config.js                 (wave 0) adapter pages/assets = process.env.ATLAS_BUILD_DIR ?? 'build'
    scripts/check-build.ts           (wave 0) same env var; samples a planned page and both flagship pages
    DEVKIT.md                        contributor guide (KIT.md stays the component reference)
    scripts/atlas/                   cli.ts root.ts load.ts commands/*.ts templates/<name>/**
                                     atlas.config.json ({"python": "python"}) + git-ignored atlas.config.local.json
    tests/atlas/                     generated fixtures (ops/, autograd/), inside site/ so lanes see them
    src/lib/server/assets.ts         (wave 0) NOT_LOADED also matches reference.* and *.ref.*
    src/lib/atlas/                   the DevKit runtime; pure TS except ui/** and adapters/*/Capture.svelte
      types/                         §4 (wave 0; frozen) + contract.test.ts
      core/       tensor.ts rng.ts io.ts compare.ts format.ts tex.ts (sym() only; no KaTeX import)
      ops/        registry.ts index.ts (CORE_OPS) manifest.ts <op>.ts *.test.ts
      graph/      executor.ts explain.ts receptive.ts refs.ts legacy.ts
      schema/     spec.ts ref.ts facts.ts tape.ts (hand-written validators) ops.json (generated by W1-A)
      autograd/   var.ts ops.ts check.ts nn.ts optim.ts loss.ts tospec.ts
      train/      trainer.ts metrics.ts helpers.ts
      data/       points2d.ts series.ts (seeded; TS is the source of truth for synthetic data)
      engine/     protocol.ts host.ts client.ts runner.ts
      process/    record.ts tape.ts (TapeStore) io.ts adapters.ts library/{sampling,euler,rollout}.ts
      lesson/     define.ts narrate.ts facts.ts plan.ts compile.ts lint.ts fixtures/
      runtime/    runtime.ts sources.ts selection.ts local.ts (localRunner) fake.ts (fakeRuntime for demos/tests)
      adapters/<id>/  adapter.ts (pure) + Capture.svelte
      ui/         LessonPlayer StageHost TapeScrubber AutoModelPage AutoOverview AutoInspector .svelte
                  runtime.svelte.ts focus.svelte.ts registry.ts packages.ts kitspec.ts strings.ts
                  train/session.svelte.ts
                  views/<key>/{meta.ts, View.svelte}
    src/lib/explainers/<slug>/       ONE package per model (contributor-owned; found by glob)
      explainer.ts lesson.ts [trainer.ts] [process.ts] [worker.ts] [ops.ts + ops/<op>.ts] [claims.test.ts]
      [views/<key>/] [adapters/<id>/] [sections/<Snippet>.svelte] [content/{En,Zh}.svelte]
    src/routes/m/[slug]/             (changed) +page.server.ts (new), +page.ts, +page.svelte
    src/routes/dev/atlas/            view gallery (demo data, badged Illustrative)
    src/routes/dev/atlas/lab/[slug]/ Lesson Lab (no prerendered entries unless ATLAS_INCLUDE_DRAFTS=1)
    src/routes/dev/atlas-train/      the training engine's performance page
    static/models/<slug>/            assets (CONTRACTS format + 1.1 additions)
    .atlas/                          git-ignored: reports, parity tables, fixtures for Python twins, builds
  notes/<slug>.md  notes/review-<slug>.md  notes/devkit-<pkg>.md   (notes/shots/ stays git-ignored)
```

**Why packages live in `src/lib/explainers/<slug>/`:**

- No new alias is needed.
- Node and Vite resolve `svelte` upward from `site/node_modules`.
- Discovery is a glob.
- Adding a model edits no shared file.

**Path resolution.** The repo is used from the main tree and from lane copies at `site/.work/<name>/site`.

- `scripts/atlas/root.ts` and `atlas_kit/paths.py` resolve two paths:
  - `ATLAS_SITE`: the site folder the command runs in (a lane's copy, inside a lane);
  - `ATLAS_ROOT`: the nearest ancestor of `ATLAS_SITE` that contains `CONTRACTS.md` and `pipeline/`. This works from `site/.work/<name>/site`, where `site/..` has no `pipeline/`.
- The CLI passes both to Python as environment variables.
- No new script uses a `../../site` relative path.
- Node tests inside `site/` read their fixtures from `site/tests/atlas/`, relative to their own file.

**Dependency rules** (`atlas check` enforces them):

- These files import only from `src/lib/atlas/**` (outside `ui/`) and `src/lib/i18n/types.ts`:
  - `atlas/{core,ops,graph,schema,autograd,train,data,engine,process,lesson,runtime}/**`;
  - `adapters/*/adapter.ts`;
  - every package's `explainer.ts`, `lesson.ts`, `trainer.ts`, `process.ts`, `worker.ts` and `ops/*.ts`.

  The drawing adapter may also import `src/lib/models/tinycnn.ts`. No `.svelte`, no `$app` and no `$lib/kit`, not even `import type`: `PredictQuestion` is copied into `atlas/types/core.ts`.
- Syntax is erasable only: no `enum`, `namespace`, parameter properties or `import =`. Imports are relative with `.ts` extensions.
- The gate is mechanical: the Node dry run imports every pure file under type stripping, which rejects anything else.
- `ui/**` may import the kit.

### 3.2 What moves out of `tinycnn.ts` / `tinygpt.ts` (and what does not)

| Today | In the kit | When |
|---|---|---|
| `forward()`: a 7-op `if` chain over `spec.layers` | `graph/executor.ts` + `ops/*.ts`, **copied** with the same loop order | wave 1a |
| `conv2dValid`, `maxpool`, `dense`, `softmax` kernels | `ops/conv2d.ts` (adds `padding`), `maxpool2d.ts`, `linear.ts` (alias `dense`), `softmax.ts` (`axis`). Conv and linear get `accumulate: 'bias-last' \| 'bias-first'` (default `bias-last`, as `tinycnn.ts`) | wave 1a |
| `convDetail`, `poolDetail`, `denseDetail`, `softmaxDetail` | `OpDef.explain(unit) → UnitExplanation` | wave 1a |
| `receptiveField` | `OpDef.receptive` + `GraphModel.receptiveField` | wave 1a |
| `preprocessStrokes` / `preprocessDrawing` / `preprocessPhoto` | `adapters/drawing28/adapter.ts` **imports** them from `models/tinycnn.ts`; nothing moves | wave 1b |
| TinyGPT `linear`, `layernorm`, attention loops, `gelu`, embeddings | `ops/{linear,layernorm,attention_scores,causal_mask,attention_mix,gelu,embedding,position_embedding,add}.ts`. `legacy.ts` sets `accumulate: 'bias-first'` on TinyGPT's linears | wave 1a |
| `mulberry32` (three copies) | `core/rng.ts`. The old copies stay | wave 1a |
| `trainBPE` steps, `generate().steps` | `process/adapters.ts`: `bpeTape` and `generateTape`, wrappers that change nothing | wave 1a |
| `tinycnn_finetune.ts` hand-written gradients | stays | — |

**What `graph/legacy.ts` (`normalizeSpec`) does with the two shipped specs.** It works at load time and never rewrites files.

- **Any v1 spec** (no `atlas` key) is read as spec 1.1. `parity.tolerance` and `parity.passed` are read as `bar` and `pass`.
- **TinyCNN:**
  - It synthesizes `inputs: [{ name: 'input', adapter: 'drawing28', kind: 'drawing', shape: [1, 28, 28], axes: ['c', 'y', 'x'] }]`.
  - The explicit `input` layer becomes op `input` and reads that declared input.
  - `dense` resolves through its alias.
- **TinyGPT:**
  - It synthesizes `inputs: [{ name: 'input', adapter: 'text-bpe', kind: 'text', shape: ['T'], dtype: 'i32', axes: ['t'] }]` and `dims.T = { min: 1, max: spec.context }`.
  - String-valued `params.weight` and `params.bias` become `w` entries.
  - `linear_tied` becomes `linear` with `w.weight = 'tok_emb.weight'` and no bias. `transposed` is **dropped**, because the stored matrix is already `[out, in]`.
  - Every `linear` gets `accumulate: 'bias-first'`.
  - `causal_mask` stays a separate layer.

**How existing pages keep working:**

- Waves 1 and 2 change nothing in:
  - `src/lib/models/`;
  - `src/lib/pages/`;
  - `src/routes/m/tinycnn|tinygpt/`;
  - `src/routes/train/**`;
  - `static/models/{tinycnn,tinygpt}/`.
- The kit reads the shipped specs through `legacy.ts`.
- `/m/[slug]` keeps rendering "Planned" for every slug without an included package.
- Static route folders keep precedence.
- Legacy parity scripts run only with `--no-write` from the kit.
- After wave 3, a separate parity-gated task may turn `tinycnn.ts` into a thin wrapper over the kit.

---

## 4. Interfaces

The main session writes the TypeScript in §4.1–§4.10 into `site/src/lib/atlas/types/*.ts` **verbatim in wave 0**, plus `types/index.ts` and `types/contract.test.ts`. The types are frozen during waves 1 and 2. A change goes through the main session and is broadcast to every lane with `atlas worktree --sync`.

### 4.1 `types/core.ts`

```ts
import type { I18n, Text } from '../../i18n/types.ts';
export type { I18n, Text };
export type Provenance = 'live' | 'precomputed' | 'illustrative';
export type Kind = 'trace' | 'train' | 'process';
export type FArray = Float32Array | Float64Array;
/** CONTRACTS tensor (row-major; images [C,H,W]; sequences [T,D]). Training toys may use Float64Array. */
export interface Tensor { shape: number[]; data: FArray }
export interface IntTensor { shape: number[]; data: Int32Array; dtype: 'i32' }
export type AnyTensor = Tensor | IntTensor;
/** CONTRACTS trace, unchanged. Sub-steps are `${layerId}#${sub}` ('L0.attn.scores#h2', 'lstm1#f') and follow
 *  their layer in `order`. '#' never appears in a layer id; layer ids may contain dots. */
export interface Trace { order: string[]; t: Record<string, Tensor>; meta?: TraceMeta }
export interface TraceMeta { dims: Record<string, number>; weights: string; provenance: Provenance; ms?: number; where?: 'worker' | 'main' | 'node' }
/** inclusive cell box */
export interface Box { y0: number; x0: number; y1: number; x1: number }
export type FetchLike = (url: string) => Promise<{ ok?: boolean; status?: number; json(): Promise<unknown>; arrayBuffer(): Promise<ArrayBuffer>; text?(): Promise<string> }>;
/** Structural copy of kit PredictQuestion (atlas never imports the kit); ui/kitspec.ts asserts assignability both ways. */
export interface PredictQuestion { id: string; question: I18n; options: I18n[]; answer: number; explain: I18n }
/** mulberry32. next() in [0,1); int(n) = floor(next()·n); fork(label) = mulberry32((seed ^ fnv1a32(label)) >>> 0), independent of
 *  how much was consumed. normal(): Box–Muller, u1 = 1 − next(), u2 = next(), z = sqrt(−2 ln u1)·cos(2π u2), no caching.
 *  Python ak.Rng is bit-identical for next(), int() and fork() only. normal() and other draws that use log/cos/sqrt may differ in
 *  the last ulp across engines: processes that use them set RecordOpts.recordDraws, and Python references replay the recorded draws.
 *  Test vectors: mulberry32(42) → 0.6011037519, 0.4482905590, 0.8524657935, 0.6697340414; mulberry32(1) → 0.6270739406, 0.0027357212, 0.5274470400. */
export interface Rng { readonly seed: number; next(): number; normal(): number; int(n: number): number; fork(label: string): Rng; getState(): number; setState(s: number): void }
```

### 4.2 `types/spec.ts`: spec 1.1, a trimmed superset of CONTRACTS

Every v1 key keeps its meaning, and old specs stay valid. The kit readers (`specToLayers`, `ProvenanceBadge`, root-layout `liveSpecs`) get a `ModelSpec` built by `ui/kitspec.ts` (§5.1), never an `AtlasSpec` directly.

```ts
import type { I18n, Kind } from './core.ts';
export type Dim = number | string;                       // symbolic: 'T' tokens, 'N' nodes, 'P' points
export type Axis = 'c' | 'y' | 'x' | 't' | 'd' | 'h' | 'tq' | 'tk' | 'n' | 'f' | 'class' | 'vocab' | 'p' | 'xy' | 'k';
export interface TensorEntry { name: string; shape: number[]; offset: number; length: number }
/** kind is a catalog InputKind: 'drawing' | 'text' | 'points2d' | 'series' | 'graph' | 'game' | … */
export interface InputDecl { name: string; adapter: string; kind: string; shape: Dim[]; dtype?: 'f32' | 'i32'; axes?: Axis[] }
export interface AtlasLayer {
  id: string;                                            // no '#'
  op: string; label: I18n;
  in: string[];                                          // layer ids or declared input names
  outShape: Dim[];
  /** op attributes (v1 name kept), validated against OpDef.attrs */
  params?: Record<string, unknown>;
  /** weight slot -> tensor name. Resolution: w[slot] ?? (typeof params[slot] === 'string' ? params[slot] : `${id}.${slot}`) */
  w?: Record<string, string>;
  desc?: I18n; axes?: Axis[];
  sub?: string[];                                        // recorded as `${id}#${sub}`
  code?: { file: string; line: number; end?: number };
  ui?: { zoom?: string; display?: string; hidden?: boolean };
  teach?: { lesson?: 'step' | 'fold' | 'skip'; title?: I18n; narration?: { what?: I18n; how?: I18n } };
  rf?: { size: number; jump: number };                   // v1 key written by TinyCNN; read as is
}
export interface ParityRecord {
  against: string;
  max_abs_diff: number | null; checked: string | null;  // null until `atlas parity` runs
  bar?: number; relative?: boolean; pass?: boolean; scope?: 'forward' | 'train' | 'process'; script?: string;
  cases?: number; edgeCases?: string[]; perLayer?: Record<string, number>; details_max_abs_diff?: number; steps?: number;
  tolerance?: number; passed?: boolean;                 // TinyGPT's legacy names, read as bar / pass
  [extra: string]: unknown;
}
export interface AtlasSpec {
  slug: string; name: I18n; version: string; params: number; precision: 'fp32' | 'f64' | 'fp16' | 'mixed';
  input?: { kind: string; shape: Dim[]; notes?: string };   // v1; the exporter writes it for single-input models
  tensors: TensorEntry[]; layers: AtlasLayer[];
  training?: { dataset?: string; steps?: number; metrics?: Record<string, number | string>; [k: string]: unknown };
  parity: ParityRecord;
  atlas?: { spec: '1.1'; kinds: Kind[]; exporter: string };
  inputs?: InputDecl[]; outputs?: string[];
  dims?: Record<string, { min: number; max: number; label: I18n }>;
  files?: Record<string, string>;                         // as TinyGPT already writes
  [extra: string]: unknown;                               // classes, vocabSize, eotId, …
}
```

**Groups** are inferred, not declared. A group is a maximal run of layers whose ids share a prefix before the first `.`, where that prefix matches `/^[A-Za-z]+\d+$/` (`L0`, `B3`). Groups with the same op sequence are repeats.

**Asset files beside `spec.json`:**

- **`reference.json` (`atlas-ref-1`).** This is TinyGPT's existing layout, generalized.
  - Layout: `{ format: 'atlas-ref-1', model, version, against: 'pytorch-fp32-cpu', note, cases: [{ id, label, edge, inputs: { <name>: { shape, dtype, b64 } }, order, tensors: { <id or id#sub>: { shape, dtype: 'float32', b64 } } }] }`.
  - `b64` is base64 of little-endian values. `-Infinity` is stored as is.
  - `graph/refs.ts` also reads the two legacy layouts (TinyCNN float lists, TinyGPT `prompts`).
  - Pages never load it.
  - A binary sidecar is added only if a reference passes 4 MB. Nothing in the batch does.
- **`facts.json` (`atlas-facts/1`):** `{ format: 'atlas-facts/1', facts: { <key>: { value, unit?, source, measured } } }`. It is written by the exporter and read by narration as `{fact:<key>}`.
- **`tapes/<name>.tape.json` + `.bin` (`atlas-tape/1`):** see §4.4. A tape is precomputed by the same TS code in Node, or written by the Python `TapeWriter` for references.

### 4.3 `types/op.ts`: op definition, unit explanation, graph model

```ts
import type { AnyTensor, Box, FetchLike, I18n, Tensor, Trace } from './core.ts';
import type { AtlasLayer, AtlasSpec, Axis, Dim } from './spec.ts';
export type OpCategory = 'io' | 'linear' | 'conv' | 'activation' | 'pool' | 'norm' | 'attention' | 'embedding' | 'shape' | 'merge' | 'recurrent' | 'graph' | 'output';
/** The zoom view is chosen by the kind of arithmetic, not by op type. */
export type ZoomKind = 'weighted-sum' | 'reduce' | 'elementwise' | 'normalize' | 'softmax' | 'lookup' | 'index-map' | 'gate' | 'none';
export type UnitPick = 'argmax' | 'argmin' | 'argmaxAbs' | 'center' | 'first' | 'last' | 'strongestChannel' | { unit: number[] };
/** Bilingual narration templates (syntax §4.6). */
export interface NarrationSet { what: I18n; how?: I18n; why?: I18n; look?: I18n; repeat?: I18n; unit?: I18n }
export interface AttrDecl { type: 'int' | 'float' | 'bool' | 'string' | 'int[]' | 'float[]'; default?: unknown; required?: boolean; doc?: string }
export interface OpCtx { layer: AtlasLayer; dims: Record<string, number>; w(slot: string): Tensor; emit(sub: string, t: Tensor): void }
export interface UnitQuery<P> { layer: AtlasLayer; unit: number[]; ins: AnyTensor[]; out: Tensor; w(slot: string): Tensor; p: P; trace: Trace; spec: AtlasSpec }
/** The arithmetic behind ONE output number. Generalizes convDetail/poolDetail/denseDetail/softmaxDetail and TinyGPT's q·k. */
export interface UnitExplanation {
  kind: ZoomKind; layer: string; unit: number[];
  reads: { tensor: string; cells: number[]; box?: Box }[];          // drives linked highlights in every view
  terms?: { x: number; w: number; p: number; xAt?: number; wAt?: number }[];   // weighted sums: x·w = p
  groups?: { label: I18n; from: number; to: number; sum: number }[];          // per input channel / head / neighbour
  layout?: { rows: number; cols: number };
  steps?: { label: I18n; value: number; tex?: string }[];                      // normalize: mean, var, (x−μ)/σ, ·γ, +β
  curve?: { fn: string; x: number; y: number };                                // elementwise: the point on the curve
  bias?: number;
  values: Record<string, number>;      // named scalars for narration vars and {{name}} in the formula
  result: number;                      // recomputed here
  traced: number;                      // read from the trace; |result − traced| ≤ 1e-5 is a gate
}
export interface OpDef<P extends Record<string, unknown> = Record<string, unknown>> {
  op: string; version: number;
  /** declared attributes: the source of schema/ops.json, which the Python mirrors are checked against */
  attrs: Record<string, AttrDecl>;
  aliases?: Record<string, Partial<P>>;                 // { dense: {} }, { linear_tied: {} }, { gelu_tanh: { approx: 'tanh' } }
  category: OpCategory; label: I18n;
  weights?: { slot: string; optional?: boolean }[];
  subs?: { name: string; label: I18n; axes?: Axis[] }[];
  inferShape(ins: Dim[][], p: P, w: Record<string, number[]>): Dim[];
  axes?(ins: Axis[][], p: P): Axis[];
  /** f64 accumulation, f32 storage (as tinycnn.ts/tinygpt.ts do). Record sub-steps with ctx.emit. */
  forward(xs: AnyTensor[], p: P, ctx: OpCtx): Tensor;
  unit?: { axes(out: Axis[]): Axis[]; scan?: 'raster' | 'rows' | 'time' | 'none'; pick?: UnitPick };
  explain?(q: UnitQuery<P>): UnitExplanation;
  receptive?(box: Box, p: P): Box;
  /** KaTeX in the kit's convention: colour roles with sym(role, tex) from atlas/core/tex.ts (the same output as kit/tex.ts),
   *  live values as {{name}} from explain().values, filled by kit fillTex in the UI. */
  formula?: string;
  narration: NarrationSet;
  /** template variables from structure: kernel, inChannels, products = C·k·k, … */
  vars?(layer: AtlasLayer, spec: AtlasSpec): Record<string, string | number>;
  /** a repeat that deserves its own lesson step (conv: inChannels 1 → >1) */
  distinct?(p: P, prev: P): boolean;
  zoom: ZoomKind; color?: 'diverging' | 'sequential';
  parity: { bar: number };
  pytorch?(layer: AtlasLayer): string;                 // one line for the Code tab
  terms?: string[];                                    // glossary ids
}
export interface ResolvedOp { def: OpDef; params: Record<string, unknown> }   // alias params merged under layer params
export interface OpRegistry { get(op: string): ResolvedOp; has(op: string): boolean; list(): string[] }
export interface GraphModel {
  spec: AtlasSpec; weights: Record<string, Tensor>; ops: OpRegistry;
  forward(input: AnyTensor | Record<string, AnyTensor>, o?: { record?: 'all' | 'layers'; dims?: Record<string, number> }): Trace;
  explain(trace: Trace, layer: string, unit: number[]): UnitExplanation;
  receptiveField(layer: string, box: Box): Box | null;
  layer(id: string): AtlasLayer;
}
// W1-A: ops/registry.ts createRegistry(...ops: OpDef[]): OpRegistry;  ops/index.ts CORE_OPS: OpDef[]
//       ops/manifest.ts (node ops/manifest.ts --write) -> schema/ops.json {op, version, attrs, aliases, weights, subs}
//       graph/executor.ts fromParts(spec, weights: ArrayBuffer | Record<string, Tensor>, ops): GraphModel;
//                         loadGraph(fetch: FetchLike, baseUrl: string, ops): Promise<GraphModel>
//       graph/legacy.ts normalizeSpec(raw: unknown): AtlasSpec (§3.2);  graph/refs.ts readReference(fetch, dir): Promise<RefCase[]>
//       core/compare.ts maxDiff(a, b, { relative?: boolean }): { max: number; at: number; nan: boolean }  (NaN or unmatched ±∞ fails; −∞ == −∞)
```

**The shared op set (`CORE_OPS`, W1-A): 20 ops plus 3 aliases.** It covers every op the two existing models use, plus every op that two or more batch models need. The elementwise activations are always core: each is a one-line definition on the shared elementwise zoom.

| Op | Attributes | Weights / subs / semantics |
|---|---|---|
| `input` | — | reads a declared input |
| `conv2d` | `kernel`, `stride=1`, `padding=0`, `inChannels`, `outChannels`, `accumulate='bias-last'` | `weight [O,C,k,k]`, `bias` |
| `linear` (+ `dense`, `linear_tied`) | `accumulate='bias-last'` | `weight [out,in]`, optional `bias` |
| `relu`, `tanh`, `sigmoid`, `silu`, `gelu` (+ `gelu_tanh`) | `gelu.approx = 'none' \| 'tanh'` | — |
| `maxpool2d` | `kernel`, `stride` | — |
| `flatten`, `concat` (`axis`), `add` | | — |
| `embedding`, `position_embedding` | | `weight` |
| `layernorm` | `eps` | `weight`, `bias`; biased variance |
| `attention_scores` | `nHead`, `dHead`, `scale`, `causal=false` | out `[H,T,T]` |
| `causal_mask`, `softmax` (`axis=-1`), `attention_mix` (`nHead`, `dHead`) | | — |
| `lstm` | `hidden` | `weight_ih [4H,In]`, `weight_hh [4H,H]`, `bias_ih`, `bias_hh` (both kept). PyTorch gate order i, f, g, o: c<sub>t</sub> = f⊙c<sub>t−1</sub> + i⊙g; h<sub>t</sub> = o⊙tanh(c<sub>t</sub>); h<sub>0</sub> = c<sub>0</sub> = 0. Subs `i f g o c h`, each `[T,H]` |

**Local ops (rule of two).**

- A local op lives in `src/lib/explainers/<slug>/ops/<op>.ts`, with a Python mirror in `pipeline/<slug>/ops_<op>.py`.
- The package's `ops.ts` exports `[...CORE_OPS, ...local]`.
- `atlas check --all` rejects a name that clashes with another package's op unless the files are identical.
- An op moves into `CORE_OPS` when a second model needs it. The spec does not change.

| Model | Local ops |
|---|---|
| tinyvit | `to_tokens`, `cls_token`, `select_token` |
| gcn | `gcn_conv` |
| tinyclip | `l2norm`, `mean_pool`, `similarity` |
| tinyforecaster | `conv1d` (dilated, causal) |
| char-rnn (optional) | `gru`: weights `weight_ih [3H,In]`, `weight_hh [3H,H]`, `bias_ih`, `bias_hh`; r, z, n gates; n = tanh(W<sub>in</sub>x + b<sub>in</sub> + r ⊙ (W<sub>hn</sub>h + b<sub>hn</sub>)); h′ = (1 − z)⊙n + z⊙h; subs `r z n h` |

### 4.4 `types/process.ts`: process, frame, tape

```ts
import type { I18n, Provenance, Rng, Tensor } from './core.ts';
export interface FieldDef {
  kind: 'tensor' | 'scalar' | 'int' | 'index' | 'text' | 'json'; label: I18n;
  shape?: (number | string)[]; scale?: 'diverging' | 'sequential' | 'categorical';
  role?: 'activation' | 'param' | 'grad' | 'optimizer' | 'data' | 'target' | 'output';
  /** 'snapshot' (default): copied when present in a frame. 'changes': mutated in place; every edit must be reported in frame.changes */
  track?: 'snapshot' | 'changes';
  /** Derived fields are never stored, saved or sent between threads ("store causes, recompute effects"). They are recomputed
   *  (memoized) wherever the ProcessDef module is loaded: the main thread lazily imports the same trainer.ts/process.ts the worker runs. */
  derive?: (s: any) => Tensor | number;
  tol?: number | 'exact';                               // parity tolerance for this field
}
export type Schema = Record<string, FieldDef>;
/** Schema without functions: what tape files and worker messages carry. */
export type SchemaJson = Record<string, Omit<FieldDef, 'derive'> & { derived?: boolean }>;
export interface PhaseDef {
  label: I18n;
  kind: 'input' | 'forward' | 'loss' | 'backward' | 'update' | 'observe' | 'act' | 'reward' | 'message' | 'aggregate' | 'sample' | 'merge' | 'output' | 'other';
  formula?: { tex: string; values: (f: Frame) => Record<string, number> };   // tex uses {{name}} and sym()
  narration?: { what: I18n; unit?: I18n };
}
export interface Change { field: string; key?: string; index?: number[]; from?: number; to?: number; note?: I18n }
/** key: an entry of a record-valued field (field 'params', key 'net.0.weight'). */
export interface Focus { field: string; key?: string; index?: number[] | number[][]; role?: 'in' | 'out' | 'param' | 'grad' | 'q' | 'k' | 'v' | 'accent'; label?: I18n }
export interface Frame<S = Record<string, unknown>> {
  phase: string; step?: number; sub?: number;
  state: Partial<S>; changes?: Change[]; focus?: Focus[];
  metrics?: Record<string, number>; caption?: I18n;
  /** every value returned by ctx.rng (and its forks) while making this frame, when RecordOpts.recordDraws is set */
  draws?: number[];
}
export type Control<O = Record<string, unknown>> =
  | { type: 'set'; opt: Partial<O> } | { type: 'input'; input: unknown }
  | { type: 'pause' } | { type: 'resume' } | { type: 'step'; n: number } | { type: 'reset'; seed?: number };
export interface RunCtx<O> { rng: Rng; opt: O; detail: 'coarse' | 'fine'; signal: { cancelled: boolean } }
export interface ProcessDef<I = unknown, O = Record<string, unknown>, S = Record<string, unknown>> {
  id: string; kind: 'inference' | 'training' | 'algorithm'; title: I18n;
  phases: Record<string, PhaseDef>; schema: Schema; defaults: O;
  /** Pure: same input, opt and seed => same frames. 'set'/'input' controls arrive via gen.next(control). Ties break to the smallest index. */
  run(input: I, ctx: RunCtx<O>): Generator<Frame<S>, void, Control<O> | undefined>;
}
/** `where` is evaluated on the thread that holds the tape (the main thread for worker sources); cursors are never sent to a worker. */
export interface CursorQuery { phase?: string; step?: number; sub?: number; layer?: string; last?: boolean; where?: (f: Frame<any>, s: any) => boolean }
export type Cursor = 'first' | 'last' | number | CursorQuery;
export interface Tape<S = any> {
  process: string; seed: number; opt: unknown; provenance: Provenance; version: string;
  readonly length: number;
  frame(i: number): Frame<S>;
  state(i: number): S;                                   // nearest keyframe + changes; pure in i
  find(c: Cursor): number;                               // −1 if not recorded (yet); callers clamp
  segments(): { phase: string; from: number; to: number }[];
  metric(name: string): Float64Array;
}
export interface LiveTape<S = any> extends Tape<S> {
  readonly done: boolean;
  /** resolves with the index of the first frame matching c once it is recorded, or the last index if the run ends without it */
  ensure(c: Cursor, o?: { budgetMs?: number }): Promise<number>;
  control(c: Control): void;
  subscribe(fn: (length: number) => void): () => void;   // throttled to animation frames
  close(): void;
}
export interface RecordOpts { seed?: number; detail?: 'coarse' | 'fine'; keyEvery?: number /* 64 */; maxFrames?: number /* 20000 */; keep?: 'all' | { every: number }; recordDraws?: boolean }
export interface TapeHeader {
  format: 'atlas-tape/1'; process: string; version: string; seed: number; opt: unknown; provenance: Provenance;
  schema: SchemaJson; phases: Record<string, { label: I18n; kind: PhaseDef['kind']; tex?: string }>; frames: unknown[];
}
// W1-C, process/*.ts:
//   tape.ts   class TapeStore implements Tape { push(frame) } — keyframes, changes, derived memo (LRU). Used by the local recorder AND by W1-B2's engine client.
//   record.ts record(p, input, opt?, ro?): Tape;  recorder(p, input, opt?, ro?): LiveTape (advances in ≤ 8 ms slices)
//   io.ts     saveTape(t): { header: TapeHeader; blob: ArrayBuffer };  loadTape(h, blob, p?: ProcessDef): Tape
//             (p re-attaches derive and formula.values; without p, e.g. Python-written tapes, derived fields are unavailable)
//   adapters.ts traceToTape(spec, trace), scanTape(model, trace, layer, unit?, o?), bpeTape(r), generateTape(r)
//   library/sampling.ts (temperature/top-k, TinyGPT semantics), library/euler.ts (ODE particles), library/rollout.ts
```

### 4.5 `types/autograd.ts` and `types/train.ts`: kind B

```ts
// autograd.ts: tensor-level reverse mode, f64 by default. Every VJP is written with the same ops,
// so grad(..., { createGraph: true }) is itself differentiable (PINN u'' and its parameter gradients).
import type { I18n, Rng, Tensor } from './core.ts';
import type { AtlasSpec } from './spec.ts';
export interface Var { readonly value: Tensor; grad: Var | null; readonly requiresGrad: boolean; readonly name?: string }
/** named() uses PyTorch state_dict names ('net.0.weight') and [out, in] layouts, so twins and Focus keys match. */
export interface Module { params(): Var[]; named(): Record<string, Var>; toSpec?(o: { slug: string; inputAxes?: string[] }): { spec: AtlasSpec; weights: Float32Array } }
export interface Optimizer { lr: number; step(): void; set(o: Partial<{ lr: number; momentum: number; betas: [number, number]; weightDecay: number }>): void; state(): Record<string, { m?: Tensor; v?: Tensor }> }
export type Rec = (id: string, v: Var) => void;          // named intermediates for views (tape field `acts`)
// W1-B1, autograd/*.ts:
//   var.ts:   param(name, t), input(t, requiresGrad?), grad(ys, xs, { createGraph?, seeds?, allowUnused? }), backward(loss, { createGraph? }), noGrad(f), zeroGrad(ps)
//   ops.ts:   op.{ add sub mul div neg scale pow sqrt exp log tanh sigmoid relu silu gelu sin cos abs matmul linear sum mean reshape
//                  transpose broadcastTo concat slice gather logsumexp softmax logSoftmax conv1d layerNorm }
//   loss.ts:  mse, crossEntropy, bceWithLogits, infoNCE
//   nn.ts:    Linear, MLP(name, sizes, act, rng), Embedding, LSTMCell (both biases), Conv1d({dilation, causal}), LayerNorm
//   optim.ts: SGD, Momentum, Adam/AdamW (m and v exposed for the optimizer glyphs)
//   check.ts: gradcheck(f, xs, { eps, tol, second? }) -> { maxRelErr, worst }
//   tospec.ts: Linear/MLP stacks -> AtlasSpec + f32 weights (the kind-A explainer on a live model)

// train.ts
import type { Module, Optimizer, Rec, Var } from './autograd.ts';
import type { Schema } from './process.ts';
export interface TrainOpt { seed: number; lr: number; batch: number; steps: number; snapshotEvery: number; optimizer: 'sgd' | 'momentum' | 'adam'; [k: string]: unknown }
export interface TrainState {
  step: number; loss: number; parts?: Record<string, number>;
  params: Record<string, Tensor>;                        // copied at snapshot frames
  grads?: Record<string, Tensor>; opt?: Record<string, { m?: Tensor; v?: Tensor }>;   // fine detail only
  batch?: Record<string, Tensor>; acts?: Record<string, Tensor>;                      // fine detail only
  [derived: string]: unknown;
}
export interface TrainerDef<D = unknown, M extends Module = Module, B = unknown, O extends TrainOpt = TrainOpt> {
  id: string; title: I18n; defaults: O;
  model(rng: Rng, opt: O): M;                            // use rng.fork('init') so compared runs share initial weights
  batch(data: D, step: number, rng: Rng, opt: O): B;     // use rng.fork('batch')
  loss(m: M, b: B, rec: Rec, opt: O): { loss: Var; parts?: Record<string, Var> };
  optimizer?(m: M, opt: O): Optimizer;                   // default from opt.optimizer
  derived?: Schema;                                      // decision grid, vector field, u(t), …
  metrics?(m: M, data: D): Record<string, number>;
}
// W1-B1: train/trainer.ts trainingProcess(def): ProcessDef<D, O, TrainState>
//   phases: 'init'; coarse 'snapshot' every snapshotEvery (metrics on every step); fine 'forward' → 'loss' → 'backward' → 'update'.
//   { type:'set', opt:{lr} } applies at the next step; { type:'input' } swaps the data (a new tape segment).
// train/helpers.ts updateOf(at, key, index) -> { old, g, lr, new } (SGD sources only; throws for Adam so narration cannot lie)
```

### 4.6 `types/lesson.ts`: lesson, steps, narration

```ts
import type { I18n, PredictQuestion, Provenance } from './core.ts';
import type { Cursor, Focus, Frame, Tape } from './process.ts';
import type { UnitExplanation } from './op.ts';
import type { ExplainerRuntime, SelectionState } from './runtime.ts';
export interface At { index: number; frame?: Frame; state?: unknown; tape?: Tape; unit?: UnitExplanation; layer?: string }
/** Template: {name|fmt}. Names resolve in order: vars() result; built-in paths (§5.3); fact:<key> (facts.json). No expressions.
 *  fmt: n3 (3 significant digits; default for numbers) | int | pct | signed | shape | label | tok.
 *  en and zh must use the same set of names (AK403). An unresolved or non-finite value renders "…" and fails lint (AK401). */
export type TmplVal = number | string | { v: number; fmt?: string; focus?: Focus };
export type Tmpl<R> = I18n & { vars?: (rt: R, at: At) => Record<string, TmplVal> };
export type Narration<R> = I18n | Tmpl<R>;
export interface StepDef<R extends ExplainerRuntime = ExplainerRuntime> {
  id: string; title: I18n; narration: Narration<R>; look?: Narration<R>;
  source?: string;                                       // default: the explainer's first source
  at?: Cursor | ((rt: R) => Cursor);
  select?: (rt: R) => Partial<SelectionState>;           // the lesson's own selection (never the explore view's)
  view: string | { key: string; props?: (rt: R, at: At) => Record<string, unknown> };
  play?: 'view' | { from: Cursor; to: Cursor; interval?: number; phases?: string[] };   // 'view': the view animates itself (autoplay + onend)
  focus?: (rt: R, at: At) => Focus[];
  provenance: Provenance | { mode: Provenance; note: I18n };
  predict?: PredictQuestion;                             // stage dimmed until answered; Play waits
  task?: { prompt: I18n; done: (rt: R, at: At) => boolean; hint?: I18n };
  compare?: { a: string; b: string; label: [I18n, I18n] };
  kindOf?: 'limits' | 'breakIt';                         // marks the honest-limits step the lesson shape rule asks for
  terms?: string[];
  waitForDone?: boolean; linger?: number; duration?: number;
}
/** Auto step ids: 'input' | 'op:<layer>' | 'group:<id>' | 'output' | 'train:<motif>@<source>' | 'proc:<motif>@<source>' */
export type AutoRef<R> = { auto: string; override?: Partial<StepDef<R>> } | { autoAll: { from?: string; until?: string; skip?: string[] } };
export interface FixtureDef { id: string; label: I18n; raw: unknown; expect?: { empty?: boolean } }
export interface LessonDef<R extends ExplainerRuntime = ExplainerRuntime> {
  id?: string;                                           // deep-link prefix, default 'lesson'
  title: I18n; summary: I18n;                            // summary -> catalog lesson.summary (promote)
  steps: (StepDef<R> | AutoRef<R>)[];
  fixtures?: FixtureDef[];                               // adapter edge fixtures are always added
}
export interface CompiledStep {
  id: string; origin: 'auto' | 'author'; autoId?: string;
  title: I18n; narration(rt: ExplainerRuntime, at: At): I18n; look?(rt: ExplainerRuntime, at: At): I18n | undefined;
  source: string; at(rt: ExplainerRuntime): Cursor; view: string; props(rt: ExplainerRuntime, at: At): Record<string, unknown>;
  play?: StepDef['play']; focus(rt: ExplainerRuntime, at: At): Focus[];
  provenance: { mode: Provenance; note?: I18n }; select?(rt: ExplainerRuntime): Partial<SelectionState>;
  predict?: PredictQuestion; task?: StepDef['task']; compare?: StepDef['compare']; kindOf?: StepDef['kindOf']; terms: string[];
  waitForDone: boolean; linger?: number; duration?: number;
}
export interface LintIssue { code: string; level: 'error' | 'warn'; step?: string; fixture?: string; lang?: 'en' | 'zh'; message: string; fix?: string }
// W1-C: lesson/define.ts defineLesson, auto, autoAll, defineExplainer;  lesson/plan.ts planLesson(spec | null, ops, manifest) -> StepDef[]
//       lesson/compile.ts compileLesson(def, { spec, ops, manifest }) -> { id: string; steps: CompiledStep[]; issues: LintIssue[] }
//       lesson/narrate.ts render(tmpl, rt, at, lang) -> { text: string; placeholders: { name: string; value: unknown; focus?: Focus }[] }
//       lesson/lint.ts lintLesson(compiled, makeRuntime, fixtures) -> LintIssue[]   (Node and the Lesson Lab)
//       lesson/fixtures.ts withFixtures(pkg, fn) — the helper claims.test.ts files use
```

**Claims are tests.** A factual sentence that is not computed live ("the bag-of-words model gives the same score to both captions") is backed by an assertion in `explainers/<slug>/claims.test.ts`. The assertion runs over every fixture and preset through `withFixtures`. `atlas check` runs these tests.

### 4.7 `types/runtime.ts`: package modules, runner, runtime

The runtime is pure TS, so Node can build it for dry runs. In the browser, `ui/runtime.svelte.ts` wraps the same object in `$state.raw` and replaces trace and tape objects rather than mutating them (KIT rule 5).

```ts
import type { AnyTensor, FetchLike, I18n, Provenance, Tensor, Trace } from './core.ts';
import type { AtlasSpec } from './spec.ts';
import type { GraphModel, OpDef } from './op.ts';
import type { Cursor, LiveTape, ProcessDef, Tape } from './process.ts';
import type { ExplainerManifest, InputAdapter } from './explainer.ts';
import type { LessonDef } from './lesson.ts';
export interface Preprocessed { tensors: Record<string, AnyTensor>; steps: { id: string; label: I18n; tensor?: Tensor; note?: I18n }[]; empty: boolean; notes?: I18n[] }
export type SourceDef =
  | { kind: 'trace'; inputs?: string[]; assets?: string }   // assets: another model's folder under static/ ('models/tinycnn'), read via legacy.ts
  | { kind: 'scan'; of: string; layer: string | ((rt: ExplainerRuntime) => string); unit?: (rt: ExplainerRuntime) => number[]; order?: 'raster' | 'peak-first'; stride?: number }
  | { kind: 'train'; trainer: string; opt?: Record<string, unknown>; detail?: 'coarse' | 'fine'; precomputed?: string }
  | { kind: 'process'; process: string; input?: (rt: ExplainerRuntime) => unknown; opt?: Record<string, unknown>; seed?: number; run?: 'main' | 'worker'; precomputed?: string }
  | { kind: 'tape'; file: string }                                                               // precomputed only
  | { kind: 'live-model'; of: string };                                                          // kind-A trace on a train source's current weights
export interface SourceState { kind: SourceDef['kind']; status: 'idle' | 'loading' | 'running' | 'ready' | 'error'; provenance: Provenance;
  trace?: Trace | null; tape?: Tape | LiveTape | null; model?: GraphModel | null; progress?: { at: number; target: number }; error?: string }
export interface SelectionState { layer?: string; unit?: number[]; ch?: number; head?: number; t?: number; node?: number; [k: string]: unknown }
/** Everything a package contributes, already imported. Built by ui/packages.ts (Vite globs) in the browser and by
 *  scripts/atlas/load.ts (dynamic import by path) in Node. Both loaders wrap every trainer.ts export with trainingProcess()
 *  and register it as process 'train:<name>', so the runtime itself only ever runs ProcessDefs. */
export interface PackageModules {
  slug: string; manifest: ExplainerManifest; lesson: LessonDef<any> | null;
  ops: OpDef[];                                          // the package's ops.ts default export, else CORE_OPS
  processes: Record<string, ProcessDef<any, any, any>>;
  adapters: Record<string, InputAdapter>;                // kit adapters + the package's local ones
  worker?: () => Worker;                                 // browser only: the package's worker.ts (Vite ?worker)
}
export interface OpenSource { slug: string; source: string; process: string; input: unknown; opt: Record<string, unknown>; seed: number; detail: 'coarse' | 'fine'; recordDraws?: boolean }
/** Runs 'train' and 'process' sources.
 *  localRunner (runtime/local.ts, W1-C): in-process with recorder(); used in Node, tests, and for run: 'main'.
 *  engineRunner (engine/runner.ts, W1-B2): in pkg.worker(); frames stream into a TapeStore on the main thread. */
export interface SourceRunner { open(o: OpenSource, pkg: PackageModules): LiveTape }
export interface ExplainerRuntime {
  slug: string; spec: AtlasSpec | null; model: GraphModel | null;
  input: unknown; pre: Preprocessed | null;
  sources: Record<string, SourceState>;
  sel: SelectionState;                                   // the lesson's own selection
  fact(path: string): number | string | I18n | undefined; // built-in paths (§5.3) + facts.json
  num(path: string): number;                             // NaN when missing or not a number (lint AK401)
  ensure(source: string, c: Cursor): Promise<number>;    // load/run until the cursor exists; returns a clamped frame index
  setInput(raw: unknown): void;                          // re-runs sources; steps re-resolve their cursors
}
// W1-C: runtime/runtime.ts createRuntime(pkg: PackageModules, o: { fetch: FetchLike; base: string; input?: unknown; runner?: SourceRunner }): Promise<ExplainerRuntime>
//       runtime/fake.ts fakeRuntime(partial): ExplainerRuntime (view demos, tests)
```

### 4.8 `types/view.ts`: view registry and stage contract

```ts
import type { I18n, Kind, Provenance } from './core.ts';
import type { At } from './lesson.ts';
import type { Focus } from './process.ts';
import type { ExplainerRuntime, SelectionState } from './runtime.ts';
/** views/<key>/meta.ts (pure TS, readable by Node lint); the component is views/<key>/View.svelte (lazy). */
export interface ViewMeta {
  key: string;                                    // 'zoom.weighted-sum', 'plane2d'; local views register as '<slug>/<key>'
  title: I18n; kinds: Kind[];
  accepts?(o: { field?: string; zoom?: string; axes?: string[] }): boolean;   // default view per field/zoom/axes
  describe(p: StageProps): I18n;                  // text summary: live region + "show as table"
  /** demo props for the gallery and the Lesson Lab, built on runtime/fake.ts. Contract (types/contract.test.ts):
   *  resolves in Node, describe(demo) is non-empty in en and zh, provenance is 'illustrative'. */
  demo(): Promise<Omit<StageProps, 'ctx'>>;
  minHeight?: number;                             // the stage reserves it (no layout jump)
}
export interface StageProps {
  rt: ExplainerRuntime; source: string; at: At;
  focus: Focus[]; sel: SelectionState;
  ctx?: { shown: boolean; reduced: boolean; playing: boolean; done(): void };   // kit WalkContext subset
  provenance: Provenance; autoplay: boolean; onend: () => void;
  props: Record<string, unknown>;
}
// W1-D1: ui/registry.ts — metas via import.meta.glob (inside AutoModelPage's lazy chunk only), components lazy;
//        package views from '$lib/explainers/*/views/*/meta.ts' registered as '<slug>/<key>'.
```

### 4.9 `types/explainer.ts`: the manifest

```ts
import type { I18n, Kind, PredictQuestion } from './core.ts';
import type { SourceDef, Preprocessed } from './runtime.ts';
import type { NarrationSet } from './op.ts';
export interface Credit { name: string; url: string; license: string; usedFor: string; attribution: string }   // licenses.json row
export interface TryThisDef { id: string; title: I18n; setup: { preset?: string; input?: unknown; opt?: Record<string, unknown> }; predict: I18n; look: I18n; goto?: { step?: string; layer?: string } }
export interface QuizDef { q: I18n; answerFact?: string; tolerance?: number; options?: I18n[]; answer?: number; explain: I18n }
export interface ExplainerManifest {
  slug: string; type: 'model'; kinds: Kind[];
  quality: 'draft' | 'reviewed' | 'flagship';
  /** lab only (hello-xor, demo-*): exempt from the catalog rules (AK701), never routed under /m/ */
  devOnly?: boolean;
  assets?: { dir: string; auto: string[]; lazy?: Record<string, string> };  // dir relative to static/; auto = loaded on every visit
  input?: { adapter: string; presets?: string; start?: unknown; options?: Record<string, unknown> };
  inputs?: { name: string; adapter: string; options?: Record<string, unknown> }[];   // multi-input (TinyCLIP)
  output?: { view: string; layer?: string; props?: Record<string, unknown> };
  sources?: Record<string, SourceDef>;                   // default { fwd: { kind: 'trace' } } when kinds has 'trace'
  overview?: { layout: 'pipeline' | 'stack'; collapse?: string[] };
  page: { goals: I18n[]; prequestion: PredictQuestion; tryThis?: TryThisDef[]; quiz?: QuizDef[];
          misconception?: { claim: I18n; truth: I18n; step?: string }; lessonNote?: I18n };
  credits: Credit[];                                     // promote appends to static/licenses.json
  glossary?: { id: string; term: I18n; def: I18n; related?: string[] }[];   // promote prints the glossary.ts patch
  /** promote prints the models.ts patch, plus tech-zh.ts entries for runtime/params/size */
  catalog?: { hook?: I18n; shows?: I18n; runtime?: I18n; params?: I18n; size?: I18n; prereqs?: string[] };
  budget?: { autoLoadBytes?: number };
}
/** Input adapters: atlas/adapters/<id>/adapter.ts (pure) + Capture.svelte. kind is a catalog InputKind. */
export interface InputAdapter<Raw = unknown> {
  id: string; kind: string;
  preprocess(raw: Raw, spec: unknown, opts: Record<string, unknown>): Preprocessed;
  presets?(assets: Record<string, unknown>): { id: string; label: I18n; raw: Raw }[];
  edgeFixtures(): { id: string; label: I18n; raw: Raw }[];   // drawing: blank, dot, small; text: '', 1 token, > context; points: 0, one class; series: flat; graph: isolated node
  layer0: { view: string; narration: NarrationSet };
}
```

Adapter to catalog `InputKind`:

| Adapter | InputKind |
|---|---|
| `drawing28` | `drawing` |
| `text-bpe`, `text-char` | `text` |
| `points2d` | `points2d` |
| `series` (curve1d capture; PINN observation points) | `series` |
| `graph` (local to gcn) | `graph` |
| `gridworld` (local editor) | `game` |
| `shapes-captions` (local to tinyclip) | `drawing` + `text` |

**File conventions inside a package.** `explainer.ts` never imports Svelte, so these files are found by name, not by import:

| File | Effect |
|---|---|
| `sections/<Snippet>.svelte` | replaces one of the 15 `ModelPageShell` snippet props (`Facts`, `Input`, `Lesson`, `Overview`, `Inspector`, `Unit`, `Output`, `TryThis`, `Training`, `Article`, `Code`, `Links`, `MoreLinks`, `Checkpoint`, `Credits`) |
| `content/En.svelte`, `content/Zh.svelte` | the article |
| `views/<key>/` | a local view, registered as `<slug>/<key>` |
| `adapters/<id>/` | a local adapter |
| `ops.ts` | the package's op list |
| `trainer.ts` / `process.ts` | named `TrainerDef` / `ProcessDef` exports |
| `worker.ts` | generated by `atlas new` when the package has train or process sources (4 lines, §4.10) |
| `claims.test.ts` | claim tests (§4.6) |

### 4.10 `types/engine.ts`: one worker protocol

This protocol is for new code. The four hand-written worker stacks stay as they are.

```ts
import type { Control, Frame, SchemaJson } from './process.ts';
export type EngineRequest =
  | { id: number; channel: string; type: 'call'; method: string; args: unknown }
  | { id: number; channel: string; type: 'stream'; method: 'open'; args: { process: string; input: unknown; opt: unknown; seed: number; detail: 'coarse' | 'fine'; recordDraws?: boolean }; maxFps?: number }
  | { channel: string; type: 'control'; control: Control }
  | { channel: string; type: 'cancel' };
export type EngineReply =
  | { id: number; type: 'result'; data: unknown }
  | { id: number; type: 'schema'; schema: SchemaJson }
  | { id: number; type: 'frames'; frames: Frame[]; done?: boolean }
  | { id: number; type: 'error'; code: string; message: string }
  | { type: 'status'; status: 'loading' | 'ready' | 'error'; message?: string };
// Transfer rules (W1-B2):
//  - frames carry stored fields only: never derived fields, never functions. The schema is sent once per stream.
//  - each tensor in a batch of frames is COPIED into its own fresh ArrayBuffer (deduplicated by identity within the batch),
//    and those buffers are transferred. Arena, parameter or optimizer memory is never transferred (that would detach the
//    worker's own state), and no buffer is listed twice (DataCloneError).
//  - cursors (and CursorQuery.where) never cross threads: the main thread evaluates them on its TapeStore.
// Per-package worker entry, generated by `atlas new` (src/lib/explainers/<slug>/worker.ts):
//    import { serve } from '../../atlas/engine/host.ts';
//    import * as trainers from './trainer.ts';          // omitted when absent
//    import * as processes from './process.ts';         // omitted when absent
//    serve({ trainers, processes });
// ui/packages.ts: import.meta.glob('$lib/explainers/*/worker.ts', { query: '?worker', import: 'default' }).
// Each package gets its own worker bundle (Vite's default iife format inlines its imports, which is right per package).
// W1-B2 spikes this first. If it fails, the main session sets worker.format: 'es' in vite.config.ts and the engine uses one
// generic worker with lazy per-package imports.
// Semantics taken from NtpClient: a newer request on a channel supersedes the older one (resolves null); main-thread fallback
// through localRunner when the worker fails; frames batched to maxFps (default 30); loops yield every ≤ 8 ms; hidden tab ⇒ pause + notice.
```

### 4.11 Python API (`pipeline/atlas_kit`, W1-E1 and W1-E2)

The SDK targets Python ≥ 3.10, so it uses no `typing.Self`, `tomllib` or `ExceptionGroup`. Importing `atlas_kit` sets `torch.backends.cudnn.allow_tf32 = False` and `torch.backends.cuda.matmul.allow_tf32 = False`. References and goldens are always computed on the CPU.

```python
import atlas_kit as ak

class SpecBuilder:
    # every call computes the PyTorch value for the current case and records the layer
    def input(self, name: str, value: Tensor, *, adapter: str, kind: str, axes: str, shape: list | None = None) -> Node
    # CORE_OPS mirrors; attrs are checked against site/src/lib/atlas/schema/ops.json (AK106 on unknown or missing attrs)
    def conv2d(self, id, x, module: nn.Conv2d, **meta) -> Node              # stride/padding read from the module
    def linear(self, id, x, module: nn.Linear | None = None, *, weight=None, bias=None, tied_to: str | None = None, **meta) -> Node
    def relu(self, id, x, **meta); def tanh(...); def sigmoid(...); def silu(...); def gelu(self, id, x, approx="none", **meta)
    def maxpool2d(self, id, x, kernel=2, stride=None, **meta); def flatten(self, id, x, **meta); def concat(self, id, xs, axis, **meta)
    def softmax(self, id, x, axis=-1, **meta); def add(self, id, a, b, **meta); def layernorm(self, id, x, module, **meta)
    def embedding(self, id, ids, module, **meta); def position_embedding(self, id, ids, module, **meta)
    def attention_scores(self, id, q, k, n_head, causal=False, **meta); def causal_mask(self, id, s, **meta); def attention_mix(self, id, w, v, n_head, **meta)
    def lstm(self, id, x, module: nn.LSTM, **meta) -> Node   # gate subs from a reference loop checked against module(x) ≤ 1e-5 (AK104); both biases kept
    def op(self, type: str, id: str, inputs: list[Node], *, params: dict, weights: dict[str, Tensor], ref: Callable, **meta) -> Node   # local ops
    def output(self, *nodes: Node) -> None
    # **meta: label=(en, zh), desc=(en, zh), ui={...}, teach={...}, code=("pipeline/<slug>/model.py", line)
    # ids may not contain '#'; sub-steps are written as "<id>#<sub>"

@dataclass
class Case: id: str; label: str; input: Any; edge: bool = False

def export_model(describe: Callable[[SpecBuilder, Case], None], cases: list[Case], *, slug: str, name: tuple[str, str],
                 kinds=("trace",), out: str | None = None,         # default $ATLAS_SITE/static/models/<slug>
                 module: nn.Module | None = None, module_input: Callable[[Case], Any] | None = None,   # proves describe() == module(x) (AK103)
                 extras: dict | None = None, training: dict | None = None, facts: dict | None = None,
                 presets: dict | None = None, bar: float = 1e-4, file_mb: float = 5, total_mb: float = 15) -> ExportReport
    # CPU only (AK102 if a tensor or module is on CUDA). The first describe() call fixes the graph and weights; later calls must give
    # the same graph (AK101). Writes spec.json (1.1, parity null), weights.bin (tied tensors de-duplicated), reference.json (atlas-ref-1),
    # facts.json, presets.json, with LF line endings and UTF-8 without BOM (Windows Python would otherwise write CRLF).
    # Enforces budgets (AK105). The CLI then runs parity.

class Rng: ...                     # next(), int(), fork() bit-identical to core/rng.ts (same test vectors); normal() not cross-engine exact
class TapeWriter:                  # atlas-tape/1, the same layout as TS saveTape
    def __init__(self, process: str, schema: dict, seed: int, opt: dict, key_every: int = 64): ...
    def frame(self, phase: str, state: dict, step=None, sub=None, changes=None, metrics=None, draws=None) -> None
    def write(self, stem: str) -> None
def train_reference(fixture: str, twin: str, out: str) -> None
    # python -m atlas_kit.train_reference --fixture $ATLAS_SITE/.atlas/<slug>/train_fixture.json --twin pipeline/<slug>/twin.py
    # reads the init params and batches written by Node (TS is the source of truth for init and synthetic data), runs float64 torch
    # for K steps (create_graph where the twin asks), writes losses, step-1 grads (and double grads) and final params
def validate(dir: str) -> list[str]   # structural only (offsets, lengths, ids, budgets); the full schema check is TS (`atlas check`)
ak.data.quickdraw(classes, split)                         # wraps pipeline/tinycnn/common.py loaders (no download)
ak.data.tinystories_chars(path, n_chars)                  # pipeline/data/tinystories (already downloaded)
ak.data.karate()                                          # networkx.karate_club_graph(); UNWEIGHTED A = nx.to_numpy_array(g, weight=None);
                                                          # labels from the 'club' attribute; Zachary 1977 citation
```

**The `twin.py` contract for kind B** is about 20 lines:

```python
def build(cfg: dict) -> nn.Module                          # same architecture and state_dict names as the TS trainer
def loss(model, batch: dict, cfg: dict) -> tuple[Tensor, dict]   # may call torch.autograd.grad(create_graph=True)
```

**The Python process reference for kind C.**

- `pipeline/<slug>/reference.py` uses `ak.Rng` and `ak.TapeWriter`. It writes `$ATLAS_SITE/.atlas/<slug>/<process>.ref.tape.json|.bin`.
- **Draws.** When a process draws with `normal()`, Beta/Gamma or another transcendental transform, it records its draws (`recordDraws`). The reference then consumes `.atlas/<slug>/<process>.draws.json` instead of drawing its own.
- **Near ties.** At each discrete choice the reference computes its own answer. If its answer differs from the TS choice only because the two candidates are within `1e-12` relative (UCB and Q ties), the check reports AK205 as a warning and the reference adopts the TS choice. Any other mismatch is AK204.

### 4.12 CLI (`npm run atlas -- <cmd>` from `site/` or from a lane's `site/`; W1-E1 and W1-E2)

| Command | What it does |
|---|---|
| `doctor` | Checks: <ul><li>Node ≥ 24 and the site `node_modules`;</li><li>the Python from `atlas.config.local.json`, else `atlas.config.json` (`"python"` on PATH): Python ≥ 3.10 and `import torch, numpy, networkx`;</li><li>playwright `msedge`;</li><li>free ports;</li><li>git `-c safe.directory=...` on exFAT.</li></ul> It prints the exact fix for each failure. |
| `new model <slug> --kind trace\|train\|process[,…] --input <adapter> [--template drawing-classifier\|char-seq\|points2d-train\|series-train\|graph-node\|process-blank] [--dev]` | Scaffolds `pipeline/<slug>/`, `src/lib/explainers/<slug>/` (including `worker.ts` when needed) and `notes/<slug>.md`. Goals and outline comments are seeded from the catalog `hook`/`shows`. The slug must be in the catalog, unless `--dev`, which makes a lab-only `devOnly` package |
| `new op <name> [--local <slug>] [--like <op>]` / `new view <key> [--local <slug>]` | An op file + test + Python mirror stub; a view folder (`meta.ts` + `View.svelte` with a demo) |
| `worktree <name> [--sync]` | Makes `site/.work/<name>/site`: no symlinks; skips `node_modules`, `.svelte-kit`, `build`, `.work`, `.atlas`; patches that copy's `vite.config.ts` `fs.allow` (never merged back); records a hash snapshot. `--sync` refreshes every non-owned path from main and refuses (AK703) if the lane changed one |
| `merge <name> [--dry-run]` | Copies the lane's owned paths back into the main tree. Refuses (AK702) if the lane changed anything outside them. Never copies `vite.config.ts` |
| `export <slug>` | Runs `pipeline/<slug>/export.py` with the configured Python (`PYTHONPATH=$ATLAS_ROOT/pipeline`, `ATLAS_SITE`, `ATLAS_ROOT`), then `parity`, then `lesson draft` |
| `parity <slug> [--write] [--json]` | <ul><li>A: every traced id and sub-step at the dim extremes, and `explain` equals the trace.</li><li>B: TS → Python twin fixtures.</li><li>C: the reference tape.</li></ul> Writes `.atlas/<slug>/parity.json`, and writes `spec.parity` (with `perLayer`) for kit packages. For `tinycnn`/`tinygpt` it runs the legacy scripts with `--no-write` and compares their numbers with the shipped spec. `--write` on a legacy model is refused unless run from the main tree by the main session |
| `run <slug> <source> [--fixture id] [--frames a..b] [--save] [--precompute]` | Prints a process or trainer tape as a table in Node, before any UI exists. `--save` writes the Python-twin fixture or the synthetic data set to `.atlas/<slug>/`. `--precompute` writes `static/models/<slug>/tapes/` (provenance `precomputed`) |
| `lesson <slug> draft\|lint\|print [--lang zh] [--fixture id]` | `draft`: the auto plan with step ids. `lint`: §6.2 rules over every fixture in en and zh. `print`: narration tables for translators and reviewers |
| `check <slug>\|--all [--fast]` | All gates (§6.3). `--fast` skips the build and e2e. A per-slug build goes to `.atlas/build-<slug>/` through `ATLAS_BUILD_DIR`; `--all` builds once into `.atlas/build-all/` |
| `e2e <slug> [--full]` | Playwright (msedge, headless) on the sampled matrix (§6.3). `--full` runs the whole matrix in chunks of ≤ 8 minutes. Screenshots go to `notes/shots/<slug>/` |
| `review <slug>` | Writes `notes/review-<slug>.md`: the checklist, report summary, screenshot list, and every narration resolved on 3 presets in en and zh |
| `promote <slug>` | Appends `credits` to `static/licenses.json` (re-read, append only). Prints a ready-to-apply patch for `catalog/models.ts` (status, `lesson`, catalog fields), `catalog/tech-zh.ts` and `catalog/glossary.ts`, which the main session applies. Then runs `check:catalog` |

There is no `preview` command. Contributors run `npm run dev -- --port <port>` from the port table in `DEVKIT.md`.

**Error format.** Every command reports errors as:

```
AK201  parity: layer "B1.attn.weights" max |diff| 3.2e-02 > 1e-4
  where   case "house#2", head 3, row 5, col 2: js 0.412, torch 0.380
  likely  PyTorch applied a mask the spec does not know (spec says causal=false)
  fix     set causal at pipeline/tinyvit/export.py:41, or teach the op
  docs    site/DEVKIT.md#ak201
```

**Defined codes** (the hundreds digit is the area):

| Area | Codes |
|---|---|
| Environment | AK001 Node/npm; AK002 Python or packages; AK003 port busy; AK004 git safe.directory |
| Export | AK101 graph differs between cases; AK102 CUDA/TF32 during export; AK103 describe() ≠ module(x); AK104 recurrent gate reference ≠ module; AK105 budget at export; AK106 op attrs vs ops.json |
| Parity | AK201 layer over bar (A); AK202 explain ≠ trace; AK203 twin mismatch (B); AK204 reference mismatch (C, first divergence); AK205 near-tie divergence (warning) |
| Assets | AK301 schema; AK302 inferShape ≠ outShape at a dim extreme; AK303 size budget |
| Lessons | AK401 placeholder unresolved or non-finite; AK402 literal multi-digit number in narration (warning); AK403 en/zh placeholder sets differ; AK404 cursor not found on a fixture; AK405 view not registered; AK406 missing zh (error for titles and labels, warning for narration at `draft`); AK407 lesson shape; AK408 claim test failed; AK409 `reviewed` lesson made only of auto steps |
| Page | AK501 TODO markers in the manifest; AK502 step without provenance; AK503 dependency rule or `Math.random`; AK504 empty `describe()` |
| E2E | AK601 console error; AK602 horizontal scroll at 360 px; AK603 long task |
| Ownership | AK701 slug not in catalog; AK702 change outside owned paths; AK703 sync refused; AK704 credits missing |

---

## 5. Auto-explainer

### 5.1 What is generated from a spec

**Route mechanics for `/m/[slug]`** (W1-D1):

- **`+page.server.ts` (new).**
  - It finds packages with a server-side glob of `explainers/*/explainer.ts`.
  - A package is **included** when its catalog status is `live` or `ATLAS_INCLUDE_DRAFTS=1`. The dev server sets that variable implicitly. `promote` flips the status after review.
  - For an included package it embeds `spec.json` and measures the bytes of `assets.auto`.
  - `devOnly` packages are never included.
- **`+page.ts`.**
  - `entries` = every catalog slug except `STATIC_ROUTES = ['tinycnn', 'tinygpt']`.
  - A live slug with neither a package nor a static folder still returns 404.
  - For an included package, `load` returns `component: (await import('$lib/atlas/ui/AutoModelPage.svelte')).default`.
- **`+page.svelte`.** It renders `data.component` or the existing Planned page.

This keeps the 101 planned pages free of the auto-page code. `AutoModelPage`, the view registry and KaTeX are only in the lazy chunk, and the runtime loads in `onMount` (KIT rule 4).

**Lesson Lab** (`/dev/atlas/lab/[slug]/`, W1-D1):

- It has no prerender `entries` unless `ATLAS_INCLUDE_DRAFTS=1`, so production builds publish no draft lab pages.
- Lazy chunks of draft packages may still be emitted into `build/_app`, but no page links to them. `check:build` lists them.
- The lab reads `?spec=models/<dir>` on the client in `onMount`, because reading `url.searchParams` during prerender throws.

**Adapting to the kit.** `ui/kitspec.ts` turns an `AtlasSpec` into the kit's `ModelSpec`:

- symbolic dims become the current values;
- `max_abs_diff: null` becomes "no tolerance", so `ProvenanceBadge` gets `undefined`, not `null`.

It also asserts that atlas `PredictQuestion` and kit `PredictQuestion` are mutually assignable. The kit itself is not changed.

`AutoModelPage` fills `ModelPageShell`. A package overrides any of the 15 snippet props with `sections/<Snippet>.svelte`.

| Shell prop | Generated from | The author supplies |
|---|---|---|
| `goals`, `prequestion`, `lessonNote` | — | `page.*` (required; `check` fails on TODO markers, AK501) |
| `facts` | catalog + spec (params, precision, parity bar and measured value) + build-time bytes | — |
| `input` | the input adapter's `Capture` + presets; layer 0 = `Preprocessed.steps` | preset choice |
| `lesson` | `compileLesson(lesson.ts)` with `auto` steps from `planLesson` → `LessonPlayer` → kit `Walkthrough` | edits in `lesson.ts` |
| `overview` | `AutoOverview`: `pipeline` (kit `ModelOverview` via `specToLayers`) for ≤ 20 layers; `stack` (collapsible inferred groups plus a residual line) for blocks | optional `overview` hints |
| `inspector` | `AutoInspector` on the kit `Inspector`: Values/Stats from the trace; Math from `op.formula` filled with `explain()` values; Code from `layer.code` / `op.pytorch` | — |
| `unit` | the zoom view for `op.zoom` (or `layer.ui.zoom`) | optional custom view |
| `output` | `output.view` (e.g. `kit.probbars`) | — |
| `tryThis`, `checkpoint` | rendered from `page.tryThis` / `page.quiz` (predict first, then look; deep links `#try-<id>`) | the content |
| `training` | a `train` source → `train-controls` + `loss-curve` + `plane2d`/`curve1d`; or `spec.training` facts | — |
| `article` | `content/En.svelte` / `Zh.svelte` if present | the prose |
| `code` | `layer.code` + the op kernel name | — |
| `links`, `moreLinks`, `credits` | catalog prerequisites; `credits` | — |

**Per-layer zoom views.** Nine zoom views cover every op, chosen by arithmetic. They are built new, modeled on the bespoke components that exist today:

| Zoom view | Modeled on | Serves |
|---|---|---|
| `zoom.weighted-sum` | ConvDetail, DenseDetail, DotProduct | conv2d, linear, attention_scores, attention_mix, similarity, conv1d |
| `zoom.reduce` | PoolDetail | maxpool2d, mean_pool |
| `zoom.elementwise` | ReluDetail, GeluPlot | relu, tanh, sigmoid, gelu, silu, add |
| `zoom.normalize` | BlockView LayerNorm steps | layernorm, l2norm |
| `zoom.softmax` | SoftmaxDetail | softmax |
| `zoom.lookup` | TinyGPT embedding views | embedding, position_embedding, cls_token |
| `zoom.index-map` | FlattenDetail | flatten, concat, to_tokens, select_token, causal_mask |
| `zoom.gate` | (new) gate bars + a cell-state line | lstm, gru |
| `zoom.formula` | FormulaCard | any `PhaseDef.formula` (kind C/B steps) |

`gcn_conv` uses a local `gcn/aggregate` view.

**Displays by axes.** The `layer.tensor` view, also used for thumbnails:

| Axes | Display |
|---|---|
| `c,y,x` | tiles (TileStack) |
| `t,d` | token grid with chip headers |
| `h,tq,tk` | head grid (AttentionMatrix) |
| `class` / `vocab` | probability bars (sequential) or a logit strip (diverging) |
| `p,xy` | `plane2d` |
| `t` | `curve1d` |
| `n,f` | the gcn local view |
| otherwise | TensorView |

**Shell size.**

- Auto page target: ≤ 220 KB gzip.
- Planned pages must stay within 5 KB of their wave-0 baseline. `check:build` samples `m/resnet-18` and enforces this.

### 5.2 Auto-lesson planner (`lesson/plan.ts`, simplified)

The planner walks `spec.layers` in order:

1. **Input step** (`input`): the adapter's `layer0` narration ("Your drawing becomes 784 numbers").
2. **First occurrence** of an op type gives a full step (`op:<id>`: what + how + look). If `op.unit.scan` is set, the step plays a `scan` source and waits for `done`.
3. **Distinct repeat** (`op.distinct`, e.g. conv `inChannels` 1 → >1) gives a short step with the `repeat` beat.
4. **Repeated groups** (inferred, §4.2): the first group's layers follow rules 2–3. The later groups give one summary step (`group:<id>`, "the same block, N more times"). Their layers get no steps of their own.
5. **Shape-only ops** (`category: 'shape'`) get a step only when the axes kind changes (`c,y,x` → vector).
6. **Output step**: the answer plus its limits ("a share among N known classes, not certainty").
7. **Spec hints win:** `teach.lesson`, `teach.title` and `teach.narration`.

There is no pattern collapsing and no budget folding. Lint warns (AK407) when the count falls outside 5–12 steps, and the author edits `lesson.ts`.

Kind-B and kind-C motifs come from the sources:

| Source kind | Motifs |
|---|---|
| train | `train:data@s`, `train:predict@s` (live-model trace of one example), `train:loss@s`, `train:grad@s` (one weight, fine phase), `train:update@s`, `train:watch@s` (run to step n) |
| process | `proc:event@s` (first event of each phase, with `PhaseDef.formula`), `proc:run@s` (play to the end) |

**Reported metrics, not gates** (W1-C reports them at the wave-1a integration):

- how many of the hand-made TinyCNN lesson ids (`pixels, slide, kernels, relu, combine, pool, grow, flatten, dense, softmax`) the auto plan from the shipped spec covers (expected 7–8; `kernels` and `grow` are author steps);
- the TinyGPT auto step count.

The hand-written lessons stay the shipped ones.

### 5.3 Narration language and built-in facts

**Templates** use `{name|fmt}`.

- en and zh use the same placeholder set (AK403).
- There is no `eval` and there are no expressions; computed values come from `vars()`.
- AK402 warns (it does not block) on literal numbers with two or more digits or with a decimal point. A quoted number that disagreed with the screen was the most common reviewer finding.

**Built-in paths:**

| Group | Paths |
|---|---|
| Per layer | `layer.<id>.<stat>`, stat ∈ {`max`, `min`, `argmax`, `negShare`, `zeroShare`, `shape`, `strongestChannel`}. Parsed right to left: the last segment is the stat, the rest is the id, which may contain dots and `#sub` (`layer.L0.attn.scores#h2.max`) |
| Output | `top.label` `top.p` `runnerUp.label` `margin` |
| Selection | `sel.layer` `sel.ch` `sel.cell` `sel.value` |
| Unit | `unit.<name>` (`explain()` values at the selection) |
| Training | `train.step` `train.loss` `train.<metric>` |
| Process | `frame.phase` `frame.<metric>` `run.length` |
| Options | `opt.<name>` (source options) |
| Exported facts | `fact:<key>` (facts.json) |

**Op templates** are short, reviewed and bilingual. They are reviewed once in wave 1 (R0, §8.2), because they reach every auto page. Example, `conv2d.what` (en): "A kernel is {kernelCount} learned numbers. It slides over the grid one cell at a time; at each stop the window is multiplied by the kernel, the products are added, and a bias is added. That sum is one cell of the {outShape|shape} map."

### 5.4 How a contributor overrides any part

| Level | Contributor writes | Effect |
|---|---|---|
| L0 | Python only (`describe()`, `teach=`/`ui=` hints in builder calls) | Auto page + auto lesson; quality stays `draft` |
| L1 | `explainer.ts` + `lesson.ts`: `auto('op:patch', { title, narration })`, `autoAll({ until: 'op:pos' })`, own `StepDef`s; `claims.test.ts` | The normal path; required for `reviewed` |
| L2 | `views/<key>/` local view (≈ 20–150 lines on kit parts) | A custom stage, referenced as `'<slug>/<key>'` |
| L3 | `sections/<Snippet>.svelte` | Replaces one shell section |
| L4 | `src/routes/m/<slug>/` static folder | Fully custom page, as TinyCNN is today |

- `auto()` steps track the op templates, so wording fixes reach every page. Author-owned steps are never rewritten.
- **Tier rule (automated, AK409):** a lesson made only of auto steps cannot be `reviewed`.
- Everything else about quality is in the human review checklist (§6.4).

---

## 6. Contributor workflow

### 6.1 End to end (kind A shown; kinds B and C skip the Python steps)

| Step | Command / action | Output |
|---|---|---|
| 1 | Read `site/DEVKIT.md` and the catalog entry; run `npm run atlas -- doctor` | environment OK |
| 2 | In our parallel waves: `npm run atlas -- worktree tinyvit`, then work inside `site/.work/tinyvit/site`. Then `npm run atlas -- new model tinyvit --kind trace --input drawing28 --template drawing-classifier` | package skeleton + checklist |
| 3 | Write `pipeline/tinyvit/model.py`; run `train.py` with the configured Python (data via `ak.data.quickdraw`, fixed seed, `runs/<run>/best.pt`) | checkpoint + metrics |
| 4 | Write `export.py` (`describe()` + `export_model`), then `npm run atlas -- export tinyvit` | assets, parity, draft lesson |
| 5 | `npm run dev -- --port 5191` → `/dev/atlas/lab/tinyvit/` and `/m/tinyvit/` (drafts are included in dev) | live page with auto lesson |
| 6 | Edit `explainer.ts` (goals, prequestion, tryThis, quiz, credits) and `lesson.ts` (narration, own steps); add `claims.test.ts` and a local view if needed | hot reload in the lab |
| 7 | `npm run atlas -- lesson tinyvit lint`, then `check tinyvit`, then `e2e tinyvit` | gates green, screenshots |
| 8 | Write `notes/tinyvit.md` ("What", "Decisions", "Final numbers", "How to re-run", "Kit issues"); `npm run atlas -- review tinyvit` | review packet |
| 9 | The teaching reviewer completes the checklist (wave 3). The main session runs `merge`, `promote` (applying the printed catalog patch) and `check --all`, then commits locally | live page |

For **kind B or C without Python**:

- write `trainer.ts` or `process.ts`;
- check it with `atlas run <slug> <source> --frames 0..40`;
- then write the lesson.

A twin or reference is optional and gives the stronger badge.

### 6.2 Checklist (each item names the gate that enforces it)

1. **Provenance.**
   - Every stage has a badge from its step or source (AK502).
   - `illustrative` needs a note.
   - `spec.parity` is written only by `atlas parity`.
   - Badge texts:
     - A: "matches PyTorch within {bar}";
     - B with a twin: "Live · f64 in your browser · matches PyTorch fp64 to 7 significant digits over 50 steps";
     - B without a twin: "core ops match PyTorch fp64 (op tests)";
     - C: "same steps as an independent Python run".
2. **Honest numbers.**
   - Placeholders for live values; literal multi-digit numbers warn (AK402).
   - Measured facts come from `facts.json`.
   - Every factual sentence is either live or covered by a claim test (AK408).
   - No `Math.random` outside `core/rng.ts` (AK503).
   - A narrated update rule must match the optimizer that ran (`updateOf` throws for Adam).
3. **Bilingual.**
   - en complete.
   - zh is required for every title and UI label. `I18n.zh` is optional in the type, so lint enforces this (AK406).
   - zh narration is required at `reviewed`.
   - `lesson print --lang zh` gives translators a table.
   - Terms come from the glossary.
   - Kit-side UI strings live in `atlas/ui/strings.ts`, not in the shared `i18n/strings.ts`.
4. **a11y.**
   - Every view's `describe()` is non-empty for every fixture (AK504).
   - Keyboard works (inherited from Walkthrough; views do not trap keys).
   - Reduced motion: no autoplay, and every `play` has a step list.
   - Nothing is understandable only while animating.
5. **Phone layout.**
   - 360 px with no horizontal scroll; touch input works.
   - Stage `minHeight` declared.
   - Auto-loaded bytes ≤ 5 MB.
   - Auto page shell ≤ 220 KB gzip.
6. **Size budgets.** Each file ≤ 5 MB; each model ≤ 15 MB; phone tapes ≤ 2 MB (AK303).
7. **Licenses.**
   - `credits` lists every dataset, ported code or idea-with-code.
   - `promote` appends them to `licenses.json`.
   - No code from GPL, NC or unlicensed projects (AK704).
8. **Parity.**
   - A: every traced id and sub-step ≤ 1e-4 at dim extremes and edge cases. `explain` equals the trace ≤ 1e-5 (all units, or 2,000 sampled).
   - B: SGD for 50 steps ≤ 1e-10 relative; the page's optimizer for 50 steps ≤ 1e-7 relative; step-1 grads (and double grads) ≤ 1e-10 relative.
   - C: frames equal the Python reference (discrete fields exact, floats ≤ field `tol`, draws replayed). `state(i)` equals a straight replay for 200 random i.
9. **Edge inputs.** Adapter edge fixtures plus model ones: no NaN, no crash, and the "empty" note is shown.
10. **Lesson shape (AK407).**
    - 5–12 steps (warning outside).
    - Narration 2–4 sentences and ≤ 90 English words.
    - At least one `predict`, one `task` and one step with `kindOf: 'limits' | 'breakIt'`.
    - Deep-link ids unique.
11. **Notes.** `notes/<slug>.md` has "What", "Decisions", "Final numbers", "How to re-run" and "Kit issues".
12. **Ownership.** No file outside the package's owned paths changed (`merge --dry-run`, AK702).
13. **Size of the page code.** The `check` report counts page-specific lines (files under `explainers/<slug>/`, excluding tests). The target is ≤ 1,000.

### 6.3 Gates (`atlas check <slug>`)

Every command stays under 9 minutes (CONTRACTS rule 4).

| Gate | Bar | Blocking | Time |
|---|---|---|---|
| Schema | <ul><li>spec 1.1, ref-1, tape-1 and facts-1 validate (hand-written TS validators)</li><li>slug = folder = catalog slug (unless `devOnly`)</li><li>ops registered</li><li>`inferShape` matches `outShape` at dim extremes</li></ul> | yes | seconds |
| Parity A / B / C | §6.2 item 8 | yes | ≤ 2 min |
| Op tests | goldens: JS vs torch fp64 on the same f32 inputs, \|a − b\| ≤ 1e-6 + 1e-6·\|b\|; explain = trace; templates resolve en/zh | yes | < 1 min |
| Lesson lint + Node dry run + claim tests | every step × every fixture × en/zh: cursors resolve, placeholders finite, views registered, terms exist; `claims.test.ts` passes | yes (zh narration: warning at `draft`) | ≤ 2 min |
| Process determinism | same seed twice → identical frame hash; `keyEvery=1` equals `keyEvery=64` (catches unreported mutations) | yes | < 1 min |
| Honesty lint | no `Math.random`; dependency rules (Node imports every pure file); literal-number warnings | yes / warn | seconds |
| Budgets | §6.2 items 5–6 | yes | seconds |
| Site | `npm run check` 0/0; `check:catalog`; build into `.atlas/build-<slug>/`; `check:build` (no third-party URLs, shell budgets) | yes (skipped by `--fast`) | ≤ 6 min |
| E2E (sampled matrix) | <ul><li>(1) every step at 1280 px, en, light;</li><li>(2) every step at 360 px, en, dark;</li><li>(3) zh text pass over every step: no "…", no English fallback in narration, no screenshots;</li><li>(4) reduced motion, one pass over the step lists.</li></ul> Live-training steps run with `?atlas-e2e=1`, which caps training at a fixed step count. Checks: no console errors; every step renders with a badge; no horizontal scroll at 360 px; no main-thread long task > 50 ms from the engine and none > 100 ms from any cause (1× CPU) | yes | ≤ 8 min |
| Paperwork | notes complete; credits complete | yes | seconds |

`check --all` runs every package's non-build gates, then builds once into `.atlas/build-all/` and runs `check:build`. It also runs the legacy parity scripts (`--no-write`).

### 6.4 Review and merge (local only; no GitHub push until the user asks)

**Lanes.**

- Each package works in its own `site/.work/<name>/site` copy, on its own port (static table in `DEVKIT.md`).
- Python writes into `pipeline/<slug>/` in the main tree, which is disjoint between packages. Its assets go to `$ATLAS_SITE`, the lane.
- A blocking kit bug is worked around inside the package (local op, local view, author step, section override) and logged under "Kit issues". Lanes never edit the kit.

**Review (wave 3).** A teaching reviewer (the round-3 "first-year student" protocol) uses `notes/review-<slug>.md` and the Lesson Lab. They tick a 10-item checklist in that file:

1. Measurable goals.
2. One idea per step with a focused "look at" hint.
3. No term before it is defined.
4. Narration numbers match the screen on 3 presets.
5. Every factual sentence is live or covered by a claim test.
6. The first occurrence of an idea is worked out fully; repeats are summarized.
7. A misconception or limits step.
8. Try-this experiments ask for a prediction first.
9. Natural Chinese with glossary terms.
10. Credits are correct.

The reviewer edits only that package's files and sets `quality: 'reviewed'`.

**Merge.** The main session:

1. runs `atlas merge <slug>` (owned paths only);
2. runs `atlas promote <slug>` and applies the printed catalog patch;
3. runs `atlas check --all`;
4. commits, one commit per model;
5. updates `LOG.md`.

**Kit changes** (`atlas/**`, `scripts/atlas/**`, `pipeline/atlas_kit/**`) are separate commits.

- They must keep every package's gates and the legacy parity green.
- They merge before the models that need them.
- They reach running lanes through `worktree --sync`.

### 6.5 External contributors and CI (written now, used once the repo is public)

**`CONTRIBUTING.md`** (W1-E2) describes the path for people outside the maintainers' computer:

1. fork and clone;
2. `cd site && npm ci`;
3. `pip install -r pipeline/requirements.txt` (only for kind A or twins);
4. `npm run atlas -- doctor`;
5. `npm run atlas -- new model <slug>`, for a slug already in the catalog. New catalog entries are proposed in an issue first; `--dev` is for experiments;
6. work on a git branch. No `.work` copies are needed; those exist for our parallel agents;
7. `npm run atlas -- check <slug>` and `e2e <slug>`;
8. open a PR with the `atlas review` packet.

Maintainers run `promote`. Contributors never edit `catalog/*`, `licenses.json` or `atlas/**`. Kit changes are proposed as separate PRs.

**CI** (designed, off until the user agrees to push):

- Linux, Node 24, Playwright Chromium;
- `check --all` and the sampled e2e for changed packages;
- CODEOWNERS: maintainers own `src/lib/atlas/**`, `src/lib/kit/**`, `scripts/atlas/**` and `pipeline/atlas_kit/**`; the contributor plus a maintainer own each `src/lib/explainers/<slug>/**` and `pipeline/<slug>/**`.

---

## 7. The next batch, built with the DevKit

None of these needs a new download. They use:

- Quick, Draw! (already downloaded);
- TinyStories (already downloaded);
- networkx's built-in karate club (installed in the venv);
- seeded synthetic generators.

**Flagged for the user's OK, not in this batch:**

- a public-domain real series (e.g. NOAA CO₂) for the forecaster;
- Speech Commands for the later KWS model.

**Built in two rounds.**

- **Wave 2a:** one model of each kind (TinyViT, MLP playground, gridworld). It proves the kit.
- **K0:** kit fixes from their "Kit issues".
- **Wave 2b:** the other six.

| Round | Model (slug, port) | Kinds | Sources | Ops / processes | Auto-generated | Custom (est. page lines) | Data | Size |
|---|---|---|---|---|---|---|---|---|
| 2a | TinyViT (`tinyvit`, 5191) | A | <ul><li>`fwd` trace;</li><li>`attn` scan (one query row);</li><li>`cnn` trace with `assets: 'models/tinycnn'` (the compare step runs both models live on the same drawing)</li></ul> | <ul><li>shared: conv2d (stride = kernel = 7), position_embedding, add, layernorm, linear, attention_scores (non-causal), softmax, attention_mix, gelu;</li><li>local: `to_tokens`, `cls_token`, `select_token`</li></ul> | stack overview (4 inferred block groups), inspector, all zooms, draft | `views/patch-grid` (attention row over the drawing as a 4×4 patch overlay); 3 local ops; lesson edits; compare step (≈ 600) | Quick, Draw! 10 classes, same split and `preprocessStrokes` as TinyCNN | d = 48, 4 blocks, 4 heads, **MLP ratio 4** (as TinyGPT): ≈ 117k params, ≈ 470 KB weights; references ≈ 0.6 MB |
| 2a | MLP playground (`mlp-playground`, 5192) | B (+A on live weights) | <ul><li>`train` (2→[4,4]→2, tanh/ReLU, Adam; coarse every 5 steps);</li><li>`sgd1` (same trainer, SGD, fine detail for the slowed-down step);</li><li>`pt` live-model trace of one point;</li><li>`hiLr` (lr = 3)</li></ul> | autograd: linear, tanh, relu, crossEntropy; `Module.toSpec` → `zoom.weighted-sum` on one point | train motifs; `plane2d`, `network`, `train-step`, `loss-curve`, `train-controls` | `views/neuron-grid` (per-neuron heatmaps, derived field `grids`); spiral `breakIt`; presets (≈ 500). Reimplemented on our core, borrowing TF Playground layout ideas (Apache-2.0; credited) | `atlas/data/points2d.ts` (moons, circles, spiral, xor) + brush | no shipped weights (seeded init); a spec for architecture + parity |
| 2a | Gridworld + bandits (`gridworld`, 5193) | C | <ul><li>`bandit` (choose → pull → update; you / ε-greedy / UCB / Thompson; `recordDraws`, because Thompson uses Beta draws);</li><li>`q` (act → reward → update → end; `Q` tracked by changes; derived V and policy);</li><li>`vi` (value-iteration sweeps)</li></ul> | processes in `process.ts`; `PhaseDef.formula` for the UCB bonus and the Bellman target | proc motifs, `zoom.formula`, `tape.table` | `views/grid-board` (editable walls/goal/pit, Q triangles, arrows), `views/arm-bars` (estimates, Beta posteriors, regret); local adapter `adapters/gridworld` (InputKind `game`) (≈ 900) | editor presets (cliff, maze) | presets.json ≈ 2 KB |
| 2b | Char LSTM (`char-rnn`, 5194) | A + C | <ul><li>`fwd` trace (LSTM with gate subs);</li><li>`unroll` process over the trace (gates → cell → hidden → predict per character);</li><li>`sample` (library/sampling, TinyGPT semantics);</li><li>`fwdGru` only if time allows</li></ul> | shared: embedding, lstm, linear, softmax; local `gru` optional | `zoom.gate`, overview, inspector, draft | `views/rnn-timeline` (hidden `[H×T]` heatmap under the characters); precomputed `unitTop.json` (top contexts per unit); "a cell decays after a quote" `breakIt` (≈ 600) | TinyStories characters (already downloaded) | vocab ≈ 70, emb 24, h = 64: LSTM ≈ 29k params (117 KB); optional GRU ≈ 23.5k (94 KB) |
| 2b | GCN, karate club (`gcn`, 5195) | A + B | <ul><li>`fwd` trace;</li><li>`mp` scan of `gcn_conv` (self-loop → normalize Â → transform XW → message per incoming edge → aggregate → activate);</li><li>`retrain` (the learner picks the labelled nodes)</li></ul> | <ul><li>local: `gcn_conv` (sub `xw`; Â explained per neighbour);</li><li>shared: tanh, linear, softmax;</li><li>autograd matmul with a constant Â</li></ul> | inspector, draft, train motifs | `views/graph` (node-link, focus-linked to the adjacency grid), local `aggregate` view, adapter `adapters/graph` (edit edges), 2-D embedding on `plane2d`; 8-layer over-smoothing `breakIt` (≈ 900) | `ak.data.karate()`, **unweighted** A (no download; cite Zachary 1977); offline 2-D layout | 34 → 4 → 2 GCN + 2 → 2 head: 156 params; karate.json ≈ 3 KB |
| 2b | 2-D flow matching (`flow-2d`, 5196) | B + C (+A on live weights) | <ul><li>`train` (vθ(x,y,t) MLP 3→64→64→64→2, SiLU, Adam 1e-3, batch 256; derived vector field on 20×20 at the selected t);</li><li>`pairs` (fine: x₀, x₁, t, x_t, target vs predicted arrow);</li><li>`sample` (library/euler, 400 particles, K = 50);</li><li>`sampleFew` (K = 4);</li><li>precomputed `train-smiley` tape for phones</li></ul> | autograd linear/silu/concat/mse; library euler; `toSpec` → concat + linear + silu trace | train + proc motifs; `plane2d` layers (points, vectors, trajectories, time slider) | lesson; "fewer steps blur" compare; a link to diffusion (≈ 500) | painted points + presets (smiley, moons, ring) | 8,706 params ≈ 35 KB per preset (3 presets ≈ 105 KB) |
| 2b | TinyCLIP, synthetic (`tinyclip`, 5197) | A (+B-lite on the N×N logits) | <ul><li>`img` and `txt` traces (multi-input);</li><li>`sim` (local similarity op over a batch of 6 pairs);</li><li>`nce` (autograd infoNCE on the logit matrix only)</li></ul> | <ul><li>shared: conv2d, relu, maxpool2d, flatten, linear, embedding;</li><li>local: `mean_pool`, `l2norm`, `similarity` (learned temperature)</li></ul> | per-tower inspector and zooms, draft | `views/similarity` (N×N with drawing thumbnails and caption labels, temperature slider), `sections/Overview.svelte` (two lanes), adapter `adapters/shapes-captions`; "bag of words ignores order" `breakIt` + claim test (≈ 900) | synthetic shapes × size × place, drawn as strokes and rendered with **TS** `preprocessStrokes`; template captions with a ≈ 24-word vocab; `atlas run tinyclip synth --save` writes the training set for Python | ≈ 16k params, ≈ 64 KB |
| 2b | Time-series forecaster (`tinyforecaster`, 5198; covers the TCN idea of `tiny-tcn`) | B + C (+A gates) | <ul><li>`windows` process (L = 24, H = 1 or 8);</li><li>`train` LSTM (h = 16) or TCN (k = 2, dilations 1/2/4/8, 8 channels);</li><li>`rollout` (library/rollout, error vs truth);</li><li>`naive` seasonal baseline</li></ul> | shared: lstm (`zoom.gate`), linear; local: `conv1d` (dilated causal; `receptive`) | train + proc motifs; `curve1d` (editable, window overlays) | `views/tcn-rf` (dilation tree on the series); baseline compare; random-walk `breakIt` (≈ 700) | `atlas/data/series.ts` (sine, trend, seasonal + noise, square, random walk) + draw a curve | LSTM ≈ 1.2k params, TCN ≈ 0.45k; presets < 20 KB |
| 2b | PINN, damped oscillator (`pinn`, 5199) | B (createGraph) + C (`deriv`) | <ul><li>`plain` and `pinn` trainers sharing init via `rng.fork('init')`: MLP 1→32→32→1 tanh, Adam; derived u, u′, u″ and the residual on 200 points;</li><li>`deriv` (one t: autograd vs finite difference);</li><li>`wrongC`</li></ul> | autograd `grad(…, { createGraph: true })` twice; mse | train motifs, `curve1d`, `loss-curve` | `views/loss-bars` (data / physics / initial-condition stacked), `views/residual-strip`; λ slider task; wrong-c `breakIt` (≈ 600) | analytic underdamped solution; 6–10 noisy points in the first 40 % of time; input via the `series` adapter (draggable observation points) | ≈ 1.15k params; no weights shipped |

**Lesson outlines.** Each is the auto draft plus the builder's edits; step ids are given in short form.

- **tinyvit (11):** pixels → patches (scan over 16 patches) → patch embedding (49 → 48) → position + [CLS] → q·k for [CLS] (predict: "where will the [CLS] row look?") → attention row on the drawing → mix V → MLP → blocks 2–4 summary → classes → compare with TinyCNN on the same drawing.
- **mlp-playground (10):** points → one neuron is a soft line → a layer is several lines → combining gives curved regions → loss per point (predict) → gradient on every edge → one SGD update → watch to step 300 → spiral with 2 neurons underfits (task: add neurons) → lr = 3 diverges (compare).
- **gridworld (6 + 10):**
  - Bandits: pull 10 times yourself → running average → greedy gets stuck → ε explores → UCB bonus with live counts → regret compare.
  - Gridworld: world → Q-table → ε-greedy choice (u vs ε) → reward → target r + γ max Q′ → update Q ← Q + αδ → one episode → values spread back over episodes → policy arrows → edit the maze and relearn, γ 0.5 vs 0.95.
- **char-rnn (11):** characters → h as memory → forget gate → input gate + candidate → cell c = f⊙c + i⊙g for one unit → output gate → next-character bars → unroll through time → a unit that tracks something (precomputed) → sampling → `breakIt`: decay.
- **gcn (10):** graph and adjacency linked → one-hot features → self-loops → Â = D^-1/2(A+I)D^-1/2 with live degrees → XW → messages into the selected node → tanh → two hops → 2-D embedding by faction → task: pick 2 labelled nodes and retrain; compare: over-smoothing.
- **flow-2d (11):** paint data → noise → straight pairs → x_t on a time slider → the job (x,y,t) → arrow (predict) → loss on one pair → one step → watch the field → Euler sampling → 4 steps vs 50 → link to diffusion.
- **tinyclip (10):** two towers → image vector → caption vector → normalize → cosine for one pair → N×N matrix → row and column softmax → the gradient pulls the diagonal → task: write 3 captions → `breakIt`: "circle above square" equals "square above circle".
- **tinyforecaster (9):** series → windows → receptive field (TCN) or gates (LSTM) → forecast arithmetic → loss → train live → rollout errors compound → beat the seasonal-naive baseline? (live numbers) → `breakIt`: random walk.
- **pinn (10):** the law → data points → a plain net wanders → u′, u″ by autodiff vs finite difference → residual strip → loss bars → plain vs PINN side by side → extrapolation → task: the λ slider → `breakIt`: a wrong c gives a confident wrong fit.

**Parity per model.**

- **A** (tinyvit, char-rnn, gcn, tinyclip): ≤ 1e-4 for every id, including gate subs, on CPU references. Edge cases: blank and dot drawings; T = 1 and T = max text; an isolated node.
- **B:** the §6.2 bars. For PINN, u′, u″ and the physics-loss gradients ≤ 1e-10 relative against `torch.autograd.grad(create_graph=True)`.
- **C:**
  - gridworld via `pipeline/gridworld/reference.py`: positions, actions and episodes exact; Q ≤ 1e-12. Thompson and ε draws are replayed from the TS tape, and near ties follow the AK205 rule.
  - Euler sampling against the twin.
  - Rollout frames identical.

**Catalog follow-ups.** These are applied at wave-3 integration through the patches `promote` prints:

- `gcn`: the hook says "Edit a molecule". It should say karate club; molecules stay a later preset.
- `mlp-playground`: the runtime should say "own autograd core in a worker", with a matching `tech-zh.ts` entry (`check:catalog` requires it).
- `tinyforecaster`: its prerequisite is the planned `scalar-lstm`. Replace it with `char-rnn`, where LSTM gates are taught, or keep it and accept a link to a planned page. The user decides at integration; the default is `char-rnn`.
- `tinyvit`: replace the design-time params/size strings ("≈0.2M (design)") with the measured values (and `tech-zh.ts` entries where they are not language-neutral).
- `tiny-tcn` stays P2. Its TCN idea lives in the forecaster page.

**Later models.** These are mostly reuse; each new op goes through `new op`.

| Model | Mostly reuses | New |
|---|---|---|
| word2vec | B on embedding + dot; C pair generation | — |
| TinyBERT | shared attention ops, non-causal | promote `cls_token` / `select_token` from tinyvit (second user) |
| U-Net | conv2d padding | `upsample`, skip `concat` |
| Matrix factorization | B | — |
| KWS | — | an `audio` adapter and a `logmel` op (needs the Speech Commands download OK) |
| Concept packages (gradient descent, backprop, optimizers…) | kinds B and C as they are | `AutoConceptPage` and the `/train/[slug]` route change |

---

## 8. Build plan

The waves are:

1. wave 0;
2. wave 1a, then integration 1a;
3. wave 1b, then integration 1b with review R0 and the afternoon test;
4. wave 2a;
5. K0;
6. wave 2b;
7. wave 3, which can start on the 2a models while 2b runs.

**Rules for every wave:**

- Packages in one wave own disjoint paths.
- A package may stub another package's exports only inside its own folder (for example `lesson/fixtures/stub-ops.ts`).
- Acceptance items that need another package's code are listed under the integration gate, not under the package.

### 8.0 Wave 0: main session, sequential, ≈ 2 hours

Wave 0 starts **after the round-3 merge commit**. It records that commit's hash in the CONTRACTS addendum and re-checks the `WalkStep`, `WalkContext` and `PredictQuestion` shapes against §4.

**Owned files:**

- `site/src/lib/atlas/types/{core,spec,op,process,autograd,train,lesson,runtime,view,explainer,engine,index}.ts`, verbatim from §4, plus `types/contract.test.ts`.
- `site/package.json` (scripts only), `site/.gitignore`, the root `.gitignore`.
- `site/svelte.config.js` and `site/scripts/check-build.ts`: the `ATLAS_BUILD_DIR` variable; a planned-page budget at baseline + 5 KB; sample pages `m/tinycnn`, `m/tinygpt`, `m/resnet-18`.
- `site/src/lib/server/assets.ts`: `NOT_LOADED` matches `reference.*` and `*.ref.*`.
- `pipeline/parity/tinycnn.parity.ts` and `tinygpt.parity.ts`: a `--no-write` flag. The default behaviour is unchanged.
- `CONTRACTS.md` (additive "DevKit additions": spec 1.1 keys, `atlas-ref-1`, the `#` sub-step separator, the CPU/no-TF32 rule, lanes and `ATLAS_SITE`/`ATLAS_ROOT`), `LOG.md`, `research/devkit/**`.

**Acceptance:**

1. `npm run check` is 0/0. `node -e "import('./src/lib/atlas/types/index.ts')"` loads under type stripping.
2. The default build is unchanged: 192 pages, and `check:build` OK.
3. `ATLAS_BUILD_DIR=.atlas/build-test npm run build` works.
4. Both legacy parity scripts with `--no-write` leave `spec.json` byte-identical (hash before and after), and still pass.
5. Shell-size baselines are recorded in `LOG.md` (`/m/resnet-18/`, `/m/tinycnn/`).

**Decisions to confirm with the user before wave 0** (recommended defaults in brackets):

| | Decision | Default |
|---|---|---|
| D1 | Spec 1.1 additive keys, `atlas-ref-1` (base64 JSON) and the `#` separator, as a CONTRACTS addendum | yes |
| D2 | Packages under `src/lib/explainers/<slug>/` | yes |
| D3 | Draft packages are built only with `ATLAS_INCLUDE_DRAFTS=1`. A package goes live when `promote` flips its catalog status | yes |
| D4 | The hand-written TinyCNN and TinyGPT lessons stay as they are | yes |
| D5 | Karate club via networkx, not a download | yes |
| D6 | The forecaster stays synthetic | yes; a real series only with the user's OK |
| D7 | MLP playground reimplemented, not ported | yes |
| D8 | Wave 2 in two rounds (3 + 6) | yes |
| D9 | The licence for code and lesson text before any public push | MIT for code; CC BY 4.0 for lesson text; datasets keep their own licences per `licenses.json`. **The user decides** |

### 8.1 Wave 1a: foundation cores (4 packages in parallel)

Every package works in `site/.work/<pkg>/site`. Wave 1 copies these by hand (the CLI does not exist yet): skip `node_modules`, `.svelte-kit`, `build`, `.work` and `.atlas`, and patch the copy's `vite.config.ts` `fs.allow`. Every package treats `types/**` as read-only and writes `notes/devkit-<pkg>.md`. Ports run from 5201 upward.

**W1-A: runtime A, shared ops, validators (TS)**

- **Owns:**
  - `site/src/lib/atlas/{core,ops,graph}/**`
  - `site/src/lib/atlas/schema/{spec,ref,facts}.ts` and `schema/ops.json`
  - `pipeline/atlas_fixtures/ops/**`, `site/tests/atlas/ops/**`
  - `notes/devkit-runtime.md`
- **Deliverables:**
  - `core/`: tensor helpers, rng with the test vectors, io (`loadJson` that strips a BOM, `fsFetch`), NaN-safe `compare` with a relative mode, `format`, `tex.ts` (`sym`).
  - All 20 `CORE_OPS`, each with forward, attrs, explain, receptive (spatial ops), formula, bilingual narration, vars/distinct, parity bar and a `pytorch` line.
  - `ops/manifest.ts`, which generates `schema/ops.json`.
  - `graph/` (executor, explain, receptive, refs, legacy per §3.2).
  - Hand-written validators.
  - Torch goldens on CPU: 8 seeded cases per op, f32 inputs, fp64 reference.
  - Node tests.
- **Acceptance:**
  1. The executor on the TinyCNN spec equals the `tinycnn.ts` forward **bit for bit** on the 3 reference inputs, the 60 class presets and the 12 held-out presets. Against the reference it is within 1e-4 (2.86e-6 expected).
  2. The executor on the normalized TinyGPT spec matches `tinygpt.ts` within 1e-6 relative (|a − b| ≤ 1e-6·max(1, |b|)) at T = 1, 9, 22, 37 and 64. The T = 1 and 64 cases are tokenized fixed strings, so no PyTorch file is needed. Against the shipped PyTorch reference (T = 9, 22, 37) it is within 1e-4. If the local, git-ignored `verify_ref.json` exists, T = 64 against PyTorch is reported but does not gate.
  3. `explain` equals the trace (≤ 1e-5) for every unit of every TinyCNN layer and 2,000 sampled TinyGPT units.
  4. Every op passes its goldens (|a − b| ≤ 1e-6 + 1e-6·|b|).
  5. Both shipped specs validate after `normalizeSpec`.
  6. Node imports every file; `npm run check` 0/0.

**W1-B1: autograd core and trainer (pure TS)**

- **Owns:**
  - `site/src/lib/atlas/{autograd,train,data}/**`
  - `pipeline/atlas_fixtures/autograd/**`, `site/tests/atlas/autograd/**`
  - `notes/devkit-autograd.md`
- **Deliverables:**
  - The autograd core (§4.5) with `createGraph`.
  - nn (state_dict names, LSTMCell with both biases), optim, losses, gradcheck.
  - `tospec.ts`, `trainingProcess`, `train/helpers.ts` (`updateOf`).
  - `points2d` and `series` generators.
  - Torch fp64 fixtures on CPU.
- **Acceptance:**
  1. gradcheck and gradgradcheck pass for every op (relative ≤ 1e-6).
  2. Values, grads and double grads match torch fp64 within 1e-10 relative.
  3. MLP 2→8→8→2, compared with torch fp64:
     - 50 SGD steps ≤ 1e-10 relative;
     - 50 Adam steps ≤ 1e-7 relative (losses and params).
  4. Tanh MLP u′, u″ and the physics-loss parameter grads match `torch.autograd.grad(create_graph=True)` within 1e-10 relative.
  5. The flow MLP at batch 256 runs ≥ 50 steps/s in Node. If not, record the measured rate and the fallback (a smaller batch plus a precomputed tape).
  6. `trainingProcess` passes the determinism checks of W1-C's harness. Before integration, it uses a local copy of that check.

**W1-C: kind-C runtime and lesson core (pure TS)**

- **Owns:**
  - `site/src/lib/atlas/{process,lesson,runtime}/**`
  - `site/src/lib/atlas/schema/tape.ts`
  - `notes/devkit-lesson.md`
- **Deliverables:**
  - The tape layer:
    - `TapeStore`, `record` / `recorder` (LiveTape with async `ensure`), `saveTape` / `loadTape`;
    - `traceToTape`, `scanTape`, `bpeTape`, `generateTape`, `library/*`.
  - The lesson layer:
    - `define` (`defineLesson`, `auto`, `autoAll`, `defineExplainer`), `narrate` (right-to-left fact paths), `facts`;
    - `plan` (§5.2), `compile`, `lint`, `withFixtures`.
  - The runtime layer: `createRuntime(pkg, …)`, `localRunner`, `fakeRuntime`.
  - Lint fixtures that replay the 4 past reviewer findings: a wrong metric quoted, NaN on a 1-token input, a stale step index, a missing provenance badge.
- **Acceptance:**
  1. Determinism: the same seed twice gives an identical frame hash, and `state(i)` equals a replay for 200 random i. This is tested on a synthetic process with `track: 'changes'` and `recordDraws`.
  2. `bpeTape` reproduces `trainBPE` on `pipeline/parity/bpe_cases.json`.
  3. Every lint fixture is caught with its AK code.
  4. `compileLesson` output passes a contract test with `fakeRuntime`.
  5. A full Node dry run of a 6-step fixture lesson (stub ops) in en and zh.

**W1-E1: Python SDK core and CLI skeleton**

- **Owns:**
  - `pipeline/atlas_kit/**`, `pipeline/requirements.txt`
  - `site/scripts/atlas/**`
  - `notes/devkit-tools.md`
- **Deliverables:**
  - The SpecBuilder machinery, `export_model`, the reference writer (`atlas-ref-1`), `Rng`, `TapeWriter`, `numerics.py` (CPU, no TF32), `paths.py`, `validate`, `data/*`, AK1xx errors.
  - Mirrors for `input`, `linear`, `relu`, `tanh`, `sigmoid`.
  - The CLI skeleton: `root.ts`, `load.ts` (Node package loader), `doctor`, `new` (+ templates, `--dev`, `worker.ts` generation), `run`.
  - `atlas.config.json` plus the maintainer's git-ignored local config.
- **Acceptance:**
  1. Python `Rng` matches the test vectors for `next`, `int` and `fork`.
  2. `export_model` refuses CUDA tensors (AK102).
  3. `doctor` is green here, and it reports a readable failure with the Python path unset.
  4. `ATLAS_ROOT` resolves correctly from the main tree and from a `.work` lane.
  5. `new model demo-x --dev --kind process` scaffolds files that Node imports.

**Integration 1a (main session).**

1. Merge A, B1, C and E1.
2. Run every Node test together.
3. Check `toSpec` → executor forward = Module forward (≤ 1e-6 relative).
4. Run `planLesson` on the real ops and record the TinyCNN mapping and the TinyGPT step count in `LOG.md`.
5. Run `atlas run` on the demo process and on a fixture trainer.
6. Check `new model demo-x --dev --kind process`, then `check demo-x --fast` (as far as the 1a gates exist).

Owners' findings are fixed by the main session or carried into the 1b packages as named items.

### 8.2 Wave 1b: engine, UI, views, tools (4 packages in parallel)

**W1-B2: engine, kind-B views, performance page**

- **Owns:**
  - `site/src/lib/atlas/engine/**`, `site/src/lib/atlas/ui/train/**`
  - `site/src/lib/atlas/adapters/{points2d,series}/**`
  - `site/src/lib/atlas/ui/views/{plane2d,curve1d,loss-curve,train-step,network,train-controls}/**`
  - `site/src/routes/dev/atlas-train/**`
  - `notes/devkit-engine.md`
- **Deliverables:**
  - The `?worker` glob spike, done first (fallback per §4.10).
  - Engine host, client and `engineRunner` (frames → `TapeStore`), following the transfer rules.
  - `TrainSession`; the two adapters with `Capture`; the 6 views (meta + View + demo); the performance page.
- **Acceptance:**
  1. No main-thread long task > 50 ms from the engine while training the flow MLP on `/dev/atlas-train/` (1× CPU).
  2. The main-thread fallback works.
  3. Pause and lr changes act within one frame.
  4. Frames from the worker give the same `TapeStore` states as `localRunner` (hash equal).
  5. Two packages' workers build as separate bundles.
  6. The view contract test passes.

**W1-D1: player, stage, auto page, routes, Lesson Lab**

- **Owns:**
  - `site/src/lib/atlas/ui/*.svelte` and `ui/*.ts` (LessonPlayer, StageHost, TapeScrubber, AutoModelPage, AutoOverview, AutoInspector, runtime.svelte.ts, focus.svelte.ts, registry.ts, packages.ts, kitspec.ts, strings.ts)
  - `site/src/lib/atlas/adapters/{drawing28,text-bpe,text-char}/**`
  - `site/src/routes/m/[slug]/**`, `site/src/routes/dev/atlas/lab/**`
  - `notes/devkit-ui.md`
- **Deliverables:**
  - `LessonPlayer`: compiled steps → `WalkStep[]` + a shared `stage` snippet. `onEnter` applies `select` and `ensure` on the lesson selection only.
  - `StageHost`: clamped cursor, "Computing… k/n", badge, lazy view, `TapeScrubber`, task and predict gating via `ctx.done`.
  - `AutoModelPage`, `AutoOverview`, `AutoInspector`; `kitspec.ts`; the browser package loader; the 3 adapters.
  - The route mechanics of §5.1.
  - The Lesson Lab: one page per slug, with fixture and language selectors, an inline lint list, a `spec.parity.perLayer` strip, and a "copy deep link" button; `?spec=` read client-side.
- **Acceptance:**
  1. `/dev/atlas/lab/tinycnn/?spec=models/tinycnn` renders a full auto page and auto lesson from the shipped assets.
  2. No source file under `routes/m/tinycnn`, `lib/pages/tinycnn` or `lib/models` changed. The rendered text of `/m/tinycnn/` is identical, ignoring hashed asset names. Its shell stays ≤ 300 KB.
  3. All 192 existing pages still build, and every planned slug still renders Planned within its shell baseline + 5 KB.
  4. Keyboard, reduced motion and deep links work in the lab lesson.
  5. `npm run check` 0/0.

**W1-D2: kind-A views and gallery**

- **Owns:**
  - `site/src/lib/atlas/ui/views/{zoom.weighted-sum,zoom.reduce,zoom.elementwise,zoom.normalize,zoom.softmax,zoom.lookup,zoom.index-map,zoom.gate,zoom.formula,layer.tensor,compare,tape.table}/**` and `ui/views/kit.*/**`
  - `site/src/routes/dev/atlas/+page.svelte` (the gallery)
  - `notes/devkit-views.md`
- **Deliverables:** every view with meta, View and a demo on `fakeRuntime`; the gallery.
- **Acceptance:**
  1. The view contract test passes.
  2. The gallery shows every view (including W1-B2's after integration) with no console errors at 360 and 1280 px, in light and dark.
  3. Every zoom view renders a real `UnitExplanation` from TinyCNN or TinyGPT in the lab (checked at integration).

**W1-E2: SDK op mirrors, full CLI, docs, tutorial**

- **Owns:**
  - `pipeline/atlas_kit/**` and `site/scripts/atlas/**` (continuing W1-E1)
  - `site/DEVKIT.md`, `CONTRIBUTING.md`
  - `site/src/lib/explainers/hello-xor/**`, `pipeline/hello-xor/**`, `site/static/models/hello-xor/**` (a few KB; dev-only, lab route only)
  - `notes/devkit-tools.md`
- **Deliverables:**
  - All `CORE_OPS` mirrors, validated against W1-A's `schema/ops.json`.
  - The remaining commands (`export`, `parity`, `lesson`, `check`, `e2e`, `worktree`, `merge`, `review`, `promote`); the 6 remaining templates.
  - `DEVKIT.md`: the afternoon walkthrough, the three kinds, lessons, the error codes, the port table, and the op and view reference (generated from `ops.json` and view metas by a script inside `scripts/atlas/`).
  - `CONTRIBUTING.md`.
  - `hello-xor` (dev-only): a 2-2-1 tanh MLP on XOR covering A (exported trained weights), B (live training) and C (one SGD step in fine phases).
- **Acceptance:**
  1. Re-describing TinyCNN from `pipeline/tinycnn/model_final.pt` (output to `.atlas/tmp/`, never to `static/`) gives the same layer ids, ops, `outShape` and params, and a **byte-identical** `weights.bin`.
  2. Every Python mirror agrees with `ops.json` and with its TS op on the goldens.
  3. `atlas parity tinycnn` reports 2.86e-6 (±1e-7) through the legacy script with `--no-write`, and `spec.json` is unchanged.
  4. `worktree` / `merge --dry-run` / `--sync` behave as specified on a scratch lane.

**Integration 1b (main session).**

1. Merge B2, D1, D2 and E2.
2. Run `npm run check`, `npm run build`, `check:build`, the legacy parity scripts (`--no-write`) and `npm run atlas -- check hello-xor` (every gate).
3. **R0 teaching review:** a reviewer reads every op template (en and zh) and the `hello-xor` lesson, and fixes only `ops/*.ts` narration and `hello-xor`.
4. **Afternoon test:** a fresh agent that has only `DEVKIT.md` builds `hello-xor-2` (`--dev`, sigmoid output) to the `draft` gates. Time it and log it.

### 8.3 Wave 2a and 2b: models built with the kit

**Common ownership for each package** (nothing else may change; `atlas merge --dry-run` checks it):

- `pipeline/<slug>/**`
- `site/src/lib/explainers/<slug>/**`, including its local ops, views, adapters, sections and claim tests
- `site/static/models/<slug>/**`
- `notes/<slug>.md`, `notes/shots/<slug>/**`

**Common inputs:** the wave-1 kit (frozen; issues go into "Kit issues" in `notes/<slug>.md`) and §7.

**Common acceptance:**

- `npm run atlas -- check <slug>` green;
- `atlas e2e <slug>` green with screenshots;
- notes complete;
- ≤ 1,000 page-specific lines in the `check` report;
- `merge --dry-run` clean;
- quality `draft`, with every author item done (goals, predict, task, a limits or breakIt step, zh narration, claim tests).

| Round | Pkg | Slug (port) | Specific acceptance |
|---|---|---|---|
| 2a | M1 | tinyvit (5191) | Test accuracy reported honestly (target ≥ 92 %; TinyCNN is 96.81 %); parity ≤ 1e-4 including blank and dot; the compare step shows both models live on the same drawing |
| 2a | M2 | mlp-playground (5192) | Twin parity at the §6.2 bars; the default preset loss drops ≥ 50 % within 300 steps; no NaN in 2,000 steps on any preset, including spiral with lr 3 (divergence is shown, not crashed) |
| 2a | M3 | gridworld (5193) | Python reference tape equality (Q-learning and value iteration; SARSA if included) with draws replayed; bandit regret curves deterministic per seed; an unreachable-goal fixture shows the right message |
| — | K0 | kit fixes | **Owns** `site/src/lib/atlas/**`, `site/scripts/atlas/**`, `pipeline/atlas_kit/**`. Fixes the "Kit issues" of M1–M3 and keeps every gate green; the main session merges, then 2b lanes are created (or synced) from the fixed kit |
| 2b | M4 | char-rnn (5194) | Gate subs ≤ 1e-4 at T = 1 and T = 128; the `sample` tape equals a Python loop with the same seed (greedy and sampled; sampling uses `next()` only) |
| 2b | M5 | gcn (5195) | Forward ≤ 1e-4 with unweighted Â; retrain twin parity; editing an edge re-runs live; the isolated-node fixture is handled |
| 2b | M6 | flow-2d (5196) | Twin parity (SGD 20 steps, Adam 20 steps at the §6.2 bars); Euler frames equal the twin; the phone path uses the precomputed tape (≤ 2 MB) |
| 2b | M7 | tinyclip (5197) | Per-tower parity ≤ 1e-4; the synthetic set is rendered by TS `preprocessStrokes`, with a check that Python reads the identical bytes; the bag-of-words claim is a claim test |
| 2b | M8 | tinyforecaster (5198) | The LSTM uses the shared `lstm` (gate parity); TCN local `conv1d` goldens; twin parity; the baseline comparison uses live numbers |
| 2b | M9 | pinn (5199) | Double-grad parity ≤ 1e-10 relative; plain and PINN share init (`fork('init')`); the λ task and the wrong-c `breakIt` hold as claim tests |

Wave 2b runs up to 6 lanes, using `check --fast` while building. Full builds and e2e runs are spread so that at most two run at once.

### 8.4 Wave 3: review and integration

**R1–R9 (parallel; R1–R3 can start during wave 2b).** One teaching reviewer per model.

- **Owns:** that model's package paths and `notes/review-<slug>.md`.
- **Deliverables:** the completed checklist (§6.4) and fixes inside the package.
- **Acceptance:** verdict "ship"; gates still green; quality set to `reviewed`.

**I1 (main session, after each R).** Integration.

- **Owns:**
  - `site/src/lib/catalog/{models,tech-zh,glossary}.ts`
  - `site/static/licenses.json`
  - `LOG.md`
  - git
- **Deliverables:**
  - `atlas merge` and `atlas promote` per model, with the printed patches applied: status live, `lesson` info, the §7 catalog follow-ups, credits, glossary.
  - One local commit per model.
  - `atlas check --all` (one build), `check:build`, legacy parity (`--no-write`).
- **Acceptance:**
  - 192 pages plus `/dev/atlas/` and `/dev/atlas-train/` build. Going live adds no page, because the 9 slugs were already prerendered as planned pages.
  - No third-party URLs.
  - Every live model card links its lesson.
  - Nothing is pushed to GitHub (on hold at the user's request).

**K1 (optional, after I1).** Kit consolidation.

- **Owns:** `site/src/lib/atlas/**`.
- **Work:** promote local ops or views that are now used twice; regenerate `ops.json` and the docs.
- **Out of scope unless the user asks:** migrating TinyCNN or TinyGPT onto the kit, and concept packages.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Over-abstraction slows the first models | <ul><li>Every layer has an escape hatch: a local op, a local view, `sections/`, `play: 'view'`, a plain author step, or a static route (L4).</li><li>`hello-xor` covers A, B and C before any model.</li><li>Wave 2a builds one model per kind, and K0 fixes the kit before the other six start.</li></ul> |
| Auto lessons and generic zooms teach worse than the hand-made pages | <ul><li>Op templates are distilled from the hand lessons and reviewed once (R0).</li><li>Auto-only lessons cannot be `reviewed` (AK409).</li><li>The checklist requires a limits step.</li><li>The hand-made pages are not replaced.</li></ul> |
| Teaching errors and false claims (reviewers fixed them in 4 of 5 pages) | <ul><li>Placeholders and `facts.json`; the AK402 warning.</li><li>Claim tests over all fixtures.</li><li>A Node dry run over edge fixtures.</li><li>The Lesson Lab shows every narration per fixture and language.</li><li>`updateOf` refuses to narrate an SGD rule for an Adam step.</li></ul> |
| Kit work collides with round 3 | <ul><li>Wave 0 starts after the round-3 merge and re-checks the Walkthrough types.</li><li>The kit uses Walkthrough unchanged.</li><li>No package edits `src/lib/kit/**` or the shared `i18n/strings.ts` (atlas has its own strings).</li></ul> |
| Undefined seams between parallel packages | <ul><li>`PackageModules` and `SourceRunner` are in the frozen types.</li><li>`TapeStore` is shared by the local and remote runners.</li><li>`ops.json` has one owner (W1-A).</li><li>The view `demo()` contract has a test.</li><li>Cross-package acceptance is moved to the integration gates.</li><li>Wave 1 is split into 1a and 1b.</li></ul> |
| Parallel builders on one exFAT tree (`.svelte-kit` races, dev servers locking `build/`, CRLF, Write-tool failures) | <ul><li>`.work` copies per package; a port per package.</li><li>Builds go into `.atlas/build-*` via `ATLAS_BUILD_DIR`.</li><li>The CLI writes files itself (LF, no BOM) and stops dev servers before building.</li><li>`doctor` prints the `safe.directory` form.</li><li>The Write-tool fallback: write to a temporary folder, then `cp`.</li></ul> |
| Lanes cannot see main-tree code, or vice versa | <ul><li>`ATLAS_SITE` and `ATLAS_ROOT` everywhere.</li><li>Fixtures live in `site/tests/atlas/`.</li><li>Python writes assets into the lane.</li><li>`worktree --sync` brings kit fixes into lanes and refuses if a lane touched kit files.</li></ul> |
| The kit rewrites shipped assets | <ul><li>The legacy parity scripts run with `--no-write`.</li><li>Re-descriptions write to `.atlas/tmp/`.</li><li>`spec.json` hashes are checked in wave 0 and at every integration.</li></ul> |
| Float differences between engines (TF32 on the GPU; reduction order; libm) | <ul><li>References and goldens are CPU-only, with TF32 off.</li><li>Goldens compare against fp64 on the same f32 inputs.</li><li>Kind-B bars are relative, and the strictest bar uses SGD.</li><li>Draws are recorded and replayed; near ties are handled by AK205.</li><li>Ties break to the smallest index.</li></ul> |
| JS training speed (flow-2d batch 256; forecaster) | <ul><li>Fused tensor kernels and a per-step buffer arena (never transferred).</li><li>Benchmark gates in W1-B1 and W1-B2.</li><li>Precomputed tapes from the same TS code for phones.</li><li>Defaults shrink when the gate is missed.</li></ul> |
| Double-backward bugs or slowness (PINN) | <ul><li>gradgradcheck plus torch fp64 parity for every op in wave 1a.</li><li>Fallback 1: forward-mode 2-jets for MLPs (`jet2`, about 80 lines, only if the gate fails).</li><li>Fallback 2: jax-js (MIT), per brief §7.</li></ul> |
| Vite worker bundling (`iife` inlines everything) | Per-package worker entries found by a `?worker` glob, spiked first in W1-B2. The fallback is `worker.format: 'es'` set by the main session |
| Page-shell size (TinyCNN is already at 297.9 / 300 KB; planned pages share `/m/[slug]`) | <ul><li>`AutoModelPage`, the registry and KaTeX are only in a lazy chunk returned by `+page.ts`.</li><li>The auto-page target is ≤ 220 KB.</li><li>`check:build` enforces the planned-page baseline + 5 KB.</li></ul> |
| Preprocessing mismatch (the raster path fell to 51–77 % on small drawings) | <ul><li>The drawing28 adapter always uses `preprocessStrokes` / `preprocessPhoto` from `tinycnn.ts`.</li><li>TinyCLIP's synthetic images are rendered by the same TS function.</li><li>Edge fixtures (blank, dot, small) run in `check`.</li></ul> |
| Parity hides local bugs or holds only for short inputs | <ul><li>Per-layer bars.</li><li>Cases at the dim extremes.</li><li>NaN/±∞-safe compare.</li><li>Badges show the bar, not the best case.</li><li>`explain` equals the trace on every unit.</li></ul> |
| Op explosion | Rule of two (gru stays local); composition preferred; promotion only in K1 |
| Licensing of borrowed ideas (TF Playground Apache-2.0; PINN notebook MIT; Zachary 1977) | `credits` are required (AK704); `promote` appends; reimplement rather than port; ported code carries headers and a NOTICE |
| External contributors cannot run the kit | <ul><li>Python from PATH by default, plus a local override.</li><li>`requirements.txt` for Python ≥ 3.10.</li><li>A Node-only path for kinds B and C.</li><li>`CONTRIBUTING.md`.</li><li>`doctor` prints fixes.</li><li>The licence decision (D9) comes before any push.</li></ul> |
| Scope creep: the kit becomes the project | <ul><li>v1 covers only what the batch needs: 20 ops and the §7 views.</li><li>No concept pages.</li><li>12 CLI commands.</li><li>Lines per page are reported for every model.</li></ul> |
| New downloads | None in the batch. The CLI flags `download` or `urlretrieve` in `pipeline/<slug>/`. A real series and Speech Commands wait for the user's OK |

---

## 10. Decisions from review

I checked every critique point against the code or the machine before deciding. "Verified" means I reproduced the fact.

### 10.1 Accepted (changed in this version)

| # | Critique point | Verified | What changed |
|---|---|---|---|
| 1 | The runtime contract has no way to load package code; the train-source seam is split across W1-B, W1-C and W1-D | yes (§4.7 of v1 had only strings) | `PackageModules`, `SourceRunner`, `OpenSource` and `createRuntime(pkg, { runner })` are in the frozen types. Loaders wrap trainers as `train:<name>` processes, so W1-C never imports W1-B code. `TapeStore` is shared by `localRunner` and `engineRunner`. `LiveTape.ensure` is async (§4.4, §4.7) |
| 2 | Dev-only packages break the slug rules | yes (hello-xor and demo-x are not in the catalog) | `devOnly` manifests and `atlas new --dev` are exempt from AK701 and never routed under `/m/` (§4.9, §4.12) |
| 3 | The legacy parity scripts rewrite shipped assets | yes (`writeFileSync` at tinycnn.parity.ts:129 and tinygpt.parity.ts:175; the root layout embeds `parity.checked` in every page) | Wave 0 adds `--no-write`. The kit always uses it, `--write` is main-session only, and `spec.json` hashes are checked (§4.12, §8.0) |
| 4 | Pipeline checks cannot see lane code | yes (`site/.work/<pkg>/site/..` has no `pipeline/`) | `ATLAS_SITE`/`ATLAS_ROOT` resolution; fixtures moved to `site/tests/atlas/`; Python writes into `$ATLAS_SITE` (§3.1) |
| 5 | Shared files that no package owns | yes (`pages: 'build'`, `BUILD = 'build'`, `NOT_LOADED = {reference.json}`) | `svelte.config.js`, `check-build.ts`, `assets.ts` and both `.gitignore` files belong to wave 0. UI strings live in `atlas/ui/strings.ts`, not the shared file. `vite.config.ts` is untouched (per-package workers). No validator dependency (hand-written validators) |
| 6 | One `linear` cannot match both legacy kernels bit for bit | yes (`s = 0 … + b[o]` vs `s = b ? b[o] : 0`) | `accumulate` attribute on `linear`/`conv2d`, set to `bias-first` for TinyGPT by `legacy.ts`. The TinyGPT bar is 1e-6 relative |
| 7 | `linear_tied` `transposed: true` | yes (tok_emb is `[1024, 76]` = `[out, in]`) | `legacy.ts` drops `transposed` and maps to `linear` with no bias (§3.2) |
| 8 | TinyGPT has no `input` layer and no `dims` | yes | `legacy.ts` synthesizes `inputs` and `dims.T = {1..64}`; layer `in` may name a declared input |
| 9 | Sub-step ids with dots are ambiguous | yes (`L0.attn.q`) | Separator `#`; fact paths parsed right to left (§4.1, §5.3) |
| 10 | The formula syntax does not match the kit | yes (`fillTex` uses `{{name}}`; `\sym` is not a KaTeX macro) | `{{name}}` + `sym()` from `atlas/core/tex.ts` (same output, no KaTeX import) |
| 11 | The examples do not type-check | yes | `Focus.key`, state_dict names, `rt.num()`; examples rewritten with zh text |
| 12 | `AtlasSpec` cannot go into the kit unchanged; `tolerance={null}` | yes (`ModelSpec.outShape: number[]`; the badge tests only `!== undefined`) | `ui/kitspec.ts` adapter; the kit is unchanged |
| 13 | A byte-identical `/m/tinycnn/` build is unrealistic | agreed (hashed chunk names change) | Acceptance is now "no source change, same rendered text, shell ≤ 300 KB" |
| 14 | "192 + 9 pages" double-counts | yes (the 9 are among the 101 planned pages) | Fixed in I1 |
| 15 | "40 presets"; T = 1/64 references are git-ignored | yes (60 + 12 presets; `verify_ref.json` ignored) | W1-A uses 60 + 12 presets; T = 1/64 compare TS against TS, with PyTorch only at 9/22/37 |
| 16 | Rng is not bit-identical for `normal()` | agreed (libm can differ in the last ulp) | Exactness is limited to `next`/`int`/`fork`; `recordDraws` and replay; AK205 near-tie rule |
| 17 | Catalog input kinds and the forecaster prerequisite | yes (`game`, `series`, `scalar-lstm`) | Adapter → InputKind table (§4.9); prerequisite follow-up (§7) |
| 18 | TinyViT params assume MLP ratio 2 | yes (my count: ratio 4 gives ≈ 117k) | Ratio 4 stated; ≈ 117k params, ≈ 470 KB |
| 19 | 15 snippet props, not 13 slots | yes | All 15 are overridable via `sections/<Snippet>.svelte` |
| 20 | Vite iife workers inline the whole glob | yes (`inlineDynamicImports` for iife in Vite's config.js) | Per-package `worker.ts` + a `?worker` glob, spiked first; fallback `format: 'es'` |
| 21 | `check:catalog` needs `TECH_ZH` for a new runtime text | yes (check-catalog.ts) | `tech-zh.ts` is in I1's owned files; `catalog.runtime` is `I18n`; `promote` prints the entry |
| 22 | networkx karate edges are weighted | yes (`{'weight': 4}`) | `ak.data.karate()` builds an unweighted A |
| 23 | Tensor transfer rules; functions cannot cross postMessage | agreed | Copy-and-dedupe transfer rules; `SchemaJson`; derived fields recomputed where the module is loaded; `loadTape(h, blob, p?)` |
| 24 | No JSON Schema validator available | yes (no `jsonschema`, no `ajv`) | Hand-written TS validators, with structural checks only in Python |
| 25 | Onboarding for people outside the maintainers' computer | agreed; this is the user's stated goal | `CONTRIBUTING.md`, `requirements.txt`, Python-on-PATH default + local override, Python 3.10 limits, the D9 licence decision, CI design (§6.5) |
| 26 | No sync path from kit to lanes | agreed | `worktree --sync` (AK703) and the K0 step between 2a and 2b |
| 27 | Numerics policy | yes (`cudnn.allow_tf32 = True` here) | CPU-only, TF32 off on import (AK102); fp64 goldens on f32 inputs with relative bars; relative kind-B bars, SGD for the strictest one |
| 28 | Recurrent-op semantics | agreed | Both biases, gate order and GRU's `r ⊙ (W_hn h + b_hn)` are specified (§4.3) |
| 29 | Lab route in production; `searchParams` during prerender | yes (root `prerender = true`) | No lab entries without `ATLAS_INCLUDE_DRAFTS`; `?spec` read in `onMount`; unlinked draft chunks noted |
| 30 | Command time limits; differing long-task bars | agreed | One build for `--all`; a sampled e2e matrix with an `?atlas-e2e=1` training cap; one long-task rule (50 ms from the engine, 100 ms from anything) |
| 31 | zh review of shared op templates; `zh` optional in the type | yes | R0 review at integration 1b; lint AK406 enforces zh |
| 32 | Planned-page shell size on `/m/[slug]` | agreed | The auto page is a lazy component returned by `+page.ts`; planned baseline + 5 KB enforced |
| 33 | Type-only kit import; erasable syntax | agreed | `PredictQuestion` copied into atlas; erasable-only rule enforced by Node imports |
| 34 | Parallelism: acceptance chains; two sources of truth for op attrs; W1-B views in W1-D's tree; legacy parity races; pinning wave 0 | agreed | Waves 1a/1b with integration gates; W1-A owns `ops.json`; a view contract test; `--no-write` and main-tree-only legacy runs; wave 0 pinned to the post-merge commit |

**Simplifications taken:**

- Concept pages cut from v1.
- Planner shrunk (no patterns or folding; the TinyCNN mapping is a metric).
- TinyGPT re-description, `transformer_block` and `sequential` dropped.
- TinyGPT's base64 layout adopted as `atlas-ref-1`.
- Spec 1.1 trimmed (no `groups`, `tokenGrid`, `tapes[]`, `facts` key, code auto-fill, f16/u8 stores or `log` keep).
- `facts.ts`, `{=literal}` and `Recipe` dropped; claims became tests.
- The CLI cut to 12 commands, `docs`/`list`/`preview`/`tape` removed, and 38 explicit AK codes instead of open ranges.
- W1-B and W1-D split.
- Wave 2 in two rounds; `gru` moved to a local op; compare kept in reduced form (below).
- The signed rubric replaced by a checklist; AK409 is the only automated tier rule.
- A static port table and a single Lab page per slug.

### 10.2 Rejected or only partly taken

| Critique point | Decision | Why |
|---|---|---|
| Defer the `live-model` source | Rejected | MLP playground is a wave-2a model and needs it for "one neuron is a soft line" on live weights. It is a thin source over `toSpec` + the executor, which W1-B1 and integration 1a already test. Dropping it would force a second, bespoke forward view |
| Defer `compareWith` cross-model runs | Partly | Replaced by a plain `trace` source with `assets: 'models/tinycnn'`. W1-A's acceptance already loads the TinyCNN spec through `legacy.ts`, so the cost is nearly zero, and the TinyViT-vs-TinyCNN step is one of the batch's best teaching moments |
| Drop the Python re-description of TinyCNN | Rejected (TinyGPT part accepted) | TinyCNN is 14 layers with a real checkpoint. A byte-identical `weights.bin` is the cheapest proof that the conv/pool/dense mirrors and the writer are right before nine models depend on them. Output goes to `.atlas/tmp/`, so no frozen asset is touched |
| Cut `worktree`, `merge` and `promote` to later scripts | Partly | `worktree` (with `--sync`) and `merge` stay: the parallel waves need the ownership check (AK702/AK703); in round 2 the main session had to diff every copy against main by hand (LOG, round-2 merge plan). `promote` shrinks to appending licences plus a printed catalog patch, because rewriting TS catalog files by script is fragile |
| Drop AK402 (the literal-digit lint) | Partly | Kept as a non-blocking warning that ignores single digits, so zh text such as "第 1 块" and structural "3×3" do not trip it. It targets the most common past reviewer finding (a quoted number that disagreed with the screen). The `{=…}` syntax is dropped |
| Collapse AK codes to about 15 | Partly | 38 are defined, because each gate needs a distinct, documented code for the "fix" line. The open ranges are gone; only listed codes exist |
| Shrink the planner to four rules | Partly | One more rule is kept, a summary step for repeated groups. Without it, TinyViT and later TinyBERT drafts would carry no mention of blocks 2–4 |
| "A JSON Schema validator needs the user's package OK" | Conclusion accepted, reason rejected | LOG "Open items" shows npm packages are already approved, and CONTRACTS allows pip installs. Hand-written validators are chosen anyway, to keep the kit dependency-free and extractable |
| Cut `AutoConceptPage` permanently | Deferred, not cut | Kinds B and C fit gradient-descent and optimizer concept pages well. The route change waits until after wave 2, as a later item (§7 later models) |
