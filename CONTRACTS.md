# Model Atlas — build contracts (v0)

Every build agent reads this file first. Its contents are fixed; changing it requires the main session.

## Places and tools

- **Root:** the repo root (the folder that holds `CONTRACTS.md`, `pipeline/` and `site/`). Quote paths that contain spaces in bash.
- **Python:** your Python, 3.10 or newer with torch, numpy and networkx (`pip install -r pipeline/requirements.txt`). The DevKit CLI finds it as `site/DEVKIT.md` section 1 describes. Training may use a GPU. pip-install extra packages only when needed, and note them in your notes file.
- **Node 24:** runs `.ts` files directly through type stripping (`node file.ts`). Avoid enums and namespaces in shared TS so this keeps working. npm 11.
- **Background:** the Model Atlas brief is `research/design_brief.md` (long; read the sections your task names).

## Layout

```
model-atlas/
  CONTRACTS.md        this file
  notes/<stream>.md   each agent's working notes: what was done, decisions, final numbers, how to re-run
  pipeline/
    data/             downloaded datasets (never shipped, never committed)
    tinycnn/          TinyCNN data, train, export and fine-tune scripts
    tinygpt/          tokenizer, train and export scripts for TinyGPT
    parity/           node scripts: JS forward pass vs PyTorch reference
  site/               SvelteKit 2 + Svelte 5 app (adapter-static)
    static/models/<slug>/      shipped model assets (format below)
    static/licenses.json       every third-party dataset, model, code or font used
    src/lib/models/<slug>.ts   pure-TS model code (no Svelte, no DOM): load + forward with a full trace
    src/lib/kit/               shared components and utilities (foundation stream)
    src/lib/catalog/           catalog data for areas, models, concepts and the glossary
    src/routes/...             pages
    KIT.md                     kit API docs (written by the foundation stream)
```

## Rules for every agent

1. **Fully static.** Nothing may be requested from a third-party host at runtime: no Google Fonts, no CDNs, no analytics. Self-host everything. This is for durability, and because learners in China cannot reach Google or Hugging Face.
2. **Never present made-up numbers as real.** Every panel that shows model numbers carries a provenance badge:
   - `live`: computed in the browser right now
   - `precomputed`: computed offline from the same model
   - `illustrative`: made-up example values, clearly labeled
3. **Keep model assets small** so they can live in the repo for v0: each file ≤ 5 MB, and each model ≤ 15 MB in total.
4. **Keep every shell command under 9 minutes** (Bash timeout 600000 ms). Split longer training into resumable chunks. Foreground `sleep` is not available.
5. **Write working notes to `notes/<stream>.md` as you go**, so the work survives a context loss.
6. **Record every third-party source** (dataset, pretrained model, copied or ported code, font) in `site/static/licenses.json`. The file is a JSON array of `{ "name", "url", "license", "usedFor", "attribution" }`. Read it and append; create it as `[]` if it is missing. Other agents write to it too, so re-read it right before writing and only append.
7. **If an editor tool fails to write a file in place**, write it into a temporary folder first, then copy it into place with the shell.
8. **Bilingual text.** All user-visible strings are bilingual `{ en, zh }`. English must be complete; Chinese may be shorter but must not be missing for UI labels.
9. **Write plainly.** Short sentences, no hype, explain terms on first use.

## Model asset format (`site/static/models/<slug>/`)

- **`spec.json`:**
  ```json
  { "slug": "tinycnn", "name": {"en": "...", "zh": "..."}, "version": "2026-10-05",
    "params": 6722, "precision": "fp32",
    "input": {"kind": "drawing", "shape": [1, 28, 28], "notes": "..."},
    "tensors": [ {"name": "conv1_1.weight", "shape": [8, 1, 3, 3], "offset": 0, "length": 72} ],
    "layers":  [ {"id": "conv1_1", "op": "conv2d", "label": {"en": "Conv 1-1", "zh": "卷积 1-1"},
                  "in": ["input"], "outShape": [8, 26, 26], "params": {"kernel": 3, "stride": 1, "padding": 0}} ],
    "training": { "dataset": "...", "steps": 0, "metrics": {} },
    "parity": { "against": "pytorch-fp32", "max_abs_diff": 0.0, "checked": "2026-10-05" } }
  ```
  - `tensors` index into `weights.bin`. `offset` and `length` count floats, not bytes.
  - `layers` are listed in display order.
  - Model-specific extras go in their own keys (`classes`, `heldOutClasses`, `vocabSize`, `context`, …).
- **`weights.bin`:** all tensors as float32 little-endian, concatenated.
- **`reference.json`:** a few inputs and the PyTorch output of every layer id, used by the parity tests. Pages never load it, but it is shipped so anyone can re-check parity.
- **Other files as needed:** `presets.json`, `tokenizer.json`, `checkpoints/…`, `replay.json`, `finetune.json`, …

## Trace format (returned by `src/lib/models/<slug>.ts`)

- `load(fetchFn, baseUrl)` returns a model object. In the browser `fetchFn` is `fetch`; in Node the parity test passes a function that reads files.
- `forward(...)` returns `{ order: string[], t: Record<string, { shape: number[], data: Float32Array }> }`.
  - There is one entry per layer id in `spec.layers`, plus finer sub-steps where a page needs them (e.g. per-head attention).
  - Arrays are row-major. Images are channels-first `[C, H, W]`. Sequences are `[T, D]`.
- Deterministic. Synchronous is fine for tiny models. No DOM access, so the same code runs in a Worker or in Node.
- **Parity:** a node script in `pipeline/parity/` runs the TS forward pass on `reference.json` inputs and compares every traced tensor with PyTorch. The pass bar is max |diff| ≤ 1e-4 (fp32). The measured value is written into `spec.parity`.

## Site conventions (fixed now; details in `site/KIT.md` once the foundation stream writes it)

- **Base path:** `BASE_PATH` env var (GitHub Pages project site, e.g. `/model-atlas`). Always use `base` from `$app/paths` for links and asset URLs.
- **Routes:** `/`, `/atlas/`, `/atlas/[area]/`, `/m/<slug>/` (static route folders for live models: `/m/tinycnn/`, `/m/tinygpt/`; a `[slug]` route renders "Planned" pages for the rest), `/train/`, `/train/<slug>/`, `/glossary/`, `/about/`, `/dev/kit/` (component gallery).
- **Slugs:**
  - Live v0 models: `tinycnn`, `tinygpt`.
  - Live v0 concepts: `tokenization`, `next-token-prediction`, `transfer-learning`.

## DevKit additions (wave 0, 2026-10-06)

Additive: nothing above changes, and every v1 asset stays valid. The full specification is `research/devkit/DEVKIT_DESIGN.md`; this section lists the rules every DevKit agent must follow.

- **Base commit:** `e4718c964f2439d27f053de51067e127e21e21f9` (the round-3 merge, "Round 3: guided step-by-step lessons on TinyGPT, tokenization, next-token prediction and transfer learning"). Wave 0 started from it.

### Frozen types

- `site/src/lib/atlas/types/{core,spec,op,process,autograd,train,lesson,runtime,view,explainer,engine}.ts` are design §4.1–§4.10, plus `index.ts` (type-only re-exports) and `contract.test.ts`.
- They are **read-only during waves 1 and 2**. A change goes through the main session, is recorded here, and reaches lanes with `atlas worktree --sync`.
- **Deviations from the §4 text** (each one was needed to type-check; nothing else changed except a 2-line header comment per file):
  1. `train.ts` imports `I18n`, `Rng` and `Tensor` from `./core.ts`. The design prints `autograd.ts` and `train.ts` as one block, so `train.ts` had no import for them.
  2. `lesson.ts`: `export type AutoRef<R extends ExplainerRuntime> = …` (the design has `AutoRef<R>`). `Partial<StepDef<R>>` needs the constraint; without it svelte-check reports "Type 'R' does not satisfy the constraint 'ExplainerRuntime'".
- **Notes for implementers:**
  - `OpDef<P>` requires `P extends Record<string, unknown>`. Write `P` as a type alias (`type ConvP = { kernel: number }`), not an interface; then `OpDef<ConvP>` goes into `OpDef[]` without a cast.
  - **View metas:** `views/<key>/meta.ts` **default-exports** its `ViewMeta` (`export default { … } satisfies ViewMeta`). `contract.test.ts` checks every kit view (`atlas/ui/views/<key>/`) and package view (`explainers/<slug>/views/<key>/`, key `<slug>/<key>`): key = folder, `demo()` resolves in Node, `describe(demo)` is non-empty in en and zh, demo provenance is `'illustrative'`.
- **Kit re-check at the base commit:**
  - kit `PredictQuestion` and atlas `PredictQuestion` are mutually assignable (identical shape).
  - kit `WalkContext` (`index, step, playing, visible, shown, reduced, done`) is assignable to `StageProps.ctx`.
  - A `CompiledStep` maps onto a kit `WalkStep`: `title` (I18n) is a `Text`; `narration` becomes a `() => Text` closure; `look` needs a fallback, because `CompiledStep.look` may return `undefined`; `waitForDone`, `linger` and `duration` are the same.
  - `Walkthrough` props: `steps`, `index`, `id`, `label`, `ready`, `stage(step, ctx)`, `aside(ctx)`, `end`, `stageMinHeight`, `syncHash`, `onstep`. `narrationFirst` is a deprecated no-op: narrow layouts always show the narration first.
  - A `StepPlayer` inside a lesson stage passes `counter` (e.g. `{ en: 'Part {i} of {n}', zh: '第 {i} / {n} 部分' }`) or `position(i, n)`, so its counter is not mistaken for the lesson's steps.

### Spec 1.1 (additive keys; `AtlasSpec` in `types/spec.ts`)

- Top level: `atlas: { spec: '1.1', kinds, exporter }`, `inputs` (declared inputs with `adapter`, `kind`, `shape`, `dtype`, `axes`), `outputs`, `dims` (symbolic dims such as `T` with `min`/`max`), `files`.
- Layers: `w` (weight slot → tensor name), `axes`, `sub`, `code`, `ui`, `teach`, `desc`. `outShape` may hold symbolic dims. A layer's `in` may name a declared input.
- `parity`: `max_abs_diff` and `checked` may be `null` until `atlas parity` runs; `bar`, `pass`, `relative`, `scope`, `perLayer`, `cases`, `edgeCases`, `steps`. TinyGPT's `tolerance`/`passed` are read as `bar`/`pass`.
- v1 specs (no `atlas` key) are read through `graph/legacy.ts` at load time. **No shipped file is rewritten.**

### Sub-steps, references, facts, tapes

- **Sub-step separator `#`:** sub-step trace ids are `${layerId}#${sub}` (`L0.attn.scores#h2`, `lstm1#f`) and follow their layer in `order`. `#` never appears in a layer id; layer ids may contain dots. Fact paths (`layer.<id>.<stat>`) are parsed right to left.
- **`reference.json` (`atlas-ref-1`):** `{ format: 'atlas-ref-1', model, version, against: 'pytorch-fp32-cpu', note, cases: [{ id, label, edge, inputs: { <name>: { shape, dtype, b64 } }, order, tensors: { <id or id#sub>: { shape, dtype: 'float32', b64 } } }] }`. `b64` is base64 of little-endian values; `-Infinity` is stored as is. The two legacy layouts (TinyCNN float lists, TinyGPT `prompts`) stay readable.
- Pages never load references: `src/lib/server/assets.ts` leaves `reference.*` and `*.ref.*` (e.g. `q.ref.tape.json`) out of the measured download size.
- **`facts.json` (`atlas-facts/1`)** and **tapes (`atlas-tape/1`, `tapes/<name>.tape.json` + `.bin`)**: formats in design §4.2 and `TapeHeader` in `types/process.ts`.
  - The `atlas-tape/1` file layout (frames, `$t` tensor refs into the 8-byte-aligned `.bin`, `$f` for non-finite numbers, `key: true` keyframes, `bin {file, bytes}`) is in `notes/devkit-tools.md`. The TS writer (`atlas/process/io.ts` `saveTape`) and the Python `atlas_kit.TapeWriter` write identical files (integration 1a).
  - Storage rule for `track: 'changes'` fields: a full value is stored at keyframes and when it is a real replacement; a value that the running value plus the frame's `changes` already explain is not stored again. Empty `draws` lists are omitted.

### Numerics

- References and goldens are computed **on the CPU with TF32 off**: `torch.backends.cudnn.allow_tf32 = False` and `torch.backends.cuda.matmul.allow_tf32 = False`. `atlas_kit` sets both on import. PyTorch's default for `cudnn.allow_tf32` is `True`.
- Training may use the GPU. Exporting a CUDA tensor or module is refused (AK102).
- Kind-B parity runs against PyTorch fp64 (relative bars, design §6.2).

### Lanes, paths and builds

- **Lanes:** packages work in `site/.work/<name>/site` copies: no symlinks; skip `node_modules`, `.svelte-kit`, `build`, `.work` and `.atlas`; patch the copy's `vite.config.ts` `fs.allow` and **never merge that line back**. A lane changes only its owned paths (design §8).
- **`ATLAS_SITE`** is the site folder a command runs in (a lane's copy inside a lane). **`ATLAS_ROOT`** is the nearest ancestor of `ATLAS_SITE` that contains `CONTRACTS.md` and `pipeline/`. The CLI passes both to Python. New scripts never use a `../../site` relative path. Node test fixtures live in `site/tests/atlas/`. Python writes assets into `$ATLAS_SITE`.
- **`ATLAS_BUILD_DIR`** (read by `svelte.config.js` and `scripts/check-build.ts`; default `build`): lanes and `atlas check` build into `site/.atlas/build-<name>/`. In Git Bash: `MSYS_NO_PATHCONV=1 ATLAS_BUILD_DIR=.atlas/build-x npm run build`, then the same variable for `npm run check:build`. Stop dev servers before a build (on some file systems a running dev server locks files).
- **Shell budgets in `check:build`:** the sample pages include `/m/tinycnn/` and `/m/tinygpt/` (300 KB gzip each; 298.9 and 298.2 KB at wave 0; 254.8 and 255.3 KB since the prep for wave 2b, "Catalog in page data" below). Every planned `/m/<slug>/` page must stay within its wave-0 shell baseline (110,912 B) + 5 KB. `m/resnet-18` always gates; under `ATLAS_INCLUDE_DRAFTS` the other planned pages only warn.
- **Git-ignored:** `site/.atlas/` and `site/scripts/atlas/atlas.config.local.json`.

### Legacy parity and scripts

- `pipeline/parity/tinycnn.parity.ts` and `tinygpt.parity.ts` take **`--no-write`**. Every check still runs, `spec.json` is left untouched, and the record that would have been written is printed as one `PARITY_JSON {…}` line (plus a "shipped vs measured" line). Without the flag they behave as before and rewrite `spec.parity`, including today's date.
- The kit and every lane run them **only with `--no-write`**. Only the main session runs them without it, from the main tree.
- `npm run check:atlas`: `node --test` over `src/lib/atlas/**/*.test.ts`, `src/lib/explainers/**/*.test.ts` and (since integration 1a) the CLI tests `scripts/atlas/tests/*.test.ts`. The CLI tests use the configured Python when it can import `atlas_kit` and torch, and skip those checks otherwise.
- `npm run atlas -- <cmd>`: `node scripts/atlas/cli.ts` (W1-E1). Built since wave 1a: `doctor`, `new`, `run`. Since wave 1b (W1-E2) every command is built: `export`, `parity`, `lesson`, `check`, `e2e`, `worktree` (`--sync`), `merge` (`--dry-run`), `review`, `promote`. The guide for contributors is `site/DEVKIT.md` (its tables are generated by `node scripts/atlas/docs.ts --write`).

### Types revision 1b (integration 1b, 2026-10-06)

Optional additions only: every object written for the wave-0 types still type-checks (`contract.test.ts` has a test for
that), and nothing was renamed or removed.

- `op.ts`: `OpCtx.has?(slot)` and `UnitQuery.has?(slot)`. The executor provides both; `ops/common.ts optW()` uses `has`
  when present and otherwise still catches `MissingWeightError`.
- `process.ts`:
  - `ProcessRun<S, O>` = `Generator<Frame<S>, void, Control<O> | undefined>`, the return type to give `run()` when frames
    differ in shape. `atlas/process/define.ts defineProcess()` is an identity helper that infers I and O.
  - `ProcessDef.recordDraws?` (the runtime already read it untyped) and `ProcessDef.liveSpec?(state, opt)` (trainingProcess
    adds it; 'live-model' sources call it).
  - `Tape.phases?`: the phases with their formulas. TapeStore had it; the recorder and the engine's tapes now expose it
    too, so `zoom.formula`, `tape.table` and the scrubber find phase formulas on live tapes.
  - `LiveTape.paused? / truncated? / error? / where? / notice?`: what the recorder and the engine already reported.
- `runtime.ts`: `Preprocessed.tokens?` (display text per input position) and `vocab?` (text of any id; `string[]` or a
  function); `recordDraws?` on the `train` and `process` SourceDefs; `SourceRunner.dispose?()`.
- `explainer.ts`: `CaptureProps<Raw>`, the props of every adapter's `Capture.svelte`: `rt`, `raw`, `presets` (may be
  empty: a Capture may then show its adapter's generated presets), `options` (the manifest's `input.options`),
  `onchange(raw, { live? })`, `compact`, `label`.
- `train.ts`: `TrainerDef.metrics?(m, data, opt?)`; trainingProcess passes the run's options.
- `engine.ts`: the open request's args may carry `keyEvery`, `maxFrames`, `keep` and `autorun` (W1-B2's recorder options).
- Not changed (kept for R0 or a later revision): `TrainOpt.optimizer` has no `'adamw'` (use `adam` + `decoupled: true`);
  `InputAdapter.preprocess` gets no assets (text-bpe takes the tokenizer from `options.tokenizer`); `StageProps` has no
  `compare` or focus scope (they travel in `props`: StageHost passes `props.compare` to the `compare` view);
  `UnitExplanation.layout` still serves both conv kernels and linear's flattened channels.

### Build configuration (integration 1b)

- `site/vite.config.ts` pins the Svelte client runtime (`svelte-runtime`) and the SvelteKit client runtime
  (`kit-runtime`) to one chunk each (`build.rollupOptions.output.manualChunks`). Without it every new lazy chunk that
  used a different mix of runtime helpers re-split the chunks the flagship pages preload (303.0 KB for `/m/tinycnn/` at the
  1b merge). Do not remove it; new code that changes the flagship shells must still pass `check:build`.
- The dev server ignores `.work/`, `.atlas/` and `build/` (`server.watch.ignored`); watching the lane copies kept the main
  tree's dev server from answering for minutes.

### Types revision K0 (K0, 2026-10-06, commit adcb914)

Additive, like revision 1b: every object written for the earlier types still type-checks, nothing was renamed or
removed, and `types/contract.test.ts` now also accepts an I18n value from a step's `vars()`.

- `lesson.ts`:
  - `TmplVal` = `number | string | I18n | { v, fmt?, focus? }`. An I18n value prints as is in each language (plural
    forms, conditional sentences); `atlas/lesson/say.ts` has `plural()`, `num()` and `pick()` to build them.
  - `StepDef.play` (the object form) adds `unit?: I18n` (what the scrubber's marks count, e.g. "episode") and
    `by?: 'step' | 'sub'` (the frame field that holds the count; default `step`).
  - `StepDef.task.waiting?: I18n`: the status line until the task is done (default "Not yet: do what the task says.").
- `process.ts`: `FieldDef.derive(s, opt?)`: derived fields get the run's options (the tape's `opt`).
- `runtime.ts`:
  - The `process` SourceDef adds `detail?: 'coarse' | 'fine'` (the run's `RunCtx.detail`, default `coarse`).
  - `SourceParity` and `ParityFile`, and `PackageModules.parity?` (below).
- `view.ts`: `StageProps.setInput?(raw)`: a view changes the learner's input; the page re-runs every source and the lesson
  re-resolves its steps, as after an edit in the input widget.
- `explainer.ts`: `PageSection` (`input | lesson | overview | inspector | unit | output | tryThis | training | article |
  code`) and `manifest.page.sections?: { titles?, intros?, hide? }`. Without a spec (kinds B and C) the model sections
  are left out unless a `sections/<Name>.svelte` override fills them.
- Kit, outside atlas (`src/lib/kit/types.ts`): `WalkStep.waitText?: Text | (() => Text)` (shown while Play waits on the
  step) and `keepPlay?: boolean` (presses inside the stage do not pause Play). Opt-in: the TinyCNN and TinyGPT lessons
  do not set them. Also changed in the kit, not a type: `fmt3` and `fmtFull` print every minus sign as U+2212 (also in
  the exponent) and `texNum` turns them back into ASCII for TeX.
- **`atlas-parity/1`** (`explainers/<slug>/parity.json`): `{ format, slug, checked, sources: { <source>: SourceParity } }`
  with `SourceParity` = `{ scope: 'train' | 'process', against, pass, max, bar?, relative?, steps?, frames?, fixtures?,
  checked }`. Written only by `atlas parity <slug> --write` (`--fixtures` adds every fixture input), one record per train
  source with a twin and per process source with a Python reference. It is a package file (a lane writes its own; the
  main session writes it for merged packages). The stage badge of a train or process step reads its source's record and
  shows the design §6.2 text ("Same steps as an independent Python run (also on N other inputs)", "f64 in your browser ·
  matches PyTorch fp64 to D significant digits over S steps"); a source without a passing record shows only "Live". The
  Lesson Lab's parity panel lists the records and the lesson sources without one.
- Not changed (open for a later revision): a `UnitPick` for the last-query rule (`pickUnit(..., { axes })`), `OpDef.vars`
  typed as I18n, `TrainOpt.optimizer` `'adamw'`, assets for `InputAdapter.preprocess`.

### Catalog in page data (prep for wave 2b, 2026-10-06)

- Model pages load none of the catalog tables (`models.ts`, `concepts.ts`, `areas.ts`, `tech-zh.ts`). The root
  `+layout.server.ts` gives every page `page.data.site` (`SiteCounts`, the footer's counts) and, on `/m/<slug>/` routes
  and the Lesson Lab, `page.data.catalog` (`ModelPageCatalog`: the entry, its area name, its technical strings as
  bilingual text, and compact `CardData` for its concepts, prerequisites and next models), computed by
  `src/lib/catalog/slices.ts` (server code only). `ModelPageShell` reads it (or its `catalog` prop), `SiteFooter` reads
  `site`; `CatalogCard` renders `CardData` without lookups, and `ModelCard` / `ConceptCard` build theirs with `modelCard()`
  (`catalog/cards.ts`) and `conceptCard()` (`catalog/concepts.ts`).
- `check:build` prints the catalog modules each sample page preloads and **fails if `/m/tinycnn/` or `/m/tinygpt/`
  preloads any**. So entries going live, new catalog text and new tech-zh rows never grow the flagship shells (checked by
  giving six entries a status change, lesson info and new runtime and params strings with tech-zh rows: both shells
  unchanged within 6 B of chunk-hash noise). List pages (home, atlas, area pages, `/train/`, `/dev/kit/`) and the
  `/m/[slug]` route node still import the tables they show. The glossary stays in every page that uses `Term`.
- The 404 fallback shell has no server data: its footer leaves the counts out.

### Types revision K1 (DevKit K1, 2026-10-07, commit 012cef5)

Additive, like revisions 1b and K0: every object written for the earlier types still type-checks and nothing was renamed
or removed. The behaviour that came with it is in DEVKIT.md sections 5-10 and 17.

- `runtime.ts`, `SourceDef`:
  - `train` adds `runOnEmpty?: boolean`. A train source no longer runs on an empty input (the adapter says `empty`): it
    holds an empty, finished tape and its steps show the adapter's empty note. `runOnEmpty: true` runs it anyway (a
    trainer that does not learn from the learner's input).
  - `process`: `run?: 'main' | 'worker'` is now read in the browser: `'main'` records on the main thread (its input may
    hold what a worker cannot receive, such as `rt.model`); unset or `'worker'` runs in the package worker. In Node a
    worker source whose input cannot be sent to a worker fails as the page would ("cannot be sent to a worker", AK404).
  - `process` adds `after?: string` (opened once that train or process source has run to its end; `input(rt)` may read
    its frames) and `reference?: boolean` (`false`: `atlas parity` runs no Python reference for it; the badge says
    "Live").
  - `live-model` adds `at?: 'first' | 'last'` (`'first'`: the train source's starting weights; default `'last'`).
  - `OpenSource.run?: 'main' | 'worker'` carries the source's `run` to the runner.
- `lesson.ts`: `StepDef.task.doneText?: I18n` (the status line once the task is done; default "Done."); `At.source?:
  string` (the step's source; `{opt.<name>}` resolves from its options and its process defaults before it has run).
- `process.ts`: `ProcessDef.metricTol?: number | 'exact' | Record<string, number | 'exact'>` (parity's tolerance for
  `frame.metrics` against the Python reference; default 1e-12 relative).
- `train.ts`: `TrainerDef.twin?: { sgd?: Partial<O> }` (the options of the plain-SGD twin run; default the page's options
  with `optimizer: 'sgd'`).
- `view.ts`: `StageProps.shared?: Record<string, unknown>` (one object per lesson step, shared by the views of that step,
  both panels of a compare included; reactive in the browser, `{}` in Node). `ViewSlot` now passes `StageProps.setInput`
  (K0 type; before K1 only StageHost did, so views fell back to `rt.setInput`).
- `explainer.ts`: `manifest.page.metrics?: Record<string, { label: I18n; fmt?: 'pct' | 'int' | 'n3' | 'n4' }>` (names
  and formats of training metrics in the auto Training section).
- `spec.ts`: `Axis` adds `'nk'` (the second node index of an `[n, nk]` node-by-node table) and `'b'` (an item of a batch).
- Kit, outside atlas (opt-in; the TinyCNN and TinyGPT lessons do not set them): `WalkStep.pending?: () => boolean` (the
  screen-reader announcement waits until the step has its numbers, at most 10 s), `Inspector` prop `maxRows?: number`
  (a tall 2-D tensor shows a window of that many rows), `ProbBars` prop `ids?: boolean` (the `#id` next to each token;
  default true).
- Formats and CLI, not types: `atlas parity --write` records one twin per trainer and options in `atlas-parity/1` (a
  source with its own options gets its own record); a process input longer than 8,000 characters reaches a Python
  reference as `--input @<file>` (read it with `ak.json_arg`); tapes go to `.atlas/<slug>/parity/`.
- Not changed (open): `UnitQuery.sub` for local ops, a spec answer-row hint for masked models (the kit reads a `[T, V]`
  output's answer at its last row), `TrainOpt.optimizer` `'adamw'`, assets for `InputAdapter.preprocess`.

### Types revision K2 (DevKit K2, 2026-10-08, commits 98d2439 and e05172b)

Additive, like revisions 1b, K0 and K1: every object written for the earlier types still type-checks and nothing was
renamed or removed. The behaviour that came with it is in DEVKIT.md sections 5-10 and 18.

- `lesson.ts`: `At.skipped?: boolean`: the empty input skipped the step (its source did not run). The narration leaves
  out the sentences whose values do not exist yet instead of printing "…" (`lesson/narrate.ts` render).
- `explainer.ts`:
  - `AnswerRow` = `'last' | 'first' | number | ((rt: ExplainerRuntime) => number)` and `manifest.output.row?: AnswerRow`:
    the row of a `[T, V]` sequence output that is the model's one answer (default `'last'`, a causal language model's
    next token; a negative number counts from the end; a function gives a masked model's `[MASK]` row). The side
    panel's answer, `top.*`, the auto output step, `kit.probbars` and compare read it. `manifest.output.view` is now
    optional.
  - `manifest.input.presetsFromAdapter?: boolean`: no presets file; the adapter's `presets({})` makes the presets in TS,
    and they reach `rt.presets`, the Capture and the lint fixtures as a file's would.
  - `manifest.page.trainingCurve?: { metric?, metrics?, labels?: Record<string, I18n>, log?: boolean | 'auto', smooth?,
    refs?: { y: number; label?: I18n }[] }`: the loss-curve props of the auto Training section (default: the loss). A
    metric recorded only at snapshot frames (a `metrics()` value) draws through them.
- `process.ts`:
  - `PhaseDef.formula.values(f)` returns `Record<string, number | string>`: a value may be a string, a setting as it was
    set (`fmtSetting(0.05)` is `'0.05'`).
  - `ProcessDef.trainer?: string`: the trainer whose frames this process makes. `trainingProcess()` sets it, and a
    process that spreads `...trainingProcess(t)` inherits it. When its frames equal the trainer's (`train/wraps.ts`
    checks), `atlas parity` runs the trainer's twin with the source's options, so its steps get the "= PyTorch fp64"
    badge.
- `train.ts`: `TrainerDef.batch(data, step, rng, opt, model?)`: the network as it is before this step (an RL round played
  with the current policy). The training loop always passes it; other callers (claim tests) may not.
- **Concept packages, not yet in the frozen types.** A concept package is an ordinary package folder whose
  `explainer.ts` declares `type: 'concept'` (a slug of `catalog/concepts.ts`; AutoConceptPage renders it at
  `/train/<slug>/`) and which adds `concept.ts`. Until a types revision admits them:
  - the package spells its manifest type `type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' }`
    (the scaffold writes it), and `isConceptManifest()` in `atlas/ui/concept/panels.ts` tells the two kinds apart;
  - `ConceptDef` (`hook?: I18n | StepDef`, `toy?`, `real?` and `compare?: StepDef`, `pitfalls?: { claim, truth, step? }[]`,
    `further?: { label: I18n; href }[]`) lives in `atlas/ui/concept/panels.ts`. `concept.ts` uses frozen types only
    (`StepDef`, `I18n`), because packages may not import `ui/` (AK503);
  - `concept.ts` is loaded by `atlas/ui/concept/packages.ts` (`loadConcept(slug)`), not through `PackageModules`.
  - Proposed for the next revision: `ExplainerManifest.type: 'model' | 'concept'`, `ConceptDef` in `types/`, and
    `PackageModules.concept?: ConceptDef`.
- Kit, outside atlas, not types: `kit/keyscroll.ts` (`use:keyScroll` for a box that scrolls), `ink()` in `kit/color.ts`
  with WCAG luminance (`inkOn`, `contrast`, `luminance`), and the `-text` color tokens (KIT.md rules 9 and 16).
- Not changed (open): the concept types above; the slug `logit-lens` is both a planned model (`models.ts`) and a planned
  concept (`concepts.ts`), so one of them needs a new slug before either becomes a package; `UnitQuery.sub` for local
  ops; `TrainOpt.optimizer` `'adamw'`; assets for `InputAdapter.preprocess`.
