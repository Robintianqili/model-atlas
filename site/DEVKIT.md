# Model Atlas DevKit: build an illustration (site/DEVKIT.md)

This guide is for someone who knows PyTorch and wants to add a model to Model Atlas (模型图鉴): a page that
teaches how the model works through a guided, step-by-step lesson with real numbers. You do not need to know Svelte.
You write a few TypeScript files of plain data and functions (and, for a trained model, a Python export script);
the kit generates the page, the lesson player, the views and the checks.

Read sections 1 to 4 first. Section 4 walks through a whole package in an afternoon. The rest is reference: look
things up when you need them. `site/KIT.md` documents the Svelte component kit underneath (you rarely need it).
The design behind the kit is `research/devkit/DEVKIT_DESIGN.md`. A concept page (`/train/<slug>/`) is a package too
(section 18).

**Building with Claude Code.** The repository's project skill `.claude/skills/model-atlas-builder/` gives an agent this
guide's routine in short form: the fast check routine, reuse before you write, the teaching bar, the chart and
accessibility rules, and the builder and reviewer roles. It points back here for the details.

## Contents

1. [Set up](#1-set-up)
2. [The three kinds of illustration](#2-the-three-kinds-of-illustration)
3. [What a package is](#3-what-a-package-is)
4. [Walk-through: your first package](#4-walk-through-your-first-package)
5. [Kind A: export a trained model](#5-kind-a-export-a-trained-model)
6. [Kind B: train live in the browser](#6-kind-b-train-live-in-the-browser)
7. [Kind C: a process (an algorithm, step by step)](#7-kind-c-a-process-an-algorithm-step-by-step)
8. [The lesson](#8-the-lesson)
9. [Views, sections and other ways to customize](#9-views-sections-and-other-ways-to-customize)
10. [Checks and gates](#10-checks-and-gates)
11. [Lanes, ports and merging](#11-lanes-ports-and-merging)
12. [CLI reference](#12-cli-reference)
13. [Error codes](#13-error-codes)
14. [Op reference](#14-op-reference)
15. [View reference](#15-view-reference)
16. [File formats](#16-file-formats)
17. [Troubleshooting and machine notes](#17-troubleshooting-and-machine-notes)
18. [Concept packages](#18-concept-packages)

---

## 1. Set up

You need:

- **Node 24** (it runs `.ts` files directly) and npm. From `site/`: `npm install` (it also installs the browser test's
  tools: `playwright-core`, and `@axe-core/playwright`, MPL-2.0, for the accessibility scan; section 10).
- **Python 3.10 or newer** with `torch`, `numpy` and `networkx`, only for kind A (exporting a trained model) and for
  the optional Python twins: `pip install -r pipeline/requirements.txt` (a CPU build of torch is enough).
- Git.

Then, from `site/`:

```
npm run atlas -- doctor
```

`doctor` checks Node, the site packages, Python and its packages, Playwright (for screenshots), the ports and git, and
prints the exact fix for anything that fails. Every CLI command is `npm run atlas -- <command>`; `npm run atlas` alone
lists them.

**Which Python the CLI uses.** In this order: the `ATLAS_PYTHON` environment variable; `site/scripts/atlas/atlas.config.local.json`
(git-ignored, one machine); `site/scripts/atlas/atlas.config.json` (committed: `{"python": "python"}`, the Python on
your PATH).

The CLI runs Python with `PYTHONPATH=<repo>/pipeline` and with `ATLAS_SITE` (the site folder you work in) and
`ATLAS_ROOT` (the repo root) set, so `import atlas_kit as ak` works and files land in the right place, also inside a
lane (section 11). Kinds B and C need no Python at all unless you add a twin or a reference.

**Running a pipeline script.** Run your own scripts (training, data preparation) with the configured Python too, never
with a bare `python`:

```
npm run atlas -- py pipeline/my-xor/train.py --seed 0      # also: py -m <module> ..., py -c "<code>"
```

`py` uses the same Python, `PYTHONPATH`, `ATLAS_SITE` and `ATLAS_ROOT` as `export` and `parity`. The script path is
found relative to the folder you run npm from, else relative to the repo root; the script runs in the folder you ran
npm from, its output streams through and the exit code is the script's.

---

## 2. The three kinds of illustration

Every lesson plays a **tape**: a list of recorded frames. What makes the frames is the **kind**:

| | A: inference trace | B: live training | C: process |
|---|---|---|---|
| What you show | a trained model working on the learner's input, layer by layer | a small network learning in the browser | an algorithm, one event at a time |
| You write | `pipeline/<slug>/model.py`, `train.py`, `export.py` (PyTorch) | `trainer.ts` (TypeScript, the kit's autograd) | `process.ts` (a TypeScript generator) |
| The kit makes | `spec.json` + `weights.bin` + `reference.json`; the forward pass in TS; one frame per layer; zoom views for one output number | the training loop: a frame per step (coarse) or per forward / loss / backward / update (fine) | the recorder, the tape store, the player |
| Proof ("parity") | every layer ≤ 1e-4 against PyTorch fp32 on the CPU | optional twin: the same training in PyTorch float64 (SGD ≤ 1e-10, Adam ≤ 1e-7 relative) | optional Python reference: the same frames |
| Examples | TinyViT, char-RNN, GCN inference | MLP playground, PINN, flow matching | Q-learning, bandits, sampling, BPE merges |

A package can mix kinds. `hello-xor` (the kit's own example) uses all three on one tiny network: the PyTorch-trained
weights (A), the same network trained live (B), and one gradient-descent step slowed down into its phases (C).

**Sources** connect the kinds to the lesson. A package declares named sources in its manifest; each lesson step plays
one of them:

| Source kind | What it is |
|---|---|
| `trace` | the exported model run on the learner's input (kind A) |
| `scan:<layer>` | implicit: one frame per output cell of a layer (a kernel sliding over the input) |
| `train` | a `TrainerDef` from `trainer.ts` run live (kind B) |
| `process` | a `ProcessDef` from `process.ts` run live (kind C) |
| `tape` | a tape file recorded ahead of time (`atlas run --precompute`) |
| `live-model` | a kind-A trace on a train source's weights, as far as that source has run (section 6.2) |

---

## 3. What a package is

One folder per model, found by name; adding a model edits no shared file:

```
site/src/lib/explainers/<slug>/       the page (TypeScript; you own it)
  explainer.ts      the manifest: kinds, input, sources, goals, credits          (required)
  lesson.ts         the guided lesson: steps                                     (required for a real page)
  trainer.ts        kind B: TrainerDef exports
  process.ts        kind C: ProcessDef exports
  worker.ts         generated: runs trainer.ts / process.ts in a Web Worker      (4 lines; do not edit)
  claims.test.ts    tests for every factual sentence that is not computed live
  ops.ts, ops/      local ops (only when no core op fits; section 5.6)
  adapters/<id>/    a local input adapter (adapter.ts + Capture.svelte)
  views/<key>/      a local view (meta.ts + View.svelte)
  sections/, content/   optional Svelte overrides (section 9)
  concept.ts        a concept package only: the page around the lesson (section 18)
pipeline/<slug>/                      Python (kind A, twins, references)
site/static/models/<slug>/            assets written by export.py (never by hand)
notes/<slug>.md                       your working notes (What, Decisions, Final numbers, How to re-run, Kit issues)
```

**Rules for the TypeScript files** (`atlas check` enforces them; AK503):

- Import only from `src/lib/atlas/**` (not `atlas/ui/`), `src/lib/i18n/types.ts` and your own package, with relative
  paths that end in `.ts`: `import { MLP } from '../../atlas/autograd/nn.ts'`. No `$lib/...`, no `$app/...`, no
  `.svelte` files, no npm packages, no `node:*` (tests may use `node:*`).
- Erasable TypeScript only (Node strips the types): no `enum`, no `namespace`, no constructor parameter properties.
- No `Math.random`: draw from the `rng` the kit gives you. Every number on the page must be reproducible.
- Every user-visible string is bilingual: `{ en: '...', zh: '...' }`. English is complete; Chinese is required for
  titles and labels (AK406).

**Slugs.** A real page uses a slug from the catalog (`src/lib/catalog/models.ts`; `atlas new` checks it, AK701). For
experiments, `--dev` makes a lab-only package (`devOnly: true` in the manifest): it is never routed under `/m/` and
only appears in the Lesson Lab. A **concept package** (`npm run atlas -- new concept <slug>`) is the same kind of folder
for a slug of `src/lib/catalog/concepts.ts`, routed at `/train/<slug>/` (section 18).

**Ids are per package.** Trainer ids, process ids, source names, step ids, fixture ids and the names of local adapters,
ops and views only need to be unique inside your package: each package registers its own (two packages can both have a
trainer `xor`, registered as `train:xor` in each). One kind of id is site-wide: predict question ids
(`page.prequestion.id`, a step's `predict.id`), because the page remembers the learner's answers by them; prefix them
with your slug, as hello-xor does (`hello-xor-pre`).

---

## 4. Walk-through: your first package

This builds a small package with all three kinds, like `hello-xor`. Open the hello-xor files next to this guide; they
are the worked answer:

- `site/src/lib/explainers/hello-xor/`: `explainer.ts`, `lesson.ts`, `trainer.ts`, `process.ts`, `claims.test.ts`, `adapters/xor-point/`
- `pipeline/hello-xor/`: `model.py`, `train.py`, `export.py`, `twin.py`, `reference.py`

All commands run from `site/` (or from your lane's `site/`, section 11).

### 4.1 Scaffold

```
npm run atlas -- doctor
npm run atlas -- new model my-xor --dev --kind trace,train,process --input xor-point
```

This writes `src/lib/explainers/my-xor/` (explainer.ts, lesson.ts, trainer.ts, process.ts, worker.ts),
`pipeline/my-xor/` (README.md, model.py, export.py, twin.py, reference.py) and `notes/my-xor.md`, and prints a numbered
"Next:" list. Nothing is ever overwritten. Each kind gets a working toy to replace: an MLP to export, an XOR trainer
and its twin, a coin-flip process and its reference.

`atlas new` also gives your package its dev-server port: its row in the port table (section 11), else the first free
port from 5210 up that no other package uses (`--port n` chooses it). The notes file records it and "Next:" prints it.

**A variant of an existing package** (a different output, another dataset, a second optimizer) starts best as a copy:

```
npm run atlas -- new model my-xor --like hello-xor --dev
```

copies `src/lib/explainers/hello-xor/` and the sources of `pipeline/hello-xor/` (not its checkpoint, metrics or data:
train and export make your own), renames `hello-xor` to `my-xor` everywhere (also `helloXor` and `hello_xor`), writes
fresh notes and prints what it left out. The copied `claims.test.ts` still states facts about hello-xor: rewrite each
claim. Trainer and process ids need no renaming (section 3).

Templates give a closer starting point: `--template drawing-classifier | char-seq | points2d-train | series-train |
graph-node | process-blank` (section 12). A template sets `--kind` and `--input` for you. A trainer or process receives
the learner's **raw** input (`rt.input`: the adapter's `{ preset }` or the painted points), not the tensors: read it with
the adapter's own helper, as the templates do (`pointsOf(raw)` from `atlas/adapters/points2d/adapter.ts`, `seriesOf(raw)`
from `atlas/adapters/series/adapter.ts`).

`xor-point` is not a kit adapter, so copy hello-xor's local adapter: `adapters/xor-point/adapter.ts` (pure TS: raw input
-> tensors) and `Capture.svelte` (the input widget). Kit adapters (`drawing28`, `text-bpe`, `text-char`, `points2d`,
`series`, `graph`) need no copy; for any other `--input`, `atlas new` says which files to write.

### 4.2 Kind A: train in PyTorch, then export

1. Write the model in `pipeline/my-xor/model.py` (plain PyTorch). Train it with a fixed seed and save the checkpoint
   (hello-xor: `npm run atlas -- py pipeline/hello-xor/train.py` writes `model_final.pt` in about a second; `py` runs
   the configured Python, section 1).
2. Write `describe()` in `export.py`: one builder call per step of `forward()`. Every call computes the PyTorch value
   for the current case and records the layer:

   ```python
   def describe(b: ak.SpecBuilder, case: ak.Case) -> None:
       x = b.input("input", case.input, adapter="xor-point", kind="points2d", axes="f",
                   label=("Your point", "你的点"))
       h = b.linear("fc1", x, model.fc1, label=("Hidden layer", "隐藏层"))
       h = b.tanh("act1", h, label=("Tanh", "Tanh"))
       y = b.linear("fc2", h, model.fc2, label=("Output score", "输出分数"))
       b.output(y)                     # if forward() ends with torch.sigmoid, add  y = b.sigmoid("out", y)  first
   ```

   Values carry no batch dimension: images `[C, H, W]`, sequences `[T, D]`, a point `[2]`.
3. List the reference cases: typical inputs plus edge cases (blank, extreme, the shortest and longest sequence) with
   `edge=True`. Then:

   ```
   npm run atlas -- export my-xor
   ```

   This runs `export.py` with the configured Python. `export_model()` proves `describe()` equals `model(x)` on every
   case (AK103), writes `site/static/models/my-xor/` (spec.json, weights.bin, reference.json, facts.json,
   presets.json), then the CLI runs `parity my-xor --write` (every layer of every case ≤ 1e-4 against PyTorch; the
   measured number goes into `spec.parity`) and `lesson my-xor draft` (the auto lesson's step ids).

### 4.3 Kind B: a trainer

`trainer.ts` exports a `TrainerDef`: how to build the model, how to get a batch, the loss. The kit runs the loop.

```ts
export const xor: TrainerDef<unknown, MLP, { x: Tensor; y: Tensor }, XorOpt> = {
	id: 'xor',
	title: { en: 'Train the XOR network', zh: '训练 XOR 网络' },
	defaults: { seed: 1, lr: 0.05, batch: 4, steps: 300, snapshotEvery: 5, optimizer: 'adam', hidden: 2 },
	model: (rng, opt) => new MLP('net', [2, opt.hidden, 1], 'tanh', rng),
	batch: () => XOR,
	loss: (m, b, rec) => ({ loss: mse(m.forward(input(b.x), rec), input(b.y)) }),
	metrics: (m) => ({ correct: correct(m) })
};
```

**An output activation.** `MLP` has no activation after its last layer unless you give one:
`new MLP('net', [2, 2, 1], 'tanh', rng, { outAct: 'sigmoid' })` ends in a sigmoid (recorded as `net.3`, as
`nn.Sequential(..., nn.Sigmoid())` names it), and `outAct: 'softmax'` ends in a softmax over the last axis. `forward()`
applies it; `logits()` stops before it, for a loss that wants the raw scores:

```ts
model: (rng, opt) => new MLP('net', [2, opt.hidden, 1], 'tanh', rng, { outAct: 'sigmoid' }),
loss: (m, b, rec) => ({ loss: bceWithLogits(m.logits(input(b.x), rec), input(b.y)) }),
```

The output activation has no parameters, so the twin's state_dict names do not change, and `toSpec()` includes it, so
a `live-model` step shows it as a layer (section 6.2).

Run it in Node before any page exists:

```
npm run atlas -- run my-xor train --frames 0..20
```

The optional `pipeline/my-xor/twin.py` is the same network in PyTorch (`build(cfg)` and `loss(model, batch, cfg)`, with
the same state_dict names: `MLP('net', ...)` -> `net.0.weight`, `net.2.weight`, ...). `npm run atlas -- parity my-xor`
then proves 50 steps of SGD (≤ 1e-10) and of your optimizer (≤ 1e-7) match PyTorch float64.

### 4.4 Kind C: a process

`process.ts` exports a `ProcessDef`: a generator that yields one frame per event. Each phase gets a label, a narration
(with `{frame.<metric>}` placeholders) and optionally a formula:

```ts
phases: {
	update: {
		label: { en: 'Update', zh: '更新' },
		kind: 'update',
		formula: { tex: 'w \\leftarrow w - \\eta g = {{old}} - {{lr}} \\times {{g}} = {{new}}', values: (f) => f.metrics! },
		narration: { what: { en: 'The weight goes from {frame.old} to {frame.new}.', zh: '这个权重从 {frame.old} 变成 {frame.new}。' } }
	}
},
*run(input, ctx): Run {
	// ... compute, then
	yield { phase: 'update', step: 0, state: { W2 }, metrics: { lr, old, g, new: w } };
}
```

```
npm run atlas -- run my-xor run --frames 0..12
```

The optional `pipeline/my-xor/reference.py` writes the same frames from Python (`ak.Rng`, `ak.TapeWriter`);
`atlas parity` compares them frame by frame.

### 4.5 The manifest and the lesson

Fill `explainer.ts`: goals, the pre-question, the lesson note, the sources, credits (section 3; hello-xor's is a
complete example). Then the lesson (section 8). Start from the auto plan and replace steps with your own:

```
npm run atlas -- lesson my-xor draft          # the auto steps and their ids
npm run atlas -- lesson my-xor lint           # every step x every input fixture x en/zh
npm run atlas -- lesson my-xor print          # every narration, resolved, as tables
```

A good lesson has 5 to 12 steps, one idea each, at least one `predict` (the learner guesses before seeing), one `task`
(the learner does something) and one honest-limits step (`kindOf: 'limits'` or `'breakIt'`). Every number in the
narration is a `{placeholder}` bound to a computed value; never type a number into the text (AK402 warns).

Write `claims.test.ts` for every sentence that is not computed live ("a third hidden unit fixes this start"):
hello-xor's file shows how (`withFixtures` runs your assertion on every input fixture).

### 4.6 Look at it

```
npm run dev -- --port <your port>         # the port atlas new printed (notes/my-xor.md); section 11
```

Open `/dev/atlas/lab/my-xor/` (the Lesson Lab: every step, a fixture and language switch, the lint list). A catalog
package also renders at `/m/<slug>/` in dev (drafts are included in dev). Edit and save: the page reloads, also in a lane
(a lane made before K0 needs `npm run atlas -- worktree <name> --sync` once; section 11). Every step
has a deep link, `#lesson-<id>`: it opens that step first, and must work as the first step entered (the kit runs the
sources it needs; section 6.2 for `live-model`). You can keep the server running: `atlas e2e` and `atlas check` reuse
it (section 10).

### 4.7 Check, then write it up

```
npm run atlas -- check my-xor --quick       # while you iterate: every gate except the build and the browser test
npm run atlas -- e2e my-xor                 # the browser test: 2 passes, an axe scan, contact sheets, the lesson print
npm run atlas -- check my-xor               # once, at the end: every gate, a package-only build and the e2e passes
```

Iterate with `--quick` and `e2e`; run the full `check` once, when the package is done. `--quick` builds nothing and
opens no browser, and it takes the parity and op-test results from a cache while their inputs are unchanged, so after
a lesson edit it takes seconds. `e2e` steps through the lesson at 1280 px in English (light) and at 390 px in Chinese
(dark, with a horizontal-scroll probe at 360 px), runs an axe accessibility scan on every step (AK604), and writes a
contact sheet per pass (every step, labelled) and `lesson-print.txt` (every narration with its numbers, in both
languages) to `notes/shots/my-xor/contact/`: look at those before you click through the page yourself.

`check` and `e2e` reuse a dev server of this site when one runs (yours on the package's port, or one they kept), else
start one on your package's port (else on the first free port from 5210 up) and keep it for the next run; it stops
itself after 20 idle minutes, or now with `npm run atlas -- e2e --stop-server`. The report names the port, and
`--port n` chooses it (section 12).

Fix every AK error (section 13 explains each code). Fill `notes/my-xor.md`: What, Decisions, Final numbers (measured
only), How to re-run, Kit issues. A draft is done when `check` is green; `quality: 'reviewed'` comes after the teaching
review (`npm run atlas -- review my-xor` writes the reviewer's packet).

---

## 5. Kind A: export a trained model

### 5.1 The builder

`import atlas_kit as ak` (Python ≥ 3.10). Importing it switches TF32 off; export runs on the CPU only (AK102).

| Call | Records |
|---|---|
| `b.input(name, value, *, adapter, kind, axes, shape=None, dims=None, dtype=None, layer=None, **meta)` | a declared input; float inputs also get an `input` layer (layer 0). `shape=['T', 3]` with `dims={'T': (1, 64, ('steps', '步'))}` declares a symbolic length. `dtype='i32'` for token ids (no layer). |
| `b.linear(id, x, module=None, *, weight=None, bias=None, tied_to=None, alias=None, accumulate=None, n_head=None)` | `y = x Wᵀ + b` on the last axis. `tied_to='tok_emb.weight'` reuses a tensor; `alias='dense'` writes op `dense`; `n_head` writes the head layout of q/k/v projections. |
| `b.conv2d(id, x, module)` | square kernel, stride and zero padding read from `nn.Conv2d` (no dilation or groups) |
| `b.relu`, `b.tanh`, `b.sigmoid`, `b.silu(id, x)`; `b.gelu(id, x, approx='none' \| 'tanh')` | elementwise activations |
| `b.maxpool2d(id, x, kernel=2, stride=None)` | max over kernel x kernel windows |
| `b.flatten(id, x)`; `b.concat(id, [a, b], axis=0)`; `b.add(id, a, b)` | shape and merge ops |
| `b.softmax(id, x, axis=-1)` | softmax on an axis |
| `b.layernorm(id, x, module)` | LayerNorm on the last axis (biased variance; eps from the module) |
| `b.embedding(id, ids, module)`; `b.position_embedding(id, ids, module)` | table lookups |
| `b.attention_scores(id, q, k, n_head, causal=False, scale=None)`; `b.causal_mask(id, s)`; `b.attention_mix(id, w, v, n_head)` | multi-head attention in three layers (q, k, v are `[T, H·dH]`) |
| `b.lstm(id, x, module)` | one `nn.LSTM` layer; records the gate sub-steps `id#i f g o c h` and proves them against the module (AK104) |
| `b.op(type, id, inputs, *, params, weights, ref)` | any other op (a local op): `ref(xs, params, weights)` returns the output, or `(output, {sub: tensor})` |
| `b.output(*nodes)` | the model's outputs (default: the last layer) |

`**meta` on every call: `label=(en, zh)`, `desc=(en, zh)`, `axes='c,y,x'`, `code=('pipeline/<slug>/model.py', line)`,
`ui={...}`, `teach={'lesson': 'step' | 'fold' | 'skip', 'title': (en, zh)}`. Layer ids may contain dots; never `#`.
Attributes are checked against `src/lib/atlas/schema/ops.json` (AK106); the full table is in section 14.

### 5.2 export_model

```python
report = ak.export_model(describe, CASES, slug="my-xor", name=("My XOR", "我的 XOR"), kinds=("trace",),
                         module=model, module_input=lambda c: c.input.unsqueeze(0),
                         facts={"trainLoss": {"value": loss, "unit": "mse", "source": "train.py"}},
                         presets={...}, training={...}, extras={"classes": [...]})
```

- `describe` runs once per `ak.Case(id, label, input, edge=False)`. The first case fixes the graph and the weights;
  every later case must record the same layers (AK101). A case that differs only in a symbolic dim (T) is how a
  sequence model gets `outShape: ['T', 76]`. Include the dim extremes (T = 1 and T = max): parity warns without them.
- `module` + `module_input` prove the recorded outputs equal `module(x)` (AK103, ≤ 1e-5).
- `facts` become `facts.json`; narration reads them as `{fact:trainLoss}`. Put measured numbers here, never in text.
- Budgets: each file ≤ 5 MB, the model folder ≤ 15 MB (AK105); nothing is written when a check fails.
- Output: `spec.json` (spec 1.1 with `parity` null until `atlas parity` runs), `weights.bin` (float32, tied tensors
  once), `reference.json` (atlas-ref-1: every layer of every case), `facts.json`, `presets.json`.

### 5.3 Parity A

`npm run atlas -- parity <slug>` runs the kit's TS forward pass on every reference case and compares every layer and
sub-step with PyTorch: max |diff| ≤ `spec.parity.bar` (1e-4) or AK201 names the layer, the case and the two values. It
also checks that each op's `explain()` (the arithmetic behind one output number, shown by the zoom views) equals the
trace on 2,000 sampled units (≤ 1e-5, AK202). `--write` (and `export`) store the result in `spec.parity`; the page
badge shows it.

### 5.4 The input adapter

An adapter turns the learner's raw input (strokes, text, a point) into the model's input tensors:
`preprocess(raw, spec, opts) -> { tensors: { input: ... }, steps, empty }`, plus `presets(assets)`,
`edgeFixtures()` (inputs every lesson must survive: blank, one token, ...) and `layer0` (the first step's view and
narration). Kit adapters: `drawing28`, `text-bpe`, `text-char`, `points2d`, `series`, `graph`. A local one lives in
`adapters/<id>/adapter.ts` (+ `Capture.svelte`, the widget). Its props are `CaptureProps<Raw>` (`atlas/types/explainer.ts`):
`rt`, `raw`, `presets` (may be empty), `options` (the manifest's `input.options`), `onchange(raw, { live? })` and
`compact` (the small copy beside the lesson). A text adapter may also fill `tokens` (display text per position) and
`vocab` (text of any id) in its result, so token views show words instead of `#id`. `presets(assets)` is called when the
manifest names a presets file (`input.presets`); the file is in `assets`. An adapter that makes its presets itself needs
no file: `input: { adapter, presetsFromAdapter: true }` calls `presets({})`.

What the kit adapters take (before you copy one into your package, check that an option does not already do it):

| Adapter | Options and files |
|---|---|
| `drawing28` | `options.frame: [x0, y0, x1, y1]` (or `raw.frame`): render the strokes inside that box (the canvas), so a drawing keeps its size and place; by default the ink is cropped and centred, as TinyCNN's training data was |
| `text-char` | the alphabet from `options.vocab` or the spec; typographic characters a phone types (curly quotes, dashes, `…`, a no-break space) become their plain forms when the alphabet lacks them (`options.normalize: false` turns it off); presets file `{ texts: [{ id, en, zh, text }] }` |
| `points2d` | `options.classes` (1 = unlabelled; such a page may pick the labelled shapes too, drawn in one color), `options.presets` (an id list) |
| `series` | presets file `{ presets: [{ id, name?, en?, zh?, seed?, noise?, n? }] }` (`name` is the generator, default the id); without a file the Capture lists the generators, `options.presets` limits them; the chip of the learner's input compares the whole preset (name, length, seed, noise) |
| `graph` | Zachary's karate club (`atlas/data/karate.ts`: the same graph as `ak.data.karate()`, factions, a layout and the helpers `graphOf`, `adjacency`, `normalizedAdjacency` (Â), `toggleEdge`, `hops`); tensors `adj` and one-hot `x`; `adapters/graph/Club.svelte` draws the club with its table for your own views |

The page's input section names each of the adapter's `steps` and draws its tensor ("what the model sees"), except thin
ones: a short list of points (at most 12 rows of at most 4 numbers) is printed as numbers, and a long one (a list of 200
points) is named with its shape. A step without a tensor is named with its `note`. `notes` show in the input section
and, while the input is empty, on the lesson's stage instead of the view.

### 5.5 Classes and outputs

Put class names in `extras={"classes": [{"id", "en", "zh"}]}`; the output step and `{top.label}` use them. The page's
output view is `output: { view: 'kit.probbars' }` in the manifest for classifiers (`props.chosen`: the id that was picked,
e.g. a sampled token; a class output marks its top class, a vocabulary output nothing by default). `output.view` is
optional.

**The answer row.** For a `[T, V]` output, `output.row` says which row is the answer: `'last'` (the default), `'first'`,
an index (a negative one counts from the end) or `(rt) => row` (a masked model's `[MASK]` row). `top.*`, the side panel,
the auto output step ("guess for position k"), `kit.probbars` (also `props.row`), `compare` and the row `layer.tensor`
opens on all read it, so a masked model needs no workaround in the package.

**The answer in the lesson's side panel.** Shown for an output with one answer: a `[class]` or `[vocab]` vector, or a
sequence's `[t, vocab]` rows (read at `output.row`). A per-item output such as a GCN's `[n, class]` (one answer per
node) and an output whose last axis is not `class` or `vocab` (an image) show none, even when the spec lists classes.
A text model's answer reads as its vocabulary word (`Preprocessed.vocab`), never `#id`.

**Axis names** (`axes='n,class'` in `describe()`; views and the scrubber name an index by them): `c` channel, `y` row,
`x` column, `t` position, `tq` query, `tk` key, `d` and `f` feature, `h` head, `n` node, `nk` the second node of a
node-by-node table (an adjacency), `b` an item of a batch, `p` patch, `class`, `vocab`, `xy` a 2-D point, `k` any other
index.

**Row names.** Sequence views (attention, the zoom tables) name each row with the token text (`rt.pre.tokens`, from a text
adapter). A model without text gives the names in `extras={"rowNames": ["[CLS]", "p0", ..., "p15"]}` (a ViT's patches, a
graph's nodes); a step's `props: { tokens }` still wins.

### 5.6 When no core op fits: a local op

```
npm run atlas -- new op gcn_conv --local my-net --like linear
```

writes `explainers/my-net/ops/gcn_conv.ts` (+ a test) and `pipeline/my-net/ops_gcn_conv.py` (the PyTorch mirror), and
the package's `ops.ts` when it has none (`export default [...CORE_OPS, gcn_conv];`; with an `ops.ts` already, add the op
to it). `--like linear` copies the core op with its imports re-pointed to the package folder, its export renamed
(`gcnConv`) and its `op:` name changed (terms keep theirs), and copies the `linear` Python mirror; its test only checks
that the copy loads: write the real tests. In `describe()`, record it with
`b.op('gcn_conv', id, [x], params=..., weights=..., ref=ops_gcn_conv.reference)`. An optional weight is read with
`optW(ctx, 'bias')` (ops/common.ts), which asks `ctx.has('bias')` (the executor provides it) and returns `null` when
the model has no such tensor. An op moves into the core when a second model needs it.

---

## 6. Kind B: train live in the browser

### 6.1 TrainerDef

| Field | Meaning |
|---|---|
| `id`, `title` | registered as process `train:<id>` of your package (ids are per package, section 3); a source `{ kind: 'train', trainer: '<id>' }` runs it |
| `defaults` | `TrainOpt`: `lr, batch, steps, snapshotEvery, optimizer: 'sgd' \| 'momentum' \| 'adam'`, plus your own fields. Extras the optimizer reads: `momentum`, `betas`, `eps`, `weightDecay`, `decoupled` |
| `model(rng, opt)` | build the `Module`: `MLP` (`new MLP(name, sizes, act, rng, { bias?, outAct? })`: no activation after the last layer unless `outAct`; `logits()` stops before it), `Linear`, `Embedding`, `LSTMCell`, `Conv1d`, `LayerNorm` from `atlas/autograd/nn.ts`, or your own class with `params()`, `named()` and, for a `live-model` source, `toSpec()` (section 6.2) |
| `batch(data, step, rng, opt, model)` | the batch for one step; `data` is the learner's **raw** input (`rt.input`): read it with the adapter's helper (`pointsOf(data)`, `seriesOf(data)`; section 4.1); `model` is the network being trained (an RL round that acts with it) |
| `loss(m, b, rec, opt)` | `{ loss, parts? }` as autograd `Var`s (`atlas/autograd/loss.ts`: `mse`, `crossEntropy`, `bceWithLogits`, `infoNCE`; ops in `atlas/autograd/ops.ts` as `op.*`) |
| `optimizer?(m, opt)` | override the optimizer |
| `metrics?(m, data)` | numbers recorded at init and snapshot frames (accuracy, ...) |
| `twin?` | `{ sgd: { lr: 1e-3 } }`: the options of `atlas parity`'s plain-SGD twin run (section 6.3) |

**An empty input.** A train source does not run while the adapter says the input is empty (no points, no text): its
steps show the adapter's empty note instead of the `init` frame of a network trained on nothing, and lint skips them on
the empty fixture. A later input starts a fresh run. `{ kind: 'train', ..., runOnEmpty: true }` runs it anyway (a
trainer that does not learn from the learner's input).

**Seeds.** The run's seed is the source's `opt.seed` (`{ kind: 'train', trainer: 'xor', opt: { seed: 5 } }`), default
1; `defaults.seed` is not used. `model()` receives the run's `rng.fork('init')`, so two sources with the same seed start
from the same weights. `batch()` receives one `rng.fork('batch')` stream for the whole run: **draw from the rng you are
given; never fork it again per step** (that would give every step the same batch).

**AdamW** is `optimizer: 'adam'` with `decoupled: true` (weight decay default 0.01). `TrainOpt` has no `'adamw'`.

**Recording activations.** `rec(id, v)` stores a named intermediate for the views (`state.acts`, fine detail). When a
loss calls the model several times (PINN: collocation points, observations, t = 0), the ids collide and the last call
wins; wrap `rec` with a prefix per call: `(id, v) => rec('obs.' + id, v)`.

### 6.2 What the tape looks like

- **coarse** (`detail: 'coarse'`): one frame per step. Phase `snapshot` every `snapshotEvery` steps and at the end
  (with `params`), phase `step` otherwise (metrics only: `step`, `loss`, `lr`, `gradNorm`). `{ step: 300 }` means "the
  weights after 300 updates".
- **fine** (`detail: 'fine'`): four frames per step: `forward` (batch, activations), `loss`, `backward` (grads),
  `update` (new params, previous params, grads, optimizer moments, lr).
- an `init` frame starts every run.
- Controls: `{ type: 'set', opt: { lr } }` acts at the next step; `{ type: 'input' }` swaps the data and keeps the
  weights (a new segment); `{ type: 'reset' }` starts again from fresh weights, on the latest input and with the options
  set so far.
- **A new input restarts the run.** When the learner changes the input, every train source starts again on it (the
  runtime sends `input` then `reset`): fresh weights, the same seed, the learner's `set` options kept, and the same tape
  object (a `TrainSession` or `train-controls` bound to it keeps working). So a step that reads the trained weights (a
  `live-model` step) always shows weights trained on the input it shows. To swap the data without restarting, send
  `{ type: 'input' }` to the tape yourself.
- A source with a `precomputed` tape (phones) uses the file for the start input only; after a new input it trains live.

`train/helpers.ts` `updateOf(at, 'net.0.weight', [0, 1])` returns `{ old, g, lr, new }` for the narration "old − lr × g
= new"; it throws unless the source ran plain SGD, so a lesson cannot claim the SGD rule for Adam. Use a second source
with `opt: { optimizer: 'sgd' }` and `detail: 'fine'` for the slowed-down step.

A `live-model` source (`{ kind: 'live-model', of: 'train' }`) traces the weights of a train source on the learner's
input (the adapter's `input` tensor) with the kind-A views. `at: 'first'` traces the starting weights (the train source's
`init` frame) instead of the latest: the same train source serves "before" and "after" steps.

- **The model needs `toSpec()`.** The kit rebuilds your `model()` with the run's options, loads the recorded weights and
  calls its `toSpec()`. `MLP` and `Linear` have one (an `MLP`'s `outAct` included). A custom module writes it with
  `stackToSpec` (`atlas/autograd/tospec.ts`): the linear layers and activations in `forward()` order. Without it the step
  fails with "its model has no toSpec()".

  ```ts
  toSpec(o: { slug: string } & ToSpecOpts) {
  	return stackToSpec([
  		{ kind: 'linear', id: 'net.0', weight: this.fc1.weight, bias: this.fc1.bias },
  		{ kind: 'act', id: 'net.1', act: 'tanh' },
  		{ kind: 'linear', id: 'net.2', weight: this.fc2.weight, bias: this.fc2.bias },
  		{ kind: 'act', id: 'out', act: 'sigmoid' }
  	], o);
  }
  ```
- **Its numbers depend on how far the train source has run.** It reads the source's latest snapshot. A learner who
  steps through the lesson has run it to the end at the train step (`at: 'last'`), so the numbers are the final
  weights'; a learner who pauses training midway sees that moment's weights; `?atlas-e2e=1` caps training at 200 steps,
  so e2e screenshots can show other numbers than your browser (`atlas e2e <slug> --cap off` runs the full length).
  Narrate them as "the weights just trained", bound to placeholders, and back any fixed claim about them with a claim
  test that runs the source to `'last'`.
- **A deep link straight to the step works.** Entered first (`#lesson-live`), the step runs its train source to the
  end first, exactly as a step `at: 'last'` on it would, and the stage says "Training the network first… step k / n"
  meanwhile; the numbers equal the ones a learner gets by stepping in.

### 6.3 The twin (optional, stronger badge)

`pipeline/<slug>/twin.py`:

```python
def build(cfg: dict) -> nn.Module                      # same architecture and state_dict names, .double()
def loss(model, batch: dict, cfg: dict) -> (Tensor, dict)   # batch values are float64 tensors in the TS shapes
def optimizer(model, cfg)                              # optional
```

Labels arrive as floats: call `.long()`. `atlas parity` runs the TS trainer in fine detail, writes its init params and
batches to `.atlas/<slug>/train_fixture.<trainer>.<optimizer>.json` (TS is the source of truth for initial weights and
synthetic data), runs `python -m atlas_kit.train_reference`, and compares losses, step-1 gradients and final params:
plain SGD for 50 steps ≤ 1e-10 relative, the page's optimizer for 50 steps ≤ 1e-7 relative (AK203).

It runs once per distinct trainer and options (`steps` and `snapshotEvery` aside): sources that differ in an option
(λ = 0, λ = 0.01, a wrong c) each get their own record, so every badge states a measured run. The plain-SGD run uses
the page's options with `optimizer: 'sgd'`; a stiff loss can diverge at the page's learning rate. Such a run is compared
up to the step before it blows up and on its step-1 gradients, with a warning; for a full 50-step comparison give the
trainer `twin: { sgd: { lr: 1e-3 } }`. The page's own optimizer run is unchanged.

**A process that wraps a trainer.** A process source that spreads `...trainingProcess(t)` and makes exactly t's frames
gets t's twin, run with the source's options, and the "= PyTorch fp64" badge after `parity --write`: the kit compares
the frames itself (`train/wraps.ts`; `trainingProcess` sets `ProcessDef.trainer`). A wrapper that adds metrics or
changes frames keeps the "Live" badge, and its twin failures are warnings. Twin answers are cached in
`.atlas/<slug>/twin-cache/` (by fixture, steps, the `.py` files and the Python), so a second parity run takes seconds.

---

## 7. Kind C: a process (an algorithm, step by step)

### 7.1 ProcessDef

```ts
export type WalkOpt = { cells: number; steps: number };
export type WalkState = { pos: number; visits: Tensor };
type Run = ProcessRun<WalkState, WalkOpt>;   // from atlas/types; spell this out (see below)

export const walk: ProcessDef<unknown, WalkOpt, WalkState> = {
	id: 'walk',
	kind: 'algorithm',
	title: { en: 'A random walk', zh: '随机游走' },
	phases: { start: {...}, move: {...} },          // label, kind, narration, formula
	schema: {
		pos: { kind: 'int', label: {...}, tol: 'exact' },
		visits: { kind: 'tensor', label: {...}, track: 'changes', tol: 'exact' }
	},
	defaults: { cells: 7, steps: 30 },
	*run(input, ctx): Run {
		const s = ctx.rng.int(2) === 0 ? -1 : 1;    // randomness only from ctx.rng
		...
		const control = yield { phase: 'move', step: t, state: { pos, visits }, changes: [{ field: 'visits', index: [pos], from: old, to: old + 1 }], metrics: { t, s } };
	}
};
```

- **Frames**: `{ phase, step?, sub?, state, changes?, focus?, metrics?, caption? }`. Put every number the narration or
  the formula needs into `metrics` of that frame: a step's cursor only runs the process up to that frame, so values
  from later frames do not exist yet.
- **Schema**: one entry per stored field. `track: 'changes'` for a big field you mutate in place (a Q-table): every
  edit must be reported in `frame.changes` (the determinism gate catches a missing one). `tol` is the parity tolerance
  for the Python reference (`'exact'` or a relative bar such as `1e-12`). `derive(state, opt)` makes a field that is
  computed, never stored; `opt` is the run's options (the defaults and the source's `opt`), so a derived heatmap can
  read the activation the run used.
- **Phases**: `kind` is one of input, forward, loss, backward, update, observe, act, reward, message, aggregate, sample,
  merge, output, other. Give each phase a `narration.what` with `{frame.<metric>}` placeholders: the auto lesson makes
  one step per narrated phase (`proc:event.<phase>@<source>`). A `formula` (KaTeX; `{{name}}` slots filled from
  `values(frame)`) shows in the `zoom.formula` view. A value may be a string: `fmtSetting(0.05)` (`atlas/core/format.ts`)
  prints a setting as it was set, '0.05', not '0.0500'.
- **The `Run` type**: declare `run()`'s return type as above (`ProcessRun<S, O>` is
  `Generator<Frame<S>, void, Control<O> | undefined>`). Without it, TypeScript merges the yielded literals into one type
  and frames with different metrics fail to type-check. `defineProcess({...})` (`atlas/process/define.ts`) is an
  identity helper that infers the input and option types; it still needs the `ProcessRun<S, O>` return type when the
  frames differ in shape.
- **Controls** arrive as the value of `yield`: `{ type: 'set', opt }`, `{ type: 'input', input }`.
- **Pure**: the same input, options and seed must give the same frames. Ties break to the smallest index.
- **recordDraws**: a process that draws with `rng.normal()` (or other transcendental transforms) sets
  `recordDraws: true` on its ProcessDef (or on its source: `recordDraws: true` or `opt: { recordDraws: true }`), so
  the Python reference replays the TS draws instead of drawing.

The source: `{ kind: 'process', process: 'walk', seed: 1, opt: {...}, input: (rt) => ..., detail: 'fine' }`.

- `input` defaults to the learner's raw input. A new learner input closes the run (it reruns on the next step that
  needs it), unless `input(rt)` gives the same input as before (compared as data): a process that ignores the learner's
  input keeps its run.
- **Where it runs.** In the browser a process runs in the package's worker when the package has one. `run: 'main'`
  keeps it on the main thread: needed when its input holds what a worker cannot receive (`rt.model`, a function). Keep
  each frame of a main-thread process well under 50 ms (AK603). Lint catches the other case in Node: a worker source
  whose input cannot be sent to a worker fails ("cannot be sent to a worker"), as the page would. (Before K1 the browser
  ignored `run: 'main'` and sent every process to the worker.)
- `after: '<source>'` opens the process once that source has run to its end, so `input(rt)` can read its frames, e.g.
  the trained weights: `input: (rt) => { const t = rt.sources.train.tape!; return t.state(t.length - 1).params; }`.
  The stage says "Training the network first… step k / n" meanwhile, and a new input reruns both.
- `detail` (default `'coarse'`) is the run's `ctx.detail`: a process that wraps `trainingProcess(...)` passes it on to
  get the forward / loss / backward / update frames.
- `reference: false`: `atlas parity` runs no Python reference for this source (one the reference cannot reproduce,
  such as drawings rendered by TS); its badge says "Live".
- A process that yields no frames on an empty input is skipped like a train source: its steps show the adapter's empty
  note (section 8.3).
- The e2e step cap (`?atlas-e2e=1`, 200 steps) applies to `opt.steps` of processes with `kind: 'training'` (train
  sources, and your own process if it declares that kind).

Library processes: `atlas/process/library/sampling.ts` (temperature / top-k / top-p), `euler.ts`, `rollout.ts`.

### 7.2 The Python reference (optional)

`pipeline/<slug>/reference.py` reimplements the process with `ak.Rng` (bit-identical to the TS rng for `next()`,
`int()` and `fork()`) and writes the frames with `ak.TapeWriter`. **CLI contract**: it accepts `--process <id>
--seed <n> --opt <json> --draws <draws.json> --input <json> --out <stem>`; `atlas parity` calls it that way after
recording the TS tape (and its draws) in `.atlas/<slug>/parity/` (its own folder: `atlas run --save` keeps yours in
`.atlas/<slug>/`). Frames must match one by one: phase, step and sub exactly, every stored field within its `tol`,
metrics within the process's `metricTol` (default 1e-12 relative). The first difference is AK204.

- **The input** is the source's `input(rt)` as JSON: typed arrays become lists, and what is not plain data (a model
  object, a function) is left out. A long input (over 8,000 characters; the Windows command line holds about 32,000)
  arrives as `--input @<file>`: read the argument with `ak.json_arg(a.input)`, which takes both forms.
- **Metrics of a float32 model** cannot match PyTorch float32 to 1e-12: set `metricTol: 1e-6` on the ProcessDef, or
  one bar per metric, `metricTol: { loss: 1e-6, picked: 'exact' }`.
- A source with `reference: false` gets no reference; one with `after:` gets its input after that source has run.

```python
w = ak.TapeWriter("walk", SCHEMA, seed=seed, opt={"cells": 7, "steps": 30}, phases=PHASES, provenance="live")
w.frame("move", {"pos": pos, "visits": visits}, step=t, changes=[{"field": "visits", "index": [pos], "from": old, "to": old + 1}],
        metrics={"t": t, "s": s})
w.write(out_stem)          # <stem>.tape.json + <stem>.tape.bin
```

A tracked field's first full value goes in without changes; after that every edit is a change (the writer refuses
unreported edits). A tracked field may start after frame 0 (a library of paths that grows): earlier keyframes leave it
out, as the TS writer does. `ak.json_bytes(obj)` and `ak.write_json(path, obj)` write an extra asset the way the kit
does (UTF-8, LF, no NaN).

---

## 8. The lesson

### 8.1 LessonDef and steps

```ts
const lesson: LessonDef = {
	title: { en, zh }, summary: { en, zh },
	steps: [ { autoAll: {} }, { auto: 'op:fc1', override: { title } }, { id: 'answer', ... } ],
	fixtures: [ { id: 'long', label: { en, zh }, raw: ... } ]       // extra inputs lint must survive
};
```

| Step field | Meaning |
|---|---|
| `id` | unique; deep link `#lesson-<id>` |
| `title`, `narration`, `look?` | bilingual text; narration is 2-4 sentences, ≤ 90 English words |
| `source` | a source name (default: the first) |
| `at` | the frame: `'first'`, `'last'`, an index, or `{ phase, step, sub, layer, last, where }` |
| `select` | the lesson's own selection: `() => ({ layer: 'fc2', unit: [0] })` |
| `view` | a view key (section 15), or `{ key, props: (rt, at) => ({...}) }` |
| `provenance` | `'live'`, `'precomputed'`, or `{ mode: 'illustrative', note }` (required: AK502) |
| `predict` | a question answered before the stage shows (`{ id, question, options, answer, explain }`) |
| `task` | `{ prompt, done(rt, at), hint?, waiting?, doneText? }`: the learner does something; `waiting` is the status line until `done` (default "Not yet: do what the task says."), `doneText` after it (default "Done.") |
| `kindOf` | `'limits'` or `'breakIt'`: the honest-limits step |
| `play` | `'view'` (the view animates itself), or `{ from, to, interval?, phases?, unit?, by? }`: a scrubber over a range of frames; `unit: { en: 'episode', zh: '回合' }` and `by: 'sub'` make its marks read "episode 12" from `frame.sub` (default "step k" from `frame.step`); on a scan, `unit: { en: 'patch', zh: '图块' }` makes it count "Patch 3 of 16" and mark "patch 2"; `unit` may be a template with `{n}` (`{ en: 'second', zh: '第 {n} 秒' }`). Until Play animates the range, the stage shows the step's own frame (its cursor, the one the narration describes); Play then runs the range from its start |
| `compare` | `{ a, b, label }`: two sources side by side. Both run to the step's cursor (`'last'` compares the ends of two runs of different length) and, while a range plays, the frames with the same phase and step. The view renders once per side (two instances): a control inside it (a toggle) keeps its state in `shared` (StageProps), one object per step, so both panels follow |
| `terms` | glossary ids: the site glossary's, or the package's own `manifest.glossary` ids (shown before promote adds them to the site glossary); lint checks them (AK411) |
| `focus`, `waitForDone`, `linger`, `duration` | highlights, timing |

**Predict and task steps with Play.** The narration, the look line and the title show before the learner answers: set
the question up there and put the answer in `predict.explain`, which shows after the answer (lint warns, AK412, when
one of them quotes the right option or repeats the explanation). While Play waits for the answer or the task, the
lesson says "Waiting for your answer…" ("Waiting for you to do the task…"), and presses and typing in the stage or in
the side panel (where a task's input often is) do not pause Play, also while a phone still shows the narration. After
the answer the line goes back to "Playing this step's animation…" while the step's own range plays, and the step stays
long enough to read what the answer revealed: about 55 ms per character of the explanation (a predict) or of the
narration as it reads once done (a task), 5 to 14 s; `linger` overrides it.

**While the learner reads.** The next step's live source (train, process, live-model, tape) starts computing as soon as
a step is entered, so its numbers are usually ready when the learner gets there. The screen-reader announcement of a
step waits (up to 10 s) until its narration has its numbers. The end card waits until the last step has its numbers.

**Where things are.** The stage sits left of the narration at desktop width and below it on a phone, so a narration
says "in the stage" or "in the table", never "above", "below" or "on the left". Lint cannot catch a wrong direction;
reviewers do.

### 8.2 Auto steps

The planner reads the spec and the sources and proposes steps; `lesson draft` lists their ids:

| Id | Step |
|---|---|
| `input` | the input as numbers (the adapter's layer-0 narration and view: the tokens for a text model) |
| `op:<layer>` | the first occurrence of an op (what + how + look), a distinct repeat |
| `group:<id>` | "the same block, N more times" for repeated groups (`L1`, `B2`, ...), naming the blocks as their layer labels do ("Block 2 to Block 4") |
| `output` | the answer and what it does not mean: a classifier's classes, or a language model's next token among its vocabulary (the token's text, never an id) |
| `train:data@<source>`, `train:loss@<source>`, `train:grad@<source>`, `train:update@<source>`, `train:watch@<source>` | training motifs (loss, grad and update need a fine source) |
| `proc:event.<phase>@<source>` | the first event of each narrated phase |
| `proc:run@<source>` | play the whole run |

`{ autoAll: { from, until, skip } }` takes a range of the plan; `{ auto: '<id>', override: {...} }` takes one step and
changes fields. Auto steps follow the op templates, so wording fixes in the kit reach every page; your own steps are
never rewritten. A lesson made only of unedited auto steps cannot be `reviewed` (AK409).

### 8.3 Narration templates

`{name}` or `{name|fmt}`. Names resolve in order: your `vars(rt, at)` result; then the built-in paths; then
`fact:<key>` (facts.json). No expressions: compute values in `vars()`. en and zh must use the same names (AK403).
An unresolved or non-finite value shows "…" and fails lint (AK401).

**A step the empty input skips** (its source did not run: no points, no text) leaves out every sentence of its
narration that has a missing value ("…", empty, NaN), ends with "Its numbers appear once there is an input." and hides
its look line. So put the step's idea in sentences without numbers, and the numbers in sentences of their own.

A `vars()` value may be a number, a string or an `{ en, zh }` text, printed as is in each language, so a sentence can
change with the numbers. `atlas/lesson/say.ts` has the helpers:

```ts
import { num, pick, plural } from '../../atlas/lesson/say.ts';
vars: (rt) => ({
	pts: plural(n, { one: '{n} point', many: '{n} points', zh: '{n} 个点' }),   // "1 point", "200 points"
	gap: num(loss - lossAfter, 'n2'),                                          // printed as {gap|n2} would print it
	verdict: pick(acc > 0.9, { en: 'it learned the shape', zh: '它学会了这个形状' }, { en: 'it did not', zh: '它没学会' })
})
```

A non-finite `n` or `v` comes back as NaN, so lint still reports it (AK401). For numbers you format yourself,
`atlas/core/format.ts` has `fmtNum` (a count as itself, "50", anything else with 3 significant digits), `fmtSetting`
(a setting as it was set, "0.05") and `paren` (a negative factor in parentheses, "0.500 × (−0.300)").

| Format | Output |
|---|---|
| (none) | an exact integer as itself ("9", "6,722"); anything else with 3 significant digits |
| `n3` | 3 significant digits |
| `n2`, `n4`, `n5`, `n6` | that many significant digits (`{loss\|n4}` -> "0.6991") |
| `int` | a rounded integer with separators |
| `pct` | a share as a percentage ("18.3%") |
| `signed` | with a sign ("+0.12") |
| `shape`, `label`, `tok` | text as is |

| Built-in path | Value |
|---|---|
| `layer.<id>.<stat>` | stat of a traced layer: `max`, `min`, `argmax`, `negShare`, `zeroShare`, `shape`, `strongestChannel` (the id may contain dots and `#sub`) |
| `top.label`, `top.p`, `runnerUp.label`, `runnerUp.p`, `margin` | the output's best two classes |
| `sel.layer`, `sel.ch`, `sel.cell`, `sel.value` | the lesson selection |
| `unit.<name>` | a value of the selected unit's explanation |
| `train.step`, `train.loss`, `train.<metric>` | the latest frame of the first train source |
| `frame.phase`, `frame.step`, `frame.<metric>`, `run.length` | the step's frame (or the latest process frame) |
| `opt.<name>` | an option of the step's source: its `opt`, else its process's (or trainer's) defaults; it resolves before the source has run |
| `fact:<key>` | a measured number from facts.json |

Every minus sign is the typographic minus U+2212, also in exponents ("−7.70e−4").

**Narrating a small change.** Three significant digits can hide a change: a loss that goes from 0.6991 to 0.6987 reads
"falls from 0.699 to 0.699". Narrate the change itself (`drop: loss - lossAfter` in `vars()`: "falls by 3.76e−4 to
0.699"), or give both numbers more digits (`{loss|n4}` and `{lossAfter|n4}`: "from 0.6991 to 0.6987"). Lint warns
(AK410) when two different numbers of one narration print the same and the narration does not state their difference.

**Counting from 0.** Views and the auto narration count indices from 0, as PyTorch does: channels, heads, rows, columns,
positions, queries, keys, features, nodes, units and training steps (the first update is `{frame.step}` = 0); they add
"(counting from 0)" where the learner reads an index. Counters count from 1: "Step 3 of 9", "Frame 2 of 40",
"Output 5 of 676" (a scan's place). The built-in paths are 0-based too (`sel.ch`, `sel.cell`, `layer.<id>.argmax`,
`layer.<id>.strongestChannel`): when your narration quotes one, say "channel {ch} (counting from 0)"; for an ordinal
("the 4th head"), the explain values `headNo`, `qNo`, `kNo`, `tNo`, `jNo` are 1-based.

### 8.4 Claims are tests

A factual sentence that is not computed live ("more steps do not help") needs an assertion in
`explainers/<slug>/claims.test.ts`, run over every fixture:

```ts
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
test('step "stuck": more steps do not help', async () => {
	const pkg = (await loadPackage('my-xor', SITE)).pkg;
	await withFixtures(pkg, async (rt) => { await rt.ensure('stuck', 'last'); assert.ok(...); }, { fetch: fsFetch, base: join(SITE, 'static') });
});
```

`atlas check` runs it (AK408 on failure). `withFixtures` records each distinct run once per test file and shares it
between fixtures and tests (the same process, input, options, seed and detail): a source that ignores the learner's
input, or a preset equal to the start input, is not trained again. A control you send to a tape gives that runtime a
private run first; pass `{ runner: localRunner }` (`atlas/runtime/local.ts`) for a fresh run every time.

**Node and the browser can differ in the last digits.** Lint, `lesson print`, the review tables and the claim tests run
in Node; the page runs in the browser. Plain arithmetic is identical, but `Math.exp`, `Math.tanh`, `Math.log` and the like
may differ in the last bit between engines, and a chaotic run (learning rate 3) amplifies that: the MLP playground's
loss after one step at lr 3 is 9.881493015551671 in Edge and 9.88149301555167 in Node. Narrate only what survives such a
change (3 significant digits of a stable run; the direction and size of a chaotic one, not its digits), and test claims
about a chaotic run with a tolerance or on what does not depend on the last digits.

### 8.5 Lint

`npm run atlas -- lesson <slug> lint` enters every step on every fixture (lesson fixtures, the adapter's edge fixtures,
presets) in en and zh, on a real runtime in Node. Errors: AK401 AK403 AK404 AK405 AK406 AK409 AK411 AK501 AK502;
warnings: AK402 (a typed number), AK407 (lesson shape: 5-12 steps, a predict, a task, a limits step, 2-4 sentences, at
most 90 words also once the placeholders are filled in; at `reviewed` a missing predict, task or limits step is an
error), AK410 (two different numbers that print the same; not for values that print at a saturation bound, ±1.00, 0%
or 100%, such as two saturated gates) and AK412 (a predict step's narration, look or title gives the answer away). AK401
is also an error when a `vars()` value prints "…" on a step that ran, and a warning when a `vars()` string prints "NaN",
"undefined" or "null". The page's start input is a lint fixture (`start`) unless another fixture has the same input.
Lint renders a skipped step as the page does (section 8.3). Lint cannot judge meaning: read `lesson <slug> print` (or
`contact/lesson-print.txt` after `e2e`) as a learner would.

Lint records each distinct run once and shares it between fixtures (the report counts runs recorded and shared). While
you iterate, `lesson <slug> lint --cap 200` caps live training at 200 steps, as e2e does; the narration then quotes
other numbers and a cursor past the cap is reported. It is never a gate: `atlas check` lints at full length.

---

## 9. Views, sections and other ways to customize

A **view** draws one step's stage from the runtime and the step's frame. Kit views cover most needs (section 15):
`layer.tensor` (any tensor), the `zoom.*` views (the arithmetic of one output number, chosen by the op's zoom kind),
`zoom.formula` (a phase formula), `tape.table`, `loss-curve`, `plane2d`, `train-step`, `kit.probbars`, ...

| Level | You write | Effect |
|---|---|---|
| L0 | Python only (`teach=`/`ui=` hints) | auto page and auto lesson |
| L1 | `explainer.ts` + `lesson.ts`, `claims.test.ts` | the normal path (required for `reviewed`) |
| L2 | `views/<key>/` (`npm run atlas -- new view <key> --local <slug>`) | your own stage, key `'<slug>/<key>'` |
| L3 | `sections/<Snippet>.svelte` | replaces one section of the page |
| L4 | `src/routes/m/<slug>/` | a fully custom page, as TinyCNN's |

A view folder holds `meta.ts` (pure TS: `export default { key, title, kinds, describe(props), demo(), minHeight } satisfies ViewMeta`)
and `View.svelte` (the drawing). `describe()` is the text for screen readers and must never be empty (AK504);
`demo()` gives illustrative props for the gallery (`/dev/atlas/`).

**A view that changes the input** (a maze editor, a "use these points" button) calls `setInput(raw)` from its props
(`StageProps.setInput`): the page re-runs every source and the lesson re-resolves its steps, exactly as after an edit in
the input widget. (`rt.setInput(raw)` does the same on the page.) Before K1 the stage dropped `setInput` on its way to
the view. **State shared by the panels of a compare** goes in `shared` (StageProps): one object per lesson step.

Notes on kit views:

- `loss-curve`'s `compare: ['other']` runs the other sources as far as its own; `log: true` starts at the smallest
  positive value (a loss of exactly 0 sits on the axis); `refs: [{ y, label }]` draws dashed reference lines (an
  equilibrium value, a baseline). Its readout prints steps as counts ("step 50"), and with several lines "lowest" names
  its line.
- `plane2d` reads the exact value at a data point from `pointValues` (a `[n]` tensor, e.g. a derived field) or
  `valueAt(x, y, state)`; without them its probe reads the grid between cell centres and says "≈".
- `zoom.gate`: `props.gate: 'f' | 'i' | 'g' | 'o' | 'c' | 'h'` puts one gate (or the cell or output box) in focus, so
  steps about different gates do not show the same stage.
- `zoom.weighted-sum`: `props.names: { x, w, p, i }` renames the product table's columns (when x and y mean something
  else in your lesson).
- `zoom.formula`: the numbers again under the formula only with `props.names: { key: { en, zh } }` (or `chips: true`).
- `zoom.normalize` draws an L2 normalization (explain values `x` and `len`) as the row and its length, LayerNorm (`mean`,
  `std`) as before.
- `curve1d`: `yLabel` is drawn along the y axis; `legend: true` gives a single curve a legend.
- `kit.probbars`: `chosen` (above, section 5.5).
- A long formula breaks after every second `+` of a long sum, so live numbers fit a phone.

**Before you write a view.** Look for a kit view that already draws it (section 15, the gallery at `/dev/atlas/`), then a
kit view with props (`layer.tensor`, `loss-curve`, `curve1d`, `plane2d`, `tape.table`, `compare`), then a local view
that composes kit parts: `ViewSlot` (a kit view inside yours), the parts in `atlas/ui/views/kit.parts/` (`CurvePlot`,
`RowBars`, `SumBars`, `TensorPanel`, `TermTable`, `Num`, `Notice`) and the components in `$lib/kit/` (`FormulaCard`,
`ProbBars`, `TensorGrid`; KIT.md section 7). Write new drawing code last.

**Charts and colors** (KIT.md section 5). Use the kit's two numeric scales and nothing else:
`makeScale('diverging', dom)` for signed values (blue below 0, gray at 0, orange above, a symmetric domain) and
`makeScale('sequential', dom)` (one purple ramp) for shares, probabilities and attention; classes get `categorical(k)`
(a colorblind-checked order) with the class name next to the color. Label both axes (with the unit), keep numbers to scale (a bar's length is its value; a log axis
says so), and keep one domain for panels that are compared. Text takes the `-text` tokens (section 13, AK604).

**Accessibility of a view** (AK604; KIT.md rules 9 and 16). A box that scrolls gets `use:keyScroll`
(`$lib/kit/keyscroll`); text is colored with the `-text` tokens and written on a colored cell with `scale.ink(v)` or
`inkOn(rgb)`; text is never faded with `opacity` (use `--muted` and a dashed border); an `<svg>` that holds buttons is
`role="group"`; ARIA table cells sit inside `role="row"`; a `<dl>` holds only `<dt>`/`<dd>` groups. While you iterate,
`npm run atlas -- e2e <slug> --axe warn` reports without failing; read `contact/axe.txt`.

**The auto page** (what you get without a section override):

- **Training.** A train source: the Train button, its loss curve at the last training snapshot (a run that samples after
  training is not shown at a sampling frame), and its numbers there. An exported model: the dataset, the steps, the
  parameters and `spec.training.metrics`. Metric names are readable in both languages (`test_acc` reads "Right on the
  test set"); name your own with `page: { metrics: { my_key: { label: { en, zh }, fmt: 'pct' } } }` (`pct`, `int`,
  `n3`, `n4`). `page.trainingCurve` sets what its curve draws: `{ metric, metrics, labels, log, smooth, refs }`.
- **Inside a layer** shows a tall `[T, D]` layer 32 rows at a time.
- **The arithmetic** of an op without a zoom view (`zoom: 'none'`) shows the op's formula with its explained values
  under the layer.
- **The facts** list every input of a multi-input model and, without a spec, the precision of JavaScript numbers (f64).

**The page's sections.** Without a spec (kinds B and C), the auto page leaves out the model sections (overview, inside a
layer, the arithmetic, output, code) unless a `sections/<Snippet>.svelte` override fills them. `manifest.page.sections`
titles, introduces or hides any section:

```ts
page: { ..., sections: { titles: { inspector: { en: 'The Q-table', zh: 'Q 表' } }, intros: { ... }, hide: ['code'] } }
```

---

## 10. Checks and gates

`npm run atlas -- check <slug>` runs every gate. It has three modes:

- `check <slug> --quick`, while you iterate: schema, parity, op tests, lesson, determinism, honesty, budgets, types,
  paperwork and ownership, with no build and no browser. Parity and the op tests come from
  `.atlas/<slug>/gate-cache.json` while their inputs are unchanged.
- `check <slug>`, once, when the package is done: the same gates without the cache, then site (a package-only build)
  and e2e (the default passes).
- `check <slug> --fast`: the quick gates without types and without the cache (every gate re-run, no build).

**The cache** of `--quick` counts as inputs the package's files except `lesson.ts`, `claims.test.ts` and its `.svelte`
and `.md` files (unless a package test imports them), its static assets, `pipeline/<slug>/` (without `runs/` and
`logs/`), `pipeline/atlas_kit`, `src/lib/atlas`, `scripts/atlas` and the configured Python. A lesson edit keeps the
cached results, so `--quick` takes seconds then; a lesson fixture added meanwhile reaches parity and the op tests in the
full check.

**Several packages.** `check a b` gives the packages one package-only build and one e2e sequence. `check --all
[<slug>...]` runs the fast gates of every package, the kit gates (honesty, the docs gate: DEVKIT.md's generated tables
against the code, `node scripts/atlas/docs.ts --check`, which `npm run check:atlas` runs too; every Node test of the
kit; both legacy parity scripts with `--no-write`), `npm run check`, one full build and one e2e sequence over the
packages given (`--e2e a,b` or `--e2e all`; `--passes all` unless you say otherwise). Use it once at integration.

`--only a,b` picks gates; `--json` prints the report (also in `.atlas/<slug>/check.json`); `--port n`,
`--keep-server`, `--no-keep-server` and `--stop-server` go to the dev server (below). A failed site step keeps its whole
output in `.atlas/build-<slug>.<step>.log`.

| Gate | Passes when |
|---|---|
| schema | the manifest is complete (no TODO: AK501; credits: AK704; catalog slug: AK701, against `concepts.ts` for a concept package), spec/reference/facts/tapes validate (AK301), shapes hold at the dim extremes (AK302) |
| parity | kind A ≤ the bar on every layer (AK201) and explain = trace (AK202); twins (AK203); references (AK204; "reference failed" when the reference did not run) |
| op tests | the package's own tests (local ops, views, a concept package's `concept.test.ts`) pass |
| lesson | lint is clean in en and zh; `claims.test.ts` passes (AK408) |
| determinism | every process and train source (at the source's `detail`): same seed twice = same frames; keyframes every frame = every 64 frames; random replays agree |
| honesty | every pure file loads in Node; the import rule and the `Math.random` ban hold (AK503) |
| budgets | file ≤ 5 MB, model folder ≤ 15 MB, tape ≤ 2 MB, auto-loaded bytes ≤ 5 MB (AK303) |
| types | svelte-check of the package's own files: 0 errors, 0 warnings (the whole site: `npm run check`, in `check --all`) |
| site | `check:catalog`, a package-only build into `.atlas/build-<slug>/`, `check:build` |
| e2e | `atlas e2e` (AK601 AK602 AK603 AK604) on the default passes |
| paperwork | `notes/<slug>.md` has the five sections (a warning for dev-only packages; a missing file is a warning, as in a fresh clone of the public repository, which has no `notes/`) |
| ownership | inside a lane: nothing outside the owned paths changed (AK702) |

**The package-only build.** `ATLAS_ONLY=<slug>[,<slug>...]` makes the build prerender the home page and the packages'
pages (`/m/<slug>/`, `/train/<slug>/`, `/dev/atlas/lab/<slug>/`, the ones that exist; `ATLAS_ONLY_PAGES=/a/,/b/` adds
pages). Its bundles equal a full build's, so `check:build` still checks for third-party URLs and every catalog module,
and the shell size of each page built (300 KB). `npm run atlas -- build <slug>[,<slug>...]` makes it on its own, after
the type check of the packages and `check:catalog` (`build --all`: `npm run check` and the full build). A build without
`ATLAS_ONLY` is unchanged.

**The browser test** (`atlas e2e`, section 12) runs the default passes: 1280 px in English (light) and 390 px in
Chinese (dark), whose horizontal-scroll check also probes 360 px after every step (AK602). Each pass writes a contact
sheet, and the run writes `lesson-print.txt` and the axe results to `notes/shots/<slug>/contact/`. **Accessibility
(AK604):** axe scans every step of every pass against WCAG 2.1 A and AA; serious and critical violations fail, moderate
and minor ones warn. `src/lib/atlas/ui/a11y.test.ts` (in `check:atlas`) guards the colors: `ink()` on both ramps and
every text token at 4.5:1 in both themes, so a new text token needs a line there.

**The dev server.** `check` and `e2e` reuse a running dev server of this site: one they kept (recorded in
`.atlas/dev-server.json`), or one on the package's port that answers the site's identity probe (a token in
`.svelte-kit/atlas-dev-probe.txt`; a server of another lane never matches). Else they start one and keep it: it stops
itself after 20 idle minutes, or now with `npm run atlas -- e2e --stop-server`; `--no-keep-server` stops it when the
command ends. A server they do not keep exits when the atlas process dies, so a killed command leaves nothing
listening. Builds may run beside it; on EBUSY or EPERM (a locked file) the site gate stops a server it kept and
retries once.

The report also counts the page's own lines of code (target ≤ 1,000): files under `explainers/<slug>/` except tests,
without blank lines, comment-only lines and `<style>` blocks (it also prints all non-blank lines). The budgets gate
counts as auto-loaded the `assets.auto` files, the presets file and the spec and weights of a trace source on another
model's folder (`{ kind: 'trace', assets: 'models/tinycnn' }`).

**Quality levels** (`quality` in the manifest): `draft` (gates green), `reviewed` (the teaching review signed off;
zh narration required, lint rules become errors), `flagship`.

---

## 11. Lanes, ports and merging

Parallel builders each work in a **lane**: a copy of the site at `site/.work/<name>/site`, with its own dev-server port.

```
npm run atlas -- worktree tinyvit --from-git HEAD  # the lane, from the last commit (owns the model's paths, or --owns a,b)
cd .work/tinyvit/site && npm run dev -- --port 5191
npm run atlas -- worktree tinyvit --sync           # take kit fixes from main (refused, AK703, if you edited a kit file)
npm run atlas -- merge tinyvit --dry-run           # what would be copied back; AK702 for any change outside your paths
npm run atlas -- merge tinyvit                     # (the main session) copy the owned files into main
```

- `--from-git <rev>` (HEAD, a branch, a commit) takes the lane's files from a committed revision: it reads
  `git archive` in memory (no tar program) and applies the same skips; `lane.json` records the `rev`. Use it whenever
  another workflow may be editing the main tree. Without it the lane copies the main tree's files as they are.
  `worktree <name> --sync --from-git <rev>` refreshes the non-owned files from a revision.
- The git-ignored `scripts/atlas/atlas.config.local.json` (the machine's Python) is copied from main into every lane;
  it is never synced, compared or merged.
- The copy skips `node_modules`, `.svelte-kit`, `build`, `.work`, `.atlas` and `package-lock.json`
  (no symlinks). Node finds the packages of the main `site/node_modules` upward.
- Its `vite.config.ts` gets `server.fs.allow: ['.', '../../..']` and loses main's `.work` pattern in
  `server.watch.ignored` (it matched every file of the lane, so the lane's dev server never reloaded; lane setup, never
  merged back). `worktree --sync` re-patches an older lane.
- `site/.work/<name>/lane.json` records the owned paths and a hash snapshot, so `merge` and `--sync` know what the lane
  changed and what main changed.
- A model lane owns `pipeline/<slug>/**`, `site/src/lib/explainers/<slug>/**`, `site/static/models/<slug>/**`,
  `notes/<slug>.md` and `notes/shots/<slug>/**`. Python writes `pipeline/<slug>/` in the main tree directly (it is
  yours alone); its assets go to the lane's `static/` (the CLI sets `ATLAS_SITE`).
- Lanes never edit the kit. A blocking kit bug is worked around inside the package (local op, local view, author step,
  section override) and written under "Kit issues" in your notes.
- The view table of this guide (section 15) lists kit views only, so merging a package with a local view needs no
  `docs.ts --write`.
- Builds go to `.atlas/build-<name>/` (`npm run atlas -- build <slug>`, or
  `MSYS_NO_PATHCONV=1 ATLAS_BUILD_DIR=.atlas/build-x npm run build` in Git Bash); a dev server may keep running beside
  them (section 10).

**Integrating lanes** (the main session). `merge <lane> --dry-run` for each lane; `merge` the lanes whose owned files
do not overlap, and copy an owned file that two lanes changed by hand. Then `promote` each package and apply the
printed catalog patch; run `npm run check`, `npm run check:atlas`, `npm run check:catalog` and
`node scripts/atlas/docs.ts --check`; `npm run atlas -- check <slugs> --quick`; and one
`npm run atlas -- check --all --e2e <slug>,<slug>` over the merged packages (one full build, the five `all` passes).
Commit one package per commit, then delete each merged lane and its builds.

**Ports.** A package or lane in the table runs on its row. Any other package gets a port from `atlas new`: the first
free one from 5210 up that no table row and no other package's notes use, recorded in `notes/<slug>.md`
(`npm run dev -- --port N`); `--port n` chooses it. `atlas check` and `atlas e2e` first reuse a dev server of this
site (section 10), else start one: on `--port n` (refused with AK003 when something else listens there), else on the
package's port when it is free, else on the first free port from 5210 up; the report names it.

<!-- atlas:generated ports -->
| Package or lane | Port | Dev server |
|---|---|---|
| `tinyvit` | 5191 | `npm run dev -- --port 5191` |
| `mlp-playground` | 5192 | `npm run dev -- --port 5192` |
| `gridworld` | 5193 | `npm run dev -- --port 5193` |
| `char-rnn` | 5194 | `npm run dev -- --port 5194` |
| `gcn` | 5195 | `npm run dev -- --port 5195` |
| `flow-2d` | 5196 | `npm run dev -- --port 5196` |
| `tinyclip` | 5197 | `npm run dev -- --port 5197` |
| `tinyforecaster` | 5198 | `npm run dev -- --port 5198` |
| `pinn` | 5199 | `npm run dev -- --port 5199` |
| `w1-a` | 5201 | `npm run dev -- --port 5201` |
| `w1-b1` | 5202 | `npm run dev -- --port 5202` |
| `w1-c` | 5203 | `npm run dev -- --port 5203` |
| `w1-e1` | 5204 | `npm run dev -- --port 5204` |
| `w1-b2` | 5211 | `npm run dev -- --port 5211` |
| `w1-d1` | 5212 | `npm run dev -- --port 5212` |
| `w1-d2` | 5213 | `npm run dev -- --port 5213` |
| `w1-e2` | 5214 | `npm run dev -- --port 5214` |
| `hello-xor` | 5214 | `npm run dev -- --port 5214` |
| anything else | 5210 and up: `atlas new` picks a free one and records it in `notes/<slug>.md` | `npm run dev -- --port <that port>` |
<!-- /atlas:generated -->

---

## 12. CLI reference

<!-- atlas:generated commands -->
| Command | Usage | What it does |
|---|---|---|
| `doctor` | `npm run atlas -- doctor [--node-only] [--json]` | check Node, the site packages, Python (+ torch, numpy, networkx, atlas_kit), playwright msedge, ports and git; print fixes |
| `new` | `npm run atlas -- new model <slug> --kind trace\|train\|process[,...] [--input <adapter>] [--template <name>] [--dev] [--port n] \| new model <slug> --like <package> [--dev] [--port n] \| new concept <slug> [--kind trace\|train\|process[,...]] [--input <adapter>] [--dev] [--port n] \| new concept <slug> --like <package> \| new worker <slug> \| new op <name> [--local <slug>] [--like <op>] \| new view <key> [--local <slug>]` | scaffold a model package (manifest, lesson, trainer/process, Python side, notes with its dev-server port), a concept package (type concept with concept.ts, routed at /train/<slug>/; --like turns a model package into a concept), a renamed copy of an existing package, a worker entry, an op (TS + test + Python mirror) or a view (meta + component); never overwrites |
| `py` | `npm run atlas -- py <script.py \| -m module \| -c code> [args...]` | run a pipeline script with the configured Python (the one export and parity use), PYTHONPATH=pipeline, ATLAS_SITE and ATLAS_ROOT set |
| `export` | `npm run atlas -- export <slug> [--no-parity] [--no-draft] [-- <export.py args>]` | run pipeline/<slug>/export.py with the configured Python, then parity --write and lesson draft |
| `parity` | `npm run atlas -- parity <slug> [--write] [--json] [--dir <folder under static/>] [--steps n] [--fixtures]` | kind A against reference.json, kind B against the twin (one run per trainer and options), kind C against the Python reference (--fixtures: on every fixture input; files in .atlas/<slug>/parity/); --write records spec.parity and explainers/<slug>/parity.json (the badge); legacy scripts with --no-write |
| `run` | `npm run atlas -- run <slug> <source> [--fixture id] [--frames a..b] [--seed n] [--opt '<json>'] [--detail coarse\|fine] [--save] [--precompute]` | print a process or trainer tape in Node, its input from a real runtime on the fixture (--opt overrides source options, e.g. '{"hidden":3}') |
| `lesson` | `npm run atlas -- lesson <slug> draft\|lint\|print [--lang en\|zh] [--fixture id] [--json] [--cap n]` | the auto plan, the lint over every fixture (identical runs shared between fixtures; --cap n caps training while you iterate, never a gate), or the narration tables |
| `check` | `npm run atlas -- check <slug> [<slug>...] [--quick\|--fast] [--passes <list>] [--json] [--only gate,gate] [--port n] [--keep-server\|--no-keep-server] [--stop-server] \| check --all [<slug>...] [--e2e <slugs\|all>] [--passes <list>] [--fast] [--json]` | every gate (schema, parity, op tests, lesson, determinism, honesty, budgets, types, site, e2e, paperwork, ownership) with a package-only build and the default e2e passes; --quick while iterating (no build or browser; parity and op tests from a cache while their inputs are unchanged); --fast without types, site and e2e; several slugs share one build and one e2e sequence; --all runs the fast gates of every package, the kit tests, docs --check, legacy parity, one full build and one e2e sequence over the packages given (--passes all) |
| `build` | `npm run atlas -- build <slug>[,<slug>...] [--no-types] \| build --all [--no-types]` | a package-only build (ATLAS_ONLY: the home page and the pages of the packages) into .atlas/build-<slug>/ after svelte-check of the packages and check:catalog, then check:build; --all: npm run check and the full build |
| `e2e` | `npm run atlas -- e2e <slug>[,<slug>...] [--passes default\|all\|full\|<width>-<lang>-<scheme>[-reduced],...] [--full] [--phone px] [--url <path>] [--port n] [--base <url>] [--keep-server\|--no-keep-server] [--stop-server] [--axe fail\|warn\|off] [--no-contact] [--wait ms] [--shots <dir>] [--cap n\|off]` | Playwright (Edge, headless) through the lesson; default passes 1280 px en light and 390 px zh dark (with a 360 px scroll probe), all or full for more; an axe scan of every step (AK604); contact sheets, lesson-print.txt and axe.txt in notes/shots/<slug>/contact/; reuses a dev server of this site or starts one and keeps it (it stops itself after 20 idle minutes); training capped at 200 steps (--cap) |
| `worktree` | `npm run atlas -- worktree <name> [--owns <glob,...>] [--from-git <rev>] \| worktree <name> --sync [--from-git <rev>]` | make a lane copy site/.work/<name>/site from the main tree or from a committed revision (hash snapshot; the local CLI config is copied, never merged), or refresh its non-owned files (AK703) |
| `merge` | `npm run atlas -- merge <name> [--dry-run]` | copy a lane's owned files back into the main tree; refuses (AK702) if the lane changed anything else |
| `review` | `npm run atlas -- review <slug>` | write notes/review-<slug>.md (checklist, check summary, screenshots, every narration on 3 inputs in en and zh) |
| `promote` | `npm run atlas -- promote <slug> [--dry-run]` | append credits to static/licenses.json, print the catalog patch (models.ts, tech-zh.ts, glossary.ts), run check:catalog |
<!-- /atlas:generated -->

Notes:

- **py**: `py <script.py> [args]` (also `py -m <module>`, `py -c "<code>"`) runs a pipeline script with the configured
  Python and the atlas environment (section 1).
- **new**: `new model <slug> --kind trace,train,process --input <adapter> [--template <name>] [--dev] [--port n]`, or
  `new model <slug> --like <package> [--dev] [--port n]` for a renamed copy of an existing package (section 4.1). The
  port (the table row, else the first free unclaimed port from 5210) goes into the notes and the "Next:" list. Templates:
  `drawing-classifier` (A, drawings, a CNN on Quick, Draw!), `char-seq` (A + C, a character LSTM on TinyStories and a
  sampling process), `points2d-train` (B, an MLP classifier on 2-D points with its twin), `series-train` (B + C, a
  forecaster on windows of a series), `graph-node` (B, a two-layer GCN on a graph with a local adapter), `process-blank`
  (C, a minimal process with its reference). `new worker <slug>` adds `worker.ts` to a package made before the engine
  existed. `new op` and `new view` are in sections 5.6 and 9.
- **run** prints a process or trainer tape as a table: `--frames a..b`, `--seed`, `--detail fine`, `--opt '{"lr":0.1}'`,
  `--save` (the tape and draws in `.atlas/<slug>/`), `--precompute` (a phone tape in `static/models/<slug>/tapes/`).
- **parity** for `tinycnn` and `tinygpt` (no package) runs their legacy scripts with `--no-write` and leaves the shipped
  `spec.json` untouched; `--write` is refused for them outside the main tree. For a package, `--write` also writes
  `explainers/<slug>/parity.json` (atlas-parity/1: every twin and reference by source), which the stage badge of train
  and process steps reads ("matches PyTorch fp64 to 7 significant digits over 50 steps", "same steps as an independent
  Python run"), also without a spec.json. `--fixtures` runs every process reference on every fixture input too
  (lesson fixtures, edge fixtures, presets), not only on the default input.
- **e2e** reuses a dev server of this site or starts one and keeps it (section 10; `--base http://localhost:<port>`
  uses yours as given, `--port n` picks the port, `--no-keep-server` stops it at the end, `--stop-server` alone stops
  the kept one), opens the Lesson Lab (dev packages), `/m/<slug>/` or `/train/<slug>/` (concept packages; `--url <path>`
  opens another page), and steps through the lesson once per pass.
  - **Passes** are named `<width>-<lang>-<scheme>[-reduced]`: width `desktop` (1280 px), `phone` (390), `narrow` (360)
    or a number of px; `-reduced` is one pass over the step list with reduced motion. Presets: `default` (desktop-en-light
    and phone-zh-dark; the 390 px pass also probes horizontal scroll at 360 px after every step, so AK602 keeps its gate
    width), `all` (+ narrow-en-dark, desktop-zh-light and desktop-en-light-reduced; use it at integration) and `full`
    (+ desktop-en-dark, narrow-zh-light and phone-en-light; also `--full`). `--passes` takes names and presets in a list;
    `--phone px` adds `<px>-en-light`. Several slugs (`e2e a,b`) run as one sequence on one server.
  - **Axe** scans every step of every pass (AK604): `--axe warn` reports without failing while you iterate, `--axe off`
    skips it; `check` always fails on it.
  - **Output** in `notes/shots/<slug>/` (or `--shots <dir>`), replaced on each run: `e2e-<pass>-<nn>.png`, and in
    `contact/` a sheet per pass with every step labelled (`<pass>.png`; `--no-contact` skips them), `lesson-print.txt`
    (title, narration and numbers per step and language), `axe.txt` and `axe-<pass>.json`. `atlas review` lists these
    first.
  - **Timing.** Dev-server timings are pessimistic; long tasks are measured while stepping, not while loading, only once
    the page's web fonts have loaded, and not during an axe scan or the 360 px probe (e2e starts Edge with
    `--expose-gc` and collects axe's garbage inside the scan's own window). A step may say "Computing…" for up to
    60 s. Screenshots are taken with the site header unstuck. `--cap n` caps live training at n steps instead of 200,
    `--cap off` not at all.
- **build** `build <slug>[,<slug>...]` makes the package-only build of section 10 (`--no-types` skips the type check;
  `build --all` the full build).
- **worktree** `--from-git <rev>` makes the lane from a commit (section 11).
- **lesson** `lint` and `print` share identical runs between fixtures; `--cap n` caps training while you iterate (not a
  gate).
- **run** gets the source's input from a real runtime on the fixture (spec, model, presets, the adapter), so a process
  whose `input(rt)` reads `rt.model` records what the page records; `--fixture` takes lesson fixture ids, `edge:<id>`
  and `preset:<id>`.
- **review** writes `notes/review-<slug>.md`; **promote** (the main session) appends credits to
  `static/licenses.json` (a source whose url is already listed is not added twice) and prints the catalog patch, the
  entry's current `hook` and `shows` to check against the page, a tech-zh row that already says nearly the same (reuse
  its wording), and what the new catalog text costs: glossary terms reach every page that uses `Term` (the flagships
  too, with their 300 KB budget); models and tech-zh text reach the list pages and the `/m/[slug]` route only (page
  data, since the prep for wave 2b). For a concept package, promote prints the `concepts.ts` patch instead and never
  touches `models.ts` (section 18).
- **new concept** scaffolds a concept package (section 18).
- The reference sections of this file are generated: `node scripts/atlas/docs.ts --write` (and `--check`, which
  `npm run check:atlas` and `check --all` run: a kit op, kit view, command, code or port added without regenerating
  fails). Package views are not listed.

---

## 13. Error codes

Every command prints problems the same way:

```
AK201  parity: layer "B1.attn.weights" max |diff| 3.2e-02 > 1e-4
  where   case "house#2", flat index 41 of [4×9×9]: js 0.412, torch 0.380
  likely  PyTorch applied a mask the spec does not know
  fix     set causal at pipeline/tinyvit/export.py:41, or teach the op
  docs    site/DEVKIT.md#ak201
```

<!-- atlas:generated codes -->
| Code | Meaning |
|---|---|
| [AK001](#ak001) | Node or npm |
| [AK002](#ak002) | Python or Python packages |
| [AK003](#ak003) | port busy |
| [AK004](#ak004) | git refuses the repo |
| [AK101](#ak101) | graph differs between cases |
| [AK102](#ak102) | CUDA or TF32 during export |
| [AK103](#ak103) | describe() differs from module(x) |
| [AK104](#ak104) | recurrent gate reference differs from the module |
| [AK105](#ak105) | budget at export |
| [AK106](#ak106) | op attributes differ from ops.json |
| [AK201](#ak201) | layer over the parity bar (kind A) |
| [AK202](#ak202) | explain() differs from the trace |
| [AK203](#ak203) | twin mismatch (kind B) |
| [AK204](#ak204) | reference mismatch (kind C, first divergence) |
| [AK205](#ak205) | near-tie divergence (warning) |
| [AK301](#ak301) | schema |
| [AK302](#ak302) | inferShape differs from outShape at a dim extreme |
| [AK303](#ak303) | size budget |
| [AK401](#ak401) | placeholder unresolved or non-finite |
| [AK402](#ak402) | literal multi-digit number in narration (warning) |
| [AK403](#ak403) | en/zh placeholder sets differ |
| [AK404](#ak404) | cursor not found on a fixture |
| [AK405](#ak405) | view not registered |
| [AK406](#ak406) | missing zh |
| [AK407](#ak407) | lesson shape |
| [AK408](#ak408) | claim test failed |
| [AK409](#ak409) | reviewed lesson made only of auto steps |
| [AK410](#ak410) | two different numbers print the same (warning) |
| [AK411](#ak411) | step term in no glossary |
| [AK412](#ak412) | predict narration gives the answer away (warning) |
| [AK501](#ak501) | TODO markers in the manifest |
| [AK502](#ak502) | step without provenance |
| [AK503](#ak503) | dependency rule or Math.random |
| [AK504](#ak504) | empty describe() |
| [AK601](#ak601) | console error |
| [AK602](#ak602) | horizontal scroll at 360 px |
| [AK603](#ak603) | long task |
| [AK604](#ak604) | accessibility violation (axe: serious or critical) |
| [AK701](#ak701) | slug not in catalog |
| [AK702](#ak702) | change outside owned paths |
| [AK703](#ak703) | sync refused |
| [AK704](#ak704) | credits missing |
<!-- /atlas:generated -->

### AK001

Node or npm is missing or too old, or a site package is not installed. Install Node 24 and run `npm install` in `site/`.

### AK002

Python is not configured, too old (< 3.10), or lacks torch / numpy / networkx (or a data file is missing). Run
`npm run atlas -- doctor` for the exact fix; set `"python"` in `site/scripts/atlas/atlas.config.local.json`.

### AK003

A dev-server port is busy: `atlas e2e --port n` (or `check --port n`) names a port something already listens on, or the
dev server did not answer. Stop the other server, pick another `--port`, or
let e2e choose (omit `--port`).

### AK004

Git refuses the repo (exFAT "dubious ownership").

### AK101

The graph differs between cases: `describe()` recorded other layers, inputs, params or weights for a later case, or an
axis changed size without a declared symbolic dim. Record the same layers for every case (no branching on the input);
declare varying lengths with `shape=['T', ...]` and `dims=`.

### AK102

A CUDA tensor or module reached export, or TF32 was switched back on. Call `model.cpu()` before export; train on the GPU
if you like, but references are CPU float32.

### AK103

`describe()` differs from `module(x)`. A step of `forward()` is missing, reordered or different (an activation, a
reshape, an eps). Compare the layers one by one; `module_input` must give the module the same input (with a batch dim).

### AK104

The LSTM gate reference differs from the module (more than 1e-5), or the module is not a plain one-layer,
unidirectional `nn.LSTM` with `batch_first=False`. Call `model.eval()`; describe custom cells as a local op.

### AK105

A file would exceed 5 MB or the model folder 15 MB. Use fewer or shorter reference cases, fewer presets, or a smaller
model.

### AK106

A builder call gave attributes the op does not declare, missed a required one, or gave the wrong type (checked against
`schema/ops.json`, section 14). Remove the attribute, or add it to the TS op and regenerate ops.json.

### AK201

A traced layer differs from PyTorch by more than the bar (1e-4), or the kit trace lacks an id the reference has. The
message names the layer, the case and both values. Usually an attribute is missing in the spec (a mask, a scale, an
eps, padding) or the layout differs; fix `describe()` or teach the op. Also a warning when the cases miss a symbolic
dim's min or max.

### AK202

An op's `explain()` (the arithmetic behind one output number) differs from the trace. `explain()` must recompute the
unit with the same arithmetic as `forward()`.

### AK203

The Python twin does not match the TS trainer: losses, step-1 gradients or final params differ, the state_dict names
differ, or the twin failed. Make `twin.py` mirror `trainer.ts` exactly (architecture, activation, loss reduction,
optimizer settings, label dtype).

### AK204

The Python reference tape differs from the TS tape; the message names the first frame and field. Usually a different
choice at a tie (break ties to the smallest index), a different draw (replay the recorded draws), or a frame emitted in
another order. Step both with `npm run atlas -- run <slug> <source> --frames a..b`.

### AK205

A near tie: the reference's choice differed only because two candidates were within 1e-12; it adopts the TS choice
(warning).

### AK301

A schema problem: a malformed spec, reference, facts or tape file, a missing file (export first), a manifest or a
loader problem, or a bad command line. The message says which file and key.

### AK302

`inferShape` disagrees with a layer's `outShape` at a symbolic dim's min or max (for example T = 65 overflows a
64-row position table). Fix the dims or the spec.

### AK303

A size budget: a file over 5 MB, a model folder over 15 MB, a phone tape over 2 MB, or auto-loaded assets over 5 MB.

### AK401

A narration placeholder is unresolved or not a finite number on some fixture (the learner would read "…"), or a step
function threw. Bind it in `vars()`, put the number in the frame's metrics, or guard the empty input.

### AK402

A number typed into narration text (warning). Use a placeholder bound to the live value or `{fact:<key>}`; quoted
numbers drift from the screen. Single digits are fine.

### AK403

The en and zh texts of a narration use different placeholder names.

### AK404

A step's cursor matches no frame on some fixture, or the step cannot be entered (its source failed or has no frames).
Use a cursor query (`{ phase, step, last }`) instead of a fixed index; check the source runs on that input.

### AK405

A step uses a view key that is not registered. See section 15 (or add a local view).

### AK406

Chinese is missing: titles, labels, goals and predict texts need zh (error); narration needs zh at `reviewed` (a
warning in drafts).

### AK407

Lesson shape: 5-12 steps, 2-4 sentences per narration and at most 90 English words (counted in the text and again once
the placeholders are filled in, on every fixture), at least one predict, one task and one limits step, unique ids, and
`auto`/`autoAll` ids that exist in the plan. A missing predict, task or limits step is an error at `reviewed` and a
warning in drafts; the step count, words and sentences stay warnings (a two-part page, such as bandits + gridworld with
17 steps, may accept the warning); a duplicate id or an unknown auto id is always an error.

### AK408

A claim test failed (`claims.test.ts`), or a `reviewed` lesson has none.

### AK409

A `reviewed` lesson made only of unedited auto steps.

### AK410

Two different numbers of one narration print the same text (warning), as "the loss falls from 0.699 to 0.699" when it
went from 0.6991 to 0.6987, and no placeholder of the narration holds their difference. Narrate the change itself (a
`vars()` entry such as `drop: before - after`, which silences the warning), or give both numbers more digits with
`{name|n4}` (section 8.3). Values that print at a saturation bound (±1.00, 0%, 100%, "<0.1%") are left alone: two
saturated gates that both print 1.00 are two quantities, not a hidden change.

### AK411

A step's `terms` names an id that is in neither the site glossary (`src/lib/catalog/glossary.ts`) nor the package's
`manifest.glossary`. Use a glossary id, or add `{ id, term, def }` to `manifest.glossary` (the step shows it before
`promote` proposes it for the site glossary).

### AK412

A predict step's narration, look line or title gives the answer away (warning): they show before the learner answers,
and one of them quotes the right option or repeats a run of six words of the explanation. Set the question up there and
put the answer in `predict.explain`.

### AK501

TODO markers remain in the manifest or the narration, or `notes/<slug>.md` misses one of its five sections. A missing
`notes/<slug>.md` is only a warning: the public repository has no `notes/` folder, and a package's notes are written
while it is built.

### AK502

A step has no provenance (its stage would show no badge), or `illustrative` without a note; in the browser test, a
stage without a badge.

### AK503

The dependency rule or the `Math.random` ban: a pure file imports something other than `src/lib/atlas/**` (outside
`ui/`), `src/lib/i18n/types.ts` or its own package, uses a `$lib`/`$app` alias, a `.svelte` file, an npm package or
`node:*`, or does not load in Node (non-erasable TypeScript). Tests (`*.test.ts`, `tests/atlas/**`, `fixtures/`) are
exempt.

### AK504

A view's `describe()` is empty (screen readers would get nothing).

### AK601

The browser test saw a console error, a page error, or a missing page or lesson.

### AK602

Horizontal page scroll at 360 px. Use `grid-template-columns: minmax(0, 1fr)` and let wide views scroll in their own box.

### AK603

A main-thread long task while stepping through the lesson: over 100 ms fails, over 50 ms warns. Move work to the worker
(leave `run` unset, or `run: 'worker'`) or yield every ≤ 8 ms. The bar is load-sensitive: the same page shows 2-3×
longer tasks while other processes keep the CPU busy, so re-run e2e on a quiet machine before you chase a 50-100 ms
warning. A first formula used to cost a font-swap layout; the page now fetches the KaTeX fonts while idle, and e2e
measures after they have loaded.

### AK604

The axe accessibility scan (WCAG 2.1 A and AA, every lesson step of every e2e pass) found a serious or critical
violation; moderate and minor ones warn. `notes/shots/<slug>/contact/axe.txt` lists each rule with its steps, nodes and
help link (`axe-<pass>.json` has the details). While you iterate, `atlas e2e <slug> --axe warn` reports without
failing; `atlas check` always fails on it. The kit has a fix for each common finding (KIT.md rules 9 and 16):

- **A scroll box the keyboard cannot reach** (scrollable-region-focusable): `use:keyScroll` from `$lib/kit/keyscroll`,
  not a fixed `tabindex`. It adds a tab stop only while the box overflows and holds nothing focusable. CodeBlock, Tex,
  FormulaCard, TensorGrid and the kit views already use it.
- **Text below 4.5:1** (color-contrast): color text with the `-text` tokens (`--div-neg-text`, `--div-pos-text`,
  `--q-text`, `--k-text`, `--v-text`, `--real-text`, `--generated-text`); the plain tokens are for fills and lines.
  Text on a colored cell takes `scale.ink(v)` or `inkOn(rgb)` (`$lib/kit/color`; 4.5:1 or more). Never fade text with
  `opacity`: use `--muted` and a dashed border. Native buttons need the site's colors (the browser's dark-theme gray
  fails).
- **An `<svg>` that holds buttons** (nested-interactive): `role="group"`, not `role="img"`.
- **An ARIA table** (aria-required-parent, aria-required-children): its cells and headers sit inside `role="row"`; a
  CSS subgrid keeps the grid layout.
- **A definition list** (definition-list): a `<dl>` holds only `<dt>`/`<dd>` groups; a label goes outside it.

### AK701

The slug is not in the catalog (use `--dev` for experiments; new catalog entries are proposed first), or a dev-only
package was promoted.

### AK702

A change outside the lane's owned paths (merge refused), an owned file changed both in the lane and in main, or a
write the lane may not do (legacy parity `--write`).

### AK703

`worktree --sync` refused: the lane changed a file it does not own. Undo that edit (copy main's version back), then sync.

### AK704

Credits are missing or incomplete: every dataset, ported code or idea-with-code needs a `credits` row (name, url,
license, usedFor, attribution).

---

## 14. Op reference

The 20 core ops (`src/lib/atlas/ops/`, `CORE_OPS`) and their Python builder mirrors. The zoom view is chosen by the
kind of arithmetic.

<!-- atlas:generated ops -->
20 ops. Python: `b.<op>(id, ...)` builder calls (`pipeline/atlas_kit/spec.py`); attributes are checked against this table (AK106). Parity bar for every op: 0.0001.

| Op | Label (en / zh) | Category | Zoom view | Attributes | Weights (`?` optional) | Sub-steps | The auto step starts with |
|---|---|---|---|---|---|---|---|
| `input` | Input<br>输入 | io | `zoom.none` | `name`: string | - | - | This is where your input enters the model, as a tensor of shape {outShape}. |
| `conv2d` | Convolution<br>卷积 | conv | `zoom.weighted-sum` | `kernel`: int (required)<br>`stride`: int = 1<br>`padding`: int = 0<br>`inChannels`: int<br>`outChannels`: int<br>`accumulate`: string = "bias-last" | `weight`, `bias`? | - | A kernel is {kernelIs}. |
| `linear`<br>aliases `dense`, `linear_tied` | Linear (fully connected)<br>线性层（全连接） | linear | `zoom.weighted-sum` | `accumulate`: string = "bias-last"<br>`inFeatures`: int<br>`outFeatures`: int<br>`nHead`: int<br>`dHead`: int | `weight`, `bias`? | - | Each of the {outFeatures\|int} outputs is a weighted sum of all {inFeatures\|int} inputs: every input is multiplied by its own learned weight, and the products are added up{biasClause}. |
| `relu` | ReLU<br>ReLU | activation | `zoom.elementwise` | - | - | - | ReLU keeps positive numbers and replaces negative numbers with 0. |
| `tanh` | Tanh<br>Tanh | activation | `zoom.elementwise` | - | - | - | Tanh squashes every number into the range from −1 to 1. |
| `sigmoid` | Sigmoid<br>Sigmoid | activation | `zoom.elementwise` | - | - | - | The sigmoid squashes every number into the range from 0 to 1, so it can be read as a share or as how far a switch is open. |
| `silu` | SiLU<br>SiLU | activation | `zoom.elementwise` | - | - | - | SiLU multiplies each number by its own sigmoid, x · σ(x). |
| `gelu`<br>aliases `gelu_tanh` (approx="tanh") | GELU<br>GELU | activation | `zoom.elementwise` | `approx`: string = "none" | - | - | GELU multiplies each number by a share between 0 and 1 that grows with the number: big positive numbers pass almost unchanged, big negative numbers become almost 0, and numbers near 0 are partly let through. |
| `maxpool2d` | Max pooling<br>最大池化 | pool | `zoom.reduce` | `kernel`: int = 2<br>`stride`: int | - | - | Max pooling looks at each {kernel\|int}×{kernel\|int} block of a map and keeps only the largest of its {windowCount\|int} numbers. |
| `flatten` | Flatten<br>展平 | shape | `zoom.index-map` | - | - | - | Flatten lays the {inShape} grid out as one long list of {count\|int} numbers, channel by channel and row by row. |
| `concat` | Concatenate<br>拼接 | merge | `zoom.index-map` | `axis`: int = 0 | - | - | Concatenation joins {parts\|int} tensors end to end along one axis. |
| `add` | Add<br>相加 | merge | `zoom.elementwise` | - | - | - | Add sums two tensors cell by cell: each output number is a + b, with a taken from “{aName}” and b from “{bName}”. |
| `embedding` | Embedding<br>嵌入 | embedding | `zoom.lookup` | - | `weight` | - | An embedding is a table with one row of {dim\|int} learned numbers for each of the {vocab\|int} possible ids. |
| `position_embedding` | Position embedding<br>位置嵌入 | embedding | `zoom.lookup` | - | `weight` | - | A position embedding is a table with one learned row of {dim\|int} numbers for each position, {positions\|int} rows in all, so the model can read at most {positions\|int} tokens at once. |
| `layernorm` | Layer normalization<br>层归一化 | norm | `zoom.normalize` | `eps`: float = 0.00001 | `weight`?, `bias`? | - | Layer normalization rescales each row of {features\|int} numbers so that its mean is 0 and its spread (standard deviation) is 1. |
| `attention_scores` | Attention scores<br>注意力分数 | attention | `zoom.weighted-sum` | `nHead`: int (required)<br>`dHead`: int (required)<br>`scale`: float<br>`causal`: bool = false | - | - | Attention scores measure how well each query matches each key. |
| `causal_mask` | Causal mask<br>因果掩码 | attention | `zoom.index-map` | `fill`: string = "-Infinity" | - | - | The causal mask hides the future: for query i, every key after position i gets the score −∞, so after the softmax its weight is exactly 0. |
| `softmax` | Softmax<br>Softmax | output | `zoom.softmax` | `axis`: int = -1 | - | - | Softmax turns a row of scores into shares between 0 and 1 that add up to 1: it takes e to the power of each score, then divides each result by their sum. |
| `attention_mix` | Weighted sum of values<br>值的加权和 | attention | `zoom.weighted-sum` | `nHead`: int (required)<br>`dHead`: int (required) | - | - | Each position now gathers information from the positions it attends to. |
| `lstm` | LSTM<br>LSTM（长短期记忆） | recurrent | `zoom.gate` | `hidden`: int (required)<br>`inputSize`: int | `weight_ih`, `weight_hh`, `bias_ih`?, `bias_hh`? | i f g o c h | An LSTM reads a sequence one step at a time and keeps a memory, the cell state c, of {hidden\|int} numbers. |
<!-- /atlas:generated -->

---

## 15. View reference

<!-- atlas:generated views -->
21 kit views. Use the key as a step's `view`; open `/dev/atlas/` for the gallery with demo data. A package's own views
(`explainers/<slug>/views/<key>/`, key `'<slug>/<key>'`) are in the gallery too; they are not listed here.

| Key | Title (en / zh) | Kinds | Min height (px) |
|---|---|---|---|
| `compare` | Two runs side by side<br>两次运行并排对比 | trace, train, process | 300 |
| `curve1d` | Curve over time<br>随时间变化的曲线 | train, process | 240 |
| `kit.attention` | Attention weights, one head at a time<br>注意力权重，一次一个头 | trace | 320 |
| `kit.probbars` | Output as probability bars<br>用概率条显示输出 | trace | 260 |
| `kit.tokens` | The text as tokens<br>文本切成的词元 | trace | 90 |
| `layer.tensor` | A whole layer, drawn by its axes<br>按轴画出整个层 | trace, train, process | 320 |
| `loss-curve` | Loss curve<br>损失曲线 | train, process | 260 |
| `network` | The network<br>网络 | train | 280 |
| `plane2d` | Points on a plane<br>平面上的点 | train, process | 320 |
| `tape.table` | Every frame of a run, as a table<br>把一次运行的每一帧列成表 | train, process | 360 |
| `train-controls` | Training controls<br>训练控制 | train | 150 |
| `train-step` | One training step<br>一步训练 | train | 300 |
| `zoom.elementwise` | One number in, one number out<br>一个数进，一个数出 | trace | 420 |
| `zoom.formula` | A formula with live numbers<br>代入实时数值的公式 | trace, train, process | 200 |
| `zoom.gate` | Gates decide what to keep<br>门决定保留什么 | trace | 520 |
| `zoom.index-map` | Numbers move, or get masked<br>数值换位置，或被遮住 | trace | 380 |
| `zoom.lookup` | Look up a row of a table<br>在表中查出一行 | trace | 420 |
| `zoom.normalize` | Normalize a row<br>把一行归一化 | trace | 480 |
| `zoom.reduce` | A window shrinks to one number<br>一个窗口缩成一个数 | trace | 460 |
| `zoom.softmax` | Scores become shares<br>分数变成份额 | trace | 460 |
| `zoom.weighted-sum` | Weighted sum, worked out<br>加权和的计算过程 | trace | 520 |
<!-- /atlas:generated -->

---

## 16. File formats

- **spec.json (spec 1.1)**: `{ slug, name, version, params, precision, tensors: [{name, shape, offset, length}],
  layers: [{id, op, label, in, outShape, params?, w?, axes?, sub?, code?, ui?, teach?, desc?}], inputs, outputs, dims,
  parity, atlas: {spec: '1.1', kinds, exporter}, files, ...extras }`. `offset`/`length` count float32 values in
  weights.bin. A layer's weight `slot` resolves to `w[slot]`, else `<id>.<slot>`. Sub-step ids are `<id>#<sub>`.
- **reference.json (atlas-ref-1)**: `{ format: 'atlas-ref-1', model, version, against: 'pytorch-fp32-cpu', cases:
  [{id, label, edge, inputs: {name: {shape, dtype, b64}}, order, tensors: {id: {shape, dtype, b64}}}] }`, b64 =
  little-endian values. Pages never load it.
- **facts.json (atlas-facts/1)**: `{ format, facts: { key: { value, unit?, source, measured } } }`.
- **atlas-tape/1**: `<stem>.tape.json` (header: format, process, version, seed, opt, provenance, schema, phases,
  keyEvery, `bin: {file, bytes}`, frames) + `<stem>.tape.bin` (tensors, little-endian, each at a multiple of 8 bytes).
  In frames a tensor is `{"$t": "f32" | "f64" | "i32", "shape", "o": byteOffset, "n": count}` and a non-finite number
  `{"$f": "NaN" | "Infinity" | "-Infinity"}`. Keyframes (every `keyEvery`-th frame, `key: true`) carry the full value of
  every `track: 'changes'` field; other frames carry their edits in `changes`. The TS writer (`process/io.ts`) and the
  Python `ak.TapeWriter` write identical bytes. File names use `fileSafe(id)` (`train:fit` -> `train-fit`).
- **train_fixture (atlas-train-fixture/1)**: the TS init params and per-step batches a twin trains on.

---

## 17. Troubleshooting and machine notes

- **"does not load in Node"** (AK503): an import Vite understands but Node does not (`$lib/...`, a missing `.ts`
  extension, a `.svelte` file), or an `enum`. Pure files use relative `.ts` imports only.
- **A placeholder shows "…"**: the value is missing at that step's frame. Process steps run only up to their cursor:
  put the numbers the narration needs into that frame's `metrics`.
- **Two train sources start from different weights**: give them the same `opt.seed`.
- **`run()` does not type-check** in a process: declare `type Run = ProcessRun<S, O>` (the same as
  `Generator<Frame<S>, void, Control<O> | undefined>`) and annotate `*run(input, ctx): Run`.
- **Python on PATH is too old**: point `atlas.config.local.json` at a Python ≥ 3.10 with torch ≥ 2.1, and run your own
  scripts with `npm run atlas -- py <script>` (section 1).
- **A lane's dev server does not see my edits**: the lane was made before K0; run `npm run atlas -- worktree <name>
  --sync` once (it re-patches the lane's `vite.config.ts`) and restart the dev server.
- **Step 2 still shows the old weights after I changed the input**: fixed in K0 (a new input restarts every train
  source); a lesson that worked around it with a process source keeps working.
- **A `live-model` step shows other numbers in the e2e screenshots than in your browser**: e2e caps training at 200 steps
  (`?atlas-e2e=1`), and a live-model step reads the weights as far as its train source has run (section 6.2).
- **A step works when you step into it but not from its deep link**: the step needs something an earlier step did.
  Give it its own cursor and `select`, or put the numbers it needs into its own frame; `stepAt` must work as the first
  call on a fresh runtime (`src/lib/atlas/integration/deeplink.test.ts` checks the kit's examples).
- **No downloads**: the kit never downloads data. Quick, Draw! (`ak.data.quickdraw`), TinyStories
  (`ak.data.tinystories_chars`) and the karate club (`ak.data.karate`, an unweighted adjacency; in TS
  `atlas/data/karate.ts`, the same graph) are local. A new dataset needs the user's OK first.
- **"cannot be sent to a worker"** (AK404 in lint): the source runs in the package worker and its `input(rt)` holds
  something a worker cannot receive (`rt.model`, a function, a class instance). Set `run: 'main'` on the source, or pass
  plain data (ids, typed arrays of weights).
- **A process step behaves differently since K1**: `run: 'main'` used to be ignored in the browser (every process ran in
  the worker); it now keeps the source on the main thread. If e2e reports a long task on that step, remove `run: 'main'`
  (its input must then be plain data).
- **"Training the network first…" on a process step**: the source has `after: '<train source>'` and waits for that
  source to finish (a deep link runs it first).
- **A train step shows the empty note**: the input is empty and train sources do not run on an empty input (K1); give
  the source `runOnEmpty: true` if its trainer does not learn from the learner's input.
- **reference.py fails to parse `--input`**: the input was long and arrived as `--input @<file>`; read it with
  `ak.json_arg(a.input)`.
- **A dev server is still running after e2e**: it is the kept one (section 10). It stops itself after 20 idle minutes;
  `npm run atlas -- e2e --stop-server` stops it now.
- **`--quick` passed, the full check fails parity or an op test**: `--quick` reused cached results, and a lesson
  fixture added since then (or a file outside the cache's inputs) changed the answer. Run `check <slug> --fast` to re-run
  every gate without the build.
- **AK604 on a kit component** (a badge, a code block, a formula): the kit fixed the known ones; a lane made from an older
  commit needs `worktree <name> --sync` (or a new lane `--from-git HEAD`).

---

## 18. Concept packages

A concept page (`/train/<slug>/`: loss and gradient descent, backpropagation, LoRA, quantization, ...) teaches one
idea rather than one model. It is built like a model package and passes the same gates; only the page around the
lesson differs. The kit's example is `loss-gradient-descent`: a 9-step lesson on one neuron with two dials, a toy, a
real network trained live and three learning rates side by side.

### 18.1 What differs from a model package

- The folder is the usual `src/lib/explainers/<slug>/`, for a slug of `src/lib/catalog/concepts.ts`. Model and concept
  slugs must not overlap (`logit-lens` is in both catalogs: rename one side before either gets a package).
- `explainer.ts` says `type: 'concept'`. Until a types revision admits it, the template spells the manifest's type
  `Omit<ExplainerManifest, 'type'> & { type: 'concept' }`.
- `lesson.ts` is the primary path, as for a model.
- `concept.ts` holds the page around the lesson (`ConceptDef`, `atlas/ui/concept/panels.ts`): `hook` (a text, or a
  step), `toy`, `real` and `compare` (steps in the lesson's format: source, cursor, view, narration with placeholders,
  provenance), `pitfalls` (`[{ claim, truth, step? }]`) and `further` (`[{ label, href }]`).
- Every loader, source kind (also a trace on a live model's folder: `{ kind: 'trace', assets: 'models/tinycnn' }`),
  view, lint, parity check, worker and lane rule works unchanged.

### 18.2 The page

The concept page shows, in order: the goals and the predict-first question; the hook; the lesson (its side panel holds
the input widget); "Explore freely"; the toy (the input widget above the toy's stage); the real model; the comparison;
the pitfalls (`manifest.page.misconception` first; a pitfall's `step` links `#lesson-<id>`); the checkpoint (back to the
prediction, then `page.quiz`); the links (catalog models, further reading, credits). The plain text (goals, hook,
pitfalls, quiz, links) is prerendered into the HTML. A file `sections/<Name>.svelte` replaces that section (`Hook`,
`Lesson`, `Toy`, `Real`, `Compare`, `Pitfalls`, `Checkpoint` or `Links`; props `{ store, pkg, def }`). The panels share
the lesson's runtime and sources, never change the lesson's selection, and follow each new input.

Two patterns from the example:

- **A run that ignores the learner's input** (a bigger network): a process that wraps `trainingProcess(trainer)` with
  `input: () => null`. A train source would restart on every input.
- **Two runs in one plot and one narration**: one process that runs both in lockstep and records the second run as
  metrics, drawn by `loss-curve` with `props.metrics`. A step's narration only sees the sources that step ran.

### 18.3 Scaffold, check and promote

```
npm run atlas -- new concept my-concept --kind process                 # also --input <adapter>, --dev, --port n
npm run atlas -- new concept my-concept --like loss-gradient-descent   # or start from a renamed copy
npm run atlas -- check my-concept --quick
npm run atlas -- e2e my-concept                                        # opens /train/my-concept/
```

`--kind` takes `trace`, `train` and `process` in a list, as for a model. `--like` copies a concept package, or turns a
model package into a concept. `new concept` refuses a slug that is not in `concepts.ts` (AK701) and a model slug. The template writes
`concept.test.ts`, which lints the panels like lesson steps on every fixture in en and zh (without the lesson-shape
rules) and checks the Chinese, the pitfalls' step links and the further-reading links, so the op-tests gate covers
`concept.ts`. `check` checks AK701 against `concepts.ts`, and `e2e` opens `/train/<slug>/`. In dev a draft shows at
`/train/<slug>/`; the three hand-built concept routes (tokenization, next-token prediction, transfer learning) keep
their own pages. The Lesson Lab shows a concept package with the model layout (no toy, real or compare), so look at it
at `/train/<slug>/`.

`promote` appends the credits and prints the `concepts.ts` patch (status live, `lesson: { steps, summary }`, plus
interaction, engine, mode and links when the manifest's catalog block gives them), a tech-zh row for the engine, the
glossary patch and the text cost: `concepts.ts` text reaches every `/train/` page and the list pages. It never touches
`models.ts`.
