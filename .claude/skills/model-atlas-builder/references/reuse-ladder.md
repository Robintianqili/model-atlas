# Reuse before you write: what already exists

The ladder (adapted from the "Ponytail" idea of building from what is already there):

1. Does a kit view, op, adapter, library process, helper or manifest option already do it?
2. Does another package already do it? Copy it into yours and keep its credit.
3. Does the browser or the standard library do it?
4. Only then write local code: the smallest that works, with its `describe()` text and tests.

What is never cut to save lines: real numbers, bilingual text, accessibility, provenance and validation.

Paths below are under `site/src/lib/` unless they say otherwise. When this list and the code disagree, the code wins:
DEVKIT §14 and §15 are generated from it, and `/dev/atlas/` shows every view with demo data.

## Contents

1. Start from the closest package
2. Kit views
3. Parts and components for a local view
4. Ops, adapters and data
5. Processes, training and helpers
6. Manifest options that replace package workarounds

## 1. Start from the closest package

`npm run atlas -- new model <slug> --like <package>` (or `new concept <slug> --like <package>`) copies a working
package: lesson, claims, notes, local views and adapters.

| Package | Kinds | Input | Good start for | Local views |
|---|---|---|---|---|
| `hello-xor` | A + B + C | a point (`xor-point`) | the smallest example of all three kinds | none |
| `tinyvit` | A | drawing (`drawing28`) | a trained classifier on drawings, patches | `patch-grid` |
| `vae-2d` | A | drawing | an encoder-decoder, a latent space | `latent`, `pair` |
| `char-rnn` | A + C | text | a recurrent model, sampling | `timeline` |
| `tinyseq2seq` | A + C | text | encoder-decoder attention, alignment | `align`, `attend`, `encoder` |
| `tinybert` | A + C | masked text | a masked model (`output.row`) | `both-ways`, `chips`, `guesses`, `rows` |
| `tinyclip` | A + C | shapes and captions | two towers, a similarity table | `match` |
| `tinymamba` | A + C | a sequence | a state-space scan | `memory`, `scan` |
| `gcn` | A + B + C | graph (karate club) | graph inference and retraining | `graph` |
| `gnn-variants` | A + C | graph | message passing variants | `agg` |
| `mlp-playground` | B + A | 2-D points | live training of a classifier, twins | `neuron-grid` |
| `flow-2d` | B + C + A | 2-D points | a generative model trained live, Euler sampling | `plane` |
| `gan-2d` | B | 2-D points | two networks trained against each other | `arena`, `game` |
| `pinn` | B + C | series | a physics loss, several loss terms | `loss-bars`, `residual-strip` |
| `tinyforecaster` | B + C + A | series | a forecaster, rollout | none |
| `word2vec` | B + C | a corpus | embeddings trained live | `map`, `net`, `window` |
| `matrix-factorization` | B + C | ratings | recommendation, a factor map | `map`, `matrix` |
| `cart-pole` | B + C | an environment | policy gradients (RL) | `round`, `scene`, `worked` |
| `gridworld` | C | a grid | bandits and Q-learning, tables | `arm-bars`, `grid-board` |
| `alphazero-ttt` | A + C | a board | search plus a network | `board`, `play`, `tree` |
| `kalman-filter` | C | a track | a filter, predict and update | `track` |
| `classic-ml` | C | 2-D points | kNN, a decision tree, k-means | `canvas` |
| `loss-gradient-descent` | C (concept) | two dials | a concept page: toy, real network, comparison | `bowl`, `fit` |

## 2. Kit views

Use the key as a step's `view`; most take props (`{ key, props: (rt, at) => ({...}) }`).

| Need | View |
|---|---|
| any tensor, drawn by its axes (tiles, token grid, heads, bars, plane, curve) | `layer.tensor` |
| the arithmetic of one output number | `zoom.weighted-sum`, `zoom.elementwise`, `zoom.reduce`, `zoom.normalize`, `zoom.softmax`, `zoom.lookup`, `zoom.index-map`, `zoom.gate` |
| a formula with live numbers (a process phase) | `zoom.formula` |
| class or token probabilities | `kit.probbars` (`chosen`, `row`) |
| attention weights, a head at a time | `kit.attention` |
| text as tokens | `kit.tokens` |
| a loss or metric over steps, several runs, reference lines | `loss-curve` (`compare`, `metrics`, `log`, `smooth`, `refs`) |
| any value over time | `curve1d` (`yLabel`, `legend`) |
| points on a plane, a decision field | `plane2d` (`pointValues`, `valueAt`) |
| one training step: forward, loss, backward, update | `train-step` |
| the network as a diagram | `network` |
| training controls (play, learning rate, reset) | `train-controls` |
| every frame of a run as a table | `tape.table` |
| two runs side by side | `compare` (or a step's `compare: { a, b }`) |

## 3. Parts and components for a local view

A local view (`npm run atlas -- new view <key> --local <slug>`) composes before it draws:

- `ViewSlot` (`atlas/ui/ViewSlot.svelte`): a kit view inside yours.
- `atlas/ui/views/kit.parts/`: `CurvePlot`, `RowBars`, `SumBars`, `TensorPanel`, `SpatialMaps`, `TermTable`,
  `OpFormula`, `UnitHead`, `Num`, `Notice`; `tex.ts` (`alignTex`, `texValues`).
- `$lib/kit/` (KIT.md §7): `TensorGrid`, `TensorView`, `TensorStrip`, `TileStack`, `AttentionMatrix`, `ProbBars`,
  `TokenChips`, `FormulaCard`, `Tex`, `CodeBlock`, `Inspector`, `ModelOverview`, `StepPlayer`, `DrawPad`,
  `TextInputBar`, `ColorLegend`, `RoleChip`, `RoleFrame`, `Tooltip`, `Num`, `ProvenanceBadge`.
- `$lib/kit/color` (`makeScale`, `domainOf`, `categorical`, `inkOn`, `gradientCss`), `$lib/kit/keyscroll`
  (`use:keyScroll`), `$lib/kit/theme.svelte` (`themeState`, `motion.reduced`), `$lib/i18n/i18n.svelte` (`t`).
- Numbers: `fmt3`, `fmtInt`, `fmtPct`, `fmtNum`, `fmtSetting`, `paren` (`atlas/core/format.ts`).

## 4. Ops, adapters and data

- **Core ops** (DEVKIT §14): input, conv2d, linear (dense, linear_tied), relu, tanh, sigmoid, silu, gelu, maxpool2d,
  flatten, concat, add, embedding, position_embedding, layernorm, attention_scores, causal_mask, softmax,
  attention_mix, lstm. A local op starts as `new op <name> --local <slug> --like <op>`.
- **Adapters** (DEVKIT §5.4): `drawing28` (`options.frame`), `text-bpe`, `text-char` (`options.vocab`,
  `normalize`), `points2d` (`options.classes`, `options.presets`), `series` (generators, presets file), `graph` (the
  karate club with `graphOf`, `adjacency`, `normalizedAdjacency`, `toggleEdge`, `hops`; `Club.svelte`). Check the
  options before you copy an adapter.
- **Data** (`atlas/data/`): 2-D points `moons`, `circles`, `spiral`, `xor`, `gaussians`, `ring`, `smiley`
  (`pointsTensors`, `sampleBatch`); series `sine`, `trend`, `seasonal`, `square`, `randomWalk`, `windows`,
  `dampedOscillator`, `observations`; `karate.ts`. Python: `ak.data.quickdraw`, `ak.data.tinystories_chars`,
  `ak.data.karate`.

## 5. Processes, training and helpers

- **Library processes** (`atlas/process/library/`): `samplingProcess` (temperature, top-k, top-p;
  `samplingDistribution`, `pickToken`, `topEntries`), `eulerProcess` (integrate points along a field),
  `rolloutProcess` (autoregressive forecasting).
- **Training** (`atlas/autograd/`, `atlas/train/`): modules `Linear`, `MLP` (`outAct`, `logits()`), `Embedding`,
  `LSTMCell`, `Conv1d`, `LayerNorm`; losses `mse`, `crossEntropy`, `bceWithLogits`, `infoNCE`; optimizers `sgd`,
  `momentum`, `adam` (AdamW: `decoupled: true`); `trainingProcess(trainer)` to wrap a trainer in a process (it inherits
  the trainer's twin when its frames are unchanged); `updateOf(at, key, index)` for "old − lr × g = new" (plain SGD
  only); `accuracy`, `gradNorm`, `paramCount`, `decisionGrid`; `stackToSpec` for a `live-model` source.
- **Narration** (`atlas/lesson/say.ts`): `plural`, `num`, `pick`; formats `n2`-`n6`, `int`, `pct`, `signed`.
- **Tests** (`atlas/lesson/fixtures.ts`): `withFixtures` runs a claim on every fixture and shares runs.
- **Python** (`pipeline/atlas_kit`): the builder (`b.linear`, `b.conv2d`, `b.lstm`, `b.op`, ...), `export_model`,
  `ak.Rng` (bit-identical to the TS rng), `ak.TapeWriter`, `ak.json_arg`, `ak.write_json`.

## 6. Manifest options that replace package workarounds

| Instead of | Use |
|---|---|
| reading a masked model's answer row by hand | `output.row` (`'first'`, an index, `(rt) => row`) |
| a presets file the adapter could generate | `input.presetsFromAdapter: true` |
| a section override for the training curve | `page.trainingCurve` (`metric`, `metrics`, `labels`, `log`, `smooth`, `refs`) |
| renaming or hiding auto-page sections in Svelte | `page.sections` (`titles`, `intros`, `hide`) |
| custom labels for training metrics | `page.metrics` |
| a train source that must ignore the input | `runOnEmpty: true`, or a process wrapping `trainingProcess` with `input: () => null` |
| a process waiting for trained weights | `after: '<source>'` |
| a slow process on the main thread | leave `run` unset (the package worker) |
| "step k" marks in another unit | `play.unit` (`{ en: 'episode', zh: '第 {n} 回合' }`), `by: 'sub'` |
| typed glossary text in narration | `terms` with glossary ids, or `manifest.glossary` |

## Promoted in K4 (rule of two)

- `atlas/ui/views/kit.parts/Cells.svelte`: a vector as a row of cells on the diverging scale (two-tower, sasrec).
- `atlas/ui/views/kit.parts/refocus.ts`: `use:refocus` + `rememberFocus(key)` keep a slider's focus across `setInput`
  (kalman-filter, gaussian-process, hmm, ecg-autoencoder).
- `class="atlas-answer"` on what a predict question asks: hidden while the stage is inert (two-tower, sasrec).
- `lastParams(rt, source)` in `atlas/train/helpers.ts`: a train source's last weights for a process (score-2d,
  realnvp-2d).
- Narration formats `pct2` and `sci` (crnn-ocr, char-rnn).
- Autograd `op.clamp`, `op.minimum`, `op.maximum` (PPO's clipped objective) and `TrainerDef.project` (TransE's unit
  length).
- View props: `tokens`/`keys` (cross-attention, bilingual names), `axisWords`/`axisItems` (an axis the kit cannot
  name), `digits` (formula cards, train-step), `labels` (zoom.softmax), loss-curve `xLabel` and `lowest`;
  `StepDef.predictAt` (live options, answer and explanation of a predict question).
- Still local (one user so far): vq and pixel_shuffle (vq-vae), rfft/irfft (fno), greedy decoding and the date task
  (tinyseq2seq, tiny-encdec: a second user exists, a kit version is the next candidate), the series adapter with page
  state (fno, ecg-autoencoder), a kit cross-attention view with a player (tinyseq2seq, tiny-encdec).
