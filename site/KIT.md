# Model Atlas kit (site/KIT.md)

The shared component kit, visual grammar, i18n, catalog and page templates of the Model Atlas site.
Read this before building a model page or a concept page. Every component is rendered with demo data on
**`/dev/kit/`** (`npm run dev`, then open http://localhost:5173/dev/kit/): look there first.

- Stack: SvelteKit 2.70 + Svelte 5.57 (runes) + TypeScript 5.9 + Vite 7, `@sveltejs/adapter-static`, every page
  prerendered, `trailingSlash: 'always'`, `paths.base` from `BASE_PATH` (default `''`), `paths.relative: true`.
- No runtime third-party requests. Fonts (Fraunces, Atkinson Hyperlegible Next, JetBrains Mono) and KaTeX are
  bundled from npm. CJK text uses the system stack (`PingFang SC`, `Hiragino Sans GB`, `Microsoft YaHei`,
  `Noto Sans CJK SC`).

## Contents

1. [Commands](#1-commands)
2. [Folder map](#2-folder-map)
3. [Rules for page builders](#3-rules-for-page-builders) (read this)
4. [Theming and tokens](#4-theming-and-tokens)
5. [Color-scale API](#5-color-scale-api)
6. [i18n and articles](#6-i18n-and-articles)
7. [Component reference](#7-component-reference)
8. [Adding a model page](#8-adding-a-model-page)
9. [Adding a concept page](#9-adding-a-concept-page)
10. [Catalog API](#10-catalog-api)
11. [Known limits](#11-known-limits)

---

## 1. Commands

Run from `site/` (from the repo root: `cd site`).

| Command | What it does |
|---|---|
| `npm run dev` | dev server on http://localhost:5173 |
| `npm run check` | `svelte-kit sync` + `svelte-check` (must stay at 0 errors, 0 warnings) |
| `npm run build` | prerenders every page into `build/` |
| `npm run preview` | serves `build/` on http://localhost:4173 |
| `npm run check:catalog` | validates the catalog (unique slugs, links, prerequisites, bilingual labels) |
| `npm run check:build` | after a build: expected pages exist, no third-party runtime URLs, JS sizes per page |
| `BASE_PATH=/model-atlas npm run build` | build for a GitHub Pages project site |

## 2. Folder map

```
site/
  KIT.md                       this file
  scripts/check-catalog.ts     catalog lint (node runs .ts directly)
  scripts/check-build.ts       post-build checks
  static/                      favicon.svg, licenses.json, models/<slug>/ (model streams)
  src/app.html                 sets data-theme and lang before first paint (no flash)
  src/lib/styles/              tokens.css (design tokens), base.css (reset + utilities), math.css, global.ts
  src/lib/i18n/                i18n.svelte.ts (state, t, ui, setLang), strings.ts (UI dictionary), types.ts
  src/lib/kit/                 components (*.svelte) and pure TS helpers (color, format, stats, draw, spec, tex, …)
  src/lib/catalog/             areas, models, concepts, glossary, paths, types, index, tech-zh (Chinese for technical strings)
  src/lib/server/assets.ts     build-time only: measured asset sizes, spec facts, licenses.json
  src/lib/demo/cnn-demo.ts     ILLUSTRATIVE demo network for /dev/kit/ and the home hero
  src/lib/models/<slug>.ts     model code (model streams own this; no Svelte, no DOM)
  src/routes/                  /, /atlas/, /atlas/[area]/, /m/[slug]/, /train/, /train/[slug]/, /glossary/, /about/, /dev/kit/
```

Live pages are **static route folders** that take precedence over the `[slug]` routes:
`src/routes/m/tinycnn/`, `src/routes/m/tinygpt/`, `src/routes/train/tokenization/`,
`src/routes/train/next-token-prediction/`, `src/routes/train/transfer-learning/`.
Until a folder exists, links to it are reported as `[prerender] live page not built yet` (a warning, not a failure;
see `PENDING_LIVE_ROUTES` in `svelte.config.js`).

## 3. Rules for page builders

1. **Imports.** Import components one by one: `import TensorGrid from '$lib/kit/TensorGrid.svelte'`.
   `$lib/kit` (index.ts) re-exports only the pure TS helpers (color, format, stats, draw, spec, types). A barrel of
   `.svelte` files would pull KaTeX CSS into every page.
2. **Links and assets.** Always prefix with `base` from `$app/paths`: `href="{base}/m/tinycnn/"`,
   `fetch(\`${base}/models/tinycnn/spec.json\`)`. Never hard-code `/`.
3. **Provenance (CONTRACTS rule 2).** Every panel that shows model numbers carries a badge:
   `<ProvenanceBadge mode="live" precision="fp32" tolerance={spec.parity.max_abs_diff} against={spec.parity.against} />`,
   or set `provenance` on `<Section>`, `<FormulaCard>` or `<Inspector>`. Made-up numbers are `mode="illustrative"`.
   Quantized runs pass `approximate`.
4. **Load model code lazily.** Import `src/lib/models/<slug>.ts` with `await import(...)` inside `onMount`, so the
   prerendered shell stays small (catalog pages ship ~95–115 KB gzip of JS today). Never `import` a weights or trace
   file into the bundle; `fetch` it from `static/models/<slug>/`.
5. **Reactivity with typed arrays.** Keep traces in `$state.raw(...)` and replace the whole object after each run
   (`trace = model.forward(x)`). Kit views redraw when they receive a **new** array. If you mutate in place, bump the
   `version` prop of TensorGrid / TileStack.
6. **Shapes and colors.** Use `TensorView` (or TensorGrid / TensorStrip / TileStack directly) with the grammar:
   signed values `kind="diverging"`, probabilities/attention `kind="sequential"`, class colors `categorical(k)`.
   Compute shared ranges with `domainOf()` and pass them as `scopeDomains={{ layer, global }}`.
7. **Numbers.** Show numbers with `<Num value={x} />` (3 significant digits, full value on hover/focus/tap) or
   `fmt3(x)` in canvas/SVG. Shapes with `fmtShape([8, 26, 26])` → `[8×26×26]`.
8. **Text.** Every visible string is bilingual: `t({ en, zh })` for page text, `ui('key')` for kit vocabulary.
   English must be complete; Chinese may be shorter but must exist for labels. Long prose goes in article components
   (§6).
9. **Keyboard and touch.** Everything you can hover must also work with focus/keyboard and tap. Kit components
   already do; custom widgets must too. `[` and `]` are reserved for the Inspector (layer stepping), Space/arrows for
   StepPlayer when focus is inside it. A box that scrolls (`overflow: auto`: a wide table, a code block, a formula)
   gets `use:keyScroll` (`$lib/kit/keyscroll`): it takes a tab stop only while it really overflows. An `<svg>` with
   buttons inside is `role="group"`, not `role="img"` (an image's children are not interactive).
   `atlas e2e` runs axe on every lesson step (AK604).
10. **Motion.** Animations are pure functions of a step or `t` (StepPlayer). Read `motion.reduced` from
    `$lib/kit/theme.svelte` and never autoplay when it is true.
11. **Mobile.** Pages must work at 360 px (no horizontal page scroll). Use `grid-template-columns: minmax(0, 1fr)` for
    grid containers holding canvases, and let wide tensor views scroll inside their own box. Auto-load at most 5 MB
    of model files on phones.
12. **Privacy.** Inputs never leave the device. Do not add analytics or third-party requests;
    `npm run check:build` fails if a build fetches from another host.
13. **Licenses.** Every third-party dataset, model, code or font goes into `static/licenses.json`
    (re-read the file right before writing; append only). `/about/` renders it at build time. A file the site
    redistributes (a vocabulary, a font, a bundled library) also needs its licence text in `LICENSES/` at the repo root,
    named in the row's `licenseText` (the About table links to it). The project's own licences and repository link are
    in `$lib/project` (the footer and `/about/` show them).
14. **Input hand-off.** The home-page launcher stores what the learner drew or typed in sessionStorage. Read it once
    on mount: `const h = takeHandoff('drawing'); if (h) strokes = h.strokes;` (`$lib/kit/handoff`).
15. **Guided lesson first.** Every live page gets a step-by-step lesson (kit **Walkthrough**, §7) as its primary
    path: pass it as the shell's `lesson` snippet (it lands right after the input, the hero gets a "Start the
    step-by-step lesson" button, and the remaining sections sit under an "Explore freely" heading). One focused
    view per step, built from the page's own components and the learner's current input, real numbers only.
    Give the lesson its own selection state so it never moves the explore views (TinyCNN: `LessonView`). Set
    `lesson: { steps, summary }` on the catalog entry so the home page links to `#lesson`. Reference:
    `src/lib/pages/tinycnn/Lesson.svelte`.
16. **Contrast.** Text keeps 4.5:1 against what is behind it, in both themes. Color text with the `-text` tokens
    (`--div-pos-text`, `--q-text`, ...), not the fill tokens. Write on a colored cell with `scale.ink(v)` or
    `inkOn(rgb)`. Fade "not yet" items with `--muted` and a dashed border, never with `opacity` on text.

## 4. Theming and tokens

`src/lib/styles/tokens.css` defines every token for light (default) and dark. Dark applies when the OS asks for it
(unless `<html data-theme="light">`) or when `<html data-theme="dark">`. The header toggle cycles
system → opposite of the OS theme → OS theme → system (the first click always changes what you see) and stores the
choice in `localStorage['atlas.theme']` (inside try/catch); `app.html` applies it before first paint.

```ts
import { themeState, setTheme, cycleTheme, motion } from '$lib/kit/theme.svelte';
themeState.resolved   // 'light' | 'dark' (reactive: read it in an $effect to redraw canvases)
themeState.pref       // 'system' | 'light' | 'dark'
motion.reduced        // prefers-reduced-motion
```

| Token group | Tokens |
|---|---|
| Type | `--font-display` (Fraunces), `--font-body` (Atkinson Hyperlegible Next), `--font-mono` (JetBrains Mono), `--font-cjk`, `--fs-xs … --fs-3xl`, `--measure` |
| Space / shape | `--s1 … --s8` (4–72 px), `--r1 --r2 --r3 --r-pill`, `--page-max`, `--gutter`, `--header-h` |
| Surfaces / ink | `--bg --surface --surface-2 --surface-3 --ink --ink-2 --muted --rule --rule-strong --shadow-1 --shadow-2` |
| UI accent (not data) | `--accent --accent-strong --accent-soft --accent-ink` (links, focus, current layer, trainable outline) |
| Provenance / status | `--live(-soft) --pre(-soft) --illus(-soft) --planned(-soft)` |
| Data grammar | `--div-neg --div-mid --div-pos`, `--seq-lo --seq-mid --seq-hi`, `--q --k --v` (attention frames), `--real --generated`, `--param --grad`, `--mask-hatch`, `--highlight --highlight-halo`, `--cat-0 … --cat-9` |
| Data colors as text | `--div-neg-text --div-pos-text --q-text --k-text --v-text --real-text --generated-text`: the same hues, 4.5:1 or more on `--bg`, `--surface` and `--surface-2` (use them for `color:`; the plain tokens are for fills and lines) |

Utilities in `base.css`: `.container`, `.num` (mono, tabular, slashed zero), `.eyebrow`, `.muted`, `.lede`,
`.prose`, `.btn` (+ `.primary`, `.ghost`, `.icon`, `[aria-pressed=true]`), `.chip`, `.segmented`, `.panel`,
`.visually-hidden`, `.cluster`, `.stack`. Focus rings are global (`:focus-visible`). Reduced motion disables CSS
animations globally.

## 5. Color-scale API

`src/lib/kit/color.ts` (pure TS; safe in workers and Node). Ramps are built in OKLCH and converted once into 256-entry
lookup tables per theme.

```ts
import { makeScale, domainOf, scopeDomain, categorical, divergingRgb, sequentialRgb, gradientCss } from '$lib/kit/color';

const dom = domainOf('diverging', layerData);            // [-m, m] over all arrays passed
const sc = makeScale('diverging', dom, themeState.resolved);
sc.rgb(-0.3)            // [r, g, b] 0..255
sc.css(0.7)             // 'rgb(r g b)'
sc.write(v, img.data, i * 4)   // fills RGBA into ImageData (fast loop)
sc.ink(v)               // text color on top of that cell: 4.5:1 or more (inkOn)
inkOn([r, g, b])        // the same for any background; contrast(a, b), luminance(rgb): WCAG 2 formulas
sc.position(v)          // 0..1 on the ramp (0.5 = zero for diverging)
categorical(3)          // class 3 color for the current theme ('#eda100' light)
gradientCss('sequential')   // CSS linear-gradient for legends
```

| Function | Meaning |
|---|---|
| `makeScale(kind, domain, theme?)` | `kind`: `'diverging'` (blue < 0, gray 0, orange > 0; domain made symmetric) or `'sequential'` (purple, light → dark; dark theme: dark → bright) |
| `domainOf(kind, ...arrays)` | diverging → `[-max|v|, max|v|]`; sequential → `[min(0, min), max]` |
| `scopeDomain(kind, scope, own, { layer, global })` | picks the domain for `'unit' | 'layer' | 'global'` |
| `categorical(k, theme?)`, `CATEGORICAL` | 10 fixed class colors; slots 1–8 validated colorblind-aware order (adjacent CVD ΔE ≥ 8.4 both themes), 9–10 added and re-validated; always show the class name next to the color |
| `SEMANTIC[theme]` | `real`, `generated`, `q`, `k`, `v`, `accent` hex values for canvas |
| `currentTheme()` | non-reactive theme read (prefer `themeState.resolved` in components) |

Canvas code cannot read CSS variables: use the scale functions, `categorical()`, `SEMANTIC`, or (in TensorGrid
highlights) a color string `'var(--q)'`, which the component resolves.

## 6. i18n and articles

```ts
import { i18n, t, ui, setLang, toggleLang } from '$lib/i18n/i18n.svelte';
t({ en: 'Layer', zh: '层' })         // reactive; falls back to English when zh is missing
ui('navAtlas')                       // UI dictionary (src/lib/i18n/strings.ts), every key has en + zh
ui('stepOf', { i: 2, n: 9 })         // {placeholders}
i18n.lang                            // 'en' | 'zh'
```

- Default English. The toggle persists `localStorage['atlas.lang']` (try/catch) and sets `<html lang="zh-CN">`.
  `?lang=zh` in the URL also works. Prerendered HTML is English; Chinese is applied after hydration. To avoid a
  flash of English, `app.html` marks `<html data-lang-pending="zh">` before first paint when the saved language
  (or `?lang=zh`) is Chinese, which hides the page content (`body > div { visibility: hidden }`) until `initLang()`
  has rendered Chinese (it removes the attribute after a `tick()`); a 2.5 s failsafe timeout removes it too.
  Without JavaScript the attribute is never set, so the English page shows.
- `colon()` returns the label colon for the current language (`:` / full-width `：`): write
  `{ui('yourPrediction')}{colon()}`, never a hard-coded ASCII colon after a translated label.
- Add new kit vocabulary to `strings.ts` (both languages; `npm run check:catalog` checks it). Page-specific strings
  stay inline with `t({ en, zh })`.
- **Articles: plain Svelte components per language** (chosen over mdsvex: no extra preprocessor, runes and kit
  components work directly):
  ```
  src/lib/content/tinycnn/en.svelte      English (complete)
  src/lib/content/tinycnn/zh.svelte      中文 (may come later)
  ```
  ```svelte
  <script lang="ts">
    import Article from '$lib/kit/Article.svelte';
    import En from '$lib/content/tinycnn/en.svelte';
    import Zh from '$lib/content/tinycnn/zh.svelte';
  </script>
  <Article en={En} zh={Zh} />          <!-- zh optional: Chinese readers then see a notice + English -->
  ```
  Short inline bilingual markup: `<Lang>{#snippet en()}…{/snippet}{#snippet zh()}…{/snippet}</Lang>`.
- Glossary terms: `<Term id="softmax" />` (definition tooltip + link to `/glossary/#softmax`). Ids are in
  `src/lib/catalog/glossary.ts` (73 terms). Add missing terms there, in both languages.

## 7. Component reference

All components are Svelte 5 with typed props (`$props()`), callbacks as props (`onhover`, `onselect`, …), and
snippets for slots. Bindable props are marked **(bind)**. `Text` = `string | { en, zh? }`.

### Tensor views

**TensorGrid** — matrix or one feature map on canvas, crisp cells.
| Prop | Type | Default | Notes |
|---|---|---|---|
| `data` | `ArrayLike<number>` | — | row-major, length `rows*cols` |
| `rows`, `cols` | `number` | — | |
| `label` | `Text` | — | accessible name |
| `kind` | `'diverging' \| 'sequential'` | `'diverging'` | |
| `domain` | `[lo, hi]` | from scope | explicit domain wins |
| `scope` | `'unit' \| 'layer' \| 'global'` | `'unit'` | |
| `scopeDomains` | `{ layer?, global? }` | `{}` | computed by the page with `domainOf` |
| `cell` | `number` (px) | auto-fit | auto: fits the container between `minCell` (2) and `maxCell` (40); below `minCell` the grid scrolls |
| `gridLines` | `boolean` | `true` | drawn when cells ≥ 7 px |
| `labels` | `'auto' \| boolean` | `'auto'` | 3-digit values when cells ≥ 30 px |
| `highlights` | `Highlight[]` | `[]` | `{ row, col, h?, w?, style?: 'solid'\|'dashed', color? }`; color may be `'var(--q)'` |
| `masked` | `(r, c) => boolean` | — | hatched cells (causal mask, not-yet-computed cells) |
| `cursor` **(bind)** | `{ row, col } \| null` | `null` | keyboard/selection cursor |
| `interactive` | `boolean` | `true` | `false` for thumbnails (no focus, no hover) |
| `readout` | `boolean` | `true` | "Row r, column c: v" line under the grid |
| `version` | `number` | `0` | bump after in-place mutation |
| `onhover` | `(c: {row, col, value} \| null) => void` | | hover, tap, or keyboard move |
| `onselect` | `(c) => void` | | click / Enter |

Keyboard: Tab focuses the grid; arrows move; Home/End (+Ctrl); Enter selects; a live region announces the cell.
Performance: canvas code cannot read CSS variables cheaply, so TensorGrid reads each variable it needs
(`--surface-2`, `--mask-hatch`, `--highlight`, `var(--…)` highlight colors, …) once per color scheme with
`getComputedStyle` and caches it in a module-level map shared by every grid (key: `light`/`dark` from
`data-theme` or the OS scheme + the variable name). If you add a theme token that grids read, it is picked up
on the next scheme; do not call `getComputedStyle` per repaint in new canvas components either.
```svelte
<TensorGrid data={fmap} rows={26} cols={26} label="conv1 · channel 3" scope="layer"
  scopeDomains={{ layer: conv1Dom }} highlights={[{ row: r, col: c, h: 3, w: 3 }]} onhover={(c) => (cell = c)} />
```

**TensorStrip** — a vector as a strip of cells; elided in the middle ("…") with the exact shape label always shown.
Props: `data`, `kind`, `domain`, `scope`, `scopeDomains`, `maxCells` (24), `shape` (e.g. `['T', 768]`), `name`
(Text), `cellSize` (16), `vertical`, `values` (show 3-digit values; cells widen), `selected` **(bind)**,
`onhover(i, v)`, `onselect(i, v)`.
```svelte
<TensorStrip data={emb} name="token 3 embedding" shape={['T', 64]} maxCells={32} />
```

**TileStack** — `[C, H, W]` as small multiples with a selected channel.
Props: `data`, `channels`, `height`, `width`, `label`, `kind`, `scope` (`'layer'` default: one range for all tiles),
`scopeDomains`, `selected` **(bind)**, `channelLabels?: string[]`, `tile` (≈px, 64), `version`, `onselect(c)`.
Arrow keys move the selection (one tab stop).

**TensorView** — picks the view from the shape (leading 1s dropped): `[N]` strip, `[R, C]` grid,
`[C, H, W]` tile stack + enlarged channel. Props: `tensor: { shape, data }`, `label`, `kind`, `scope` (`'layer'`),
`scopeDomains`, `channel` **(bind)**, `legend` (true), `onhover`.

**TensorStats** — shape, count, min/max/mean/std, zeros %, histogram colored with the same scale.
Props: `tensor`, `kind`, `bins` (28).

**AttentionMatrix** — query rows × key columns, sequential purple, causal mask hatched, row/col highlight with
blue (query) / red (key) frames. Props: `weights` (row-major `[Tq×Tk]`), `queries: string[]`, `keys?`, `causal`,
`domain` (`[0, 1]`), `row` **(bind)**, `col` **(bind)**, `label`, `maxCell` (34), `onhover`. Token labels show when
cells ≥ 12 px; otherwise rely on the readout and a TokenChips row.

**ProbBars** — top-k horizontal bars, chosen item highlighted, cutoff lines between ranks.
Props: `items: { label, p, id?, color? }[]`, `k` (10), `sort` (true), `chosen` (id or label), `cutoffs:
{ afterRank, label }[]` (e.g. top-k, top-p), `tokens` (show ␣ ↵ and ids), `relative` (scale to max p),
`label`, `onselect(item)`.
```svelte
<ProbBars items={classes.map((c, i) => ({ label: c, p: probs[i], id: i, color: categorical(i) }))} k={5} chosen={argmax} />
```

**TokenChips** — alternating-color token chips; id and bytes on hover/focus/tap; selectable with keyboard.
Props: `tokens: TokenItem[]`, `selected` **(bind)**, `showIds`, `selectable` (true),
`marked?: number[]` (red underline, e.g. attended keys), `label`, `onselect(i, tok)`, `onhover(i)`.
`TokenItem` (types.ts) = `{ text, id?, bytes?, special?, tone? }`; `tone: 'generated'` draws the chip in the
"generated" color (tokens sampled by the model, magenta) and `tone: 'muted'` greys it out (e.g. tokens outside
the context window).

### Math

**FormulaCard** — KaTeX with per-symbol colors and live numbers.
Props: `tex`, `values` (`{{name}}` placeholders → 3-digit numbers tagged `.num-live`), `digits` (3), `display`
(true), `caption`, `where: { tex, text }[]`, `provenance`.
```svelte
<script> import { sym } from '$lib/kit/tex'; </script>
<FormulaCard tex={`${sym('out','y')} = ${sym('param','W')}${sym('in','x')} + ${sym('param','b')} = {{y}}`}
  values={{ y }} where={[{ tex: sym('param', 'W'), text: { en: 'weights', zh: '权重' } }]} provenance="live" />
```
Symbol roles (`sym(role, tex)`): `act`, `param`, `grad`, `in`, `out`, `q`, `k`, `v`, `prob`, `neg`, `pos`, `accent`.
Use `\begin{aligned} … \end{aligned}` for long formulas (narrow screens). **Tex** renders inline/display math in
prose: `<Tex tex="\\sqrt{d}" />` (props `tex`, `display`, `values` for `{{name}}` placeholders, as in FormulaCard). KaTeX JS (~70 KB gzip) is only in pages that use FormulaCard/Tex.

### Interaction

**StepPlayer** — play/pause/step/scrub over `count` steps; pure function of the step (or a continuous `t`).
Props: `count`, `step` **(bind)**, `t` **(bind)** (0 … count−1, fractional with `smooth`), `interval` (ms/step, 800),
`loop`, `autoplay`, `smooth`, `labels?: Text[]` (caption per step; with reduced motion they become a clickable step
list), `showList`, `speeds` ([0.5, 1, 2]), `label`, `children: (step, t)` snippet, `onstep(s)`,
`playing` **(bind)** (true while playing, e.g. to hatch cells not computed yet), `onend()` (playback reached the
last step and stopped; not on manual steps or loops; the Walkthrough uses it as a step's "done" signal),
`position?: (i, n) => string` (replaces the "Step i of n" text, e.g. "Position i of n" inside a lesson that has
steps of its own). Or `counter?: Text` (position text with `{i}` and `{n}`, default "Step {i} of {n}"; a player inside a lesson step
should pass e.g. `{ en: 'Part {i} of {n}', zh: '第 {i} / {n} 部分' }` so it is not mistaken for the lesson's steps).
Keyboard inside the player: Space/k play-pause, ←/→ step, Home/End. Reduced motion: no autoplay, Play = Next.
The step buttons use `aria-disabled` instead of `disabled` at the ends (a focused button that becomes `disabled`
drops keyboard focus to `<body>`), so focus stays on the button you pressed; style them with
`[aria-disabled='true']`. The "Step i of n" text is a polite live region only while not playing (no chatter).
Keys that a view inside the stage already handled (it called `preventDefault`, as TensorGrid, TileStack, TokenChips,
ModelOverview and ScopeSwitch do for arrows) do not also step the player; keys with Ctrl/Alt/Meta and typing in a text
field inside the stage are ignored too.
```svelte
<StepPlayer count={positions} bind:step labels={captions}>
  {#snippet children(s)}<TensorGrid … highlights={[windowAt(s)]} masked={(r, c) => r * W + c > s} />{/snippet}
</StepPlayer>
```

**Walkthrough** — the guided, step-by-step lesson: one focused visualization (the stage) per step, narration,
Back / Next / Play, "Step k of N", a clickable step list, deep links. Demo on `/dev/kit/#walkthrough`;
full use in `src/lib/pages/tinycnn/Lesson.svelte`.
| Prop | Type | Default | Notes |
|---|---|---|---|
| `steps` | `WalkStep[]` | — | see below |
| `index` **(bind)** | `number` | `0` | current step |
| `id` | `string` | `'lesson'` | deep-link prefix: `#<id>-<step id>` opens that step on load (and on `hashchange`); the plain `#<id>` anchor (the home page's lesson link) is re-aligned once `ready`, because content above the lesson can grow when the model loads |
| `label` | `Text` | "Step by step" | accessible name |
| `ready` | `boolean` | `true` | step hooks fire only while true (e.g. `!!trace`); when it turns true the current step's `onEnter` fires |
| `stage` | `Snippet<[WalkStep, WalkContext]>` | — | shared stage renderer (`{#if step.id === …}`); a step's own `stage` wins |
| `aside` | `Snippet<[WalkContext]>` | — | extra panel in the side column, e.g. a small copy of the input so the learner can redraw mid-lesson |
| `end` | `Snippet` | — | shown under the narration of the last step ("what next") |
| `stageMinHeight` | `number` | `300` | px |
| `syncHash` | `boolean` | `true` | keep `#<id>-<step id>` in the URL (SvelteKit `replaceState`, no history entries) |
| `narrationFirst` | `boolean` | — | deprecated, ignored: narrow layouts always show the narration first |
| `onstep` | `(i, step) => void` | — | |

`WalkStep` (types.ts): `{ id, title: Text, narration: Text | () => Text, look?: Text | () => Text, stage?,
onEnter?, onLeave?, duration?, waitForDone?, linger? }`. Narration is 2–4 short sentences (what you see, what
changed, why it matters); a function may read live values (it re-renders when they change; the live-region
announcement uses the value at step entry). `look` is the optional "Look at:" hint.
`WalkContext` (passed to the stage): `{ index, step, playing, visible, shown, reduced, done }`. The stage is
re-created on every step (`{#key}`), so components inside start fresh; start a step's animation with
`autoplay={ctx.shown && !ctx.reduced}` (`shown` latches once the stage itself has been in view during this step:
on a phone that is when the learner scrolls from the narration to the stage) and
call `ctx.done()` when it ends (e.g. `onend={ctx.done}` on a StepPlayer).
- **Play-through:** advances after `duration` (default from the English narration length, 6–16 s), or, for
  `waitForDone` steps, after `ctx.done()` plus `linger` ms (default 2500; at most `duration`, default 45 s, of
  waiting). On narrow layouts Play first gives the narration its reading time, then scrolls the stage under the
  control bar and only then counts the step's own time. A thin bar under the controls shows the time left. A touch
  that scrolls the page does not pause; a tap or a drawing stroke does. Any other pointer or key interaction inside
  the lesson (other than the Play button) pauses; so does scrolling the lesson off screen. Reduced motion: no auto-advance and
  no Play button (Next is the primary action).
- **Layout** (container query, 860 px): wide = stage left (60 %), narration + "Look at" + aside + step list right;
  the stage keeps the tallest height seen so far (reset when the width changes), so the page does not jump.
  Narrow = control bar with progress dots, then narration ("what you see") above the stage, then the end card,
  aside and "All steps" disclosure (DOM order = reading order on both layouts). The control bar is
  sticky under the site header; if a step change happens while the lesson top is scrolled away, the lesson
  scrolls back into view.
- **Keyboard** (focus inside the lesson, not typing in a field, key not already handled by a view in the stage,
  no modifiers): ←/→ previous/next, Home/End first/last. Focus stays on the control that was used; if the focused
  element was inside the old stage, focus moves to the lesson region.
- **Accessibility:** step title + narration are announced once per step in a polite, atomic live region; the step
  list and the dots are buttons with `aria-current="step"`; Back/Next use `aria-disabled` at the ends.
```svelte
<Walkthrough {steps} bind:index ready={!!S.trace} label={{ en: 'TinyCNN, step by step', zh: 'TinyCNN 分步课程' }}>
  {#snippet stage(step, ctx)}
    {#if step.id === 'slide'}<ConvDetail {S} view={LV} autoplay={ctx.shown && !ctx.reduced} onend={ctx.done} />{/if}
  {/snippet}
  {#snippet aside()}<MiniInput {S} />{/snippet}
</Walkthrough>
```

**DrawPad** — square pad for mouse, touch and pen (pointer events, coalesced points, pen pressure).
Props: `strokes` **(bind)** (`{ points: {x, y, p?}[], size }[]`, normalized 0..1), `brush` **(bind)** (fraction of
side, 0.065), `size` (max CSS px, 280; shrinks on phones), `resolution` (280), `presets: { name, strokes }[]`,
`showBrush`, `label`, `onchange(strokes)` (stroke end, undo, clear, preset), `ondraw(strokes)` (each frame while
drawing). Methods via `bind:this`: `getImageData(px?)` (**black ink on white**, theme-independent), `getInk(px?)`
(Float32 ink 0..1, row-major), `clear()`, `undo()`. Helpers in `$lib/kit/draw`: `rasterize(strokes, px)`,
`toInk(img)`, `strokeBounds(strokes)`, `polylines(lines, size)` (presets from 0..100 coordinates). Ctrl/Cmd+Z undoes
(the pad takes focus when you start drawing). The on-screen canvas follows `devicePixelRatio` (sharp on phones); exports
are rasterized separately, so `resolution` and `getInk(px)` are unaffected. Only the primary pointer draws (a second
finger does not break the stroke); a touch that starts on the pad draws instead of scrolling the page.
Model-specific preprocessing (crop, center, resize, invert) belongs to the model page and should be shown as a tensor.

**TextInputBar** — input or textarea with bilingual presets and a length cap in characters (code points).
Props: `value` **(bind)**, `maxLength` (256), `multiline`, `rows`, `label`, `placeholder`, `presets: { text,
label? }[]`, `submit` (show Run; Enter submits single-line, but not the Enter that confirms an IME candidate while
typing Chinese), `oninput(v)`, `onsubmit(v)`. Shows "stays on your device".

### Structure

**Inspector** — the shared layer inspector: panel (or drawer) with tabs Values / Stats / Math / Code, prev/next
buttons and global `[` / `]`.
| Prop | Notes |
|---|---|
| `layers: LayerNode[]` | `{ id, label, op?, outShape?, status? }` (use `specToLayers(spec)`) |
| `current` **(bind)** | layer id |
| `tensorOf(id)` | returns `{ shape, data }` → default Values (TensorView + ScopeSwitch) and Stats tabs |
| `kindOf(id)` | `'diverging'` default |
| `scope` **(bind)**, `scopeDomainsOf(id)` | color scope for the default Values view |
| `values`, `stats`, `math`, `code`, `header` | snippets taking the layer id; a tab shows when its snippet (or `tensorOf`) exists |
| `provenance`, `precision`, `tolerance` | badge in the header |
| `drawer`, `open` **(bind)** | fixed drawer (bottom sheet ≤ 720 px); Escape closes |
| `keys` (true), `tab` **(bind)**, `title` | `[` / `]` are ignored while typing in a field and with Cmd/Ctrl (browser shortcuts); AltGr layouts work |

**ModelOverview** — layer pipeline from `layers`; left→right, top→bottom under 560 px (container query); flowing
dashes (`flowing`, default on; static with reduced motion); `thumb(layer)` snippet slot; `current` **(bind)**;
`onselect(id)`; `label` (accessible name); one tab stop, arrow keys move. Nodes with `status: 'frozen' | 'trainable'`
show the lock / accent outline.

**ModelPageShell** — the per-model template (brief §4.2). See §8. Props: `slug`, `spec`, `downloadBytes`,
`provenance`, `goals: Text[]`, `prequestion: PredictQuestion`, snippets `facts input lesson overview inspector unit
output tryThis training article code links moreLinks checkpoint credits`, `titles`, `intros`, `badges` (per-section
provenance), `lessonNote` (one line next to the hero's "Start the step-by-step lesson" button), `exploreIntro`
(text under the "Explore freely" heading). With a `lesson` snippet the section `#lesson` ("Step by step") follows
the input, the hero shows the start button (→ `#lesson`), and an "Explore freely" heading (`#explore`) precedes the
remaining sections (also marked in "On this page").
Missing sections are omitted and numbered in order; "On this page" lists the present ones. Facts come
from the catalog + spec (`params`, `precision`, `parity`, `input.shape`) + measured download size. The catalog part is
the page's `ModelPageCatalog` (`catalog` prop, default `page.data.catalog`, which the root layout computes for every
`/m/<slug>/` route and the Lesson Lab; §10): the shell imports no catalog table, so a model page's JS does not grow with
the catalog (`check:build` fails if `/m/tinycnn/` or `/m/tinygpt/` preloads one). Until the page has
fetched its spec (and in the prerendered HTML) the facts use the build-time `page.data.liveSpecs[slug]`. The shell
also sets `<title>` and `<meta name="description">` (model name and hook; "(Planned)" for planned models), so pages
do not add their own.

**ConceptPageShell** — the concept template (brief §4.4). Props: `slug`, `goals`, `prequestion`, `lede`, snippets
`hook lesson toy real compare pitfalls post links further`, `titles`, `intros`, `badges`, `lessonNote`,
`exploreIntro`. `lesson` works as in ModelPageShell (after the hook; "Explore freely" before the toy). Links default
to the catalog models of the concept. Sets `<title>` and `<meta name="description">` from the catalog entry.

**Section** — numbered section: `id`, `title`, `n?`, `intro?`, `provenance?`, `precision?`, `tolerance?`, `band?`.
**FactStrip** — `facts: { label, value, mono? }[]`, `provenance`, `precision`, `tolerance`, `against`.
**PredictFirst / PredictReveal** — `q: PredictQuestion` (`{ id, question, options, answer, explain }`); the choice is
kept in memory + sessionStorage and shown again at the checkpoint. Labels use `colon()` (full-width in Chinese).
**CodeBlock** — `code`, `lang`, `highlight: number[]` (1-based line numbers), `caption`, `start`; copy button.
**Article**, **Lang**, **Term** — see §6.

### Small parts

| Component | Props | Use |
|---|---|---|
| `Num` | `value`, `digits` (3), `unit`, `plain`, `signed` | numbers; full precision on hover/focus/tap |
| `Tooltip` | `tip` or `content` snippet, `focusable` (true), `placement`, `width`, `class` | hover/focus/tap; Escape closes |
| `ProvenanceBadge` | `mode`, `precision`, `tolerance`, `against`, `approximate`, `note`, `compact` | rule 2 |
| `StatusBadge` | `status` (`live`/`precomputed`/`planned`), `tier`, `tooltip` | catalog status |
| `RoleChip` | `role` (`param` W slab, `grad` ∇ dashed, `activation`, `cache` ⟲, `optimizer` m v, `frozen` lock, `trainable`, `lora`), `label`, `rank`, `iconOnly` | tensor roles |
| `RoleFrame` | `role`, `status` (`frozen`/`trainable`), `label`, children | frames a tensor view as weight/gradient/frozen |
| `ColorLegend` | `kind`, `domain`, `label`, `width` | legend bar |
| `ScopeSwitch` | `value` **(bind)**, `options` | unit / layer / global |
| `InputIcon` | `kind`, `size` | input-type icon |
| `Card` / `ModelCard` / `ConceptCard` / `CatalogCard` | ModelCard: `model`, `compact`, `showArea`; ConceptCard: `concept`, `compact`, `showSection`; both build a `CardData` (`modelCard()` in `catalog/cards.ts`, `conceptCard()` in `catalog/concepts.ts`) and render CatalogCard: `card: CardData`, `compact`, `showArea` (no catalog lookups; for pages that must not load the catalog). Card (the generic link card): `href` (prefix with `base`), `title`, `status`, `tier`, `kind`, `area`, `family`, `inputs`, `size`, `hook`, `note`, `compact` | catalog cards |
| `SiteHeader`, `SiteFooter`, `LangToggle`, `ThemeToggle` | — | used by the root layout |

### Helpers (pure TS)

- `format.ts`: `fmt3`, `fmtFull`, `fmtInt`, `fmtCount`, `fmtBytes`, `fmtShape`, `fmtPct`, `visibleToken` (␣ ↵ ⇥),
  `utf8Hex`.
- `stats.ts`: `tensorStats`, `histogram`, `channel(data, c, h, w)` (subarray view), `topK`, `softmax(logits, T)`,
  `seededRandom(seed)`.
- `spec.ts`: `ModelSpec`, `SpecLayer`, `specToLayers(spec, inputNode?)`, `fetchSpec(fetch, baseUrl)`.
- `handoff.ts`: `saveHandoff`, `takeHandoff(kind)`.
- `predict.svelte.ts`: shared predictions state.
- `keyscroll.ts` (browser): `use:keyScroll` and `use:keyScroll={'pre, .katex-display'}` give a scroll box (or the
  matches inside it) `tabindex="0"` while it overflows and holds nothing focusable, so the keyboard can scroll it.

## 8. Adding a model page

1. **Catalog.** Find or add the entry in `src/lib/catalog/models.ts` (slug, area, name, family, tier, inputs, hook,
   shows, prereqs). Set `status: 'live'` when the page ships. Run `npm run check:catalog`.
2. **Assets.** The model stream puts files in `static/models/<slug>/` (`spec.json`, `weights.bin`, …) and code in
   `src/lib/models/<slug>.ts` (`load(fetchFn, baseUrl)` → `forward(...)` → `{ order, t }`). Parity is written into
   `spec.parity` and shown on the badge automatically.
3. **Route.** Create `src/routes/m/<slug>/+page.svelte` (the static folder overrides `/m/[slug]/`). Nothing else is
   needed: the root layout prerenders it and measures the download size (all files in the folder except
   `reference.json`; pass `downloadBytes` if the page loads less).
4. **Skeleton.**
   ```svelte
   <script lang="ts">
     import { onMount } from 'svelte';
     import { base } from '$app/paths';
     import ModelPageShell from '$lib/kit/ModelPageShell.svelte';
     import ModelOverview from '$lib/kit/ModelOverview.svelte';
     import Inspector from '$lib/kit/Inspector.svelte';
     import DrawPad from '$lib/kit/DrawPad.svelte';
     import ProbBars from '$lib/kit/ProbBars.svelte';
     import { fetchSpec, specToLayers, type ModelSpec } from '$lib/kit/spec';
     import { takeHandoff } from '$lib/kit/handoff';
     import type { Stroke } from '$lib/kit/draw';
     import { t } from '$lib/i18n/i18n.svelte';

     let spec = $state<ModelSpec | null>(null);
     let trace = $state.raw<{ order: string[]; t: Record<string, { shape: number[]; data: Float32Array }> } | null>(null);
     let current = $state('conv1_1');
     let strokes = $state<Stroke[]>([]);
     let pad = $state<ReturnType<typeof DrawPad>>();       // bind:this gives getInk(), getImageData(), …
     let model: Awaited<ReturnType<typeof import('$lib/models/tinycnn').load>> | null = null;

     onMount(async () => {
       const url = `${base}/models/tinycnn`;
       spec = await fetchSpec(fetch, url);
       const mod = await import('$lib/models/tinycnn');     // lazy: not in the page shell
       model = await mod.load(fetch, url);
       const h = takeHandoff('drawing');
       if (h) strokes = h.strokes;
       run();
     });
     function run() {
       if (!model || !pad) return;
       const x = preprocess(pad.getInk(28));                // show this tensor as "layer 0"
       trace = model.forward(x);                             // new object -> views redraw
     }
   </script>

   <ModelPageShell slug="tinycnn" {spec} provenance="live" goals={[{ en: '…', zh: '…' }]} prequestion={q}>
     {#snippet input()}<DrawPad bind:this={pad} bind:strokes onchange={run} ondraw={run} presets={…} />{/snippet}
     {#snippet overview()}{#if spec}<ModelOverview layers={specToLayers(spec)} bind:current />{/if}{/snippet}
     {#snippet inspector()}
       {#if spec}<Inspector layers={specToLayers(spec)} bind:current tensorOf={(id) => trace?.t[id]}
         provenance="live" precision={spec.precision} tolerance={spec.parity?.max_abs_diff} />{/if}
     {/snippet}
     {#snippet output()}<ProbBars items={…} chosen={…} />{/snippet}
   </ModelPageShell>
   ```
5. **Checks.** `npm run check`, `npm run build`, `npm run check:build` (the live route should now say "built"),
   then look at the page at 360 px and in both themes and languages.

## 9. Adding a concept page

1. Find the entry in `src/lib/catalog/concepts.ts`; set `status: 'live'` when it ships.
2. Create `src/routes/train/<slug>/+page.svelte` and use `ConceptPageShell`:
   ```svelte
   <ConceptPageShell slug="tokenization" goals={[…]} prequestion={q}>
     {#snippet hook()}…{/snippet}
     {#snippet toy()}…every variable visible…{/snippet}
     {#snippet real()}…the same idea on TinyGPT…{/snippet}
     {#snippet compare()}…{/snippet}
     {#snippet pitfalls()}…{/snippet}
     {#snippet further()}<ul><li>…</li></ul>{/snippet}
   </ConceptPageShell>
   ```
3. Toys that use made-up numbers carry `badges={{ toy: 'illustrative' }}` (or a ProvenanceBadge inside).

## 10. Catalog API

```ts
import { AREAS, AREA_BY_SLUG, MODELS, MODEL_BY_SLUG, CONCEPTS, CONCEPT_BY_SLUG, SECTIONS, GLOSSARY,
         GLOSSARY_BY_ID, PATHS, sortModels, modelsAccepting, conceptsBySection, conceptsForModel } from '$lib/catalog';
```

- 11 areas; 103 models (P0 5, V1 7, P1 23, P2 68; backlog rows split into one entry each); 72 concepts in 10
  sections (13 in the v1 freeze); 73 glossary terms; 5 learning paths.
- `ModelEntry`: `slug, area, name, family, tier, tierNote?, status, inputs, own, params?, size?, runtime, mode, hook,
  shows, notes?, prereqs, lesson?`. `ConceptEntry`: `slug, section, title, status, v1, tier?, interaction, engine,
  mode, links, lesson?`. `lesson: { steps, summary }` (`LessonInfo`) marks a page with a guided lesson; the home
  page's first call to action ("Start a step-by-step lesson") lists the live pages and links those with a lesson to
  `#lesson` (pages without one link to their top).
- Brief link names that differ from catalog slugs are mapped in `LINK_ALIASES` (`tiny-cnn → tinycnn`,
  `sd → stable-diffusion`, `whisper → whisper-tiny`, …).
- Build-time data available to every page through `page.data` (root `+layout.server.ts`): `liveSizes[slug]`
  (bytes) and `liveSpecs[slug]` (`params, precision, input {kind, shape}, parity {against, max_abs_diff, checked},
  version` read from `spec.json`; trimmed on purpose because it is serialized into every page); `site` (`SiteCounts`:
  the footer's model, concept and live counts); and on `/m/<slug>/` routes and the Lesson Lab `catalog`
  (`ModelPageCatalog`: the entry, its area name, its technical strings as bilingual text, and compact `CardData` for
  the concepts that use it, its prerequisites and the models that follow it), computed by `catalog/slices.ts`.
- **Which pages import the catalog.** List pages (home, `/atlas/`, area pages, `/train/`, `/glossary/`, `/dev/kit/`)
  import the tables they list. ModelPageShell and the flagship routes (`/m/tinycnn/`, `/m/tinygpt/`) read `page.data`
  instead, so an entry going live or new catalog text never grows their JS. (The `/m/[slug]` route itself still imports
  `models.ts` for its planned pages and the auto-page check.) Import `slices.ts` only from server code.
- `runtime`, `params`, `size` (models) and `engine` (concepts) are plain English strings. Show them with
  `t(techText(s))` from `$lib/catalog/tech-zh` (Chinese in `TECH_ZH`). Strings that are only numbers and units need no
  entry; for any other new or changed string add one, or `npm run check:catalog` fails.

## 11. Known limits

- Prerendered HTML is English; a saved Chinese preference applies after hydration. The content stays hidden until
  then (see §6), so Chinese readers see a short blank instead of English; after 2.5 s (very slow devices) the
  failsafe shows the page anyway.
- `404.html` is the adapter-static SPA fallback (GitHub Pages serves it for unknown paths): it boots the app and
  renders `src/routes/+error.svelte` (bilingual, links home / atlas / the TinyCNN lesson). Without JavaScript a
  `<noscript>` message from `app.html` shows instead (only on pages whose body is empty, i.e. the fallback).
- The `/m/[slug]/` and `/train/[slug]/` routes 404 for live slugs until their static folders exist (build warns).
- KaTeX ships all its font formats (woff2, woff, ttf) in `build/`; browsers download only woff2.
- No automated Playwright tests yet; checks are `check`, `check:catalog`, `check:build` and manual review at 360 px.
