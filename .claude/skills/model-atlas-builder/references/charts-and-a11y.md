# Charts and accessibility

A chart in a lesson is a measuring instrument: the learner reads numbers off it and checks them against the narration.
These rules make every chart on the site read as one system, in both themes, for colorblind readers and for screen
readers. When the `dataviz` skill is available, load it before you write chart code; the rules below are the
project's fixed choices within it (KIT.md §4-§5 has the tokens and the color API).

## Contents

1. Before you draw
2. Color
3. Axes, marks and numbers
4. Text alternatives and interaction
5. The axe gate (AK604) and its fixes
6. Design skills

## 1. Before you draw

Reuse first: `loss-curve` (several runs, `metrics`, `refs`, `log`), `curve1d`, `plane2d`, `layer.tensor`,
`kit.probbars`, `kit.attention`, `tape.table`, or the `CurvePlot`, `RowBars` and `SumBars` parts inside a local view.
A new chart type is the last rung of the ladder (reuse-ladder.md).

Pick the form from the question the step asks: change over steps is a line; a share of one whole is a bar of
probabilities; a value per cell of a grid is a heat table with the numbers printed; a comparison of two runs is two
panels on one domain or two lines on one axis.

## 2. Color

- **Signed values** (weights, gradients, activations, errors): the one diverging scale, `makeScale('diverging', dom)`:
  blue below 0, gray at 0, orange above, with a symmetric domain (`domainOf('diverging', ...)`).
- **Shares, probabilities, attention, counts**: the one sequential purple ramp, `makeScale('sequential', dom)` (light to
  dark in light mode, dark to bright in dark mode).
- **Classes and series**: `categorical(k)`, a fixed colorblind-checked order (adjacent colors stay distinct under the
  common color-vision deficiencies in both themes). Always write the class or series name next to its color; color is
  never the only cue.
- **Meaning colors** come from tokens: `--real` / `--generated`, `--q` / `--k` / `--v`, `--param` / `--grad`,
  `--highlight`. No hex colors in a view; canvas code reads `SEMANTIC[theme]` or the scale functions.
- **Text in a data color** uses the `-text` token (`--div-pos-text`, `--q-text`, ...): the plain tokens are for fills
  and lines and can fall below 4.5:1 as text.
- **Text on a colored cell** uses `scale.ink(v)` or `inkOn(rgb)`, which pick black or white at 4.5:1 or more.
- Redraw on a theme change (`themeState.resolved` in an `$effect`) and check both themes on the contact sheets
  (`desktop-en-light`, `phone-zh-dark`).

## 3. Axes, marks and numbers

- Label both axes, with the unit ("step", "loss (nats)", "probability"). Step counts print as counts ("step 50").
- Numbers to scale: a bar's length is its value from 0; a log axis says "log scale" and starts at the smallest positive
  value; a clipped or broken axis is not used.
- Panels that are compared share one domain (`scopeDomains`, the same `dom`), or the step says why they do not.
- Print the numbers the narration quotes, with the kit's formats (`fmt3`, `fmtNum`, `fmtPct`, `Num`), so the learner can
  check them; the minus sign is U+2212.
- Mark what the step is about (the selected unit, the current frame) with `--highlight` and an outline, not with color
  alone.
- Legends sit next to the data, not in a separate box far away; one legend per scale (`ColorLegend`, `gradientCss`).

## 4. Text alternatives and interaction

- Every view's `describe(props)` returns a sentence a screen reader can use (AK504): what is drawn and its key numbers.
- A canvas or SVG chart has an accessible name, and its numbers are also available as text (a table under it, the
  readout line, or `describe()`).
- Everything you can hover also works with the keyboard and a tap (KIT.md rule 9). `[` and `]` belong to the
  Inspector; Space and the arrows to a focused StepPlayer.
- Animation is a pure function of the step or of `t`; with `motion.reduced` nothing autoplays, and nothing is only
  understandable while it moves.

## 5. The axe gate (AK604) and its fixes

`atlas e2e` scans every step of every pass with axe (WCAG 2.1 A and AA). Serious and critical violations fail `check`;
moderate and minor ones warn. While iterating, `e2e <slug> --axe warn`; read `notes/shots/<slug>/contact/axe.txt`
(each rule with its steps, nodes and help link) and `axe-<pass>.json`.

| axe rule | Cause | Kit fix |
|---|---|---|
| color-contrast | text below 4.5:1 (3:1 for large text): a data color as text, text faded with `opacity`, the browser's dark-theme button gray | the `-text` tokens; `scale.ink(v)` / `inkOn(rgb)` on cells; `--muted` plus a dashed border for "not yet" items; style native buttons with the site's colors |
| scrollable-region-focusable | a box with `overflow: auto` that the keyboard cannot reach | `use:keyScroll` from `$lib/kit/keyscroll` (a tab stop only while it overflows and holds nothing focusable); never a fixed `tabindex` |
| nested-interactive | buttons inside an `<svg role="img">` | `role="group"` on the svg, with an accessible name |
| aria-required-parent / aria-required-children | ARIA table cells or headers outside a `role="row"` | wrap each row in `role="row"`; a CSS subgrid keeps the grid layout |
| definition-list | a label or a `div` directly inside a `<dl>` | only `<dt>`/`<dd>` groups inside; the label goes outside |
| label | an input without a label | a `<label>` or `aria-label` |
| aria-hidden-focus | a focusable element inside `aria-hidden` | put `aria-hidden` on the decorative element only |

Kit components (CodeBlock, Tex, FormulaCard, TensorGrid, the kit views) already pass. A new text token needs a line in
`src/lib/atlas/ui/a11y.test.ts`, which checks every text token at 4.5:1 on `--bg`, `--surface` and `--surface-2` in both
themes.

Outside the gate: `atlas e2e` scans package lesson pages only. A change to a hand-built page (`/m/tinycnn/`,
`/m/tinygpt/`, the three hand-built concepts) or a site page (home, atlas, glossary, about, `/devkit/`) needs its own
scan before a release.

## 6. Design skills

When the `frontend-design` or Design skills are installed, use them for the layout and interaction of a new view, a
section override or a concept page's panels. Keep their output inside the kit: the tokens of `tokens.css`, the three
font families, the spacing scale, `.panel` and `.btn`, the kit components, 360 px without horizontal scroll, and the
rules above. A design that needs a new token or component is a kit change (its own pull request), not a package file.
