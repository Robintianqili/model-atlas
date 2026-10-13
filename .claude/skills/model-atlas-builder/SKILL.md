---
name: model-atlas-builder
description: >-
  Build, review or integrate a Model Atlas (模型图鉴) lesson with the project's DevKit: a model package (/m/<slug>/),
  a concept package (/train/<slug>/), a lesson view, or a kit change. Use this whenever you work in the model-atlas
  repository on site/src/lib/explainers/<slug>/ (explainer.ts, lesson.ts, trainer.ts, process.ts, concept.ts,
  claims.test.ts, views, adapters) or pipeline/<slug>/, run any `npm run atlas --` command (new, worktree, check, e2e,
  lesson, parity, review, merge, promote), write or fix lesson narration or its Chinese, add a chart or a view, fix an
  AK error code or an axe (AK604) finding, review a package as the teaching reviewer, or merge lanes. Use it even when
  the request only says "add <model> to the atlas", "continue completing the atlas", "review this lesson" or "why does
  atlas check fail". It holds the fast check routine, the reuse-before-you-write ladder, the teaching bar, the chart
  and accessibility rules, and the builder and reviewer roles with time targets.
---

# Model Atlas builder

Model Atlas teaches how models work through guided, step-by-step lessons that run on the learner's own input and show
real numbers. The lesson is the product; the tiny models exist so that every step can show real numbers. The DevKit
turns a package of plain TypeScript data and functions (plus, for a trained model, a PyTorch export) into the page, the
lesson player, the views and the gates.

This skill is the short form. The authority is `site/DEVKIT.md` ("§n" below means its section n). `site/KIT.md` covers
the Svelte components and tokens, `CONTRACTS.md` the frozen types and project rules, the decisions table in `LOG.md`
what the user decided. Read DEVKIT §1-4 once; look the rest up when you need it.

## What is never cut

Speed comes from running fewer and cheaper checks and writing less code. It never comes from these:

1. **Real numbers.** Every number in a narration is a `{placeholder}` bound to live state or to `facts.json`. Every
   factual sentence that is not computed live has a claim test (`claims.test.ts`). No `Math.random`, no typed numbers.
2. **Provenance.** Every stage carries a badge: live, precomputed, or illustrative with a note.
3. **Bilingual.** Complete English and natural Chinese, for titles, labels and narration.
4. **Accessibility.** axe finds no serious or critical violation (AK604); keyboard and touch work; reduced motion is
   respected; 360 px has no horizontal scroll.
5. **Validation.** Parity (kind A within 1e-4 of PyTorch), twins and references where they exist, every gate green.
6. **Credits and licences.** Every dataset, ported code or idea-with-code is credited. No new download without the
   user's OK; nothing is fetched from another host at runtime.
7. **Ownership.** A lane changes only its owned paths. A package lane never edits the kit: it works around a kit bug
   inside the package and writes it under "Kit issues" in its notes.

## Roles and time targets

| Role | Target | Does | Skips |
|---|---|---|---|
| Builder | 35-40 min per package (long training aside) | lane, scaffold from the closest package, numbers, lesson, self-review against the teaching bar, simplify pass, one full check (its e2e is the package's only one), review packet, notes | e2e while iterating (`check --quick` is the loop); `--passes all`; clicking through every step by hand |
| Reviewer | 25-30 min | reads the contact sheets, `lesson-print.txt` and `axe.txt` first, then `lesson <slug> print` on the fixtures, clicks only where those cannot answer, fixes inside the package, `check <slug> --quick`, verdict | e2e and the full check (only when the packet is missing or stale); repeating the builder's checks; manual walks of every width, language and theme |
| Integrator (main session) | per batch | merge, promote, one `check --all` without e2e slugs (it trusts the lanes' results), one commit per package, LOG.md, delete merged lanes | e2e of the merged packages; rewriting packages beyond the catalog patch |

**Lean gates.** Run few and cheap checks, never one that already passed, and spend the time on fixing and landing:

- **Builder**: `check --quick` while iterating; ONE e2e (the default 2 passes) inside the single full check at the end.
- **Reviewer**: no e2e; read the packet (contact sheets, `lesson-print.txt`, `axe.txt`). Re-run only when the packet
  is missing or stale (no contact sheets, or a view, adapter or layout change after the check it names).
- **Integrator**: `check --all` without e2e slugs; trust the lanes' results.
- **A kit or layout change**: one default-pass e2e on about 6 packages of different kinds (`check --all <slugs>`).
- **A full-site browser sweep** (one screenshot per page) only before a public release.
- **AK603** (a long task) never blocks and is never re-run.

The full protocols, the report format and the integration routine: [references/roles.md](references/roles.md).

## The fast routine (builder)

From `site/` (a lane's `site/` in parallel work):

```
npm run atlas -- worktree <slug> --from-git HEAD      # parallel agents: a lane from the last commit; cd to the folder it prints
npm run atlas -- new model <slug> --like <closest>    # or --template <name>, or --kind ... --input ...; concepts: new concept
npm run atlas -- py pipeline/<slug>/train.py          # kind A, then: export <slug>
npm run atlas -- run <slug> <source> --frames 0..20   # kinds B and C: the tape as a table, before any page
npm run atlas -- lesson <slug> lint --cap 200         # while writing the lesson; lesson <slug> print to read it
npm run atlas -- check <slug> --quick                 # the loop: seconds after a lesson edit (parity and op tests cached)
npm run atlas -- check <slug>                         # ONCE, when the package is done: its e2e (2 passes) is the only one
npm run atlas -- review <slug>                        # notes/review-<slug>.md for the reviewer
npm run atlas -- promote <slug> --verify              # the catalog patch from manifest.catalog, checked on a copy
npm run atlas -- e2e --stop-server                    # before you finish: stop the kept dev server
```

- **Start from the closest package** with `--like` (a working lesson, claims and notes to rewrite) rather than from an
  empty template. The table of packages by kind and input is in [references/reuse-ladder.md](references/reuse-ladder.md).
- **Look, do not walk.** The full check's e2e writes a contact sheet per pass (every step, labelled),
  `lesson-print.txt` (every narration with its numbers, en and zh) and `axe.txt` to `notes/shots/<slug>/contact/`.
  Read those; open a browser only for what they cannot show (an interaction, an animation, a task's reaction). While
  you write a view, look at it on the dev server (`npm run dev -- --port <port>`), not with e2e.
- **One full check.** If it fails, fix and re-run only the gate that failed (`check <slug> --only e2e`, or
  `--only lesson`), never the gates that passed. AK603 (a long task) only warns: do not re-run for it.
  `check <slug> --fast` re-runs every gate without the cache when you suspect it (§10). A full check takes 3-9 minutes:
  run it in the background with its output in a log file.
- **Notes as you go.** `notes/<slug>.md` has What, Decisions, Final numbers (measured only), How to re-run and Kit
  issues; it survives a lost context.
- **Page code ≤ 1,000 lines** (the check report counts it; the manifest's `catalog` block does not count).
- **The catalog entry comes from the package**: name, family, inputs, mode, hook, shows, runtime, params and size go
  into `manifest.catalog`; `promote <slug> --verify` applies them to a copy of the catalog and checks it.

Check modes, e2e passes, the dev server, lanes and the AK quick fixes: [references/procedure.md](references/procedure.md).

## Reuse before you write

Before you write code, climb this ladder and stop at the first rung that works:

1. **The kit already does it**: a kit view (often with props), op, adapter or adapter option, library process
   (sampling, Euler, rollout), helper (`say.ts`, `format.ts`, `train/helpers.ts`, the data generators) or manifest
   option (`output.row`, `input.presetsFromAdapter`, `page.trainingCurve`, `page.sections`, `loss-curve` `refs`).
   Look in DEVKIT §5.4, §9, §14, §15 and the gallery at `/dev/atlas/`.
2. **Another package already does it**: copy it into your package (`new model --like`, or copy its local view or
   adapter folder) and keep its credit. When a second package needs the same thing, write it under "Kit issues" as a
   candidate for the kit (the rule of two).
3. **The browser or the standard library does it**: CSS grid and `overflow`, `<details>`, SVG, Canvas 2D, typed arrays;
   in Python, torch and numpy built-ins rather than hand-written loops.
4. **Only then local code**: the smallest local op, view or adapter that works, with its `describe()` text and tests.

Never save lines by cutting real numbers, bilingual text, accessibility, provenance or validation. The inventory of
rungs 1 and 2: [references/reuse-ladder.md](references/reuse-ladder.md).

## The teaching bar

Self-review against it before the full check; the reviewer checks the same list.

- **One idea per step**, with a "look" line that points at one thing in the stage. 5-12 steps; a narration has 2-4
  sentences and at most 90 English words.
- **Predict first.** At least one predict step: the narration sets the question up, the answer lives in
  `predict.explain` (AK412 warns when the narration gives it away).
- **A task.** The learner does something and the page reacts (`task.done`).
- **Honest limits.** One step (`kindOf: 'limits'` or `'breakIt'`) shows with live numbers what the model cannot do.
- **Numbers from live state or `facts.json`**, never typed. Narrate a small change itself (`drop: a - b`), count from 0
  where PyTorch does and say so (§8.3).
- **Terms defined before use**, with the site glossary's ids in `terms`.
- **Natural, glossary-consistent Chinese**: what a native teacher would write, the glossary's terms, the same
  placeholders as the English, no translation tone and no English left over.
- **Directions**: "in the stage", never "above" or "on the left" (the stage moves below the narration on a phone).
- **Every step works from its deep link** (`#lesson-<id>`) as the first step a learner opens.

The full checklist with the findings reviewers make most often: [references/teaching-bar.md](references/teaching-bar.md).

## Charts, views and design

When the `dataviz` skill is available, load it before you write a chart view; this is the kit's version of its rules:

- **One diverging scale** for signed values and **one sequential purple ramp** for shares, probabilities and attention
  (`makeScale` in `$lib/kit/color`); classes take `categorical(k)`, a colorblind-checked palette, with the class name
  next to the color. No other colors for data.
- **Labelled axes** with units; **numbers to scale** (a bar's length is its value, a log axis says so); one domain for
  panels that are compared.
- Reuse `loss-curve`, `curve1d`, `plane2d`, `layer.tensor` or the `CurvePlot` part before drawing a new chart.
- Every view has a non-empty `describe()` for screen readers (AK504).

When the `frontend-design` or Design skills are installed, use them for the layout and interaction of a new view or
section, inside the kit's tokens, fonts and components (KIT.md §4): no new fonts, colors or spacing scales.

## The accessibility gate (AK604)

`atlas e2e` scans every step of every pass with axe (WCAG 2.1 A and AA); `check` fails on a serious or critical
violation. Read `contact/axe.txt` from the full check; after a fix, re-run that gate alone (`check <slug> --only e2e`).
The kit's fixes: `use:keyScroll` for a box that scrolls; the `-text` color tokens for text and `scale.ink(v)` /
`inkOn(rgb)` on colored cells; `--muted` and a dashed border instead of `opacity` on text; `role="group"` for an
`<svg>` that holds buttons; `role="row"` around ARIA table cells; only `<dt>`/`<dd>` inside a `<dl>`. Details:
[references/charts-and-a11y.md](references/charts-and-a11y.md).

## Before merge: the simplify pass

When the package works, run the `simplify` skill (when available) on the package's own changes: **reuse** (did a rung
of the ladder get skipped?), **simplification** (dead options, duplicated `vars()`, state a view does not need) and
**efficiency** (heavy work on the main thread belongs in the worker, AK603; nothing recomputed per frame that a frame
can carry). Stay inside the owned paths. Then `check <slug> --quick` and the one full check.

## Concept packages

A concept page (`/train/<slug>/`) is the same kind of package with `type: 'concept'` and a `concept.ts` (hook, toy, real,
compare, pitfalls, further). `npm run atlas -- new concept <slug> [--like <package>]`; the example is
`loss-gradient-descent`. Everything above applies; the specifics are in DEVKIT §18.

## Reference files

- [references/procedure.md](references/procedure.md): commands, check modes, e2e passes and outputs, the dev server,
  lanes, the AK quick-fix table.
- [references/roles.md](references/roles.md): builder, reviewer and integrator protocols with time budgets; notes and
  report formats.
- [references/reuse-ladder.md](references/reuse-ladder.md): what the kit and the existing packages already provide.
- [references/teaching-bar.md](references/teaching-bar.md): the review checklist and the most common findings.
- [references/charts-and-a11y.md](references/charts-and-a11y.md): chart rules and the accessibility fixes in detail.
