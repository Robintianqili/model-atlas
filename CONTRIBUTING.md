# Contributing to Model Atlas (模型图鉴)

Model Atlas teaches how neural-network models work through guided, step-by-step lessons that run on the learner's own
input and show real numbers. Each model page is built with the DevKit. This file is the path for contributors; the full
guide is [`site/DEVKIT.md`](site/DEVKIT.md).

> The repository is public at [github.com/Robintianqili/model-atlas](https://github.com/Robintianqili/model-atlas), and
> the site is at [robintianqili.github.io/model-atlas](https://robintianqili.github.io/model-atlas/). Its code is under the
> MIT licence (`LICENSE`) and its lesson text, articles and figures under CC BY 4.0 (`LICENSE-CONTENT.md`); datasets,
> vocabularies, fonts and models keep their own licences (`LICENSING.md`, the licence texts in `LICENSES/`, and
> `site/static/licenses.json`).

## What you can contribute

- **A model page** for a slug that is already in the catalog (`site/src/lib/catalog/models.ts`): one package folder,
  built with the DevKit. Kind A (a trained PyTorch model), kind B (live training in the browser) or kind C (an algorithm
  step by step), or a mix.
- **A concept page** (`/train/<slug>/`) for a slug of `site/src/lib/catalog/concepts.ts`: a concept package, built the
  same way (`site/DEVKIT.md` section 18).
- **A fix** to a lesson's wording, numbers or Chinese text, inside that model's package.
- **A kit change** (`site/src/lib/atlas/**`, `site/scripts/atlas/**`, `pipeline/atlas_kit/**`): always a separate pull
  request, discussed in an issue first.

New catalog entries (a model that is not listed) are proposed in an issue first; the maintainers add them. For
experiments that are not in the catalog, `--dev` makes a lab-only package that is never published under `/m/`.

## Setup

1. Fork and clone the repository.
2. Node 24 and npm: `cd site && npm ci`.
3. Only for kind A or for Python twins and references: Python 3.10 or newer, then
   `pip install -r pipeline/requirements.txt` (a CPU build of torch is enough). The CLI uses `python` from your PATH;
   to use another one, create the git-ignored `site/scripts/atlas/atlas.config.local.json`:
   `{ "python": "/path/to/python" }`. Run your own pipeline scripts with that same Python:
   `npm run atlas -- py pipeline/<slug>/train.py`.
4. `cd site && npm run atlas -- doctor` checks everything and prints the fix for anything missing.

## Build a model page

Work on a git branch (`git switch -c model/<slug>`). The `site/.work/` lane copies in the guide exist for the
maintainers' parallel agents; you do not need them.

```
npm run atlas -- new model <slug> --like <package>   # start from a renamed copy of the closest package
npm run atlas -- new model <slug> --kind trace|train|process[,...] --input <adapter> [--template <name>]   # or from a template
npm run atlas -- new concept <slug> [--like <package>]   # a concept page instead (DEVKIT.md section 18)
npm run atlas -- py pipeline/<slug>/train.py         # kind A: train with the configured Python
npm run atlas -- export <slug>            # kind A: runs pipeline/<slug>/export.py, then parity and the lesson draft
npm run atlas -- run <slug> <source>      # kinds B and C: the tape as a table, before any page exists
npm run dev -- --port <port>              # the port `new` printed (notes/<slug>.md); then /dev/atlas/lab/<slug>/ and /m/<slug>/
npm run atlas -- lesson <slug> lint
npm run atlas -- check <slug> --quick     # while you iterate: every gate except the build and the browser test
npm run atlas -- e2e <slug>               # browser test (Edge; ATLAS_BROWSER_CHANNEL=chromium elsewhere): 2 passes, axe, contact sheets
npm run atlas -- check <slug>             # once, at the end: every gate, a package-only build and the browser test
npm run atlas -- review <slug>            # the review packet: notes/review-<slug>.md
```

Follow `site/DEVKIT.md` section 4 for the walk-through. Every step of your lesson must also work from its deep link
(`#lesson-<id>`), as the first step a learner opens. `e2e` and `check` reuse your dev server; `npm run atlas -- e2e
--stop-server` stops the one they keep.

### Building with Claude Code

The repository has a project skill for Claude Code, `.claude/skills/model-atlas-builder/`. It gives the agent this
procedure in short form: the fast check routine, reuse before you write (a kit view, op or adapter first; another
package; the browser or the standard library; only then new code), the teaching bar, the chart and accessibility rules,
and the builder and reviewer roles. Claude Code loads it when you ask it to build or review a package in this
repository; the rules below still apply to everything it writes.

### The rules your package must keep

- **Only your paths change**: `site/src/lib/explainers/<slug>/**`, `pipeline/<slug>/**`, `site/static/models/<slug>/**`
  (written by your export script, never by hand), `notes/<slug>.md`. Do not edit `site/src/lib/catalog/*`,
  `site/static/licenses.json` or the kit; the maintainers apply those changes (`npm run atlas -- promote <slug>` prints
  them).
- **Every number is real**: narration numbers come from placeholders bound to computed values or to `facts.json`;
  every factual sentence that is not computed live has a claim test (`claims.test.ts`). No `Math.random`.
- **Every stage carries a provenance badge** (live, precomputed or illustrative with a note).
- **Bilingual**: complete English; Chinese for every title and label, and for the narration before review.
- **Accessible**: the browser test's axe scan finds no serious or critical violation (AK604); use the kit's fixes
  (`site/DEVKIT.md` section 13, AK604): `use:keyScroll` for scroll boxes, the `-text` color tokens, no faded text.
- **Parity**: kind A matches PyTorch within 1e-4 on every layer; twins and references are optional and give a stronger
  badge.
- **Small and static**: each file ≤ 5 MB, each model ≤ 15 MB; nothing is fetched from another host at runtime; no new
  dataset downloads without the maintainers' agreement.
- **Credits**: every dataset, ported code or idea-with-code goes into `credits` in `explainer.ts` (name, url, licence,
  what it is used for, attribution). No code from GPL, non-commercial or unlicensed projects.
- **Notes**: `notes/<slug>.md` with What, Decisions, Final numbers (measured only), How to re-run and Kit issues.

## Pull request

Open the pull request when `npm run atlas -- check <slug>` is green. Include:

- the review packet (`notes/review-<slug>.md`) and the contact sheets from `notes/shots/<slug>/contact/`;
- the measured numbers (accuracy, parity, sizes) and anything you could not make work, with the workaround you used.

A teaching reviewer reads the lesson with the checklist in the review packet (measurable goals, one idea per step,
numbers that match the screen, claims backed by tests, an honest limits step, natural Chinese, correct credits). The
maintainers merge, run `promote` and apply the catalog patch.

### Continuous integration

On every push to `main`, the repository's GitHub Actions workflow builds the site, runs `npm run check:build` and
deploys it to GitHub Pages. CI for pull requests is planned: on Linux with Node 24 and Playwright Chromium,
`npm run atlas -- check --all` and the browser test for the packages a pull request changes. Maintainers own `site/src/lib/atlas/**`,
`site/src/lib/kit/**`, `site/scripts/atlas/**` and `pipeline/atlas_kit/**`; a model's contributor and a maintainer own
`site/src/lib/explainers/<slug>/**` and `pipeline/<slug>/**`.

## Writing style

Short sentences, plain words, terms explained on first use, no hype. The lesson is the page: one idea per step, the
learner predicts before seeing, and the page says honestly what the model cannot do.
