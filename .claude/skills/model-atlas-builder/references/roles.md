# Roles: builder, reviewer, integrator

Three roles share one package. Each has a time target; the targets come from measured waves (a builder plus a reviewer
took 2 to 2.5 hours per model before the lean roles and the K2 speed pack). Where the time goes matters more than the
total: most of the old time went into repeated full checks and manual walks of every width, language and theme.

## Contents

1. Builder (35-40 min)
2. Reviewer (25-30 min)
3. Integrator (main session)
4. Notes and reports

## 1. Builder (35-40 min, long training aside)

| Phase | Budget | What |
|---|---|---|
| Orient | 3 min | the catalog entry (hook, shows, inputs); the closest package to copy (reuse-ladder.md); DEVKIT § for the kinds you need |
| Lane and scaffold | 2 min | `worktree <slug> --from-git HEAD`; `new model <slug> --like <closest>` (or `new concept`) |
| Numbers | 8-10 min | kind A: train (fixed seed) and `export`; kinds B and C: `trainer.ts` / `process.ts`, checked with `atlas run` |
| Lesson | 12-15 min | `lesson.ts` from the teaching bar; `explainer.ts` (goals, prequestion, credits); `claims.test.ts`; `lesson lint --cap 200`, `lesson print`; `check --quick` |
| Self-review and simplify | 3-5 min | the teaching bar (teaching-bar.md) as a learner, on `lesson <slug> print`; the simplify pass on your changes |
| Full check and look | 6-8 min | the one full `check <slug>` (its e2e, on the default 2 passes, is the package's only browser test); read the contact sheets, `lesson-print.txt`, `axe.txt`; fix and re-run only the failed gate (`--only e2e`, `--only lesson`) |
| Finish | 2 min | `review <slug>`; `notes/<slug>.md`; `e2e --stop-server` |

- No e2e while iterating: `check --quick` is the loop, and the full check's e2e is the package's one browser run.
  Never re-run a gate that passed. A long task (AK603) only warns: never re-run for it.
- The builder is also the first reviewer: ship nothing you would not sign off as a first-year student reading it.
- Write down what you could not do and why ("Kit issues", open issues). Never hide a workaround.
- Long GPU training runs in the background in resumable chunks (every shell command stays under 9 minutes); write the
  lesson meanwhile.

## 2. Reviewer (25-30 min)

The reviewer is a teaching reviewer reading as a first-year student who knows school algebra. Start from what the
builder's tools already produced; open the page only for what they cannot show. The reviewer runs no e2e: the packet
is the browser's record. Only a missing or stale packet (no contact sheets, or a change to a view, an adapter or the
layout after the check it names) calls for one full `check <slug>`; fixes to narration, Chinese or claims do not make
it stale.

1. **Read the packet** (5 min): `notes/review-<slug>.md` (checklist, the gates with the mode of the check they come
   from, the catalog patch, contact files listed first), the contact sheets (`<pass>.png`), `lesson-print.txt`,
   `axe.txt` (in `notes/shots/<slug>/contact/`, or the lane's `shots/<slug>/contact/` for a lane on D:), the builder's
   notes. A wrong catalog promise (name, hook, inputs) is fixed in `manifest.catalog`, not in the packet.
2. **Read the lesson** (10 min): `lesson <slug> print` on three or more fixtures, en and zh. Tick the 10-item checklist
   (teaching-bar.md) with the step id of each finding.
3. **Targeted clicks** (5 min): the predict and task steps, any interaction, anything the sheets leave in doubt; your
   own dev server on the package's port (or the kept one).
4. **Fix inside the package** (5-8 min): narration, look lines, Chinese, claims, a view's text. Only the package's
   owned paths.
5. **One final `check <slug> --quick`** (no build, no browser), then set `quality: 'reviewed'` and write the verdict
   ("ship", or what blocks it) in the packet's reviewer notes, with each finding, its step and the fix.

No e2e and no full check after an edit: `check <slug> --quick` while fixing and once at the end.

## 3. Integrator (main session)

Only one workflow at a time edits or merges into the main tree.

1. Note the free disk space.
2. For each lane: `merge <lane> --dry-run` (ownership OK, nothing outside the owned paths). `merge` the lanes whose
   owned files do not overlap; copy a file that two lanes changed by hand.
3. `promote <slug> --apply` per package: it writes the patch from `manifest.catalog` into `models.ts` or `concepts.ts`
   (CRLF kept), `tech-zh.ts` (rows added, rows left unused removed) and `glossary.ts` (only terms another page also
   uses), appends the credits to `licenses.json` and runs check:catalog. Read its "check by hand" lines first (sizes,
   inputs, mode, similar tech-zh rows); `--dry-run` prints the same without writing. `npm run atlas -- glossary` lists
   site terms only one package uses, when the flagship budget (300 KB) runs short.
4. `npm run check`, `npm run check:atlas`, `npm run check:catalog`, `node scripts/atlas/docs.ts --check` (and
   `--write` when the kit added an op, view, command, code or port; a changed DEVKIT section needs its Chinese summary
   hash in `src/lib/devkit-site/summaries.ts`, which `node src/lib/devkit-site/check.ts` names).
5. `check <slugs> --quick`; one `check --all` without e2e slugs (one full build, every package's fast gates, the kit
   gates and the legacy parity scripts); trust the lanes' e2e. A batch that also changed the kit or the layout names
   about 6 packages of different kinds instead (`check --all <slug> <slug> ...`, the default passes), never every
   package. AK603 never blocks; never re-run a gate that already passed.
6. One commit per package (the package, its assets, `pipeline/<slug>/`, its notes and review), kit changes in separate
   commits, the requested author and the Co-Authored-By line.
7. A LOG.md entry (what merged, measured numbers, open items); delete the merged lanes (`lanes --rm <name>`) and old
   builds; note the free space again. Nothing is pushed until the user says so; a full-site browser sweep (one
   screenshot per page) runs only before a public release.

## 4. Notes and reports

`notes/<slug>.md` (paperwork gate, AK501):

- **What**: the model or idea, the kinds and sources, the lesson's steps in one line each.
- **Decisions**: what you chose and why (data, size, steps cut, views reused or written).
- **Final numbers**: measured only (accuracy, parity, sizes, page lines), with the command that measured them.
- **How to re-run**: the exact commands.
- **Kit issues**: what the kit could not do, the workaround inside the package, and a proposed kit fix.

A workflow agent's report: a plain summary; acceptance items with measured numbers (gates, parity, timings); the files
changed; what the docs must say about kit changes; open issues, each specific enough to act on.
