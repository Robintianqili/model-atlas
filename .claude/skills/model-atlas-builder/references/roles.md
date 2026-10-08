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
| Look | 3-5 min | `e2e <slug> --axe warn`; read the contact sheets, `lesson-print.txt`, `axe.txt`; fix |
| Self-review and simplify | 3-5 min | the teaching bar (teaching-bar.md) as a learner; the simplify pass on your changes |
| Finish | 5 min | the one full `check <slug>`; `review <slug>`; `notes/<slug>.md`; `e2e --stop-server` |

- The builder is also the first reviewer: ship nothing you would not sign off as a first-year student reading it.
- Write down what you could not do and why ("Kit issues", open issues). Never hide a workaround.
- Long GPU training runs in the background in resumable chunks (every shell command stays under 9 minutes); write the
  lesson meanwhile.

## 2. Reviewer (25-30 min)

The reviewer is a teaching reviewer reading as a first-year student who knows school algebra. Start from what the
builder's tools already produced; open the page only for what they cannot show.

1. **Read the packet** (5 min): `notes/review-<slug>.md` (checklist, gates, contact files listed first),
   `notes/shots/<slug>/contact/<pass>.png`, `lesson-print.txt`, `axe.txt`, the builder's notes.
2. **Read the lesson** (10 min): `lesson <slug> print` on three or more fixtures, en and zh. Tick the 10-item checklist
   (teaching-bar.md) with the step id of each finding.
3. **Targeted clicks** (5 min): the predict and task steps, any interaction, anything the sheets leave in doubt; your
   own dev server on the package's port (or the kept one).
4. **Fix inside the package** (5-8 min): narration, look lines, Chinese, claims, a view's text. Only the package's
   owned paths.
5. **One final full check** (`check <slug>`), then set `quality: 'reviewed'` and write the verdict ("ship", or what
   blocks it) in the packet's reviewer notes, with each finding, its step and the fix.

Do not re-run the full check after every edit: `check <slug> --quick` while fixing, the full check once at the end.

## 3. Integrator (main session)

Only one workflow at a time edits or merges into the main tree.

1. Note the free disk space.
2. For each lane: `merge <lane> --dry-run` (ownership OK, nothing outside the owned paths). `merge` the lanes whose
   owned files do not overlap; copy a file that two lanes changed by hand.
3. `promote <slug>` per package; apply the printed catalog patch (`models.ts` or `concepts.ts`, `tech-zh.ts`,
   `glossary.ts`; reuse the wording of a similar tech-zh row); credits are appended to `licenses.json`.
4. `npm run check`, `npm run check:atlas`, `npm run check:catalog`, `node scripts/atlas/docs.ts --check` (and
   `--write` when the kit added an op, view, command, code or port; a changed DEVKIT section needs its Chinese summary
   hash in `src/lib/devkit-site/summaries.ts`, which `node src/lib/devkit-site/check.ts` names).
5. `check <slugs> --quick`; one `check --all --e2e <slug>,<slug>` (one full build, the five `all` passes); the legacy
   parity scripts run inside it.
6. One commit per package (the package, its assets, `pipeline/<slug>/`, its notes and review), kit changes in separate
   commits, the requested author and the Co-Authored-By line.
7. A LOG.md entry (what merged, measured numbers, open items); delete the merged lanes and old builds; note the free
   space again. Nothing is pushed until the user says so.

## 4. Notes and reports

`notes/<slug>.md` (paperwork gate, AK501):

- **What**: the model or idea, the kinds and sources, the lesson's steps in one line each.
- **Decisions**: what you chose and why (data, size, steps cut, views reused or written).
- **Final numbers**: measured only (accuracy, parity, sizes, page lines), with the command that measured them.
- **How to re-run**: the exact commands.
- **Kit issues**: what the kit could not do, the workaround inside the package, and a proposed kit fix.

A workflow agent's report: a plain summary; acceptance items with measured numbers (gates, parity, timings); the files
changed; what the docs must say about kit changes; open issues, each specific enough to act on.
