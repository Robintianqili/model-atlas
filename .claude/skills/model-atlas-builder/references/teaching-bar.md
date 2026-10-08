# The teaching bar

The site is a teaching tool first: the lesson is the product. A builder self-reviews against this page before the one
full check; the reviewer ticks the same checklist in `notes/review-<slug>.md`. Read the lesson as a first-year student
who knows school algebra would, in `lesson <slug> print` (or `contact/lesson-print.txt`) and next to the contact sheets.

## Contents

1. The shape of a lesson
2. The review checklist (10 items)
3. Writing the narration
4. Chinese
5. Findings reviewers make most often

## 1. The shape of a lesson

- 5-12 steps; each has one idea, a title that names it, 2-4 sentences of narration (at most 90 English words, also once
  the placeholders are filled in) and a "look" line that points at one thing in the stage.
- At least one **predict** step: the learner commits to an answer before the stage shows it. The narration, look and
  title set the question up; the answer and why go in `predict.explain` (AK412 warns when they give it away).
- At least one **task**: the learner does something (draws, edits a point, moves a slider, picks a preset) and the
  page reacts. `task.done(rt, at)` decides when it is done; `waiting` and `doneText` say what to do and what happened.
- One **honest-limits** step (`kindOf: 'limits'` or `'breakIt'`) that shows, with live numbers, where the model fails
  or what it cannot know.
- The first occurrence of an idea is worked out fully (every product of one weighted sum); repeats are summarized.
- Every step works from its deep link (`#lesson-<id>`) as the first step a learner opens: it runs the sources it needs.
- The page's goals are measurable ("after this page you can say why ..."), and each goal has a step that shows it.

## 2. The review checklist (10 items)

1. Measurable goals: each goal says what the learner can do after the page.
2. One idea per step, each with a focused "look at" hint.
3. No term is used before it is defined (glossary terms on first use).
4. Narration numbers match the screen on 3 presets.
5. Every factual sentence is live or covered by a claim test (`claims.test.ts`).
6. The first occurrence of an idea is worked out fully; repeats are summarized.
7. A misconception or limits step tells the honest limit.
8. Try-this experiments ask for a prediction first.
9. Natural Chinese with glossary terms (no machine-translation tone, no English left over).
10. Credits are correct and complete (`static/licenses.json` rows).

The verdict is "ship" (set `quality: 'reviewed'`) or a list of what blocks it, each with the step id and the fix.

## 3. Writing the narration

- **Numbers.** Every number is a placeholder: `{name}` from `vars(rt, at)`, a built-in path (`layer.<id>.max`,
  `top.label`, `frame.<metric>`, `opt.<name>`, ...) or `{fact:<key>}`. Compute in `vars()`; the template has no
  expressions. Single digits in words are fine; a typed multi-digit number is AK402.
- **Small changes.** Three significant digits can hide a change ("falls from 0.699 to 0.699"): narrate the difference
  (`drop: a - b`) or give both numbers more digits (`{a|n4}`) (AK410).
- **Settings as set.** A learning rate of 0.05 prints "0.05", not "0.0500": `fmtSetting` in a formula slot or `vars()`.
- **Counting.** Indices count from 0 as in PyTorch and say "(counting from 0)"; counters count from 1 ("Step 3 of 9").
  In Chinese "第 5 个" is the 6th when 5 counts from 0: write "编号 5（从 0 数起）".
- **Plurals and verdicts.** A sentence that changes with the numbers uses `plural`, `num` and `pick` (`say.ts`), so "1
  of its 16 numbers is 0" and "3 of its 16 numbers are 0" both read right.
- **The empty input.** A step the empty input skips keeps only its sentences without missing values: state the idea in
  sentences without numbers and the numbers in sentences of their own.
- **Directions and colors.** Say "in the stage", "the first panel", "the outlined cell". Never "above", "below", "left"
  or "right" (the layout changes on a phone), and never "darker means more" (dark mode inverts brightness): say
  "stronger color".
- **Claims.** A sentence that is true only for this run, this model or these inputs is either computed live or backed by
  a claim test that runs on every fixture. "Real models do X" without a test is cut or softened to what the test
  shows.
- **Plain words.** Short sentences, one term per idea, each symbol defined where it first appears (Q(s, a), ε, σ, p and
  q, −log π, "feature"), the same word never in two senses ("log" as a record and as a function).

## 4. Chinese

- Natural Chinese a native teacher would write, not a translation: reorder for Chinese, cut what Chinese says without.
- Use the site glossary's terms (`src/lib/catalog/glossary.ts`; 独热编码, 嵌入, 点积, 梯度下降, 批内负样本, ...);
  keep Latin names the glossary keeps (softmax, ReLU).
- The same placeholders as the English (AK403); no English left in a zh string except names, symbols and code.
- Every title, label, goal and predict text has zh (AK406); narration needs zh before `reviewed`.
- If no native speaker has read the Chinese yet, say so in the notes; do not claim it.

## 5. Findings reviewers make most often

The ones that keep coming back in the batch reviews (`notes/review-*.md`):

1. A symbol or term used before it is defined, or a symbol defined only in the stage.
2. A narrated number the stage does not show, or that differs from it (rounding, a different fixture, a stale run).
3. A direction or color word that is wrong on a phone or in dark mode.
4. A sentence claiming more than its claim test checks ("never", "always", "real models ...").
5. Two steps that seem to contradict each other (a best of 23.5 in one step, an average of 24.2 in the next) without a
   word on why.
6. A look line pointing at the wrong place, or at nothing in particular.
7. Padded settings ("σ = 0.0400") next to a slider that says 0.04.
8. A 0-based index without "counting from 0", or a Chinese ordinal that is off by one.
9. A task that breaks once the learner does it: a new input restarts the sources, so a view must read the current tape,
   not an index it kept.
10. A narration over 90 words once the placeholders are filled in, or a live sentence whose grammar breaks at 1.
