// DevKit frozen types (research/devkit/DEVKIT_DESIGN.md §4.6): lesson, steps, narration.
// Frozen during waves 1 and 2: changes go through the main session.
import type { I18n, PredictQuestion, Provenance } from './core.ts';
import type { Cursor, Focus, Frame, Tape } from './process.ts';
import type { UnitExplanation } from './op.ts';
import type { ExplainerRuntime, SelectionState } from './runtime.ts';
export interface At { index: number; frame?: Frame; state?: unknown; tape?: Tape; unit?: UnitExplanation; layer?: string
  /** K1 (additive): the step's source name; {opt.<name>} resolves from it (its options and its process defaults) even
   *  before the source has run */
  source?: string
  /** K2 (additive): the empty input skipped the step (its source did not run): the narration leaves out the sentences
   *  whose values do not exist yet instead of printing "…" (lesson/narrate.ts render) */
  skipped?: boolean }
/** Template: {name|fmt}. Names resolve in order: vars() result; built-in paths (§5.3); fact:<key> (facts.json). No expressions.
 *  fmt: n3 (3 significant digits; default for numbers) | int | pct | signed | shape | label | tok.
 *  en and zh must use the same set of names (AK403). An unresolved or non-finite value renders "…" and fails lint (AK401). */
/** K0 (additive): an I18n value is printed as is in each language (plural forms, conditional sentences: lesson/say.ts). */
export type TmplVal = number | string | I18n | { v: number; fmt?: string; focus?: Focus };
export type Tmpl<R> = I18n & { vars?: (rt: R, at: At) => Record<string, TmplVal> };
export type Narration<R> = I18n | Tmpl<R>;
export interface StepDef<R extends ExplainerRuntime = ExplainerRuntime> {
  id: string; title: I18n; narration: Narration<R>; look?: Narration<R>;
  source?: string;                                       // default: the explainer's first source
  at?: Cursor | ((rt: R) => Cursor);
  select?: (rt: R) => Partial<SelectionState>;           // the lesson's own selection (never the explore view's)
  view: string | { key: string; props?: (rt: R, at: At) => Record<string, unknown> };
  play?: 'view' | { from: Cursor; to: Cursor; interval?: number; phases?: string[]
    /** K0 (additive): what the scrubber's marks count ('episode'), and which frame field holds the count (default 'step') */
    unit?: I18n; by?: 'step' | 'sub' };   // 'view': the view animates itself (autoplay + onend)
  focus?: (rt: R, at: At) => Focus[];
  provenance: Provenance | { mode: Provenance; note: I18n };
  predict?: PredictQuestion;                             // stage dimmed until answered; Play waits
  /** K4 (additive): the live parts of the predict question, from the learner's own input: options (the codes this patch
   *  may snap to), the answer index (which expert takes this token) and an explanation with live numbers (shown only
   *  after the answer, so it gives nothing away); merged over `predict`, which still holds the question */
  predictAt?: (rt: R, at: At) => Partial<Pick<PredictQuestion, 'options' | 'answer' | 'explain'>>;
  task?: { prompt: I18n; done: (rt: R, at: At) => boolean; hint?: I18n
    /** K0 (additive): the status line until done (default: 'Not yet: do what the task says.') */
    waiting?: I18n
    /** K1 (additive): the status line once done (default: 'Done.') */
    doneText?: I18n };
  compare?: { a: string; b: string; label: [I18n, I18n] };
  kindOf?: 'limits' | 'breakIt';                         // marks the honest-limits step the lesson shape rule asks for
  terms?: string[];
  waitForDone?: boolean; linger?: number; duration?: number;
  /** K4 (additive): lint options of this step: same lists placeholder names that may print the same as another value
   *  (AK410 stays quiet for them: two unrelated quantities that happen to coincide on a fixture) */
  lint?: { same?: string[] };
  /** R1 (redesign W2, additive): the "Why?" disclosure under the narration: a question as its summary (shown as is) and a
   *  short answer (a narration template, at most 120 words; AK417) */
  why?: { q: I18n; a: Narration<R> };
  /** R1 (additive): number-chip targets: a placeholder written {name@ref} names one of these; hover, focus or tap on its
   *  number moves the stage's ring to that focus */
  refs?: (rt: R, at: At) => Record<string, Focus>;
}
/** Auto step ids: 'input' | 'op:<layer>' | 'group:<id>' | 'output' | 'train:<motif>@<source>' | 'proc:<motif>@<source>'
 *  (Wave-0 deviation from §4.6: R is constrained to ExplainerRuntime because Partial<StepDef<R>> requires it.) */
export type AutoRef<R extends ExplainerRuntime> = { auto: string; override?: Partial<StepDef<R>> } | { autoAll: { from?: string; until?: string; skip?: string[] } };
export interface FixtureDef { id: string; label: I18n; raw: unknown; expect?: { empty?: boolean } }
export interface LessonDef<R extends ExplainerRuntime = ExplainerRuntime> {
  id?: string;                                           // deep-link prefix, default 'lesson'
  title: I18n; summary: I18n;                            // summary -> catalog lesson.summary (promote)
  steps: (StepDef<R> | AutoRef<R>)[];
  fixtures?: FixtureDef[];                               // adapter edge fixtures are always added
  /** R1 (redesign W2, additive): 2-4 chapters of 1-6 steps each, every step in exactly one chapter, in order (AK413).
   *  The rail labels them and the eyebrow reads "<chapter> · step k of n". */
  chapters?: { id: string; title: I18n; steps: string[] }[];
  /** R1 (additive): up to 3 facts the learner saw, with their real numbers, for the end card (templates over the last
   *  step's moment; default: the chapter titles) */
  recap?: Narration<R>[];
}
export interface CompiledStep {
  id: string; origin: 'auto' | 'author'; autoId?: string;
  title: I18n; narration(rt: ExplainerRuntime, at: At): I18n; look?(rt: ExplainerRuntime, at: At): I18n | undefined;
  source: string; at(rt: ExplainerRuntime): Cursor; view: string; props(rt: ExplainerRuntime, at: At): Record<string, unknown>;
  play?: StepDef['play']; focus(rt: ExplainerRuntime, at: At): Focus[];
  provenance: { mode: Provenance; note?: I18n }; select?(rt: ExplainerRuntime): Partial<SelectionState>;
  predict?: PredictQuestion; task?: StepDef['task']; compare?: StepDef['compare']; kindOf?: StepDef['kindOf']; terms: string[];
  /** K4: StepDef.predictAt (the live options, answer and explanation) */
  predictAt?(rt: ExplainerRuntime, at: At): Partial<Pick<PredictQuestion, 'options' | 'answer' | 'explain'>>;
  waitForDone: boolean; linger?: number; duration?: number;
  /** R1: StepDef.why with its answer rendered */
  why?: { q: I18n; a(rt: ExplainerRuntime, at: At): I18n };
  /** R1: StepDef.refs */
  refs?(rt: ExplainerRuntime, at: At): Record<string, Focus>;
  /** R1: the narration and the look line with their {name@ref} values in chip marks (lesson/narrate.ts chipParts);
   *  present only when a template names a ref */
  chips?: { narration?(rt: ExplainerRuntime, at: At): I18n; look?(rt: ExplainerRuntime, at: At): I18n };
}
export interface LintIssue { code: string; level: 'error' | 'warn'; step?: string; fixture?: string; lang?: 'en' | 'zh'; message: string; fix?: string }
// W1-C: lesson/define.ts defineLesson, auto, autoAll, defineExplainer;  lesson/plan.ts planLesson(spec | null, ops, manifest) -> StepDef[]
//       lesson/compile.ts compileLesson(def, { spec, ops, manifest }) -> { id: string; steps: CompiledStep[]; issues: LintIssue[] }
//       lesson/narrate.ts render(tmpl, rt, at, lang) -> { text: string; placeholders: { name: string; value: unknown; focus?: Focus }[] }
//       lesson/lint.ts lintLesson(compiled, makeRuntime, fixtures) -> LintIssue[]   (Node and the Lesson Lab)
//       lesson/fixtures.ts withFixtures(pkg, fn) — the helper claims.test.ts files use
