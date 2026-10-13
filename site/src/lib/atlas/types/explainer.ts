// DevKit frozen types (research/devkit/DEVKIT_DESIGN.md §4.9): the manifest and input adapters.
// Frozen during waves 1 and 2: changes go through the main session.
import type { I18n, Kind, PredictQuestion } from './core.ts';
import type { ExplainerRuntime, SourceDef, Preprocessed } from './runtime.ts';
import type { NarrationSet } from './op.ts';
/** K0: the auto page's sections (ModelPageShell keys) a manifest can title, introduce or hide. */
export type PageSection = 'input' | 'lesson' | 'overview' | 'inspector' | 'unit' | 'output' | 'tryThis' | 'training' | 'article' | 'code';
export interface Credit { name: string; url: string; license: string; usedFor: string; attribution: string }   // licenses.json row
export interface TryThisDef { id: string; title: I18n; setup: { preset?: string; input?: unknown; opt?: Record<string, unknown> }; predict: I18n; look: I18n; goto?: { step?: string; layer?: string } }
/** K2: which row of a sequence output [T, V] is the model's one answer: 'last' (default; a causal language model's next
 *  token), 'first', a row index (negative: from the end), or a function of the runtime (a masked model's [MASK] row). */
export type AnswerRow = 'last' | 'first' | number | ((rt: ExplainerRuntime) => number);
export interface QuizDef { q: I18n; answerFact?: string; tolerance?: number; options?: I18n[]; answer?: number; explain: I18n }
export interface ExplainerManifest {
  slug: string; type: 'model'; kinds: Kind[];
  quality: 'draft' | 'reviewed' | 'flagship';
  /** lab only (hello-xor, demo-*): exempt from the catalog rules (AK701), never routed under /m/ */
  devOnly?: boolean;
  assets?: { dir: string; auto: string[]; lazy?: Record<string, string> };  // dir relative to static/; auto = loaded on every visit
  input?: { adapter: string; presets?: string; start?: unknown; options?: Record<string, unknown>
    /** K2 (additive): no presets file: the adapter's presets({}) makes them in TS, and they reach rt.presets, the Capture
     *  and the lint fixtures as a file's would (before, TS-made presets needed a dummy presets.json) */
    presetsFromAdapter?: boolean };
  inputs?: { name: string; adapter: string; options?: Record<string, unknown> }[];   // multi-input (TinyCLIP)
  /** K2 (additive): view is optional; row is the answer row of a [T, V] output (the side panel's answer, top.*, the auto
   *  output step, kit.probbars and compare read it; default 'last') */
  output?: { view?: string; layer?: string; props?: Record<string, unknown>; row?: AnswerRow };
  sources?: Record<string, SourceDef>;                   // default { fwd: { kind: 'trace' } } when kinds has 'trace'
  overview?: { layout: 'pipeline' | 'stack'; collapse?: string[] };
  page: { goals: I18n[]; prequestion: PredictQuestion; tryThis?: TryThisDef[]; quiz?: QuizDef[];
          misconception?: { claim: I18n; truth: I18n; step?: string }; lessonNote?: I18n;
          /** K0 (additive): section titles and intros of the auto page, and sections to leave out. Without a spec
           *  (kinds B and C) the model sections (overview, inspector, unit, output, code) are left out unless a
           *  sections/<Snippet>.svelte override fills them. */
          sections?: { titles?: Partial<Record<PageSection, I18n>>; intros?: Partial<Record<PageSection, I18n>>; hide?: PageSection[] };
          /** K1 (additive): names and formats of training metrics (spec.training.metrics, a train source's metrics) in the
           *  auto Training section: { test_acc: { label: { en: 'Right on the test set', zh: '…' }, fmt: 'pct' } } */
          metrics?: Record<string, { label: I18n; fmt?: 'pct' | 'int' | 'n3' | 'n4' }>
          /** K2 (additive): the loss-curve props of the auto Training section (default: the loss): { metric: 'meanLen' },
           *  { metrics: ['lossD', 'lossG'], refs: [{ y: 1.386, label }] }, labels, log, smooth. A metric recorded only at
           *  snapshot frames (a metrics() value) draws through them. */
          trainingCurve?: { metric?: string; metrics?: string[]; labels?: Record<string, I18n>; log?: boolean | 'auto'; smooth?: number; refs?: { y: number; label?: I18n }[] };
          /** R1 (redesign wave F, additive): the step whose stage is the lesson's thumbnail in the catalog (`atlas thumbs`).
           *  Default: the first step whose view is a figure (not a text, table or tape view). */
          poster?: { step: string };
          /** R1 (additive): the cover's stage (step 0, answer first): a view key and the source it reads. Default: the layer
           *  strip for a package with a spec, else the poster step's stage. */
          cover?: { view: string; source?: string } };
  credits: Credit[];                                     // promote appends to static/licenses.json
  glossary?: { id: string; term: I18n; def: I18n; related?: string[] }[];   // promote prints the glossary.ts patch
  /** promote prints the models.ts patch, plus tech-zh.ts entries for runtime/params/size. K4 (additive): name (the
   *  page's h1 and title), family, inputs (catalog InputKinds) and mode, so the whole entry comes from the package and
   *  `promote --verify` / `--apply` can apply it; the block is data only the CLI reads (not counted as page code). */
  catalog?: { hook?: I18n; shows?: I18n; runtime?: I18n; params?: I18n; size?: I18n; prereqs?: string[]
    name?: I18n; family?: I18n; inputs?: string[]; mode?: 'live' | 'precomputed' | 'both'
    /** R1 (redesign wave F, additive): a short name for path stations (at most 24 characters). `hook` (at most 100 English
     *  and 50 Chinese characters) is copied for concepts too. lesson.minutes is the reading time the content wave sets
     *  (round(words / 180 + steps x 20 s + beats x 30 s)). */
    short?: I18n; lesson?: { minutes?: number } };
  budget?: { autoLoadBytes?: number };
}
/** Input adapters: atlas/adapters/<id>/adapter.ts (pure) + Capture.svelte. kind is a catalog InputKind. */
export interface InputAdapter<Raw = unknown> {
  id: string; kind: string;
  preprocess(raw: Raw, spec: unknown, opts: Record<string, unknown>): Preprocessed;
  presets?(assets: Record<string, unknown>): { id: string; label: I18n; raw: Raw }[];
  edgeFixtures(): { id: string; label: I18n; raw: Raw }[];   // drawing: blank, dot, small; text: '', 1 token, > context; points: 0, one class; series: flat; graph: isolated node
  layer0: { view: string; narration: NarrationSet };
}
/** Revision 1b (additive): the props of an adapter's Capture.svelte (kit adapters/<id>/ and package adapters/<id>/).
 *  The page (ui/AutoModelPage.svelte) passes rt, raw, presets (from the manifest's presets file; may be empty),
 *  options (manifest.input.options), onchange and compact. A Capture edits the learner's raw input and reports it with
 *  onchange(raw); { live: true } marks a change in progress (a stroke being drawn), which the page coalesces per frame.
 *  An empty presets list means "none given": a Capture may fall back to its adapter's generated presets. */
export interface CaptureProps<Raw = unknown> {
  rt?: ExplainerRuntime; raw?: Raw;
  presets?: { id: string; label: I18n; raw: Raw }[];
  options?: Record<string, unknown>;
  onchange?(raw: Raw, o?: { live?: boolean }): void;
  compact?: boolean;                                     // the small copy in the lesson's side panel
  label?: I18n;
}
