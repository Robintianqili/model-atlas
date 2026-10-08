// DevKit frozen types (research/devkit/DEVKIT_DESIGN.md §4.8): view registry and stage contract.
// Frozen during waves 1 and 2: changes go through the main session.
import type { I18n, Kind, Provenance } from './core.ts';
import type { At } from './lesson.ts';
import type { Focus } from './process.ts';
import type { ExplainerRuntime, SelectionState } from './runtime.ts';
/** views/<key>/meta.ts (pure TS, readable by Node lint); the component is views/<key>/View.svelte (lazy). */
export interface ViewMeta {
  key: string;                                    // 'zoom.weighted-sum', 'plane2d'; local views register as '<slug>/<key>'
  title: I18n; kinds: Kind[];
  accepts?(o: { field?: string; zoom?: string; axes?: string[] }): boolean;   // default view per field/zoom/axes
  describe(p: StageProps): I18n;                  // text summary: live region + "show as table"
  /** demo props for the gallery and the Lesson Lab, built on runtime/fake.ts. Contract (types/contract.test.ts):
   *  resolves in Node, describe(demo) is non-empty in en and zh, provenance is 'illustrative'. */
  demo(): Promise<Omit<StageProps, 'ctx'>>;
  minHeight?: number;                             // the stage reserves it (no layout jump)
}
export interface StageProps {
  rt: ExplainerRuntime; source: string; at: At;
  focus: Focus[]; sel: SelectionState;
  ctx?: { shown: boolean; reduced: boolean; playing: boolean; done(): void };   // kit WalkContext subset
  provenance: Provenance; autoplay: boolean; onend: () => void;
  props: Record<string, unknown>;
  /** K0 (additive): change the learner's input from the stage (a maze editor, a 'use these points' button). The page re-runs
   *  every source and the lesson re-resolves its steps, exactly as after an edit in the input widget. */
  setInput?(raw: unknown): void;
  /** K1 (additive): one object per lesson step, shared by the views of that step (the two panels of a step's compare
   *  render the view twice: a toggle kept here changes both). Reactive in the browser; {} in Node. */
  shared?: Record<string, unknown>;
}
// W1-D1: ui/registry.ts — metas via import.meta.glob (inside AutoModelPage's lazy chunk only), components lazy;
//        package views from '$lib/explainers/*/views/*/meta.ts' registered as '<slug>/<key>'.
