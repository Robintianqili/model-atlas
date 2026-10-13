// DevKit frozen types (research/devkit/DEVKIT_DESIGN.md §4.7): package modules, runner, runtime.
// Frozen during waves 1 and 2: changes go through the main session.
import type { AnyTensor, FetchLike, I18n, Provenance, Tensor, Trace } from './core.ts';
import type { AtlasSpec } from './spec.ts';
import type { GraphModel, OpDef } from './op.ts';
import type { Cursor, LiveTape, ProcessDef, Tape } from './process.ts';
import type { ExplainerManifest, InputAdapter } from './explainer.ts';
import type { LessonDef } from './lesson.ts';
export interface Preprocessed { tensors: Record<string, AnyTensor>; steps: { id: string; label: I18n; tensor?: Tensor; note?: I18n }[]; empty: boolean; notes?: I18n[]
  /** Revision 1b (additive), text adapters: the display text of each input position, and of any vocabulary id (views show
   *  them instead of '#id'; text-bpe fills both when it has the tokenizer). */
  tokens?: string[]; vocab?: string[] | ((id: number) => string)
  /** K4 (additive): bilingual display names of the input positions (items, clicks, nodes: what a non-text model's rows
   *  are), preferred to `tokens` by the kit views in the page's language */
  names?: I18n[] }
export type SourceDef =
  | { kind: 'trace'; inputs?: string[]; assets?: string }   // assets: another model's folder under static/ ('models/tinycnn'), read via legacy.ts
  | { kind: 'scan'; of: string; layer: string | ((rt: ExplainerRuntime) => string); unit?: (rt: ExplainerRuntime) => number[]; order?: 'raster' | 'peak-first'; stride?: number }
  | { kind: 'train'; trainer: string; opt?: Record<string, unknown>; detail?: 'coarse' | 'fine'; precomputed?: string; recordDraws?: boolean   // recordDraws: revision 1b
      /** K1 (additive): run while the adapter says the input is empty too (default: a train source does not run on an
       *  empty input, so its steps show the adapter's empty note instead of an untrained 'init' frame) */
      runOnEmpty?: boolean }
  | { kind: 'process'; process: string; input?: (rt: ExplainerRuntime) => unknown; opt?: Record<string, unknown>; seed?: number; precomputed?: string; recordDraws?: boolean
      /** K1: 'main' records on the main thread (its input may hold what a worker cannot receive, such as a model);
       *  'worker' or unset: in the package worker when there is one. Before K1 the browser ignored 'main'. */
      run?: 'main' | 'worker';
      /** K0 (additive): RunCtx.detail of the run (default 'coarse'); a process that wraps trainingProcess passes it on */
      detail?: 'coarse' | 'fine';
      /** K1 (additive): open only after this source (train or process) has run to its end, so input(rt) can read its
       *  frames, e.g. the trained weights (rt.sources[after].tape.state(last).params) */
      after?: string;
      /** K1 (additive): false = atlas parity runs no Python reference for this source (its badge then says "Live") */
      reference?: boolean }
  | { kind: 'tape'; file: string }                                                               // precomputed only
  | { kind: 'live-model'; of: string
      /** K1 (additive): 'first' traces the train source's starting weights (its init frame); default 'last' (the latest) */
      at?: 'first' | 'last' };                                                                   // kind-A trace on a train source's weights
export interface SourceState { kind: SourceDef['kind']; status: 'idle' | 'loading' | 'running' | 'ready' | 'error'; provenance: Provenance;
  trace?: Trace | null; tape?: Tape | LiveTape | null; model?: GraphModel | null; progress?: { at: number; target: number }; error?: string }
export interface SelectionState { layer?: string; unit?: number[]; ch?: number; head?: number; t?: number; node?: number; [k: string]: unknown }
/** Everything a package contributes, already imported. Built by ui/packages.ts (Vite globs) in the browser and by
 *  scripts/atlas/load.ts (dynamic import by path) in Node. Both loaders wrap every trainer.ts export with trainingProcess()
 *  and register it as process 'train:<name>', so the runtime itself only ever runs ProcessDefs. */
/** K0 (additive): `atlas parity --write`'s record of a package's twins (kind B) and references (kind C), by source name:
 *  explainers/<slug>/parity.json (format atlas-parity/1). The stage badge of a train or process step reads it. */
export interface SourceParity { scope: 'train' | 'process'; against: string; pass: boolean; max: number; bar?: number; relative?: boolean;
  steps?: number; frames?: number; fixtures?: number; checked: string }
export interface ParityFile { format: 'atlas-parity/1'; slug: string; checked: string; sources: Record<string, SourceParity> }
export interface PackageModules {
  slug: string; manifest: ExplainerManifest; lesson: LessonDef<any> | null;
  ops: OpDef[];                                          // the package's ops.ts default export, else CORE_OPS
  processes: Record<string, ProcessDef<any, any, any>>;
  adapters: Record<string, InputAdapter>;                // kit adapters + the package's local ones
  worker?: () => Worker;                                 // browser only: the package's worker.ts (Vite ?worker)
  parity?: ParityFile;                                   // K0 (additive): explainers/<slug>/parity.json, when atlas parity wrote one
}
export interface OpenSource { slug: string; source: string; process: string; input: unknown; opt: Record<string, unknown>; seed: number; detail: 'coarse' | 'fine'; recordDraws?: boolean
  /** K1 (additive): the source's `run`; 'main' keeps the run on the main thread (engine runner) */
  run?: 'main' | 'worker' }
/** Runs 'train' and 'process' sources.
 *  localRunner (runtime/local.ts, W1-C): in-process with recorder(); used in Node, tests, and for run: 'main'.
 *  engineRunner (engine/runner.ts, W1-B2): in pkg.worker(); frames stream into a TapeStore on the main thread. */
export interface SourceRunner { open(o: OpenSource, pkg: PackageModules): LiveTape
  /** Revision 1b (additive): stop every worker this runner started (engineRunner has it; localRunner needs none). */
  dispose?(): void }
export interface ExplainerRuntime {
  slug: string; spec: AtlasSpec | null; model: GraphModel | null;
  input: unknown; pre: Preprocessed | null;
  sources: Record<string, SourceState>;
  sel: SelectionState;                                   // the lesson's own selection
  fact(path: string): number | string | I18n | undefined; // built-in paths (§5.3) + facts.json
  num(path: string): number;                             // NaN when missing or not a number (lint AK401)
  ensure(source: string, c: Cursor): Promise<number>;    // load/run until the cursor exists; returns a clamped frame index
  setInput(raw: unknown): void;                          // re-runs sources; steps re-resolve their cursors
  /** K4 (additive): the input presets the runtime loaded (the presets file, or the adapter's presets with
   *  input.presetsFromAdapter); the runtime always filled it, the type did not say so */
  readonly presets?: readonly { id: string; label: I18n; raw: unknown }[];
}
// W1-C: runtime/runtime.ts createRuntime(pkg: PackageModules, o: { fetch: FetchLike; base: string; input?: unknown; runner?: SourceRunner }): Promise<ExplainerRuntime>
//       runtime/fake.ts fakeRuntime(partial): ExplainerRuntime (view demos, tests)
