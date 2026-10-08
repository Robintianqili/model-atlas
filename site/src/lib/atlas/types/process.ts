// DevKit frozen types (research/devkit/DEVKIT_DESIGN.md §4.4): process, frame, tape.
// Frozen during waves 1 and 2: changes go through the main session.
import type { I18n, Provenance, Rng, Tensor } from './core.ts';
import type { AtlasSpec } from './spec.ts';
export interface FieldDef {
  kind: 'tensor' | 'scalar' | 'int' | 'index' | 'text' | 'json'; label: I18n;
  shape?: (number | string)[]; scale?: 'diverging' | 'sequential' | 'categorical';
  role?: 'activation' | 'param' | 'grad' | 'optimizer' | 'data' | 'target' | 'output';
  /** 'snapshot' (default): copied when present in a frame. 'changes': mutated in place; every edit must be reported in frame.changes */
  track?: 'snapshot' | 'changes';
  /** Derived fields are never stored, saved or sent between threads ("store causes, recompute effects"). They are recomputed
   *  (memoized) wherever the ProcessDef module is loaded: the main thread lazily imports the same trainer.ts/process.ts the worker runs. */
  /** K0 (additive): derive gets the run's options (the tape's opt: the defaults and the source's opt, as the run started) as a 2nd argument. */
  derive?: (s: any, opt?: any) => Tensor | number;
  tol?: number | 'exact';                               // parity tolerance for this field
}
export type Schema = Record<string, FieldDef>;
/** Schema without functions: what tape files and worker messages carry. */
export type SchemaJson = Record<string, Omit<FieldDef, 'derive'> & { derived?: boolean }>;
export interface PhaseDef {
  label: I18n;
  kind: 'input' | 'forward' | 'loss' | 'backward' | 'update' | 'observe' | 'act' | 'reward' | 'message' | 'aggregate' | 'sample' | 'merge' | 'output' | 'other';
  formula?: { tex: string; values: (f: Frame) => Record<string, number | string> };   // tex uses {{name}} and sym(); K2: a value may be a string (a setting as set: fmtSetting(0.05) = '0.05')
  narration?: { what: I18n; unit?: I18n };
}
export interface Change { field: string; key?: string; index?: number[]; from?: number; to?: number; note?: I18n }
/** key: an entry of a record-valued field (field 'params', key 'net.0.weight'). */
export interface Focus { field: string; key?: string; index?: number[] | number[][]; role?: 'in' | 'out' | 'param' | 'grad' | 'q' | 'k' | 'v' | 'accent'; label?: I18n }
export interface Frame<S = Record<string, unknown>> {
  phase: string; step?: number; sub?: number;
  state: Partial<S>; changes?: Change[]; focus?: Focus[];
  metrics?: Record<string, number>; caption?: I18n;
  /** every value returned by ctx.rng (and its forks) while making this frame, when RecordOpts.recordDraws is set */
  draws?: number[];
}
export type Control<O = Record<string, unknown>> =
  | { type: 'set'; opt: Partial<O> } | { type: 'input'; input: unknown }
  | { type: 'pause' } | { type: 'resume' } | { type: 'step'; n: number } | { type: 'reset'; seed?: number };
export interface RunCtx<O> { rng: Rng; opt: O; detail: 'coarse' | 'fine'; signal: { cancelled: boolean } }
/** Revision 1b (additive): the return type of ProcessDef.run, for generators whose frames differ in shape
 *  (`function* run(input, ctx): ProcessRun<S, O> { ... }`). process/define.ts defineProcess() infers I, O and S. */
export type ProcessRun<S = Record<string, unknown>, O = Record<string, unknown>> = Generator<Frame<S>, void, Control<O> | undefined>;
export interface ProcessDef<I = unknown, O = Record<string, unknown>, S = Record<string, unknown>> {
  id: string; kind: 'inference' | 'training' | 'algorithm'; title: I18n;
  phases: Record<string, PhaseDef>; schema: Schema; defaults: O;
  /** Pure: same input, opt and seed => same frames. 'set'/'input' controls arrive via gen.next(control). Ties break to the smallest index. */
  run(input: I, ctx: RunCtx<O>): Generator<Frame<S>, void, Control<O> | undefined>;
  /** Revision 1b (additive): record every ctx.rng draw in frame.draws by default (a source's opt.recordDraws or
   *  SourceDef.recordDraws also turns it on). For processes whose Python reference replays the draws. */
  recordDraws?: boolean;
  /** Revision 1b (additive): a kind-A spec + weights for a state of this process; 'live-model' sources trace it.
   *  train/trainer.ts trainingProcess() adds it (rebuilds the Module, loads state.params, Module.toSpec). */
  liveSpec?(state: S, opt?: Partial<O>): { spec: AtlasSpec; weights: Record<string, Tensor> };
  /** K1 (additive): atlas parity's tolerance for frame.metrics against the Python reference: one bar for every metric, or
   *  one per metric name ('exact', or a relative bar such as 1e-6). Default 1e-12 relative. */
  metricTol?: number | 'exact' | Record<string, number | 'exact'>;
  /** K2 (additive): the trainer whose frames this process makes. trainingProcess() sets it, and a process that spreads
   *  `...trainingProcess(t)` inherits it: when its frames equal the trainer's (train/wraps.ts checks), atlas parity runs
   *  the trainer's twin with the source's options, so its steps get the "= PyTorch fp64" badge. */
  trainer?: string;
}
/** `where` is evaluated on the thread that holds the tape (the main thread for worker sources); cursors are never sent to a worker. */
export interface CursorQuery { phase?: string; step?: number; sub?: number; layer?: string; last?: boolean; where?: (f: Frame<any>, s: any) => boolean }
export type Cursor = 'first' | 'last' | number | CursorQuery;
export interface Tape<S = any> {
  process: string; seed: number; opt: unknown; provenance: Provenance; version: string;
  readonly length: number;
  frame(i: number): Frame<S>;
  state(i: number): S;                                   // nearest keyframe + changes; pure in i
  find(c: Cursor): number;                               // −1 if not recorded (yet); callers clamp
  segments(): { phase: string; from: number; to: number }[];
  metric(name: string): Float64Array;
  /** Revision 1b (additive): the process's phases (labels, kinds, formulas); a tape loaded from a file has tex only.
   *  TapeStore, the recorder and the engine's tapes all have it (zoom.formula, tape.table and the scrubber read it). */
  readonly phases?: Record<string, PhaseDef | { label: I18n; kind: PhaseDef['kind']; tex?: string }>;
}
export interface LiveTape<S = any> extends Tape<S> {
  readonly done: boolean;
  /** resolves with the index of the first frame matching c once it is recorded, or the last index if the run ends without it */
  ensure(c: Cursor, o?: { budgetMs?: number }): Promise<number>;
  control(c: Control): void;
  subscribe(fn: (length: number) => void): () => void;   // throttled to animation frames
  close(): void;
  /** Revision 1b (additive): what the recorder (process/record.ts) and the engine (engine/runner.ts) report.
   *  paused: the learner's intent (not running in the background); truncated: maxFrames ended the run;
   *  where: the thread that makes the frames; notice: a note for the learner (hidden tab, main-thread fallback). */
  readonly paused?: boolean; readonly truncated?: boolean; readonly error?: string | null;
  readonly where?: 'starting' | 'worker' | 'main'; readonly notice?: I18n | null;
}
export interface RecordOpts { seed?: number; detail?: 'coarse' | 'fine'; keyEvery?: number /* 64 */; maxFrames?: number /* 20000 */; keep?: 'all' | { every: number }; recordDraws?: boolean }
export interface TapeHeader {
  format: 'atlas-tape/1'; process: string; version: string; seed: number; opt: unknown; provenance: Provenance;
  schema: SchemaJson; phases: Record<string, { label: I18n; kind: PhaseDef['kind']; tex?: string }>; frames: unknown[];
}
// W1-C, process/*.ts:
//   tape.ts   class TapeStore implements Tape { push(frame) } — keyframes, changes, derived memo (LRU). Used by the local recorder AND by W1-B2's engine client.
//   record.ts record(p, input, opt?, ro?): Tape;  recorder(p, input, opt?, ro?): LiveTape (advances in ≤ 8 ms slices)
//   io.ts     saveTape(t): { header: TapeHeader; blob: ArrayBuffer };  loadTape(h, blob, p?: ProcessDef): Tape
//             (p re-attaches derive and formula.values; without p, e.g. Python-written tapes, derived fields are unavailable)
//   adapters.ts traceToTape(spec, trace), scanTape(model, trace, layer, unit?, o?), bpeTape(r), generateTape(r)
//   library/sampling.ts (temperature/top-k, TinyGPT semantics), library/euler.ts (ODE particles), library/rollout.ts
