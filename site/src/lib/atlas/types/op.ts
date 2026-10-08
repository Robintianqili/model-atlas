// DevKit frozen types (research/devkit/DEVKIT_DESIGN.md §4.3): op definition, unit explanation, graph model.
// Frozen during waves 1 and 2: changes go through the main session.
import type { AnyTensor, Box, FetchLike, I18n, Tensor, Trace } from './core.ts';
import type { AtlasLayer, AtlasSpec, Axis, Dim } from './spec.ts';
export type OpCategory = 'io' | 'linear' | 'conv' | 'activation' | 'pool' | 'norm' | 'attention' | 'embedding' | 'shape' | 'merge' | 'recurrent' | 'graph' | 'output';
/** The zoom view is chosen by the kind of arithmetic, not by op type. */
export type ZoomKind = 'weighted-sum' | 'reduce' | 'elementwise' | 'normalize' | 'softmax' | 'lookup' | 'index-map' | 'gate' | 'none';
export type UnitPick = 'argmax' | 'argmin' | 'argmaxAbs' | 'center' | 'first' | 'last' | 'strongestChannel' | { unit: number[] };
/** Bilingual narration templates (syntax §4.6). */
export interface NarrationSet { what: I18n; how?: I18n; why?: I18n; look?: I18n; repeat?: I18n; unit?: I18n }
export interface AttrDecl { type: 'int' | 'float' | 'bool' | 'string' | 'int[]' | 'float[]'; default?: unknown; required?: boolean; doc?: string }
export interface OpCtx { layer: AtlasLayer; dims: Record<string, number>; w(slot: string): Tensor; emit(sub: string, t: Tensor): void
  /** Revision 1b (additive): true when the model has a tensor for this weight slot (optional weights such as a bias). The
   *  executor provides it; ops/common.ts optW() uses it when present and otherwise catches MissingWeightError. */
  has?(slot: string): boolean }
export interface UnitQuery<P> { layer: AtlasLayer; unit: number[]; ins: AnyTensor[]; out: Tensor; w(slot: string): Tensor; p: P; trace: Trace; spec: AtlasSpec
  has?(slot: string): boolean }                                                  // revision 1b, as OpCtx.has
/** The arithmetic behind ONE output number. Generalizes convDetail/poolDetail/denseDetail/softmaxDetail and TinyGPT's q·k. */
export interface UnitExplanation {
  kind: ZoomKind; layer: string; unit: number[];
  reads: { tensor: string; cells: number[]; box?: Box }[];          // drives linked highlights in every view
  terms?: { x: number; w: number; p: number; xAt?: number; wAt?: number }[];   // weighted sums: x·w = p
  groups?: { label: I18n; from: number; to: number; sum: number }[];          // per input channel / head / neighbour
  layout?: { rows: number; cols: number };
  steps?: { label: I18n; value: number; tex?: string }[];                      // normalize: mean, var, (x−μ)/σ, ·γ, +β
  curve?: { fn: string; x: number; y: number };                                // elementwise: the point on the curve
  bias?: number;
  values: Record<string, number>;      // named scalars for narration vars and {{name}} in the formula
  result: number;                      // recomputed here
  traced: number;                      // read from the trace; |result − traced| ≤ 1e-5 is a gate
}
export interface OpDef<P extends Record<string, unknown> = Record<string, unknown>> {
  op: string; version: number;
  /** declared attributes: the source of schema/ops.json, which the Python mirrors are checked against */
  attrs: Record<string, AttrDecl>;
  aliases?: Record<string, Partial<P>>;                 // { dense: {} }, { linear_tied: {} }, { gelu_tanh: { approx: 'tanh' } }
  category: OpCategory; label: I18n;
  weights?: { slot: string; optional?: boolean }[];
  subs?: { name: string; label: I18n; axes?: Axis[] }[];
  inferShape(ins: Dim[][], p: P, w: Record<string, number[]>): Dim[];
  axes?(ins: Axis[][], p: P): Axis[];
  /** f64 accumulation, f32 storage (as tinycnn.ts/tinygpt.ts do). Record sub-steps with ctx.emit. */
  forward(xs: AnyTensor[], p: P, ctx: OpCtx): Tensor;
  unit?: { axes(out: Axis[]): Axis[]; scan?: 'raster' | 'rows' | 'time' | 'none'; pick?: UnitPick };
  explain?(q: UnitQuery<P>): UnitExplanation;
  receptive?(box: Box, p: P): Box;
  /** KaTeX in the kit's convention: colour roles with sym(role, tex) from atlas/core/tex.ts (the same output as kit/tex.ts),
   *  live values as {{name}} from explain().values, filled by kit fillTex in the UI. */
  formula?: string;
  narration: NarrationSet;
  /** template variables from structure: kernel, inChannels, products = C·k·k, … */
  vars?(layer: AtlasLayer, spec: AtlasSpec): Record<string, string | number>;
  /** a repeat that deserves its own lesson step (conv: inChannels 1 → >1) */
  distinct?(p: P, prev: P): boolean;
  zoom: ZoomKind; color?: 'diverging' | 'sequential';
  parity: { bar: number };
  pytorch?(layer: AtlasLayer): string;                 // one line for the Code tab
  terms?: string[];                                    // glossary ids
}
export interface ResolvedOp { def: OpDef; params: Record<string, unknown> }   // alias params merged under layer params
export interface OpRegistry { get(op: string): ResolvedOp; has(op: string): boolean; list(): string[] }
export interface GraphModel {
  spec: AtlasSpec; weights: Record<string, Tensor>; ops: OpRegistry;
  forward(input: AnyTensor | Record<string, AnyTensor>, o?: { record?: 'all' | 'layers'; dims?: Record<string, number> }): Trace;
  explain(trace: Trace, layer: string, unit: number[]): UnitExplanation;
  receptiveField(layer: string, box: Box): Box | null;
  layer(id: string): AtlasLayer;
}
// W1-A: ops/registry.ts createRegistry(...ops: OpDef[]): OpRegistry;  ops/index.ts CORE_OPS: OpDef[]
//       ops/manifest.ts (node ops/manifest.ts --write) -> schema/ops.json {op, version, attrs, aliases, weights, subs}
//       graph/executor.ts fromParts(spec, weights: ArrayBuffer | Record<string, Tensor>, ops): GraphModel;
//                         loadGraph(fetch: FetchLike, baseUrl: string, ops): Promise<GraphModel>
//       graph/legacy.ts normalizeSpec(raw: unknown): AtlasSpec (§3.2);  graph/refs.ts readReference(fetch, dir): Promise<RefCase[]>
//       core/compare.ts maxDiff(a, b, { relative?: boolean }): { max: number; at: number; nan: boolean }  (NaN or unmatched ±∞ fails; −∞ == −∞)
