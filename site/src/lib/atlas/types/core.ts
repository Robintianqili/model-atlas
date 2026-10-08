// DevKit frozen types (research/devkit/DEVKIT_DESIGN.md §4.1). Frozen during waves 1 and 2: changes go through
// the main session and are recorded in CONTRACTS.md "DevKit additions".
import type { I18n, Text } from '../../i18n/types.ts';
export type { I18n, Text };
export type Provenance = 'live' | 'precomputed' | 'illustrative';
export type Kind = 'trace' | 'train' | 'process';
export type FArray = Float32Array | Float64Array;
/** CONTRACTS tensor (row-major; images [C,H,W]; sequences [T,D]). Training toys may use Float64Array. */
export interface Tensor { shape: number[]; data: FArray }
export interface IntTensor { shape: number[]; data: Int32Array; dtype: 'i32' }
export type AnyTensor = Tensor | IntTensor;
/** CONTRACTS trace, unchanged. Sub-steps are `${layerId}#${sub}` ('L0.attn.scores#h2', 'lstm1#f') and follow
 *  their layer in `order`. '#' never appears in a layer id; layer ids may contain dots. */
export interface Trace { order: string[]; t: Record<string, Tensor>; meta?: TraceMeta }
export interface TraceMeta { dims: Record<string, number>; weights: string; provenance: Provenance; ms?: number; where?: 'worker' | 'main' | 'node' }
/** inclusive cell box */
export interface Box { y0: number; x0: number; y1: number; x1: number }
export type FetchLike = (url: string) => Promise<{ ok?: boolean; status?: number; json(): Promise<unknown>; arrayBuffer(): Promise<ArrayBuffer>; text?(): Promise<string> }>;
/** Structural copy of kit PredictQuestion (atlas never imports the kit); ui/kitspec.ts asserts assignability both ways. */
export interface PredictQuestion { id: string; question: I18n; options: I18n[]; answer: number; explain: I18n }
/** mulberry32. next() in [0,1); int(n) = floor(next()·n); fork(label) = mulberry32((seed ^ fnv1a32(label)) >>> 0), independent of
 *  how much was consumed. normal(): Box–Muller, u1 = 1 − next(), u2 = next(), z = sqrt(−2 ln u1)·cos(2π u2), no caching.
 *  Python ak.Rng is bit-identical for next(), int() and fork() only. normal() and other draws that use log/cos/sqrt may differ in
 *  the last ulp across engines: processes that use them set RecordOpts.recordDraws, and Python references replay the recorded draws.
 *  Test vectors: mulberry32(42) → 0.6011037519, 0.4482905590, 0.8524657935, 0.6697340414; mulberry32(1) → 0.6270739406, 0.0027357212, 0.5274470400. */
export interface Rng { readonly seed: number; next(): number; normal(): number; int(n: number): number; fork(label: string): Rng; getState(): number; setState(s: number): void }
