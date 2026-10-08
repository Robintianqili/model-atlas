// DevKit frozen types (research/devkit/DEVKIT_DESIGN.md §4.2): spec 1.1, a trimmed superset of CONTRACTS.
// Frozen during waves 1 and 2: changes go through the main session.
import type { I18n, Kind } from './core.ts';
export type Dim = number | string;                       // symbolic: 'T' tokens, 'N' nodes, 'P' points
/** K1 (additive): 'nk' the second node index of a [n, nk] node-by-node table (an adjacency), 'b' an item of a batch
 *  (TinyCLIP's 6 drawings). */
export type Axis = 'c' | 'y' | 'x' | 't' | 'd' | 'h' | 'tq' | 'tk' | 'n' | 'f' | 'class' | 'vocab' | 'p' | 'xy' | 'k' | 'nk' | 'b';
export interface TensorEntry { name: string; shape: number[]; offset: number; length: number }
/** kind is a catalog InputKind: 'drawing' | 'text' | 'points2d' | 'series' | 'graph' | 'game' | … */
export interface InputDecl { name: string; adapter: string; kind: string; shape: Dim[]; dtype?: 'f32' | 'i32'; axes?: Axis[] }
export interface AtlasLayer {
  id: string;                                            // no '#'
  op: string; label: I18n;
  in: string[];                                          // layer ids or declared input names
  outShape: Dim[];
  /** op attributes (v1 name kept), validated against OpDef.attrs */
  params?: Record<string, unknown>;
  /** weight slot -> tensor name. Resolution: w[slot] ?? (typeof params[slot] === 'string' ? params[slot] : `${id}.${slot}`) */
  w?: Record<string, string>;
  desc?: I18n; axes?: Axis[];
  sub?: string[];                                        // recorded as `${id}#${sub}`
  code?: { file: string; line: number; end?: number };
  ui?: { zoom?: string; display?: string; hidden?: boolean };
  teach?: { lesson?: 'step' | 'fold' | 'skip'; title?: I18n; narration?: { what?: I18n; how?: I18n } };
  rf?: { size: number; jump: number };                   // v1 key written by TinyCNN; read as is
}
export interface ParityRecord {
  against: string;
  max_abs_diff: number | null; checked: string | null;  // null until `atlas parity` runs
  bar?: number; relative?: boolean; pass?: boolean; scope?: 'forward' | 'train' | 'process'; script?: string;
  cases?: number; edgeCases?: string[]; perLayer?: Record<string, number>; details_max_abs_diff?: number; steps?: number;
  tolerance?: number; passed?: boolean;                 // TinyGPT's legacy names, read as bar / pass
  [extra: string]: unknown;
}
export interface AtlasSpec {
  slug: string; name: I18n; version: string; params: number; precision: 'fp32' | 'f64' | 'fp16' | 'mixed';
  input?: { kind: string; shape: Dim[]; notes?: string };   // v1; the exporter writes it for single-input models
  tensors: TensorEntry[]; layers: AtlasLayer[];
  training?: { dataset?: string; steps?: number; metrics?: Record<string, number | string>; [k: string]: unknown };
  parity: ParityRecord;
  atlas?: { spec: '1.1'; kinds: Kind[]; exporter: string };
  inputs?: InputDecl[]; outputs?: string[];
  dims?: Record<string, { min: number; max: number; label: I18n }>;
  files?: Record<string, string>;                         // as TinyGPT already writes
  [extra: string]: unknown;                               // classes, vocabSize, eotId, …
}
