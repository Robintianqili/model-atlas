// Shared helpers for op definitions (W1-A). Pure TS.
//
// Conventions every CORE_OP follows (and local ops should follow):
//  - forward accumulates in f64 (JS numbers) and stores into an array of the first float input's type
//    (Float32Array unless a Float64Array came in); loop orders copy tinycnn.ts / tinygpt.ts so the kit is bit-identical.
//  - explain() recomputes `result` with the same arithmetic as forward, so result === traced in practice.
//  - narration templates use {name|fmt} placeholders. what/how/why/look/repeat use only names returned by
//    op.vars(layer, spec); `unit` may also use names of explain().values. en and zh use the same names (AK403).
//  - formula placeholders {{name}} are keys of explain().values.
import type { AnyTensor, AtlasLayer, AtlasSpec, Dim, I18n, OpCtx, Tensor, UnitExplanation } from '../types/index.ts';
import { numel, shapeEq } from '../core/tensor.ts';
import { shapeText } from '../core/format.ts';

/** I18n shorthand. */
export const t = (en: string, zh: string): I18n => ({ en, zh });

/** Weight tensor name of a slot (frozen rule, types/spec.ts):
 *  w[slot] ?? (typeof params[slot] === 'string' ? params[slot] : `${id}.${slot}`) */
export function weightName(layer: AtlasLayer, slot: string): string {
	const w = layer.w?.[slot];
	if (w) return w;
	const p = layer.params?.[slot];
	return typeof p === 'string' ? p : `${layer.id}.${slot}`;
}

/** The error ctx.w() throws for a tensor that does not exist (the executor uses it; optW() catches it). */
export class MissingWeightError extends Error {
	slot: string;
	tensorName: string;
	constructor(slot: string, tensorName: string, layer: string) {
		super(`layer '${layer}': weight slot '${slot}' -> tensor '${tensorName}' is not in the weights`);
		this.name = 'MissingWeightError';
		this.slot = slot;
		this.tensorName = tensorName;
	}
}

/** An optional weight: the tensor, or null when the model has none (e.g. a linear layer without bias). */
export function optW(get: { w(slot: string): Tensor; has?(slot: string): boolean }, slot: string): Tensor | null {
	if (get.has) return get.has(slot) ? get.w(slot) : null; // revision 1b: OpCtx.has / UnitQuery.has
	try {
		return get.w(slot);
	} catch (e) {
		if (e instanceof MissingWeightError || (e as Error)?.name === 'MissingWeightError') return null;
		throw e;
	}
}

/** Shape of a weight slot as declared in spec.tensors (null when absent). */
export function specWeightShape(layer: AtlasLayer, spec: AtlasSpec | null | undefined, slot: string): number[] | null {
	if (!spec) return null;
	const name = weightName(layer, slot);
	const e = spec.tensors.find((x) => x.name === name);
	return e ? e.shape.slice() : null;
}

/** outShape of the layer or declared input that `name` refers to. */
export function shapeOfName(spec: AtlasSpec | null | undefined, name: string): Dim[] | null {
	if (!spec) return null;
	const l = spec.layers.find((x) => x.id === name);
	if (l) return l.outShape.slice();
	const i = spec.inputs?.find((x) => x.name === name);
	return i ? i.shape.slice() : null;
}

/** A shape in narration: "[8×26×26]", "[T×76]". The same bracketed form the lesson layer prints for {outShape}
 *  (lesson/facts.ts shapeText), so one sentence never mixes "8×24×24" with "[8×12×12]" (R0). */
export function shapeWords(shape: readonly Dim[]): string {
	return `[${shapeText(shape)}]`;
}

/** Variables every op template may use: outShape and inShape as text ("[8×26×26]", "[T×76]"). In an auto lesson the
 *  planner replaces outShape with the traced shape of the step, in the same format. */
export function baseVars(layer: AtlasLayer, spec: AtlasSpec): Record<string, string | number> {
	const v: Record<string, string | number> = { outShape: shapeWords(layer.outShape) };
	const inS = layer.in.length ? shapeOfName(spec, layer.in[0]) : null;
	v.inShape = inS ? shapeWords(inS) : v.outShape;
	return v;
}

/** A bilingual piece of narration that vars() picks from the layer's structure (a bias or none, one input channel or
 *  several, a residual add or another add). lesson/narrate.ts renders an I18n value in the step's language, as it does
 *  for {label}; OpDef.vars is typed `string | number` in the frozen types, hence the cast (R0 kit issue). Numbers inside
 *  come from the spec (structure), never from a run. An empty phrase is words('', ''). */
export function words(en: string, zh: string): string {
	return t(en, zh) as unknown as string;
}

/** An integer for a phrase built in vars(): 1024 -> "1,024" (the narration's {x|int} format). */
export function intText(v: number | string): string {
	return typeof v === 'number' ? Math.round(v).toLocaleString('en-US') : v;
}

/** The label of the layer (or the name of the declared input) that `name` refers to, as a phrase for vars(). */
export function inputName(spec: AtlasSpec | null | undefined, name: string): string {
	const l = spec?.layers.find((x) => x.id === name);
	if (l?.label?.en) return words(l.label.en, l.label.zh ?? l.label.en);
	return words(name, name);
}

/** Ops that read only the SHAPE of their input, not its values: a position embedding reads how many positions there
 *  are. Their input edges do not count as "computed from" (K0: a ViT's token rows + position rows read as a residual). */
const SHAPE_ONLY = new Set(['position_embedding']);

/** True when layer `a` is computed (directly or through other layers) from the values of the layer or declared input
 *  `b`. Edges into shape-only ops (position_embedding) are not followed. */
export function dependsOn(spec: AtlasSpec | null | undefined, a: string, b: string): boolean {
	if (!spec || a === b) return false;
	const byId = new Map(spec.layers.map((l) => [l.id, l]));
	const inputsOf = (id: string) => {
		const l = byId.get(id);
		return l && !SHAPE_ONLY.has(l.op) ? l.in : [];
	};
	const seen = new Set<string>();
	const stack = [...inputsOf(a)];
	while (stack.length) {
		const id = stack.pop()!;
		if (id === b) return true;
		if (seen.has(id)) continue;
		seen.add(id);
		stack.push(...inputsOf(id));
	}
	return false;
}

/** An add whose one input is computed from the other: a residual (skip) connection. Token + position is not one. */
export function isResidualAdd(layer: AtlasLayer, spec: AtlasSpec | null | undefined): boolean {
	const [a, b] = layer.in;
	return a !== undefined && b !== undefined && (dependsOn(spec, b, a) || dependsOn(spec, a, b));
}

/** The spec's output layer (spec.outputs[0], else the last layer). */
export function outputLayer(spec: AtlasSpec | null | undefined): AtlasLayer | undefined {
	if (!spec) return undefined;
	const id = spec.outputs?.[0] ?? spec.layers[spec.layers.length - 1]?.id;
	return spec.layers.find((l) => l.id === id);
}

/** The layer gives the model's raw output scores ("logits"): it is the output layer, or the input of an output softmax
 *  or sigmoid. A hidden linear layer's outputs are not logits. */
export function givesLogits(layer: AtlasLayer, spec: AtlasSpec | null | undefined): boolean {
	const out = outputLayer(spec);
	if (!out) return false;
	if (out.id === layer.id) return out.op !== 'softmax' && out.op !== 'sigmoid';
	return (out.op === 'softmax' || out.op === 'sigmoid') && out.in[0] === layer.id;
}

/** Glossary terms that depend on the layer's structure (R0: the add op linked "residual connection" on TinyGPT's token +
 *  position add). Not part of the frozen OpDef type: the planner (lesson/plan.ts) calls `termsFor(layer, spec)` when an
 *  op has it, else uses the static `terms` (the full list, as the docs show it). */
export interface StructuralTerms {
	termsFor?(layer: AtlasLayer, spec: AtlasSpec): string[];
}

/** The terms an auto step of `layer` links to: def.termsFor(layer, spec) when the op has it, else def.terms. */
export function termsOf(def: { terms?: string[] } & StructuralTerms, layer: AtlasLayer, spec: AtlasSpec): string[] {
	if (typeof def.termsFor === 'function') {
		try {
			return def.termsFor(layer, spec);
		} catch {
			return def.terms ? [...def.terms] : [];
		}
	}
	return def.terms ? [...def.terms] : [];
}

/** Whether the layer has a bias tensor in the spec (the optional 'bias' slot). */
export function hasBias(layer: AtlasLayer, spec: AtlasSpec | null | undefined): boolean {
	return !!specWeightShape(layer, spec, 'bias');
}

/** Bias wording shared by conv2d and linear: the clause after "the products are added" and, for the unit sentence,
 *  "the bias is" or "there is no bias, so b =". */
export function biasWords(bias: boolean): { biasClause: string; biasIs: string } {
	return bias
		? { biasClause: words(', plus one learned bias', '，再加上一个学来的偏置'), biasIs: words('the bias is', '偏置是') }
		: { biasClause: words('; this layer has no bias', '；这一层没有偏置'), biasIs: words('there is no bias, so b =', '这一层没有偏置，所以 b =') };
}

export function int(p: Record<string, unknown>, key: string, dflt?: number): number {
	const v = p[key];
	if (v === undefined || v === null) {
		if (dflt === undefined) throw new Error(`missing attribute '${key}'`);
		return dflt;
	}
	if (typeof v !== 'number' || !Number.isInteger(v)) throw new Error(`attribute '${key}' must be an integer, got ${JSON.stringify(v)}`);
	return v;
}

export function num(p: Record<string, unknown>, key: string, dflt?: number): number {
	const v = p[key];
	if (v === undefined || v === null) {
		if (dflt === undefined) throw new Error(`missing attribute '${key}'`);
		return dflt;
	}
	if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`attribute '${key}' must be a number, got ${JSON.stringify(v)}`);
	return v;
}

/** Accumulation mode of conv2d/linear: 'bias-last' (tinycnn.ts: sum from 0, add the bias at the end) or 'bias-first' (tinygpt.ts). */
export function accMode(p: Record<string, unknown>): 'bias-last' | 'bias-first' {
	const a = p.accumulate ?? 'bias-last';
	if (a !== 'bias-last' && a !== 'bias-first') throw new Error(`accumulate must be 'bias-last' or 'bias-first', got ${JSON.stringify(a)}`);
	return a;
}

/** Output array of n values, same precision as the first float input. */
export function outArray(n: number, ...like: (AnyTensor | Tensor | null | undefined)[]): Float32Array | Float64Array {
	for (const x of like) {
		if (!x) continue;
		if (x.data instanceof Float64Array) return new Float64Array(n);
		if (x.data instanceof Float32Array) return new Float32Array(n);
	}
	return new Float32Array(n);
}

/** Throw unless a tensor has the given rank. */
export function needRank(x: AnyTensor, rank: number | number[], what: string): void {
	const ok = Array.isArray(rank) ? rank.includes(x.shape.length) : x.shape.length === rank;
	if (!ok) throw new Error(`${what}: expected rank ${Array.isArray(rank) ? rank.join(' or ') : rank}, got shape [${x.shape}]`);
}

/** Dims compare for inferShape: numbers must agree; a string matches the same string. */
export function sameDim(a: Dim, b: Dim): boolean {
	return a === b;
}

export function assertShape(got: readonly Dim[], want: readonly Dim[], what: string): void {
	if (!shapeEq(got, want)) throw new Error(`${what}: shape [${got}] does not match [${want}]`);
}

/** Multiply dims that may be symbolic: numbers multiply, otherwise a product string ('T·4'). */
export function mulDims(a: Dim, b: Dim): Dim {
	if (typeof a === 'number' && typeof b === 'number') return a * b;
	return `${a}*${b}`;
}

/** Number of output units (cells) of a tensor. */
export function cells(x: Tensor): number {
	return numel(x.shape);
}

/** Read the unit's traced value (flat index) safely. */
export function tracedAt(out: Tensor, flat: number): number {
	return out.data[flat];
}

/** A unit explanation skeleton with the common fields. */
export function unitBase(kind: UnitExplanation['kind'], layer: AtlasLayer, unit: number[], result: number, traced: number): UnitExplanation {
	return { kind, layer: layer.id, unit: unit.slice(), reads: [], values: {}, result, traced };
}

/** Store a value with the precision of `like` (Math.fround for float32), as forward does when it writes into its output. */
export function store(v: number, like: Tensor): number {
	return like.data instanceof Float64Array ? v : Math.fround(v);
}

/** The emitted sub-steps of a layer, through ctx (no-op helper for symmetry). */
export function emitAll(ctx: OpCtx, subs: Record<string, Tensor>): void {
	for (const [k, v] of Object.entries(subs)) ctx.emit(k, v);
}

// ---------------------------------------------------------------- erf (GELU 'none')
// erf in f64 to ~1e-15 relative: the all-positive series erf(x) = 2/√π · e^{−x²} · Σ (2x²)^n · x / (2n+1)!!
// (no cancellation) for |x| < 6; erf(±x) = ±1 to double precision beyond.
const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);
export function erf(x: number): number {
	if (Number.isNaN(x)) return NaN;
	const a = Math.abs(x);
	if (a >= 6) return x > 0 ? 1 : -1;
	if (a === 0) return x; // keeps −0
	const x2 = a * a;
	let term = a;
	let sum = a;
	for (let n = 1; n < 500; n++) {
		term *= (2 * x2) / (2 * n + 1);
		sum += term;
		if (term < sum * 1e-17) break;
	}
	const r = TWO_OVER_SQRT_PI * Math.exp(-x2) * sum;
	return x < 0 ? -r : r;
}
