// Built-in narration paths (design §5.3) and facts.json (atlas-facts/1). Pure TS.
//
//   layer.<id>.<stat>          stat ∈ max min argmax negShare zeroShare shape strongestChannel. Parsed right to left: the
//                              last segment is the stat, the rest is the id (ids may contain dots and '#sub').
//   top.label top.p runnerUp.label runnerUp.p margin      from the output layer (spec.outputs[0], else the last traced layer);
//                              a [T, V] output is read at its answer row (K2: manifest output.row; default the last row,
//                              the next-token prediction); the label of a
//                              vocabulary output is the token's text (“␣the”), of a classifier its class name
//   sel.layer sel.ch sel.cell sel.value                    the lesson's own selection
//   unit.<name>                explain() values at the selection (also unit.result, unit.traced, unit.bias, unit.terms)
//   train.step train.loss train.<metric>                  latest frame of the first train source
//   frame.phase frame.<metric> run.length                 the step's frame (narrate.ts); without one, the latest frame of the
//                                                          first process/train source
//   opt.<name>                 the step's source options (narrate.ts), else the first source that has the option
//   fact:<key>                 facts.json
import type { I18n, Tensor, Trace } from '../types/core.ts';
import type { AtlasSpec } from '../types/spec.ts';
import type { GraphModel, UnitExplanation } from '../types/op.ts';
import type { At } from '../types/lesson.ts';
import type { ExplainerRuntime, Preprocessed, SelectionState, SourceDef, SourceState } from '../types/runtime.ts';
import type { AnswerRow } from '../types/explainer.ts';
import { visibleToken } from '../core/format.ts';

export type FactValue = number | string | I18n | undefined;
export type FactsFile = { format: 'atlas-facts/1'; facts: Record<string, { value: number | string | I18n; unit?: string; source: string; measured: string }> };

export const LAYER_STATS = ['max', 'min', 'argmax', 'negShare', 'zeroShare', 'shape', 'strongestChannel'] as const;

/** 'layer.L0.attn.scores#h2.max' -> { id: 'L0.attn.scores#h2', stat: 'max' } */
export function parseLayerPath(path: string): { id: string; stat: string } | null {
	if (!path.startsWith('layer.')) return null;
	const rest = path.slice(6);
	const dot = rest.lastIndexOf('.');
	if (dot <= 0) return null;
	return { id: rest.slice(0, dot), stat: rest.slice(dot + 1) };
}

export function shapeText(shape: readonly (number | string)[]): string {
	return `[${shape.join('×')}]`;
}

export function layerStat(t: Tensor, stat: string): number | string | undefined {
	const d = t.data;
	const n = d.length;
	switch (stat) {
		case 'shape':
			return shapeText(t.shape);
		case 'max':
		case 'min': {
			if (!n) return undefined;
			let v = d[0];
			for (let i = 1; i < n; i++) if (stat === 'max' ? d[i] > v : d[i] < v) v = d[i];
			return v;
		}
		case 'argmax': {
			if (!n) return undefined;
			let b = 0;
			for (let i = 1; i < n; i++) if (d[i] > d[b]) b = i;
			return b;
		}
		case 'negShare':
		case 'zeroShare': {
			if (!n) return undefined;
			let c = 0;
			for (let i = 0; i < n; i++) if (stat === 'negShare' ? d[i] < 0 : d[i] === 0) c++;
			return c / n;
		}
		case 'strongestChannel': {
			// channel of a [C, …] tensor holding the largest single value (as the TinyCNN page does); 0 for vectors
			if (!n) return undefined;
			if (t.shape.length < 2) return 0;
			const per = n / t.shape[0];
			let best = 0;
			let bv = -Infinity;
			for (let i = 0; i < n; i++) if (d[i] > bv) (bv = d[i]), (best = Math.floor(i / per));
			return best;
		}
	}
	return undefined;
}

/** Label for output index i: spec.classes / spec.labels entries may be strings, { en, zh } or { id, en, zh }. */
export function outputLabel(spec: AtlasSpec | null, i: number): string | I18n {
	const list = (spec?.classes ?? spec?.labels) as unknown;
	if (Array.isArray(list) && i >= 0 && i < list.length) {
		const v = list[i];
		if (typeof v === 'string') return v;
		if (v && typeof v === 'object' && typeof (v as I18n).en === 'string') return { en: (v as I18n).en, zh: (v as I18n).zh };
		if (v && typeof v === 'object' && typeof (v as { id?: string }).id === 'string') return (v as { id: string }).id;
	}
	return `#${i}`;
}

/** Label for output index i of a language model's vocabulary output: the token's text in quotes (“␣the”) when the
 *  adapter gives token texts (Preprocessed.vocab), else "token 501". Never "#501" (R0). */
export function vocabLabel(pre: Preprocessed | null | undefined, i: number): string | I18n {
	const v = pre?.vocab;
	let s: unknown;
	try {
		s = typeof v === 'function' ? v(i) : Array.isArray(v) ? v[i] : undefined;
	} catch {
		s = undefined;
	}
	return typeof s === 'string' ? `“${visibleToken(s)}”` : { en: `token ${i}`, zh: `词元 ${i}` };
}

/** The output layer id: spec.outputs[0], else the last non-sub-step id of the trace. */
export function outputId(spec: AtlasSpec | null, trace: Trace | null | undefined): string | undefined {
	const o = spec?.outputs?.[0];
	if (o && trace?.t[o]) return o;
	return trace ? [...trace.order].reverse().find((id) => !id.includes('#')) : undefined;
}

/** K2: the row of a sequence output [T, V] that holds the model's one answer (manifest output.row). 'last' (the default)
 *  is a causal language model's guess for the next token; a masked model names its [MASK] row, usually with a function
 *  of the runtime. A number counts from 0 (a negative one from the end); the result is clamped to the rows there are. */
export function answerRow(row: AnswerRow | undefined, rows: number, rt?: ExplainerRuntime): number {
	if (!(rows > 1)) return 0;
	let r = rows - 1;
	if (row === 'first') r = 0;
	else if (typeof row === 'number') r = row;
	else if (typeof row === 'function' && rt) {
		try {
			r = row(rt);
		} catch {
			r = rows - 1;
		}
	}
	if (!Number.isFinite(r)) r = rows - 1;
	r = Math.trunc(r);
	if (r < 0) r += rows;
	return Math.max(0, Math.min(rows - 1, r));
}

/** K2: the answer row of the package's output tensor `t` (its rows are every axis but the last). */
export function outputRow(rt: object | null | undefined, t: Tensor): number {
	const V = t.shape.length >= 2 ? t.shape[t.shape.length - 1] : t.data.length;
	const rows = V > 0 ? Math.floor(t.data.length / V) : 1;
	const row = (rt as { pkg?: { manifest?: { output?: { row?: AnswerRow } } } } | null | undefined)?.pkg?.manifest?.output?.row;
	return answerRow(row, rows, (rt ?? undefined) as ExplainerRuntime | undefined);
}

/** Probabilities of the output (softmax applied in f64 when the values are not already a distribution). A sequence
 *  output [T, V] (or any rank >= 2) is read at row `row` (K2; default its last row: the prediction for what comes next;
 *  outputRow() gives the package's answer row). */
export function outputProbs(t: Tensor, row?: number): Float64Array {
	const V = t.shape.length >= 2 ? t.shape[t.shape.length - 1] : t.data.length;
	const rows = V > 0 ? Math.floor(t.data.length / V) : 1;
	const r = row === undefined ? rows - 1 : Math.max(0, Math.min(rows - 1, Math.trunc(row)));
	const d = t.shape.length >= 2 ? t.data.subarray(r * V, (r + 1) * V) : t.data;
	let sum = 0;
	let neg = false;
	for (let i = 0; i < d.length; i++) {
		sum += d[i];
		if (d[i] < 0) neg = true;
	}
	const p = new Float64Array(d.length);
	if (!neg && Math.abs(sum - 1) < 1e-3) {
		for (let i = 0; i < d.length; i++) p[i] = d[i];
		return p;
	}
	let m = -Infinity;
	for (let i = 0; i < d.length; i++) if (d[i] > m) m = d[i];
	let z = 0;
	for (let i = 0; i < d.length; i++) z += p[i] = Math.exp(d[i] - m);
	for (let i = 0; i < d.length; i++) p[i] /= z;
	return p;
}

export interface FactCtx {
	spec: AtlasSpec | null;
	/** the adapter's result: a text adapter's `vocab` gives token texts for a vocabulary output's top.label */
	pre?: Preprocessed | null;
	model: GraphModel | null;
	sources: Record<string, SourceState>;
	defs?: Record<string, SourceDef>;
	sel: SelectionState;
	facts?: FactsFile | null;
	/** cache for unit explanations (keyed by trace identity + layer + unit) */
	explainMemo?: WeakMap<object, Map<string, UnitExplanation | null>>;
	/** K2: the answer row of the output tensor (manifest output.row; default its last row) */
	outputRow?: (t: Tensor) => number;
}

export function primaryTrace(ctx: FactCtx): Trace | null {
	for (const s of Object.values(ctx.sources)) if ((s.kind === 'trace' || s.kind === 'live-model') && s.trace) return s.trace;
	return null;
}

function flat(shape: number[], unit: number[]): number {
	let i = 0;
	const off = shape.length - unit.length;
	for (let d = 0; d < shape.length; d++) i = i * shape[d] + (d >= off ? unit[d - off] : 0);
	return i;
}

export function explainAt(ctx: FactCtx, trace: Trace | null, layer: string | undefined, unit: number[] | undefined, model?: GraphModel | null): UnitExplanation | undefined {
	const m = model ?? ctx.model;
	if (!m || !trace || !layer || !unit || !trace.t[layer]) return undefined;
	const key = `${layer}|${unit.join(',')}`;
	let memo = ctx.explainMemo?.get(trace);
	if (ctx.explainMemo && !memo) ctx.explainMemo.set(trace, (memo = new Map()));
	if (memo?.has(key)) return memo.get(key) ?? undefined;
	let x: UnitExplanation | null = null;
	try {
		x = m.explain(trace, layer, unit);
	} catch {
		x = null;
	}
	memo?.set(key, x);
	return x ?? undefined;
}

export function unitValue(x: UnitExplanation | undefined, name: string): number | undefined {
	if (!x) return undefined;
	if (name in x.values) return x.values[name];
	if (name === 'result') return x.result;
	if (name === 'traced') return x.traced;
	if (name === 'bias') return x.bias;
	if (name === 'terms') return x.terms?.length;
	return undefined;
}

function latestFrameOf(ctx: FactCtx, kinds: SourceState['kind'][]) {
	for (const s of Object.values(ctx.sources)) {
		if (!kinds.includes(s.kind) || !s.tape || s.tape.length === 0) continue;
		const i = s.tape.length - 1;
		return { frame: s.tape.frame(i), state: s.tape.state(i) as Record<string, unknown>, tape: s.tape };
	}
	return null;
}

/** Path resolution that depends on the step's position (At). Returns undefined when the path is not at-dependent or
 *  not resolvable from `at`, so the caller falls back to rt.fact(). */
export function atFact(path: string, at: At | undefined): FactValue {
	if (!at) return undefined;
	if (path === 'frame.phase') return at.frame?.phase;
	if (path === 'frame.step') return at.frame?.step;
	if (path === 'frame.index') return at.index;
	if (path.startsWith('frame.')) {
		const k = path.slice(6);
		const v = at.frame?.metrics?.[k];
		if (v !== undefined) return v;
		const s = (at.state as Record<string, unknown> | undefined)?.[k];
		return typeof s === 'number' || typeof s === 'string' ? s : undefined;
	}
	if (path === 'run.length') return at.tape?.length;
	if (path.startsWith('opt.')) {
		const v = (at.tape?.opt as Record<string, unknown> | undefined)?.[path.slice(4)];
		return typeof v === 'number' || typeof v === 'string' ? v : undefined;
	}
	if (path.startsWith('unit.') && at.unit) return unitValue(at.unit, path.slice(5));
	return undefined;
}

/** Built-in paths and facts.json for rt.fact(). */
export function builtinFact(ctx: FactCtx, path: string): FactValue {
	if (path.startsWith('fact:')) {
		const v = ctx.facts?.facts?.[path.slice(5)]?.value;
		return v;
	}
	const trace = primaryTrace(ctx);
	const lp = parseLayerPath(path);
	if (lp) {
		const t = trace?.t[lp.id];
		return t ? layerStat(t, lp.stat) : undefined;
	}
	if (path === 'top.label' || path === 'top.p' || path === 'runnerUp.label' || path === 'runnerUp.p' || path === 'margin') {
		const id = outputId(ctx.spec, trace);
		const t = id ? trace?.t[id] : undefined;
		if (!t || t.data.length === 0) return undefined;
		const p = outputProbs(t, ctx.outputRow?.(t));
		const order = Array.from(p.keys()).sort((a, b) => p[b] - p[a] || a - b);
		const [a, b] = order;
		const layer = ctx.spec?.layers.find((l) => l.id === id);
		const vocab = !!layer?.axes?.length && layer.axes[layer.axes.length - 1] === 'vocab' && !Array.isArray(ctx.spec?.classes ?? ctx.spec?.labels);
		const label = (i: number) => (vocab ? vocabLabel(ctx.pre, i) : outputLabel(ctx.spec, i));
		if (path === 'top.label') return label(a);
		if (path === 'top.p') return p[a];
		if (b === undefined) return undefined;
		if (path === 'runnerUp.label') return label(b);
		if (path === 'runnerUp.p') return p[b];
		return p[a] - p[b];
	}
	if (path.startsWith('sel.')) {
		const k = path.slice(4);
		const sel = ctx.sel;
		if (k === 'layer') return sel.layer;
		if (k === 'ch') return sel.ch ?? sel.unit?.[0];
		if (k === 'cell') {
			const u = sel.unit;
			if (!u || u.length === 0) return undefined;
			return u.length >= 2 ? `(${u[u.length - 2]}, ${u[u.length - 1]})` : `(${u[0]})`;
		}
		if (k === 'value') {
			const t = sel.layer ? trace?.t[sel.layer] : undefined;
			if (!t || !sel.unit) return undefined;
			const i = flat(t.shape, sel.unit);
			return i >= 0 && i < t.data.length ? t.data[i] : undefined;
		}
		const v = sel[k];
		return typeof v === 'number' || typeof v === 'string' ? v : undefined;
	}
	if (path.startsWith('unit.')) return unitValue(explainAt(ctx, trace, ctx.sel.layer, ctx.sel.unit), path.slice(5));
	if (path.startsWith('train.')) {
		const f = latestFrameOf(ctx, ['train']);
		if (!f) return undefined;
		const k = path.slice(6);
		if (k === 'step') return f.frame.step ?? (f.state.step as number | undefined);
		const m = f.frame.metrics?.[k];
		if (m !== undefined) return m;
		const s = f.state[k];
		return typeof s === 'number' ? s : undefined;
	}
	if (path.startsWith('frame.') || path === 'run.length') {
		const f = latestFrameOf(ctx, ['process', 'train', 'tape']);
		if (!f) return undefined;
		return atFact(path, { index: f.tape.length - 1, frame: f.frame, state: f.state, tape: f.tape });
	}
	if (path.startsWith('opt.')) {
		const k = path.slice(4);
		for (const [name, s] of Object.entries(ctx.sources)) {
			const v = (s.tape?.opt as Record<string, unknown> | undefined)?.[k] ?? ((ctx.defs?.[name] as { opt?: Record<string, unknown> } | undefined)?.opt?.[k]);
			if (typeof v === 'number' || typeof v === 'string') return v;
		}
		for (const d of Object.values(ctx.defs ?? {})) {
			const v = (d as { opt?: Record<string, unknown> }).opt?.[k];
			if (typeof v === 'number' || typeof v === 'string') return v;
		}
		return undefined;
	}
	return undefined;
}
