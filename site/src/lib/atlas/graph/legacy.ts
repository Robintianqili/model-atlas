// normalizeSpec(raw): AtlasSpec (design §3.2). Reads any v1 spec (no `atlas` key) as spec 1.1 at load time; it never
// rewrites a file and never mutates its argument (it works on a deep copy). Idempotent: a 1.1 spec only gets defaults.
//
// Every v1 spec:
//   - parity.tolerance / parity.passed are also read as bar / pass;
//   - `inputs` is synthesized from v1 `input` (name 'input'; adapter by kind: drawing -> drawing28, text -> text-bpe);
//     symbolic input dims get `dims` (T: 1..spec.context);
//   - an explicit `input` layer with no `in` reads the declared input (in: ['input']);
//   - string-valued params.weight / params.bias become `w` entries (removed from params);
//   - `linear_tied` becomes `linear` with w.weight = <tied tensor> and no bias; `transposed` is dropped, because the
//     stored matrix is already [out, in];
//   - `outputs` = [last layer]; `atlas` = { spec: '1.1', kinds: ['trace'], exporter: 'legacy-v1' }.
// Model-specific:
//   - tinygpt: every linear gets accumulate 'bias-first' (tinygpt.ts starts the f64 sum at the bias).
//   - axes for the two shipped models (TinyCNN: c,y,x / f / class; TinyGPT: t,d / h,tq,tk / t,vocab).
import type { AtlasLayer, AtlasSpec, Axis, Dim, InputDecl } from '../types/index.ts';

/** Kernel conventions of the hand-written legacy runtimes (tinycnn.ts sums from 0 and adds the bias last). */
const LEGACY_ACCUMULATE: Record<string, 'bias-first' | 'bias-last'> = { tinygpt: 'bias-first' };
const ADAPTER_BY_KIND: Record<string, string> = { drawing: 'drawing28', text: 'text-bpe', points2d: 'points2d', series: 'series', graph: 'graph', game: 'gridworld' };

function isObj(x: unknown): x is Record<string, unknown> {
	return typeof x === 'object' && x !== null && !Array.isArray(x);
}

function clone<T>(x: T): T {
	return typeof structuredClone === 'function' ? structuredClone(x) : (JSON.parse(JSON.stringify(x)) as T);
}

function inputAxes(kind: string, shape: Dim[]): Axis[] | undefined {
	if (kind === 'drawing' && shape.length === 3) return ['c', 'y', 'x'];
	if (kind === 'text' && shape.length === 1) return ['t'];
	if (shape.length === 2 && kind === 'points2d') return ['p', 'xy'];
	return undefined;
}

const DIM_LABELS: Record<string, { en: string; zh: string }> = {
	T: { en: 'tokens', zh: '词元数' },
	N: { en: 'nodes', zh: '节点数' },
	P: { en: 'points', zh: '点数' }
};

/** Axes of the layers of the two shipped v1 models (used by views to pick displays). */
function legacyAxes(spec: AtlasSpec, l: AtlasLayer): Axis[] | undefined {
	const r = l.outShape.length;
	const vocab = typeof spec.vocabSize === 'number' ? spec.vocabSize : undefined;
	const nClasses = Array.isArray(spec.classes) ? spec.classes.length : undefined;
	if (r === 3 && (l.op === 'attention_scores' || l.op === 'causal_mask' || (l.op === 'softmax' && typeof l.outShape[1] === 'string'))) return ['h', 'tq', 'tk'];
	if (r === 3) return ['c', 'y', 'x'];
	if (r === 2) return vocab !== undefined && l.outShape[1] === vocab ? ['t', 'vocab'] : ['t', 'd'];
	if (r === 1) {
		if (l.op === 'flatten') return ['f'];
		if (nClasses !== undefined && l.outShape[0] === nClasses) return ['class'];
		return ['f'];
	}
	return undefined;
}

export function normalizeSpec(raw: unknown): AtlasSpec {
	if (!isObj(raw)) throw new Error('normalizeSpec: spec must be a JSON object');
	const s = clone(raw) as AtlasSpec;
	if (!Array.isArray(s.layers)) throw new Error('normalizeSpec: spec.layers must be an array');
	if (!Array.isArray(s.tensors)) throw new Error('normalizeSpec: spec.tensors must be an array');
	const legacy = s.atlas === undefined;

	// parity: legacy names read as bar / pass (originals kept)
	if (isObj(s.parity)) {
		const p = s.parity;
		if (p.bar === undefined && typeof p.tolerance === 'number') p.bar = p.tolerance;
		if (p.pass === undefined && typeof p.passed === 'boolean') p.pass = p.passed;
		if (p.max_abs_diff === undefined) p.max_abs_diff = null;
		if (p.checked === undefined) p.checked = null;
	} else {
		s.parity = { against: 'unknown', max_abs_diff: null, checked: null };
	}
	if (!legacy) return s;

	const slug = String(s.slug ?? '');
	// declared inputs and symbolic dims
	if (!s.inputs && isObj(s.input)) {
		const kind = String(s.input.kind ?? 'tensor');
		const shape = (s.input.shape ?? []) as Dim[];
		const decl: InputDecl = { name: 'input', adapter: ADAPTER_BY_KIND[kind] ?? kind, kind, shape: shape.slice() };
		if (kind === 'text') decl.dtype = 'i32';
		else decl.dtype = 'f32';
		const ax = inputAxes(kind, shape);
		if (ax) decl.axes = ax;
		s.inputs = [decl];
	}
	const dims = { ...(s.dims ?? {}) };
	const symbols = new Set<string>();
	for (const i of s.inputs ?? []) for (const d of i.shape) if (typeof d === 'string') symbols.add(d);
	for (const l of s.layers) for (const d of l.outShape ?? []) if (typeof d === 'string') symbols.add(d);
	for (const sym of symbols) {
		if (dims[sym]) continue;
		const max = sym === 'T' && typeof s.context === 'number' ? s.context : undefined;
		if (max === undefined) throw new Error(`normalizeSpec: symbolic dim '${sym}' has no known maximum (add spec.dims.${sym})`);
		dims[sym] = { min: 1, max, label: DIM_LABELS[sym] ?? { en: sym, zh: sym } };
	}
	if (Object.keys(dims).length) s.dims = dims;

	const acc = LEGACY_ACCUMULATE[slug];
	const firstInput = s.inputs?.[0]?.name;
	for (const l of s.layers) {
		const params = isObj(l.params) ? (l.params as Record<string, unknown>) : undefined;
		if (l.op === 'input' && (!l.in || l.in.length === 0) && firstInput) l.in = [firstInput];
		if (!Array.isArray(l.in)) l.in = [];
		if (l.op === 'linear_tied') {
			l.op = 'linear';
			const w = typeof params?.weight === 'string' ? params.weight : undefined;
			if (w) l.w = { ...(l.w ?? {}), weight: w };
			if (params) {
				delete params.weight;
				delete params.transposed;
			}
		}
		if (params) {
			for (const slot of ['weight', 'bias']) {
				if (typeof params[slot] === 'string') {
					l.w = { ...(l.w ?? {}), [slot]: params[slot] as string };
					delete params[slot];
				}
			}
		}
		if (acc && (l.op === 'linear' || l.op === 'dense' || l.op === 'conv2d')) l.params = { ...(params ?? {}), accumulate: acc };
		else if (params && Object.keys(params).length === 0) delete l.params;
		if (!l.axes) {
			const ax = legacyAxes(s, l);
			if (ax) l.axes = ax;
		}
	}
	if (!s.outputs && s.layers.length) s.outputs = [s.layers[s.layers.length - 1].id];
	s.atlas = { spec: '1.1', kinds: ['trace'], exporter: 'legacy-v1' };
	return s;
}
