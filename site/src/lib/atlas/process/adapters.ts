// Tapes from things that already exist (design §4.4, §3.2): wrappers that change nothing.
//   traceToTape(spec, trace)                      one frame per trace id (layers, then their '#' sub-steps)
//   scanTape(model, trace, layer, unit?, o?)      one frame per output element of one layer (lazy explain())
//   bpeTape(r)                                    trainBPE(...) result -> one frame per merge
//   generateTape(r, prompt?)                      TinyGPT generate(...) result -> one frame per sampled token
// No imports from src/lib/models: the results are typed structurally here, so the dependency rules hold.
import type { I18n, Provenance, Tensor, Trace } from '../types/core.ts';
import type { AtlasSpec } from '../types/spec.ts';
import type { GraphModel, UnitExplanation } from '../types/op.ts';
import type { Cursor, Focus, Frame, Schema, SchemaJson, Tape, TapeHeader } from '../types/process.ts';
import { TapeStore } from './tape.ts';
import { flatIndex } from './values.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });

function stats(t: Tensor): { max: number; min: number; mean: number } {
	let max = -Infinity;
	let min = Infinity;
	let sum = 0;
	let n = 0;
	for (let i = 0; i < t.data.length; i++) {
		const v = t.data[i];
		if (Number.isNaN(v)) continue;
		if (v > max) max = v;
		if (v < min) min = v;
		if (Number.isFinite(v)) {
			sum += v;
			n++;
		}
	}
	return { max, min, mean: n ? sum / n : NaN };
}

// ------------------------------------------------------------------ trace
export const TRACE_SCHEMA: Schema = {
	layer: { kind: 'text', label: L('Layer', '层') },
	id: { kind: 'text', label: L('Trace id', '轨迹 id') },
	out: { kind: 'tensor', label: L('Output', '输出'), role: 'activation' }
};
export const TRACE_PHASES = {
	input: { label: L('Input', '输入'), kind: 'input' as const },
	layer: { label: L('Layer', '层'), kind: 'forward' as const },
	sub: { label: L('Inside the layer', '层内部'), kind: 'forward' as const },
	output: { label: L('Output', '输出'), kind: 'output' as const }
};

/** One frame per id in trace.order. state = { layer (id without '#sub'), id, out }. Cursor { layer: 'conv1_1' } finds the layer's
 *  own frame (sub-steps follow it). Tensors are shared with the trace, not copied (a trace is never mutated). */
export function traceToTape(spec: AtlasSpec, trace: Trace, o?: { process?: string; provenance?: Provenance }): TapeStore {
	const store = new TapeStore({
		process: o?.process ?? `trace:${spec.slug}`,
		seed: 0,
		opt: { dims: trace.meta?.dims ?? {} },
		provenance: o?.provenance ?? trace.meta?.provenance ?? 'live',
		version: spec.version,
		schema: TRACE_SCHEMA,
		phases: TRACE_PHASES
	});
	const byId = new Map(spec.layers.map((l) => [l.id, l]));
	const declared = new Set((spec.inputs ?? []).map((i) => i.name));
	const outputs = new Set(spec.outputs ?? []);
	const lastLayer = [...trace.order].reverse().find((id) => !id.includes('#'));
	if (!outputs.size && lastLayer) outputs.add(lastLayer);
	const subCount = new Map<string, number>();
	for (const id of trace.order) {
		const t = trace.t[id];
		if (!t) continue;
		const hash = id.indexOf('#');
		const layer = hash >= 0 ? id.slice(0, hash) : id;
		const def = byId.get(layer);
		const phase = hash >= 0 ? 'sub' : def?.op === 'input' || declared.has(id) ? 'input' : outputs.has(id) ? 'output' : 'layer';
		const st = stats(t);
		const f: Frame = { phase, state: { layer, id, out: t }, metrics: st };
		if (hash >= 0) {
			subCount.set(layer, (subCount.get(layer) ?? -1) + 1);
			f.sub = subCount.get(layer);
		}
		if (def?.label) f.caption = hash >= 0 ? { en: `${def.label.en} · ${id.slice(hash + 1)}`, zh: `${def.label.zh ?? def.label.en} · ${id.slice(hash + 1)}` } : def.label;
		store.push(f, { owned: true });
	}
	return store;
}

// ------------------------------------------------------------------ scan
export interface ScanOpts {
	order?: 'raster' | 'peak-first';
	/** take every stride-th index along each free dim (default 1) */
	stride?: number;
	/** cap on frames; larger scans are thinned evenly (default 4096) */
	max?: number;
	provenance?: Provenance;
}

/** The units a scan visits. `template`: entries >= 0 are fixed, -1 (or missing trailing entries) are free.
 *  [3] on a [8, 26, 26] map scans the 26×26 cells of channel 3; [-1, 5] on [T, H] scans every t for unit 5. */
export function scanUnits(shape: number[], template?: number[], o?: ScanOpts): number[][] {
	const stride = Math.max(1, Math.floor(o?.stride ?? 1));
	const tpl = shape.map((n, d) => {
		const v = template?.[d];
		return v === undefined || v < 0 ? -1 : Math.min(n - 1, Math.max(0, Math.floor(v)));
	});
	let units: number[][] = [[]];
	shape.forEach((n, d) => {
		const next: number[][] = [];
		const vals = tpl[d] >= 0 ? [tpl[d]] : Array.from({ length: Math.ceil(n / stride) }, (_, k) => k * stride);
		for (const u of units) for (const v of vals) next.push([...u, v]);
		units = next;
	});
	if (shape.some((n) => n <= 0)) return [];
	const max = o?.max ?? 4096;
	if (units.length > max) {
		const step = units.length / max;
		units = Array.from({ length: max }, (_, k) => units[Math.floor(k * step)]);
	}
	return units;
}

const SCAN_SCHEMA: Schema = {
	layer: { kind: 'text', label: L('Layer', '层') },
	unit: { kind: 'json', label: L('Output cell', '输出格') },
	value: { kind: 'scalar', label: L('Value', '数值'), role: 'activation' },
	explain: { kind: 'json', label: L('The arithmetic', '计算过程') }
};

export interface ScanState { layer: string; unit: number[]; value: number; explain?: UnitExplanation; error?: string }

/** A lazy tape over one layer's output cells: frame i explains unit i with model.explain() on first access. */
export class ScanTape implements Tape<ScanState> {
	readonly process: string;
	readonly seed = 0;
	readonly opt: unknown;
	readonly provenance: Provenance;
	readonly version: string;
	readonly units: number[][];
	readonly layer: string;
	private readonly memo = new Map<number, Frame<ScanState>>();
	private readonly out: Tensor;
	private readonly model: GraphModel;
	private readonly trace: Trace;

	constructor(model: GraphModel, trace: Trace, layer: string, template?: number[], o?: ScanOpts) {
		this.model = model;
		this.trace = trace;
		this.layer = layer;
		const out = trace.t[layer];
		if (!out) throw new Error(`scanTape: layer ${layer} is not in the trace`);
		this.out = out;
		this.process = `scan:${layer}`;
		this.provenance = o?.provenance ?? trace.meta?.provenance ?? 'live';
		this.version = model.spec.version;
		this.opt = { layer, template: template ?? null, order: o?.order ?? 'raster', stride: o?.stride ?? 1 };
		let units = scanUnits(out.shape, template, o);
		if (o?.order === 'peak-first' && units.length > 1) {
			let best = 0;
			let bv = -Infinity;
			units.forEach((u, k) => {
				const v = out.data[flatIndex(out.shape, u)];
				if (v > bv) (bv = v), (best = k);
			});
			units = [...units.slice(best), ...units.slice(0, best)];
		}
		this.units = units;
	}

	get length(): number {
		return this.units.length;
	}

	/** The scanned layer's axes (the scrubber names its jump marks by the axis the scan walks). */
	get axes(): string[] | undefined {
		try {
			return this.model.layer(this.layer).axes?.slice();
		} catch {
			return undefined;
		}
	}

	frame(i: number): Frame<ScanState> {
		if (!Number.isInteger(i) || i < 0 || i >= this.units.length) throw new RangeError(`${this.process}: frame ${i} is not recorded (length ${this.units.length})`);
		let f = this.memo.get(i);
		if (f) return f;
		const unit = this.units[i];
		const value = this.out.data[flatIndex(this.out.shape, unit)];
		const state: ScanState = { layer: this.layer, unit, value };
		const focus: Focus[] = [{ field: 'out', index: unit, role: 'out' }];
		const metrics: Record<string, number> = { value };
		try {
			const x = this.model.explain(this.trace, this.layer, unit);
			state.explain = x;
			metrics.result = x.result;
			metrics.traced = x.traced;
			for (const r of x.reads) {
				const t = this.trace.t[r.tensor];
				if (!t) continue;
				focus.push({ field: r.tensor, index: r.cells.map((c) => unflat(t.shape, c)), role: 'in' });
			}
		} catch (e) {
			state.error = e instanceof Error ? e.message : String(e);
		}
		f = { phase: 'unit', step: i, state, focus, metrics };
		this.memo.set(i, f);
		if (this.memo.size > 512) this.memo.delete(this.memo.keys().next().value as number);
		return f;
	}

	state(i: number): ScanState {
		return this.frame(i).state as ScanState;
	}

	find(c: Cursor): number {
		const n = this.units.length;
		if (c === 'first') return n ? 0 : -1;
		if (c === 'last') return n - 1;
		if (typeof c === 'number') return Number.isInteger(c) && c >= 0 && c < n ? c : -1;
		const match = (i: number) => {
			if (c.phase !== undefined && c.phase !== 'unit') return false;
			if (c.step !== undefined && c.step !== i) return false;
			if (c.layer !== undefined && c.layer !== this.layer) return false;
			if (c.sub !== undefined) return false;
			return c.where ? c.where(this.frame(i), this.state(i)) : true;
		};
		if (c.last) {
			for (let i = n - 1; i >= 0; i--) if (match(i)) return i;
			return -1;
		}
		if (c.step !== undefined && !c.where) return match(c.step) && c.step < n ? c.step : -1;
		for (let i = 0; i < n; i++) if (match(i)) return i;
		return -1;
	}

	segments() {
		return this.units.length ? [{ phase: 'unit', from: 0, to: this.units.length - 1 }] : [];
	}

	metric(name: string): Float64Array {
		const out = new Float64Array(this.units.length);
		for (let i = 0; i < out.length; i++) {
			if (name === 'value') out[i] = this.out.data[flatIndex(this.out.shape, this.units[i])];
			else out[i] = this.frame(i).metrics?.[name] ?? NaN;
		}
		return out;
	}

	schemaJson(): SchemaJson {
		return SCAN_SCHEMA as SchemaJson;
	}

	phasesJson(): TapeHeader['phases'] {
		return { unit: { label: L('One output cell', '一个输出格'), kind: 'forward' } };
	}
}

function unflat(shape: number[], flat: number): number[] {
	const idx = new Array<number>(shape.length);
	for (let d = shape.length - 1; d >= 0; d--) {
		idx[d] = flat % shape[d];
		flat = Math.floor(flat / shape[d]);
	}
	return idx;
}

export function scanTape(model: GraphModel, trace: Trace, layer: string, unit?: number[], o?: ScanOpts): ScanTape {
	return new ScanTape(model, trace, layer, unit, o);
}

// ------------------------------------------------------------------ BPE training (models/bpe.ts trainBPE)
/** Structural copy of models/bpe.ts TrainResult. */
export interface BpeTrainResult {
	merges: [number, number][];
	steps: { rank: number; pair: [number, number]; newId: number; count: number; bytes: number[]; display: string; tokenCount: number }[];
	initialTokenCount: number;
}

export const BPE_SCHEMA: Schema = {
	pair: { kind: 'json', label: L('Merged pair', '合并的一对'), tol: 'exact' },
	newId: { kind: 'int', label: L('New token id', '新词元 id'), tol: 'exact' },
	count: { kind: 'int', label: L('How often the pair occurred', '这一对出现的次数'), tol: 'exact', role: 'data' },
	bytes: { kind: 'json', label: L('Bytes of the new token', '新词元的字节'), tol: 'exact' },
	display: { kind: 'text', label: L('New token', '新词元'), tol: 'exact' },
	tokenCount: { kind: 'int', label: L('Tokens in the text', '文本中的词元数'), tol: 'exact', role: 'output' },
	vocab: { kind: 'int', label: L('Vocabulary size', '词表大小'), tol: 'exact' },
	mergeTable: { kind: 'tensor', label: L('Merges so far (a, b)', '已学到的合并 (a, b)'), shape: ['M', 2], track: 'changes', tol: 'exact' }
};

/** One 'start' frame, then one 'merge' frame per learned merge (step = rank). mergeTable [M, 2] (rows not learned yet are
 *  -1) is tracked by changes, so state(i).mergeTable rows 0..i-1 are exactly trainBPE's merges up to that point. */
export function bpeTape(r: BpeTrainResult, o?: { process?: string; provenance?: Provenance }): TapeStore {
	const M = r.steps.length;
	const store = new TapeStore({
		process: o?.process ?? 'bpe-train',
		seed: 0,
		opt: { numMerges: M },
		provenance: o?.provenance ?? 'live',
		version: '1',
		schema: BPE_SCHEMA,
		phases: {
			start: { label: L('Raw bytes', '原始字节'), kind: 'input' },
			merge: { label: L('Merge the most frequent pair', '合并最常见的一对'), kind: 'merge' }
		}
	});
	const table: Tensor = { shape: [M, 2], data: new Float64Array(M * 2).fill(-1) };
	store.push({ phase: 'start', state: { tokenCount: r.initialTokenCount, vocab: 256, mergeTable: table }, metrics: { tokenCount: r.initialTokenCount, compression: 1 } });
	for (const s of r.steps) {
		store.push({
			phase: 'merge',
			step: s.rank,
			state: { pair: [s.pair[0], s.pair[1]], newId: s.newId, count: s.count, bytes: s.bytes.slice(), display: s.display, tokenCount: s.tokenCount, vocab: s.newId + 1 },
			changes: [
				{ field: 'mergeTable', index: [s.rank, 0], from: -1, to: s.pair[0] },
				{ field: 'mergeTable', index: [s.rank, 1], from: -1, to: s.pair[1] }
			],
			metrics: { count: s.count, tokenCount: s.tokenCount, compression: r.initialTokenCount / s.tokenCount }
		});
	}
	return store;
}

// ------------------------------------------------------------------ TinyGPT generate()
/** Structural copy of models/tinygpt.ts GenerateResult. */
export interface GenerateLike {
	ids: number[];
	newIds: number[];
	steps: { token: number; p: number; pModel: number; top: { id: number; p: number }[]; u: number | null; contextStart: number }[];
}

export const GENERATE_SCHEMA: Schema = {
	ids: { kind: 'json', label: L('Token ids so far', '目前的词元 id'), tol: 'exact', role: 'data' },
	token: { kind: 'int', label: L('Chosen token', '选中的词元'), tol: 'exact', role: 'output' },
	p: { kind: 'scalar', label: L('Its probability (after temperature and cut-offs)', '它的概率（温度和截断之后）'), tol: 1e-6 },
	pModel: { kind: 'scalar', label: L('Its probability under the plain model', '原始模型给它的概率'), tol: 1e-6 },
	u: { kind: 'scalar', label: L('Random number used (NaN when greedy)', '用到的随机数（贪心时为 NaN）'), tol: 'exact' },
	top: { kind: 'json', label: L('Most likely tokens', '最可能的词元'), tol: 1e-6 },
	contextStart: { kind: 'int', label: L('First token the model saw', '模型看到的第一个词元'), tol: 'exact' }
};

/** One 'prompt' frame, then one 'sample' frame per generated token (step = 0, 1, …). */
export function generateTape(r: GenerateLike, o?: { process?: string; provenance?: Provenance; seed?: number; opt?: unknown }): TapeStore {
	const store = new TapeStore({
		process: o?.process ?? 'generate',
		seed: o?.seed ?? 0,
		opt: o?.opt ?? {},
		provenance: o?.provenance ?? 'live',
		version: '1',
		schema: GENERATE_SCHEMA,
		phases: {
			prompt: { label: L('Your prompt', '你的提示'), kind: 'input' },
			sample: { label: L('Pick the next token', '选出下一个词元'), kind: 'sample' }
		}
	});
	const prompt = r.ids.slice(0, r.ids.length - r.newIds.length);
	store.push({ phase: 'prompt', state: { ids: prompt.slice() }, metrics: { length: prompt.length } });
	const ids = prompt.slice();
	r.steps.forEach((s, k) => {
		ids.push(s.token);
		store.push({
			phase: 'sample',
			step: k,
			state: { ids: ids.slice(), token: s.token, p: s.p, pModel: s.pModel, u: s.u ?? NaN, top: s.top.map((t) => ({ id: t.id, p: t.p })), contextStart: s.contextStart },
			metrics: { p: s.p, pModel: s.pModel, u: s.u ?? NaN, bits: -Math.log2(s.pModel) }
		});
	});
	return store;
}
