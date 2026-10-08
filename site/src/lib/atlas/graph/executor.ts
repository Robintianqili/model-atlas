// The kind-A executor (design §3.2, §4.3): spec + op registry -> GraphModel.
//   fromParts(spec, weights, ops)        spec: raw JSON or AtlasSpec (normalized by legacy.ts); weights: weights.bin bytes
//                                        (float32 LE, laid out by spec.tensors) or tensors by name; ops: OpRegistry | OpDef[]
//   loadGraph(fetch, baseUrl, ops)       fetches spec.json and the weights file (spec.files.weights ?? 'weights.bin')
//   model.forward(input, { record, dims })
//       input: one tensor (the first declared input) or { <input name>: tensor }. Symbolic dims (T) are read from the
//       inputs and checked against spec.dims; every layer output is checked against its outShape.
//       record 'all' (default): layers plus sub-steps `${id}#${sub}` (those in layer.sub when given, else every emitted
//       sub-step) right after their layer; 'layers': layer outputs only.
//   model.explain(trace, 'conv1_1', [c, y, x]) -> UnitExplanation (a sub id 'lstm1#f' explains that gate)
//   model.receptiveField(layer, box)      inclusive box in the input image, or null for non-image inputs
// Precision: float32 storage unless spec.precision is 'f64' (then weights and inputs are widened to Float64Array).
import type { AnyTensor, AtlasLayer, AtlasSpec, Box, FetchLike, GraphModel, OpCtx, OpDef, OpRegistry, Tensor, Trace, TraceMeta, UnitExplanation } from '../types/index.ts';
import { flatIndex, isIntTensor, numel, resolveShape, shapeEq } from '../core/tensor.ts';
import { joinUrl, loadBytes, loadJson } from '../core/io.ts';
import { MissingWeightError, weightName } from '../ops/common.ts';
import { mergeParams, toRegistry } from '../ops/registry.ts';
import { normalizeSpec } from './legacy.ts';
import { receptiveField as receptiveWalk } from './receptive.ts';

/** Declared inputs of every trace this module produced (explain() needs token ids, which are not in the trace). */
const INPUTS = new WeakMap<Trace, Record<string, AnyTensor>>();

/** The declared inputs a trace was computed from (undefined for traces made elsewhere, e.g. sent from a worker). */
export function traceInputs(trace: Trace): Record<string, AnyTensor> | undefined {
	return INPUTS.get(trace);
}

interface Prepared {
	layer: AtlasLayer;
	def: OpDef;
	params: Record<string, unknown>;
}

function where(): TraceMeta['where'] {
	const g = globalThis as { window?: unknown; importScripts?: unknown; process?: { versions?: { node?: string } } };
	if (g.window !== undefined) return 'main';
	if (typeof g.importScripts === 'function') return 'worker';
	return g.process?.versions?.node ? 'node' : 'main';
}

function now(): number {
	return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function readWeights(spec: AtlasSpec, buf: ArrayBuffer, f64: boolean): Record<string, Tensor> {
	const total = spec.tensors.reduce((m, t) => Math.max(m, t.offset + t.length), 0);
	if (buf.byteLength < total * 4) throw new Error(`weights: ${buf.byteLength} bytes, spec.tensors need ${total * 4}`);
	const dv = new DataView(buf);
	const all = f64 ? new Float64Array(total) : new Float32Array(total);
	for (let i = 0; i < total; i++) all[i] = dv.getFloat32(i * 4, true);
	const out: Record<string, Tensor> = {};
	for (const t of spec.tensors) {
		if (numel(t.shape) !== t.length) throw new Error(`weights: tensor ${t.name} has shape [${t.shape}] but length ${t.length}`);
		out[t.name] = { shape: t.shape.slice(), data: all.subarray(t.offset, t.offset + t.length) };
	}
	return out;
}

function isTensorLike(x: unknown): x is AnyTensor {
	const t = x as AnyTensor;
	return !!t && Array.isArray(t.shape) && !!t.data && typeof (t.data as ArrayLike<number>).length === 'number';
}

export function fromParts(specIn: AtlasSpec | unknown, weightsIn: ArrayBuffer | Record<string, Tensor>, opsIn: OpRegistry | OpDef[], o: { weightsLabel?: string } = {}): GraphModel {
	const spec = normalizeSpec(specIn);
	const ops = toRegistry(opsIn);
	const f64 = spec.precision === 'f64';
	let weights: Record<string, Tensor>;
	if (weightsIn instanceof ArrayBuffer) weights = readWeights(spec, weightsIn, f64);
	else {
		weights = {};
		for (const [k, v] of Object.entries(weightsIn)) weights[k] = f64 && !(v.data instanceof Float64Array) ? { shape: v.shape.slice(), data: Float64Array.from(v.data) } : v;
	}
	const byId = new Map<string, AtlasLayer>();
	const prepared: Prepared[] = [];
	for (const layer of spec.layers) {
		if (layer.id.includes('#')) throw new Error(`layer id '${layer.id}' contains '#', which is reserved for sub-steps`);
		if (byId.has(layer.id)) throw new Error(`duplicate layer id '${layer.id}'`);
		byId.set(layer.id, layer);
		const r = ops.get(layer.op);
		prepared.push({ layer, def: r.def, params: mergeParams(r.def, r.params, layer.params) });
	}
	const declared = spec.inputs ?? [];
	const weightsLabel = o.weightsLabel ?? spec.files?.weights ?? 'weights.bin';

	const getLayer = (id: string): AtlasLayer => {
		const l = byId.get(id);
		if (!l) throw new Error(`unknown layer '${id}'`);
		return l;
	};

	function bindInputs(input: AnyTensor | Record<string, AnyTensor>): Record<string, AnyTensor> {
		const named: Record<string, AnyTensor> = isTensorLike(input) ? { [declared[0]?.name ?? 'input']: input } : { ...input };
		for (const [k, v] of Object.entries(named)) {
			if (f64 && !isIntTensor(v) && !(v.data instanceof Float64Array)) named[k] = { shape: v.shape.slice(), data: Float64Array.from(v.data) };
		}
		return named;
	}

	function inferDims(inputs: Record<string, AnyTensor>, given?: Record<string, number>): Record<string, number> {
		const dims: Record<string, number> = { ...(given ?? {}) };
		for (const d of declared) {
			const x = inputs[d.name];
			if (!x) continue;
			if (x.shape.length !== d.shape.length) throw new Error(`input '${d.name}': shape [${x.shape}] does not fit the declared [${d.shape}]`);
			d.shape.forEach((sd, i) => {
				const v = x.shape[i];
				if (typeof sd === 'number') {
					if (sd !== v) throw new Error(`input '${d.name}': axis ${i} is ${v}, the spec declares ${sd}`);
				} else if (dims[sd] === undefined) dims[sd] = v;
				else if (dims[sd] !== v) throw new Error(`dim '${sd}' is ${dims[sd]} elsewhere but ${v} in input '${d.name}'`);
			});
		}
		for (const [name, r] of Object.entries(spec.dims ?? {})) {
			const v = dims[name];
			if (v === undefined) continue;
			if (!Number.isInteger(v) || v < r.min || v > r.max) throw new Error(`dim '${name}' = ${v} is outside ${r.min}..${r.max}`);
		}
		return dims;
	}

	function forward(input: AnyTensor | Record<string, AnyTensor>, opt: { record?: 'all' | 'layers'; dims?: Record<string, number> } = {}): Trace {
		const t0 = now();
		const inputs = bindInputs(input);
		for (const d of declared) if (!inputs[d.name]) throw new Error(`missing input '${d.name}'`);
		const dims = inferDims(inputs, opt.dims);
		const record = opt.record ?? 'all';
		const t: Record<string, Tensor> = {};
		const order: string[] = [];
		for (const { layer, def, params } of prepared) {
			const ins: AnyTensor[] = layer.in.map((n) => {
				const x = t[n] ?? inputs[n];
				if (!x) throw new Error(`layer '${layer.id}': input '${n}' is neither an earlier layer nor a declared input`);
				return x;
			});
			const emitted: [string, Tensor][] = [];
			const ctx: OpCtx = {
				layer,
				dims,
				w(slot: string) {
					const name = weightName(layer, slot);
					const w = weights[name];
					if (!w) throw new MissingWeightError(slot, name, layer.id);
					return w;
				},
				has: (slot: string) => !!weights[weightName(layer, slot)],
				emit(sub: string, x: Tensor) {
					emitted.push([sub, x]);
				}
			};
			const out = def.forward(ins, params, ctx);
			let want: number[] | null = null;
			try {
				want = resolveShape(layer.outShape, dims);
			} catch {
				want = null; // a symbolic dim the inputs do not determine: skip the check
			}
			if (want && !shapeEq(out.shape, want)) throw new Error(`layer '${layer.id}' (${layer.op}): output [${out.shape}] but outShape is [${layer.outShape}] = [${want}]`);
			t[layer.id] = out;
			order.push(layer.id);
			if (record === 'all') {
				for (const [sub, x] of emitted) {
					if (layer.sub && !layer.sub.includes(sub)) continue;
					const id = `${layer.id}#${sub}`;
					t[id] = x;
					order.push(id);
				}
			}
		}
		const trace: Trace = { order, t, meta: { dims, weights: weightsLabel, provenance: 'live', ms: now() - t0, where: where() } };
		INPUTS.set(trace, inputs);
		return trace;
	}

	function explain(trace: Trace, id: string, unit: number[]): UnitExplanation {
		const hash = id.indexOf('#');
		const parent = hash >= 0 ? id.slice(0, hash) : id;
		const sub = hash >= 0 ? id.slice(hash + 1) : null;
		const layer = getLayer(parent);
		const prep = prepared.find((p) => p.layer === layer)!;
		const out = trace.t[parent];
		if (!out) throw new Error(`explain: the trace has no '${parent}'`);
		const target = sub ? trace.t[id] : out;
		if (!target) throw new Error(`explain: the trace has no '${id}' (was it recorded?)`);
		const f = flatIndex(target.shape, unit); // throws a RangeError for a unit outside the tensor
		const inputs = INPUTS.get(trace);
		const ins: AnyTensor[] = layer.in.map((n) => {
			const x = (n === parent ? undefined : trace.t[n]) ?? inputs?.[n] ?? trace.t[n];
			if (!x) throw new Error(`explain '${id}': input '${n}' is not in the trace and the declared inputs are unknown (run forward() on this model)`);
			return x;
		});
		if (!prep.def.explain) throw new Error(`op '${layer.op}' has no explain()`);
		const e = prep.def.explain({
			layer,
			unit,
			ins,
			out,
			p: prep.params,
			trace,
			spec,
			w(slot: string) {
				const name = weightName(layer, slot);
				const w = weights[name];
				if (!w) throw new MissingWeightError(slot, name, layer.id);
				return w;
			},
			has: (slot: string) => !!weights[weightName(layer, slot)]
		});
		if (!sub) return e;
		const v = e.values[sub];
		if (typeof v !== 'number') throw new Error(`op '${layer.op}' cannot explain sub-step '${sub}' (no value named '${sub}')`);
		return { ...e, layer: id, result: v, traced: target.data[f] };
	}

	const model: GraphModel = {
		spec,
		weights,
		ops,
		forward,
		explain,
		receptiveField: (layerId: string, box: Box) => receptiveWalk(spec, ops, layerId, box),
		layer: getLayer
	};
	return model;
}

/** Fetch spec.json and the weights from a model folder (e.g. `${base}/models/tinycnn`) and build the model. */
export async function loadGraph(fetch: FetchLike, baseUrl: string, ops: OpRegistry | OpDef[]): Promise<GraphModel> {
	const raw = await loadJson(fetch, joinUrl(baseUrl, 'spec.json'));
	const spec = normalizeSpec(raw);
	const file = spec.files?.weights ?? 'weights.bin';
	const buf = await loadBytes(fetch, joinUrl(baseUrl, file));
	return fromParts(spec, buf, ops, { weightsLabel: file });
}
