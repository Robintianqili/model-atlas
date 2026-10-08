// Reference cases for kind-A parity (design §4.2 reference.json).
//   readReference(fetch, dir) -> RefCase[]       loads <dir>/reference.json (or spec.files.reference)
//   parseReference(raw)       -> RefCase[]       pure; reads three layouts:
//     atlas-ref-1   { format: 'atlas-ref-1', cases: [{ id, label, edge, inputs: { name: {shape, dtype, b64} }, order, tensors }] }
//     TinyCNN v1    { inputs: [{ label, input: number[], layers: { id: { shape, data: number[] } } }] }
//     TinyGPT v1    { prompts: [{ text, ids, order, tensors: { id: { shape, dtype, b64 } } }] }
// b64 is base64 of little-endian values; −Infinity is stored as is.
import type { AnyTensor, FetchLike, Tensor } from '../types/index.ts';
import { f32FromB64, f64FromB64, i32FromB64, joinUrl, loadJson } from '../core/io.ts';

export interface RefCase {
	id: string;
	label: string;
	edge: boolean;
	inputs: Record<string, AnyTensor>;
	order: string[];
	tensors: Record<string, Tensor>;
}

interface B64 {
	shape: number[];
	dtype?: string;
	b64: string;
}

function isObj(x: unknown): x is Record<string, unknown> {
	return typeof x === 'object' && x !== null && !Array.isArray(x);
}

function decode(t: B64, what: string): AnyTensor {
	const dtype = t.dtype ?? 'float32';
	if (dtype === 'int32' || dtype === 'i32') return { shape: t.shape.slice(), data: i32FromB64(t.b64), dtype: 'i32' };
	const data = dtype === 'float64' || dtype === 'f64' ? f64FromB64(t.b64) : f32FromB64(t.b64);
	const n = t.shape.reduce((a, b) => a * b, 1);
	if (data.length !== n) throw new Error(`reference ${what}: ${data.length} values for shape [${t.shape}]`);
	return { shape: t.shape.slice(), data };
}

function decodeFloat(t: B64, what: string): Tensor {
	const x = decode(t, what);
	if ((x as { dtype?: string }).dtype === 'i32') return { shape: x.shape, data: Float32Array.from(x.data) };
	return x as Tensor;
}

/** Decode any of the three reference layouts. */
export function parseReference(raw: unknown): RefCase[] {
	if (!isObj(raw)) throw new Error('reference: not a JSON object');
	if (raw.format === 'atlas-ref-1') {
		const cases = raw.cases as unknown[];
		if (!Array.isArray(cases)) throw new Error('atlas-ref-1: cases must be an array');
		return cases.map((c, i) => {
			const cc = c as { id?: string; label?: string; edge?: boolean; inputs: Record<string, B64>; order?: string[]; tensors: Record<string, B64> };
			const inputs: Record<string, AnyTensor> = {};
			for (const [k, v] of Object.entries(cc.inputs ?? {})) inputs[k] = decode(v, `case ${i} input ${k}`);
			const tensors: Record<string, Tensor> = {};
			for (const [k, v] of Object.entries(cc.tensors ?? {})) tensors[k] = decodeFloat(v, `case ${i} ${k}`);
			return { id: String(cc.id ?? i), label: String(cc.label ?? cc.id ?? i), edge: !!cc.edge, inputs, order: cc.order ?? Object.keys(tensors), tensors };
		});
	}
	if (Array.isArray(raw.prompts)) {
		// TinyGPT v1
		return (raw.prompts as { text?: string; ids: number[]; order: string[]; tensors: Record<string, B64> }[]).map((p) => {
			const tensors: Record<string, Tensor> = {};
			for (const [k, v] of Object.entries(p.tensors)) tensors[k] = decodeFloat(v, k);
			return {
				id: `T${p.ids.length}`,
				label: p.text ?? `T=${p.ids.length}`,
				edge: false,
				inputs: { input: { shape: [p.ids.length], data: Int32Array.from(p.ids), dtype: 'i32' } },
				order: p.order.slice(),
				tensors
			};
		});
	}
	if (Array.isArray(raw.inputs)) {
		// TinyCNN v1
		return (raw.inputs as { label?: string; input: number[]; layers: Record<string, { shape: number[]; data: number[] }> }[]).map((r, i) => {
			const tensors: Record<string, Tensor> = {};
			for (const [k, v] of Object.entries(r.layers)) tensors[k] = { shape: v.shape.slice(), data: Float32Array.from(v.data) };
			const shape = r.layers.input?.shape ?? [r.input.length];
			return { id: String(i), label: r.label ?? String(i), edge: false, inputs: { input: { shape: shape.slice(), data: Float32Array.from(r.input) } }, order: Object.keys(r.layers), tensors };
		});
	}
	throw new Error('reference: unknown layout (expected atlas-ref-1, TinyCNN inputs[] or TinyGPT prompts[])');
}

/** Load and decode <dir>/reference.json (pages never call this; parity and tests do). */
export async function readReference(fetch: FetchLike, dir: string, file = 'reference.json'): Promise<RefCase[]> {
	return parseReference(await loadJson(fetch, joinUrl(dir, file)));
}
