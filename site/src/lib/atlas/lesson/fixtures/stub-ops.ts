// Stub ops for W1-C's tests (design §8: "stub other packages' exports only inside your own folder"). They follow the
// OpDef contract (attrs, inferShape, forward with f64 accumulation and f32 storage, explain, narration in en and zh)
// so the same fixture package also runs on W1-A's executor after integration. They are not CORE_OPS.
import type { AnyTensor, Tensor } from '../../types/core.ts';
import type { OpDef, UnitExplanation } from '../../types/op.ts';

const f32 = (shape: number[], data: ArrayLike<number>): Tensor => ({ shape, data: Float32Array.from(data) });
const asF = (x: AnyTensor): ArrayLike<number> => x.data;

type NoP = Record<string, never>;
type LinP = { accumulate: string };

export const stubInput: OpDef<NoP> = {
	op: 'input',
	version: 1,
	attrs: {},
	category: 'io',
	label: { en: 'Input', zh: '输入' },
	inferShape: (ins) => ins[0] ?? [],
	forward: (xs) => f32(xs[0].shape, asF(xs[0])),
	narration: { what: { en: 'The input as numbers.', zh: '输入变成的数。' } },
	zoom: 'none',
	parity: { bar: 0 }
};

export const stubEmbed: OpDef<NoP> = {
	op: 'stub_embed',
	version: 1,
	attrs: {},
	category: 'embedding',
	label: { en: 'Letter vectors', zh: '字母向量' },
	weights: [{ slot: 'weight' }],
	inferShape: (ins, _p, w) => [ins[0][0], w.weight?.[1] ?? 'D'],
	forward(xs, _p, ctx) {
		const W = ctx.w('weight');
		const [V, D] = W.shape;
		const ids = asF(xs[0]);
		const out = new Float32Array(ids.length * D);
		for (let t = 0; t < ids.length; t++) {
			const id = Math.max(0, Math.min(V - 1, ids[t]));
			for (let d = 0; d < D; d++) out[t * D + d] = W.data[id * D + d];
		}
		return { shape: [ids.length, D], data: out };
	},
	unit: { axes: () => ['t', 'd'], scan: 'none', pick: 'first' },
	explain(q): UnitExplanation {
		const [t, d] = q.unit;
		const W = q.w('weight');
		const id = q.ins[0].data[t];
		const v = W.data[id * W.shape[1] + d];
		return { kind: 'lookup', layer: q.layer.id, unit: q.unit, reads: [{ tensor: q.layer.in[0], cells: [t] }], values: { id, row: id, value: v }, result: v, traced: q.out.data[t * q.out.shape[1] + d] };
	},
	narration: {
		what: { en: 'Each letter number picks one row of a learned table: {D|int} numbers that stand for that letter.', zh: '每个字母编号从一张学来的表里取出一行：代表这个字母的 {D|int} 个数。' }
	},
	vars: (layer, spec) => ({ D: (spec.tensors.find((x) => x.name === `${layer.id}.weight`)?.shape[1] ?? NaN) as number }),
	zoom: 'lookup',
	parity: { bar: 1e-6 }
};

export const stubLinear: OpDef<LinP> = {
	op: 'stub_linear',
	version: 1,
	attrs: { accumulate: { type: 'string', default: 'bias-last' } },
	aliases: { stub_dense: {} },
	category: 'linear',
	label: { en: 'Weighted sum', zh: '加权和' },
	weights: [{ slot: 'weight' }, { slot: 'bias', optional: true }],
	inferShape: (ins, _p, w) => [...ins[0].slice(0, -1), w.weight?.[0] ?? 'O'],
	forward(xs, _p, ctx) {
		const W = ctx.w('weight');
		const b = ctx.w('bias');
		const [O, I] = W.shape;
		const x = asF(xs[0]);
		const rows = x.length / I;
		const out = new Float32Array(rows * O);
		for (let r = 0; r < rows; r++)
			for (let o = 0; o < O; o++) {
				let s = 0;
				for (let i = 0; i < I; i++) s += x[r * I + i] * W.data[o * I + i];
				out[r * O + o] = s + b.data[o];
			}
		return { shape: [...xs[0].shape.slice(0, -1), O], data: out };
	},
	unit: { axes: (a) => a, scan: 'rows', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const W = q.w('weight');
		const b = q.w('bias');
		const [, I] = W.shape;
		const o = q.unit[q.unit.length - 1];
		const r = q.unit.length > 1 ? q.unit[0] : 0;
		const x = q.ins[0].data;
		const terms = [];
		let s = 0;
		for (let i = 0; i < I; i++) {
			const p = x[r * I + i] * W.data[o * I + i];
			terms.push({ x: x[r * I + i], w: W.data[o * I + i], p, xAt: r * I + i, wAt: o * I + i });
			s += p;
		}
		const O = W.shape[0];
		return {
			kind: 'weighted-sum',
			layer: q.layer.id,
			unit: q.unit,
			reads: [{ tensor: q.layer.in[0], cells: Array.from({ length: I }, (_, i) => r * I + i) }],
			terms,
			bias: b.data[o],
			values: { sum: s, bias: b.data[o] },
			result: Math.fround(s + b.data[o]),
			traced: q.out.data[r * O + o]
		};
	},
	narration: {
		what: { en: 'Each output number is a weighted sum of the {inFeatures|int} numbers before it, plus a bias.', zh: '每个输出数都是前面 {inFeatures|int} 个数的加权和，再加一个偏置。' },
		how: { en: 'For the highlighted cell the products add up to {sum}; with the bias {bias} that gives {result}.', zh: '对高亮的格子，乘积加起来是 {sum}；加上偏置 {bias}，得到 {result}。' },
		repeat: { en: 'Another weighted sum, now with {inFeatures|int} inputs per output.', zh: '又一个加权和，现在每个输出有 {inFeatures|int} 个输入。' }
	},
	vars: (layer, spec) => ({ inFeatures: (spec.tensors.find((x) => x.name === (layer.w?.weight ?? `${layer.id}.weight`))?.shape[1] ?? NaN) as number }),
	distinct: () => false,
	zoom: 'weighted-sum',
	parity: { bar: 1e-6 }
};

export const stubTanh: OpDef<NoP> = {
	op: 'stub_tanh',
	version: 1,
	attrs: {},
	category: 'activation',
	label: { en: 'Squash with tanh', zh: '用 tanh 压缩' },
	inferShape: (ins) => ins[0],
	forward: (xs) => ({ shape: xs[0].shape.slice(), data: Float32Array.from(asF(xs[0]), (v) => Math.tanh(v)) }),
	unit: { axes: (a) => a, scan: 'none', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		let i = 0;
		for (let d = 0; d < q.out.shape.length; d++) i = i * q.out.shape[d] + q.unit[d];
		const x = q.ins[0].data[i];
		const y = Math.fround(Math.tanh(x));
		return { kind: 'elementwise', layer: q.layer.id, unit: q.unit, reads: [{ tensor: q.layer.in[0], cells: [i] }], curve: { fn: 'tanh', x, y }, values: { x, y }, result: y, traced: q.out.data[i] };
	},
	narration: { what: { en: 'tanh squashes every number into the range between minus one and one. The highlighted {x} becomes {y}.', zh: 'tanh 把每个数压到负一和一之间。高亮的 {x} 变成 {y}。' } },
	zoom: 'elementwise',
	parity: { bar: 1e-6 }
};

export const stubMeanPool: OpDef<NoP> = {
	op: 'stub_meanpool',
	version: 1,
	attrs: {},
	category: 'pool',
	label: { en: 'Average over positions', zh: '对位置求平均' },
	inferShape: (ins) => [ins[0][1]],
	forward(xs) {
		const [T, D] = xs[0].shape;
		const x = asF(xs[0]);
		const out = new Float32Array(D);
		for (let d = 0; d < D; d++) {
			let s = 0;
			for (let t = 0; t < T; t++) s += x[t * D + d];
			out[d] = s / T;
		}
		return { shape: [D], data: out };
	},
	unit: { axes: () => ['d'], scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const [T, D] = q.ins[0].shape;
		const d = q.unit[0];
		let s = 0;
		for (let t = 0; t < T; t++) s += q.ins[0].data[t * D + d];
		return { kind: 'reduce', layer: q.layer.id, unit: q.unit, reads: [{ tensor: q.layer.in[0], cells: Array.from({ length: T }, (_, t) => t * D + d) }], values: { n: T, sum: s, mean: s / T }, result: Math.fround(s / T), traced: q.out.data[d] };
	},
	narration: { what: { en: 'Averaging over all {n|int} positions turns the sequence into one vector of shape {outShape}.', zh: '对全部 {n|int} 个位置求平均，把序列变成一个形状为 {outShape} 的向量。' } },
	zoom: 'reduce',
	parity: { bar: 1e-6 }
};

export const stubSoftmax: OpDef<NoP> = {
	op: 'stub_softmax',
	version: 1,
	attrs: {},
	category: 'output',
	label: { en: 'Softmax', zh: 'Softmax' },
	inferShape: (ins) => ins[0],
	forward(xs) {
		const x = asF(xs[0]);
		let m = -Infinity;
		for (let i = 0; i < x.length; i++) m = Math.max(m, x[i]);
		let z = 0;
		const e = Array.from(x, (v) => Math.exp(v - m));
		for (const v of e) z += v;
		return f32(xs[0].shape, e.map((v) => v / z));
	},
	unit: { axes: (a) => a, scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const x = q.ins[0].data;
		const i = q.unit[0];
		let m = -Infinity;
		for (let k = 0; k < x.length; k++) m = Math.max(m, x[k]);
		let z = 0;
		for (let k = 0; k < x.length; k++) z += Math.exp(x[k] - m);
		const p = Math.exp(x[i] - m) / z;
		return { kind: 'softmax', layer: q.layer.id, unit: q.unit, reads: [{ tensor: q.layer.in[0], cells: Array.from({ length: x.length }, (_, k) => k) }], values: { logit: x[i], max: m, sumExp: z, p }, result: Math.fround(p), traced: q.out.data[i] };
	},
	narration: { what: { en: 'Softmax turns the scores into chances that add up to one.', zh: 'Softmax 把分数变成加起来等于一的概率。' } },
	zoom: 'softmax',
	parity: { bar: 1e-6 }
};

export const STUB_OPS: OpDef[] = [stubInput, stubEmbed, stubLinear, stubTanh, stubMeanPool, stubSoftmax];
