// contrastive-learning local ops (design §7). Python mirrors: pipeline/contrastive-learning/ops_clip.py. f64 accumulation, f32 storage.
//   mean_pool   word rows [..., L, E] and their ids [..., L] -> [..., E]: the average of the rows whose id is not 0
//               (pads); 0 when a caption has no known word
//   l2norm      every last-axis row divided by its length √(Σ x²), so the row has length 1 (F.normalize, eps 1e-12)
//   similarity  a [D] or [N, D] (image vectors) and b [M, D] or [D] (caption vectors) -> [N, M] scores s · (a · b);
//               s = e^t (t: the learned weight logit_scale, as CLIP stores it), capped at maxScale
import type { AnyTensor, OpDef, Tensor, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, store, t, unitBase, weightName } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;
const last = <T>(a: readonly T[]) => a[a.length - 1];

/** row k's sum over its words, the word count and the positions used */
function poolRow(x: Tensor, ids: AnyTensor, k: number, d: number) {
	const L = x.shape[x.shape.length - 2], E = last(x.shape);
	let s = 0;
	const at: number[] = [];
	for (let i = 0; i < L; i++) if (ids.data[k * L + i] !== 0) {
		s += x.data[(k * L + i) * E + d];
		at.push(i);
	}
	return { s, n: at.length, at, L, E };
}

export const mean_pool: OpDef<NoP> = {
	op: 'mean_pool', version: 1, attrs: {}, category: 'pool',
	label: t('Average of the words', '词的平均'),
	inferShape: (ins) => [...ins[0].slice(0, -2), last(ins[0])],
	axes: (ins) => [...ins[0].slice(0, -2), 'd'],
	forward(xs) {
		const [x, ids] = xs as [Tensor, AnyTensor];
		const E = last(x.shape), K = x.data.length / (E * x.shape[x.shape.length - 2]);
		const out = outArray(K * E, x);
		for (let k = 0; k < K; k++) for (let d = 0; d < E; d++) {
			const r = poolRow(x, ids, k, d);
			out[k * E + d] = r.s / Math.max(r.n, 1);
		}
		return { shape: [...x.shape.slice(0, -2), E], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const [x, ids] = q.ins as [Tensor, AnyTensor];
		const d = last(q.unit), k = q.unit.length > 1 ? q.unit[0] : 0;
		const r = poolRow(x, ids, k, d);
		const w = 1 / Math.max(r.n, 1);
		const y = store(r.s / Math.max(r.n, 1), q.out);
		const e = unitBase('weighted-sum', q.layer, q.unit, y, q.out.data[flatIndex(q.out.shape, q.unit)]);
		e.reads = [{ tensor: q.layer.in[0], cells: r.at.map((i) => (k * r.L + i) * r.E + d) }];
		e.terms = r.at.map((i) => ({ x: x.data[(k * r.L + i) * r.E + d], w, p: x.data[(k * r.L + i) * r.E + d] * w, xAt: (k * r.L + i) * r.E + d }));
		e.values = { s: r.s, n: r.n, y, d, k };
		return e;
	},
	formula: `${sym('out', 'y')}_d = \\frac{1}{n}\\sum_{t} ${sym('in', 'e')}_{t,d} = \\frac{ {{s}} }{ {{n}} } = {{y}}`,
	narration: {
		what: t('Each caption’s word rows are averaged into one row of {d|int} numbers. Padding (id 0) is skipped, and the order of the words does not matter.', '每句描述的词行被平均成一行 {d|int} 个数。填充（编号 0）不算在内，词的顺序也不影响结果。'),
		how: t('Number d of the result is the sum of number d of every word row, divided by the number of words.', '结果的第 d 个数，是每个词行的第 d 个数之和，再除以词的个数。'),
		look: t('Common words such as “a” pull every caption toward the same average.', '像 “a” 这样常见的词，会把每句描述都往同一个平均值拉。'),
		unit: t('Number {d|int} (counting from 0): the {n|int} words add up to {s}, and {s} ÷ {n|int} = {y}.', '第 {d|int} 个数（从 0 数起）：{n|int} 个词加起来是 {s}，{s} ÷ {n|int} = {y}。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), d: Number(last(layer.outShape)) }),
	zoom: 'weighted-sum', color: 'diverging', parity: { bar: 1e-4 },
	pytorch: () => '(e * m).sum(-2) / m.sum(-2).clamp(min=1)   # m = (ids != 0)',
	terms: ['embedding']
};

const EPS = 1e-12;
function lengthOf(x: ArrayLike<number>, o: number, D: number): number {
	let s = 0;
	for (let i = 0; i < D; i++) s += x[o + i] * x[o + i];
	return Math.sqrt(s);
}

export const l2norm: OpDef<NoP> = {
	op: 'l2norm', version: 1, attrs: {}, category: 'norm',
	label: t('Scale to length 1', '缩放到长度 1'),
	inferShape: (ins) => ins[0].slice(),
	axes: (ins) => ins[0].slice(),
	forward(xs) {
		const x = xs[0] as Tensor;
		const D = last(x.shape);
		const out = outArray(x.data.length, x);
		for (let o = 0; o < x.data.length; o += D) {
			const n = Math.max(lengthOf(x.data, o, D), EPS);
			for (let i = 0; i < D; i++) out[o + i] = x.data[o + i] / n;
		}
		return { shape: x.shape.slice(), data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const x = q.ins[0] as Tensor;
		const D = last(x.shape);
		const f = flatIndex(q.out.shape, q.unit);
		const o = f - last(q.unit);
		const len = Math.max(lengthOf(x.data, o, D), EPS);
		const xv = x.data[f];
		const e = unitBase('none', q.layer, q.unit, store(xv / len, q.out), q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells: Array.from({ length: D }, (_, i) => o + i) }];
		e.values = { x: xv, len, y: e.result, n: D };
		return e;
	},
	formula: `${sym('out', 'y')}_i = \\frac{ ${sym('in', 'x')}_i }{ \\sqrt{\\sum_j x_j^2} } = \\frac{ {{x}} }{ {{len}} } = {{y}}`,
	narration: {
		what: t('Every row of {d|int} numbers is divided by its length, the square root of the sum of its squares. Afterwards each row has length 1, so only its direction is left.', '每一行 {d|int} 个数都除以它的长度，也就是各数平方和的平方根。之后每一行的长度都是 1，只剩下方向。'),
		how: t('Square every number, add them up, take the square root: that is the length. Then divide each number by it.', '把每个数平方、加起来、开平方根，就是长度。再把每个数除以它。'),
		look: t('The bars keep their pattern; only their size changes.', '这些条的形状不变，只是整体大小变了。'),
		unit: t('{x} ÷ {len} = {y}.', '{x} ÷ {len} = {y}。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), d: Number(last(layer.outShape)) }),
	zoom: 'none', color: 'diverging', parity: { bar: 1e-4 },
	pytorch: () => 'F.normalize(x, dim=-1)',
	terms: []
};

type SimP = { maxScale: number };
export const scaleOf = (t0: number, max = 100) => Math.min(Math.exp(t0), max);

export const similarity: OpDef<SimP> = {
	op: 'similarity', version: 1, attrs: { maxScale: { type: 'float', default: 100, doc: 'cap on the scale e^t (CLIP: 100)' } }, category: 'merge',
	label: t('Similarity score', '相似度分数'),
	weights: [{ slot: 'logit_scale' }],
	inferShape: (ins) => [...ins[0].slice(0, -1), ...ins[1].slice(0, -1)],
	axes: (ins) => [...(ins[0].length > 1 ? ['n' as const] : []), ...(ins[1].length > 1 ? ['k' as const] : [])],
	forward(xs, p, ctx) {
		const [a, b] = xs as [Tensor, Tensor];
		const D = last(a.shape), N = a.data.length / D, M = b.data.length / D;
		const s = scaleOf(ctx.w('logit_scale').data[0], p.maxScale ?? 100);
		const out = outArray(N * M, a);
		for (let i = 0; i < N; i++) for (let j = 0; j < M; j++) {
			let c = 0;
			for (let d = 0; d < D; d++) c += a.data[i * D + d] * b.data[j * D + d];
			out[i * M + j] = s * c;
		}
		return { shape: [...a.shape.slice(0, -1), ...b.shape.slice(0, -1)], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const [a, b] = q.ins as [Tensor, Tensor];
		const D = last(a.shape), M = b.data.length / D;
		const f = flatIndex(q.out.shape, q.unit);
		const i = Math.floor(f / M), j = f % M;
		const scale = scaleOf(q.w('logit_scale').data[0], q.p.maxScale ?? 100);
		let c = 0;
		const terms: NonNullable<UnitExplanation['terms']> = [];
		for (let d = 0; d < D; d++) {
			const pr = a.data[i * D + d] * b.data[j * D + d];
			c += pr;
			terms.push({ x: a.data[i * D + d], w: b.data[j * D + d], p: pr, xAt: i * D + d, wAt: j * D + d });
		}
		const e = unitBase('weighted-sum', q.layer, q.unit, store(scale * c, q.out), q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells: terms.map((x) => x.xAt!) }, { tensor: q.layer.in[1], cells: terms.map((x) => x.wAt!) }, { tensor: weightName(q.layer, 'logit_scale'), cells: [0] }];
		e.terms = terms;
		e.values = { cos: c, s: c, scale, t: q.w('logit_scale').data[0], y: e.result, n: D, i, j };
		return e;
	},
	formula: `${sym('out', 'y')} = s\\,(${sym('in', 'a')} \\cdot ${sym('param', 'b')}) = {{scale}} \\times {{cos}} = {{y}}`,
	narration: {
		what: t('The score of a drawing and a caption: multiply their {d|int} numbers pair by pair and add up. For two vectors of length 1 this sum is the cosine of the angle between them, from −1 to 1. Then it is multiplied by a learned scale s.', '一幅画和一句描述的分数：把它们的 {d|int} 个数逐对相乘再相加。对两个长度为 1 的向量，这个和就是它们夹角的余弦，在 −1 到 1 之间。然后再乘以一个学来的缩放 s。'),
		how: t('s = e^t, where t is learned with the other weights (CLIP’s temperature, written as a scale).', 's = e^t，其中 t 和其他权重一起学出来（CLIP 的温度，这里写成缩放）。'),
		look: t('Pairs that point the same way score high; unrelated ones near 0.', '方向一致的一对分数高；不相关的接近 0。'),
		unit: t('The {n|int} products add up to the cosine {cos}; × {scale} = {y}.', '{n|int} 个乘积加起来是余弦 {cos}；× {scale} = {y}。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), d: Number(last((spec.layers.find((l) => l.id === layer.in[0])?.outShape ?? ['?']) as (number | string)[])) }),
	zoom: 'weighted-sum', color: 'diverging', parity: { bar: 1e-4 },
	pytorch: () => 'logit_scale.exp().clamp(max=100) * (b @ a)',
	terms: ['temperature']
};
