// tinymoe local op 'moe': a mixture-of-experts layer with top-k routing (GShard / Switch Transformer; Mixtral's k = 2).
//   inputs  x [T, D] (the normalized token vectors), p [T, E] (the router's chances: a softmax over the E experts)
//   weights fc [E, H, D], fc_b [E, H], out [E, D, H], out_b [E, D]   expert j: out_j · relu(fc_j · x + fc_b_j) + out_b_j
//   gate[t] = the k largest chances of p[t] rescaled to add up to 1 (ties: the lower expert), 0 for the others
//                                                                                        sub-step '#gate' [T, E]
//   e[t, j] = expert j's output for token t, computed only where gate[t, j] > 0 (0 elsewhere)   sub-step '#e' [T, E, D]
//   y[t] = Σ_j gate[t, j] · e[t, j]                                                       output [T, D]
// Accumulates in f64 and stores each number with the output's precision (f32), as PyTorch keeps them; explain([t, d]) is
// the gate-weighted sum of the chosen experts' outputs. Python mirror: pipeline/tinymoe/ops_moe.py.
import type { AnyTensor, OpDef, Tensor, UnitExplanation } from '../../../atlas/types/index.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, needRank, outArray, t, unitBase } from '../../../atlas/ops/common.ts';

export type MoeP = { k: number };

const rounder = (like: ArrayLike<number>) => (like instanceof Float64Array ? (v: number) => v : Math.fround);

/** The k experts with the largest chances (ties: the lower index) and their gates: the chances rescaled to add up to 1
 *  (the sum rounded as the output stores numbers, as PyTorch's top_p.sum() is). */
export function topK(p: ArrayLike<number>, k: number, rd: (v: number) => number = (v) => v): { i: number[]; g: number[] } {
	const i = Array.from({ length: p.length }, (_, j) => j)
		.sort((a, b) => p[b] - p[a] || a - b)
		.slice(0, k);
	const s = rd(i.reduce((a, j) => a + p[j], 0));
	return { i, g: i.map((j) => rd(p[j] / s)) };
}

/** Expert j on token s: out_j · relu(fc_j · x_s + fc_b_j) + out_b_j, accumulated in f64, each stored number rounded
 *  (rd) as PyTorch keeps it. Shared by forward() and explain(). */
function expert(w: (slot: string) => Tensor, x: AnyTensor, s: number, j: number, rd: (v: number) => number): number[] {
	const [fc, fb, ow, ob] = ['fc', 'fc_b', 'out', 'out_b'].map((k) => w(k).data);
	const [, H, D] = w('fc').shape;
	const h = new Float64Array(H);
	for (let m = 0; m < H; m++) {
		let a = 0;
		for (let i = 0; i < D; i++) a += fc[(j * H + m) * D + i] * x.data[s * D + i];
		h[m] = Math.max(0, rd(a + fb[j * H + m]));
	}
	return Array.from({ length: D }, (_, d) => {
		let a = 0;
		for (let m = 0; m < H; m++) a += ow[(j * D + d) * H + m] * h[m];
		return rd(a + ob[j * D + d]);
	});
}

export const moe: OpDef<MoeP> = {
	op: 'moe',
	version: 1,
	attrs: { k: { type: 'int', required: true, doc: 'experts per token (top-k)' } },
	category: 'merge',
	label: t('Mixture of experts (top-k)', '混合专家（取前 k 个）'),
	weights: [{ slot: 'fc' }, { slot: 'fc_b' }, { slot: 'out' }, { slot: 'out_b' }],
	subs: [
		{ name: 'gate', label: t('gates (top-k chances, rescaled)', '门控（前 k 个概率，重新缩放）'), axes: ['t', 'k'] },
		{ name: 'e', label: t('expert outputs', '专家输出'), axes: ['t', 'k', 'd'] }
	],
	inferShape(ins, p, w) {
		const [x, pr] = ins;
		if (ins.length !== 2 || x.length !== 2 || pr.length !== 2) throw new Error('moe: inputs are x [T, D] and p [T, E]');
		const [E, H, D] = w.fc ?? [];
		if (pr[1] !== E || x[1] !== D || !w.out || w.out[0] !== E || w.out[1] !== D || w.out[2] !== H) throw new Error(`moe: fc must be [E, H, ${x[1]}] and out [E, ${x[1]}, H] with E = ${pr[1]}`);
		if (!(p.k >= 1 && p.k <= E)) throw new Error(`moe: k must be 1 to ${E}`);
		return [x[0], x[1]];
	},
	axes: () => ['t', 'd'],
	forward(xs, p, ctx) {
		const [x, pr] = xs;
		needRank(x, 2, 'moe x');
		const [T, D] = x.shape;
		const E = pr.shape[1];
		const Y = outArray(T * D, x);
		const G = outArray(T * E, x);
		const Ex = outArray(T * E * D, x);
		const rd = rounder(Y);
		for (let s = 0; s < T; s++) {
			const { i: chosen, g } = topK(pr.data.subarray(s * E, (s + 1) * E), p.k, rd);
			const y = new Float64Array(D);
			chosen.forEach((j, n) => {
				G[s * E + j] = g[n];
				expert(ctx.w, x, s, j, rd).forEach((v, d) => ((Ex[(s * E + j) * D + d] = v), (y[d] += g[n] * v)));
			});
			for (let d = 0; d < D; d++) Y[s * D + d] = rd(y[d]);
		}
		ctx.emit('gate', { shape: [T, E], data: G });
		ctx.emit('e', { shape: [T, E, D], data: Ex });
		return { shape: [T, D], data: Y };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmaxAbs' },
	// unit [t, d] of the output (also [t, j] of '#gate': values.gate) or [t, j, d] of '#e' (values.e)
	explain(q): UnitExplanation {
		const [s, u1, u2] = q.unit;
		const d = q.unit.length === 3 ? u2 : u1;
		const D = q.out.shape[1];
		const E = q.ins[1].shape[1];
		const rd = rounder(q.out.data);
		const { i: chosen, g } = topK(q.ins[1].data.subarray(s * E, (s + 1) * E), q.p.k, rd);
		const outs = chosen.map((j) => expert(q.w, q.ins[0], s, j, rd));
		const terms = chosen.map((j, n) => ({ x: outs[n][d], w: g[n], p: outs[n][d] * g[n], xAt: (s * E + j) * D + d, wAt: s * E + j }));
		const y = rd(terms.reduce((a, x) => a + x.p, 0));
		const e = unitBase('weighted-sum', q.layer, q.unit, y, q.out.data[s * D + d]);
		e.reads = [{ tensor: `${q.layer.id}#e`, cells: terms.map((x) => x.xAt) }, { tensor: `${q.layer.id}#gate`, cells: terms.map((x) => x.wAt) }];
		e.terms = terms;
		const [a, b] = [terms[0], terms[1] ?? { x: 0, w: 0, p: 0 }];
		const at = (j: number) => chosen.indexOf(j);
		e.values = { n: terms.length, s: y, y, ea: chosen[0], eb: chosen[1] ?? -1, ga: a.w, gb: b.w, xa: a.x, xb: b.x, pa: a.p, pb: b.p, t: s, d, tNo: s + 1, dNo: d + 1,
			gate: u1 < E ? (at(u1) >= 0 ? g[at(u1)] : 0) : NaN,
			e: q.unit.length === 3 ? (at(u1) >= 0 ? outs[at(u1)][d] : 0) : NaN };
		return e;
	},
	formula: `${sym('out', 'y')}_{t} = ${sym('param', 'g')}_{a}\\, ${sym('act', 'E')}_{a}(${sym('in', 'x')}_t) + ${sym('param', 'g')}_{b}\\, ${sym('act', 'E')}_{b}(${sym('in', 'x')}_t) = {{ga}} \\times {{xa}} + {{gb}} \\times {{xb}} = {{y}}`,
	narration: {
		what: t(
			'A mixture of experts: instead of one MLP, the layer holds several small ones (the experts). For each token the router’s chances pick the {k|int} most likely experts, only those run, and their outputs are added, each weighted by its gate.',
			'混合专家：这一层不是一个 MLP，而是好几个小 MLP（专家）。对每个词元，路由器的概率挑出最可能的 {k|int} 个专家，只运行它们，再把它们的输出按各自的门控加权相加。'
		),
		how: t(
			'The gates are the {k|int} largest chances rescaled to add up to 1; every other expert gets 0 and is skipped. The output is y = Σ gate · expert(x).',
			'门控就是最大的 {k|int} 个概率，重新缩放到和为 1；其余专家的门控是 0，直接跳过。输出是 y = Σ 门控 · 专家(x)。'
		),
		look: t('Each token row lights up only {k|int} experts.', '每个词元那一行只点亮 {k|int} 个专家。'),
		unit: t(
			'Token {t|int}, number {d|int} (counting from 0): expert {ea|int} gives {xa} with gate {ga}, expert {eb|int} gives {xb} with gate {gb}; together {y}.',
			'第 {t|int} 个词元、第 {d|int} 个数（从 0 数起）：专家 {ea|int} 给出 {xa}，门控 {ga}；专家 {eb|int} 给出 {xb}，门控 {gb}；合起来是 {y}。'
		)
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), k: Number((layer.params as Partial<MoeP>)?.k ?? NaN) }),
	zoom: 'weighted-sum',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'top_p, top_i = p.topk(k);  gate = top_p / top_p.sum();  y = sum(g * experts[i](x) for g, i in zip(gate, top_i))',
	terms: ['mixture-of-experts']
};

export default moe;
