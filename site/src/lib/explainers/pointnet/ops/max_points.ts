// pointnet local op max_points: [n, F] -> [F], for each feature the largest value over all n points (torch
// x.max(dim=0).values; Python mirror pipeline/pointnet/ops_max_points.py). The first largest point wins a tie, as the
// core maxpool2d does. explain(): a reduce like maxpool2d's: terms[i] = { x: point i's value, w: 1 for the winner else
// 0, p: x·w }, so Σp is the max; values.argmax is the winning point (counting from 0), values.n the number of points.
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, needRank, outArray, store, t, unitBase } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;

/** the largest value of feature f over the points of x [n, F], and the first point that has it */
export function winner(x: ArrayLike<number>, n: number, F: number, f: number) {
	let m = -Infinity, at = 0;
	for (let i = 0; i < n; i++) if (x[i * F + f] > m) (m = x[i * F + f]), (at = i);
	return { max: m, at };
}

export const maxPoints: OpDef<NoP> = {
	op: 'max_points',
	version: 1,
	attrs: {},
	category: 'pool',
	label: t('Max over the points', '对所有点取最大值'),
	inferShape(ins) {
		if (ins[0].length !== 2) throw new Error(`max_points: input must be [n, F], got [${ins[0]}]`);
		return [ins[0][1]];
	},
	axes: () => ['f'],
	forward(xs) {
		const src = xs[0];
		needRank(src, 2, 'max_points input');
		const [n, F] = src.shape;
		const out = outArray(F, src);
		for (let f = 0; f < F; f++) out[f] = winner(src.data, n, F, f).max;
		return { shape: [F], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const [f] = q.unit;
		const src = q.ins[0];
		const [n, F] = src.shape;
		const w = winner(src.data, n, F, f);
		const cells = Array.from({ length: n }, (_, i) => i * F + f);
		const y = store(w.max, q.out);
		const e = unitBase('reduce', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells }];
		e.terms = cells.map((c, i) => ({ x: src.data[c], w: i === w.at ? 1 : 0, p: i === w.at ? src.data[c] : 0, xAt: c }));
		e.layout = { rows: Math.ceil(n / 8), cols: Math.min(n, 8) }; // 8 points a row, in order (a 1-column window of 72+ rows is too tall)
		e.values = { n, y, max: w.max, argmax: w.at, f };
		return e;
	},
	formula: `${sym('out', 'g')}_f = \\max_{i}\\ ${sym('in', 'h')}_{i,f} = {{y}}`,
	narration: {
		what: t(
			'For each of the {outShape} features, keep only the largest value that any point gave it. The {inShape} table of points becomes one row: the whole shape in {outShape} numbers.',
			'对 {outShape} 个特征中的每一个，只保留所有点给出的最大值。{inShape} 的点表就变成一行：整个形状用 {outShape} 个数来表示。'
		),
		how: t('Each feature is maxed on its own, over all the points. Nothing is learned here: there are no weights.', '每个特征各自对所有点取最大值。这里没有要学的东西，也没有权重。'),
		why: t('A maximum does not care in which order the points come, or how many times a point is repeated.', '最大值不在乎点按什么顺序来，也不在乎一个点重复了几次。'),
		look: t('Each output number equals the largest number of its column in the table before.', '每个输出数都等于前一张表里对应那一列中最大的数。'),
		unit: t('Feature {f|int} (counting from 0): the largest of the {n|int} values is {y}, from point {argmax|int}.', '特征 {f|int}（从 0 数起）：{n|int} 个值中最大的是 {y}，来自点 {argmax|int}。')
	},
	vars: baseVars,
	zoom: 'reduce',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'x.max(dim=0).values',
	terms: ['pooling']
};

export default maxPoints;
