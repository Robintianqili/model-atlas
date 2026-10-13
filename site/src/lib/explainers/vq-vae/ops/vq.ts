// vq-vae local op 'vq' (vector quantization): z_e [D, H, W] and the codebook [K, D] -> z_q [D, H, W]. Each cell's D
// numbers are replaced by the codebook row at the smallest squared distance Σ_d (z_e[d] − e_k[d])² (ties: the smaller
// index). Sub-steps 'dist' [K, H, W] (every squared distance) and 'idx' [H, W] (the code indices, as numbers).
// Python mirror: pipeline/vq-vae/ops_vq.py (model.py quantize()).
import type { AnyTensor, OpDef, Tensor, UnitExplanation } from '../../../atlas/types/index.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, specWeightShape, store, t, unitBase, weightName } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;

/** squared distance of cell c of z_e [D, H, W] to code k (an f64 sum in the order d = 0 … D−1, as forward) */
export function dist2(ze: AnyTensor, cb: Tensor, c: number, k: number): number {
	const D = cb.shape[1], n = ze.data.length / D;
	let s = 0;
	for (let d = 0; d < D; d++) {
		const v = ze.data[d * n + c] - cb.data[k * D + d];
		s += v * v;
	}
	return s;
}

/** the nearest code of cell c: { k, d }, ties to the smaller index */
export function nearest(ze: AnyTensor, cb: Tensor, c: number): { k: number; d: number } {
	let k = 0, d = Infinity;
	for (let j = 0; j < cb.shape[0]; j++) {
		const s = dist2(ze, cb, c, j);
		if (s < d) [k, d] = [j, s];
	}
	return { k, d };
}

export const vq: OpDef<NoP> = {
	op: 'vq',
	version: 1,
	attrs: {},
	category: 'embedding',
	label: t('Nearest code', '最近的码字'),
	weights: [{ slot: 'codebook' }],
	subs: [
		{ name: 'dist', label: t('Squared distance to every code', '到每个码字的距离平方'), axes: ['k', 'y', 'x'] },
		{ name: 'idx', label: t('The code index of each patch', '每个图块的码字编号'), axes: ['y', 'x'] }
	],
	inferShape(ins, _p, w) {
		const cb = w.codebook, x = ins[0];
		if (!cb || cb.length !== 2) throw new Error('vq: codebook must be [K, D]');
		if (x.length !== 3 || (typeof x[0] === 'number' && x[0] !== cb[1])) throw new Error(`vq: input must be [${cb[1]}, H, W]`);
		return x.slice();
	},
	axes: () => ['d', 'y', 'x'],
	forward(xs, _p, ctx) {
		const ze = xs[0], cb = ctx.w('codebook');
		const [K, D] = cb.shape, [, H, W] = ze.shape, n = H * W;
		const out = outArray(D * n, ze, cb), dist = outArray(K * n, ze, cb), idx = outArray(n, ze, cb);
		for (let c = 0; c < n; c++) {
			for (let k = 0; k < K; k++) dist[k * n + c] = dist2(ze, cb, c, k);
			const { k } = nearest(ze, cb, c);
			idx[c] = k;
			for (let d = 0; d < D; d++) out[d * n + c] = cb.data[k * D + d];
		}
		ctx.emit('dist', { shape: [K, H, W], data: dist });
		ctx.emit('idx', { shape: [H, W], data: idx });
		return { shape: [D, H, W], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'first' },
	explain(q): UnitExplanation {
		// units of z_q are [d, row, col]; the executor also asks for the sub-steps 'dist' ([k, row, col]: values.dist) and
		// 'idx' ([row, col]: values.idx) through this function
		const cb = q.w('codebook'), ze = q.ins[0], [K, D] = cb.shape, [, H, W] = q.out.shape;
		const [d, row, col] = q.unit.length === 2 ? [0, ...q.unit] : q.unit, c = row * W + col;
		const { k, d: best } = nearest(ze, cb, c);
		const y = d < D ? cb.data[k * D + d] : NaN;
		const e = unitBase('none', q.layer, q.unit, y, d < D ? q.out.data[(d * H + row) * W + col] : NaN);
		e.reads = [
			{ tensor: q.layer.in[0], cells: Array.from({ length: D }, (_, j) => j * H * W + c) },
			{ tensor: weightName(q.layer, 'codebook'), cells: Array.from({ length: D }, (_, j) => k * D + j) }
		];
		e.values = { k, d, row, col, best: store(best, q.out), y, idx: k, dist: d < K ? store(dist2(ze, cb, c, d), q.out) : NaN };
		return e;
	},
	formula: `${sym('out', 'z_q')}[{{d}}, {{row}}, {{col}}] = ${sym('param', 'e')}_{ {{k}} }[{{d}}] = {{y}}, \\quad k = \\arg\\min_j \\lVert ${sym('in', 'z_e')} - ${sym('param', 'e')}_j \\rVert^2,\\ \\lVert ${sym('in', 'z_e')} - ${sym('param', 'e')}_{ {{k}} } \\rVert^2 = {{best}}`,
	narration: {
		what: t(
			'The codebook is a table of {codes|int} learned rows, each a list of {dim|int} numbers. Each patch’s {dim|int} numbers z_e are replaced by the row nearest to them: its code.',
			'码本是一张 {codes|int} 行的表，每一行是 {dim|int} 个学来的数。每个图块的 {dim|int} 个数 z_e 都被换成离它最近的那一行：它的码字。'
		),
		how: t(
			'For every row k it adds up the squared differences, Σ (z_e − e_k)², and keeps the row with the smallest sum (ties go to the smaller index).',
			'对每一行 k，把差的平方加起来，Σ (z_e − e_k)²，留下和最小的那一行（相等时取编号小的）。'
		),
		why: t('After the snap, each patch is just one whole number: which row it took. That is the compression.', '吸附之后，每个图块只剩一个整数：它取了哪一行。这就是压缩。'),
		look: t('Patches that look alike take the same code.', '看起来相像的图块取同一个码字。'),
		unit: t('Row {row|int}, column {col|int} (counting from 0) took code {k|int} at squared distance {best}; its number {d|int} is {y}.', '第 {row|int} 行第 {col|int} 列（从 0 数起）取了码字 {k|int}，距离平方 {best}；它的第 {d|int} 个数是 {y}。')
	},
	vars(layer, spec) {
		const ws = specWeightShape(layer, spec, 'codebook');
		return { ...baseVars(layer, spec), codes: ws?.[0] ?? '?', dim: ws?.[1] ?? '?' };
	},
	zoom: 'none',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'codebook[((ze[:, None] - codebook[None, :, :, None, None]) ** 2).sum(2).argmin(1)]',
	terms: ['vector-quantization', 'codebook']
};

export default vq;
