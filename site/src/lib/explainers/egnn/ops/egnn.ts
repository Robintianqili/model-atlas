// egnn local ops (Python mirrors: pipeline/egnn/ops_egnn.py, the same arithmetic as model.py). Every atom pair (i, j) is
// one cell of an [n, n] table; the diagonal (i = j) is computed but never summed.
//   pair_dist2   x [n, D] -> d² [n, n]: d²_ij = Σ_d (x_id − x_jd)²; explain(): one term per coordinate, (x_id − x_jd)²
//   edge_concat  h [n, F], d² [n, n] -> [n, n, 2F + 1]: the input of the message network for pair (i, j): h_i, h_j, d²_ij
//   vec_sum      x [n, D], w [n, n, 1] -> [n, D]: v_i = Σ_{j≠i} (x_i − x_j) w_ij / (n − 1) (the push on atom i, divided
//                by 1 when n = 1); explain(): one term per other atom j, (x_id − x_jd) × w_ij / (n − 1)
//   nb_sum       m [n, n, M] -> [n, M]: Σ_{j≠i} m_ij (the messages atom i receives); explain(): one term per j
//   sum_atoms    y [n, C] -> [C]: Σ_i y_i (the energy: one share per atom)
// f64 accumulation, f32 storage (atlas/ops/common.ts conventions).
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, store, t, unitBase } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;
const rank = (s: unknown[], r: number, what: string) => {
	if (s.length !== r) throw new Error(`${what} must have ${r} axes, got [${s}]`);
};
/** the push of atom i along coordinate d: Σ_{j≠i} (x_id − x_jd) w_ij / max(1, n − 1), with its terms */
export function push(x: ArrayLike<number>, w: ArrayLike<number>, n: number, D: number, i: number, d: number) {
	const c = 1 / Math.max(1, n - 1);
	const terms: { j: number; x: number; w: number; p: number }[] = [];
	let s = 0;
	for (let j = 0; j < n; j++) {
		if (j === i) continue;
		const dx = x[i * D + d] - x[j * D + d], wc = w[i * n + j] * c;
		terms.push({ j, x: dx, w: wc, p: dx * wc });
		s += dx * wc;
	}
	return { s, terms };
}

export const pair_dist2: OpDef<NoP> = {
	op: 'pair_dist2',
	version: 1,
	attrs: {},
	category: 'graph',
	label: t('Squared distances of every pair', '每对原子的距离平方'),
	inferShape: (ins) => (rank(ins[0], 2, 'pair_dist2: x'), [ins[0][0], ins[0][0]]),
	axes: () => ['n', 'k'],
	forward([x]) {
		const [n, D] = x.shape;
		const out = outArray(n * n, x);
		for (let i = 0; i < n; i++)
			for (let j = 0; j < n; j++) {
				let s = 0;
				for (let d = 0; d < D; d++) s += (x.data[i * D + d] - x.data[j * D + d]) ** 2;
				out[i * n + j] = s;
			}
		return { shape: [n, n], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'argmax' },
	explain(q): UnitExplanation {
		const [i, j] = q.unit;
		const x = q.ins[0];
		const [n, D] = x.shape;
		const terms = Array.from({ length: D }, (_, d) => {
			const v = x.data[i * D + d] - x.data[j * D + d];
			return { x: v, w: v, p: v * v, xAt: i * D + d, wAt: j * D + d };
		});
		const y = store(terms.reduce((s, z) => s + z.p, 0), q.out);
		const e = unitBase('weighted-sum', q.layer, [i, j], y, q.out.data[i * n + j]);
		e.reads = [{ tensor: q.layer.in[0], cells: terms.flatMap((z) => [z.xAt, z.wAt]) }];
		e.terms = terms;
		e.values = { i, j, y };
		return e;
	},
	formula: `${sym('out', 'd^2')}_{ {{i}},{{j}} } = \\sum_d (${sym('in', 'x')}_{ {{i}},d } - ${sym('in', 'x')}_{ {{j}},d })^2 = {{y}}`,
	narration: {
		what: t('For every pair of atoms i and j, the squared distance between them: (xᵢ − xⱼ)² + (yᵢ − yⱼ)². The table has shape {outShape}.', '对每一对原子 i 和 j，算出它们之间距离的平方：(xᵢ − xⱼ)² + (yᵢ − yⱼ)²。这张表的形状是 {outShape}。'),
		how: t('Only differences of positions enter, and their squares: no direction is left.', '只用到位置之差，再平方：方向信息一点不剩。'),
		why: t('A turn or a shift of the whole molecule leaves every distance as it was.', '整个分子转动或平移，每个距离都保持不变。'),
		unit: t('Atoms {i|int} and {j|int} (counting from 0): d² = {y}.', '原子 {i|int} 和 {j|int}（从 0 数起）：d² = {y}。')
	},
	vars: baseVars,
	zoom: 'weighted-sum',
	color: 'sequential',
	parity: { bar: 1e-4 },
	pytorch: () => '((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)',
	terms: ['invariance']
};

export const edge_concat: OpDef<NoP> = {
	op: 'edge_concat',
	version: 1,
	attrs: {},
	category: 'graph',
	label: t('Inputs of every message', '每条消息的输入'),
	inferShape(ins) {
		rank(ins[0], 2, 'edge_concat: h');
		const F = ins[0][1];
		return [ins[0][0], ins[0][0], typeof F === 'number' ? 2 * F + 1 : '?'];
	},
	axes: () => ['n', 'k', 'f'],
	forward([h, d2]) {
		const [n, F] = h.shape;
		const C = 2 * F + 1;
		const out = outArray(n * n * C, h);
		for (let i = 0; i < n; i++)
			for (let j = 0; j < n; j++) {
				const o = (i * n + j) * C;
				for (let f = 0; f < F; f++) (out[o + f] = h.data[i * F + f]), (out[o + F + f] = h.data[j * F + f]);
				out[o + 2 * F] = d2.data[i * n + j];
			}
		return { shape: [n, n, C], data: out };
	},
	narration: {
		what: t('For every pair (i, j), one row of {outShape}: the numbers of atom i, the numbers of atom j and their squared distance.', '对每一对 (i, j) 拼出一行（整体形状 {outShape}）：原子 i 的数、原子 j 的数，以及它们的距离平方。'),
		why: t('The message network sees which atoms meet and how far apart they are, never where they sit.', '消息网络看到的是哪两种原子相遇、相距多远，从来看不到它们在哪里。')
	},
	vars: baseVars,
	zoom: 'none',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'torch.cat([h[:, None].expand(n, n, F), h[None, :].expand(n, n, F), d2[..., None]], -1)'
};

export const vec_sum: OpDef<NoP> = {
	op: 'vec_sum',
	version: 1,
	attrs: {},
	category: 'graph',
	label: t('Push every atom along its pairs', '沿每一对推动原子'),
	inferShape: (ins) => (rank(ins[0], 2, 'vec_sum: x'), rank(ins[1], 3, 'vec_sum: w'), ins[0].slice()),
	axes: () => ['n', 'xy'],
	forward([x, w]) {
		const [n, D] = x.shape;
		const out = outArray(n * D, x, w);
		for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) out[i * D + d] = push(x.data, w.data, n, D, i, d).s;
		return { shape: [n, D], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const [i, d] = q.unit;
		const [x, w] = q.ins;
		const [n, D] = x.shape;
		const p = push(x.data, w.data, n, D, i, d);
		const y = store(p.s, q.out);
		const e = unitBase('weighted-sum', q.layer, [i, d], y, q.out.data[i * D + d]);
		e.reads = [{ tensor: q.layer.in[1], cells: p.terms.map((z) => i * n + z.j) }];
		e.terms = p.terms.map((z) => ({ x: z.x, w: z.w, p: z.p, xAt: z.j * D + d, wAt: i * n + z.j }));
		e.values = { i, d, n, y };
		return e;
	},
	formula: `${sym('out', 'v')}_{ {{i}} } = \\frac{1}{n-1} \\sum_{j \\ne i} (${sym('in', 'x')}_{ {{i}} } - ${sym('in', 'x')}_j)\\, ${sym('in', 'w')}_{ {{i}} j} = {{y}}`,
	narration: {
		what: t('Each atom i is pushed along the line to every other atom j, by a number wᵢⱼ the network computed from their message: v = Σ (xᵢ − xⱼ) wᵢⱼ / (n − 1). The result has shape {outShape}: one arrow per atom.', '每个原子 i 沿着它与其他每个原子 j 的连线被推动，推多少由网络从它们的消息算出的数 wᵢⱼ 决定：v = Σ (xᵢ − xⱼ) wᵢⱼ / (n − 1)。结果的形状是 {outShape}：每个原子一个箭头。'),
		how: t('A positive wᵢⱼ pushes atom i away from j, a negative one pulls it closer.', 'wᵢⱼ 为正，原子 i 被推离 j；为负，就被拉近。'),
		why: t('The arrows are built from differences of positions, so they turn when the molecule turns.', '箭头是由位置之差拼成的，所以分子转动时，它们也跟着转动。'),
		unit: t('Atom {i|int}, coordinate {d|int} (counting from 0): {y}.', '原子 {i|int} 的第 {d|int} 个坐标（从 0 数起）：{y}。')
	},
	vars: baseVars,
	zoom: 'weighted-sum',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => '((x[:, None] - x[None, :]) * w).sum(1) / max(1, n - 1)'
};

export const nb_sum: OpDef<NoP> = {
	op: 'nb_sum',
	version: 1,
	attrs: {},
	category: 'graph',
	label: t('Add up the messages each atom receives', '把每个原子收到的消息加起来'),
	inferShape: (ins) => (rank(ins[0], 3, 'nb_sum: m'), [ins[0][0], ins[0][2]]),
	axes: () => ['n', 'f'],
	forward([m]) {
		const [n, , M] = m.shape;
		const out = outArray(n * M, m);
		for (let i = 0; i < n; i++)
			for (let o = 0; o < M; o++) {
				let s = 0;
				for (let j = 0; j < n; j++) if (j !== i) s += m.data[(i * n + j) * M + o];
				out[i * M + o] = s;
			}
		return { shape: [n, M], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const [i, o] = q.unit;
		const m = q.ins[0];
		const [n, , M] = m.shape;
		const cells = Array.from({ length: n }, (_, j) => (i * n + j) * M + o).filter((_, j) => j !== i);
		const terms = cells.map((c) => ({ x: m.data[c], w: 1, p: m.data[c], xAt: c }));
		const y = store(terms.reduce((s, z) => s + z.p, 0), q.out);
		const e = unitBase('reduce', q.layer, [i, o], y, q.out.data[i * M + o]);
		e.reads = [{ tensor: q.layer.in[0], cells }];
		e.terms = terms;
		e.layout = { rows: 1, cols: terms.length }; // one row: the messages from the other atoms, in order
		e.values = { i, o, n: terms.length, y };
		return e;
	},
	formula: `${sym('out', 'a')}_{ {{i}},{{o}} } = \\sum_{j \\ne {{i}} } ${sym('in', 'm')}_{ {{i}} j,{{o}} } = {{y}}`,
	narration: {
		what: t('Each atom adds up the messages from all the other atoms, one sum per number of the message: shape {outShape}.', '每个原子把来自其他所有原子的消息加起来，消息的每个数各求一个和：形状是 {outShape}。'),
		why: t('A sum does not care in which order the other atoms are listed.', '求和不在乎其他原子按什么顺序排列。'),
		unit: t('Atom {i|int}, number {o|int} (counting from 0): {n|int} messages add up to {y}.', '原子 {i|int} 的第 {o|int} 个数（从 0 数起）：{n|int} 条消息加起来是 {y}。')
	},
	vars: baseVars,
	zoom: 'reduce',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => '(m * (1 - eye(n))[..., None]).sum(1)'
};

export const sum_atoms: OpDef<NoP> = {
	op: 'sum_atoms',
	version: 1,
	attrs: {},
	category: 'pool',
	label: t('Add up the shares of all atoms', '把所有原子的份额加起来'),
	inferShape: (ins) => (rank(ins[0], 2, 'sum_atoms: y'), [ins[0][1]]),
	axes: () => ['f'],
	forward([y]) {
		const [n, C] = y.shape;
		const out = outArray(C, y);
		for (let c = 0; c < C; c++) {
			let s = 0;
			for (let i = 0; i < n; i++) s += y.data[i * C + c];
			out[c] = s;
		}
		return { shape: [C], data: out };
	},
	narration: {
		what: t('The energy is the sum of one share per atom: {inShape} becomes {outShape}.', '能量是每个原子一份的总和：{inShape} 变成 {outShape}。'),
		why: t('A sum gives the same number for any order of the atoms.', '不管原子按什么顺序排，求和都给出同一个数。')
	},
	vars: baseVars,
	zoom: 'none',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'y.sum(0)'
};
