// gcn local ops (design §7). Python mirrors: pipeline/gcn/ops_gcn.py (the same arithmetic as model.py).
//   gcn_norm   A [n, n] -> Â = D^-1/2 (A + I) D^-1/2; sub-steps loops (A + I) and deg (row sums of A + I)
//   gcn_conv   H [n, i], Â [n, n] -> Â (H Wᵀ) + b [n, o]; sub-step xw (H Wᵀ). explain(): one term per neighbour j
//              (Â_ij × (H Wᵀ)_j), the message member j sends along the edge, so zoom.weighted-sum lists the messages.
// f64 accumulation, f32 storage (atlas/ops/common.ts conventions).
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, store, t, unitBase, weightName } from '../../../atlas/ops/common.ts';
import { normalize } from '../graph.ts';

type NoP = Record<string, never>;
const square = (s: (number | string)[], what: string) => {
	if (s.length !== 2 || (typeof s[0] === 'number' && typeof s[1] === 'number' && s[0] !== s[1])) throw new Error(`${what} must be [n, n]`);
};
/** Σₖ H[r, k] · W[o, k] in f64: row r, column o of H Wᵀ */
const rowDot = (h: ArrayLike<number>, r: number, W: ArrayLike<number>, o: number, I: number) => {
	let s = 0;
	for (let k = 0; k < I; k++) s += h[r * I + k] * W[o * I + k];
	return s;
};

export const gcn_norm: OpDef<NoP> = {
	op: 'gcn_norm',
	version: 1,
	attrs: {},
	category: 'graph',
	label: t('Normalize the adjacency', '归一化邻接矩阵'),
	subs: [
		{ name: 'loops', label: t('A + I: every member also listens to itself', 'A + I：每个成员也听自己的'), axes: ['n', 'k'] },
		{ name: 'deg', label: t('Degree d (friends + 1)', '度 d（朋友数 + 1）'), axes: ['n'] }
	],
	inferShape(ins) {
		square(ins[0], 'gcn_norm: the adjacency');
		return ins[0].slice();
	},
	axes: () => ['n', 'k'],
	forward(xs, _p, ctx) {
		const n = xs[0].shape[0];
		const { loops, deg, ahat } = normalize(xs[0].data, n);
		const f32 = (a: Float64Array) => { const o = outArray(a.length, xs[0]); o.set(a); return o; }; // prettier-ignore
		ctx.emit('loops', { shape: [n, n], data: f32(loops) });
		ctx.emit('deg', { shape: [n], data: f32(deg) });
		return { shape: [n, n], data: f32(ahat) };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'argmax' },
	explain(q): UnitExplanation {
		// a sub-step unit: norm#loops [i, j] (values.loops) or norm#deg [i] (values.deg)
		const [i, j = i] = q.unit;
		const n = q.out.shape[0];
		const { loops, deg } = normalize(q.ins[0].data, n);
		const a = loops[i * n + j];
		const y = store((1 / Math.sqrt(deg[i])) * a * (1 / Math.sqrt(deg[j])), q.out);
		const e = unitBase('none', q.layer, [i, j], y, q.out.data[i * n + j]);
		e.reads = [{ tensor: q.layer.in[0], cells: [i * n + j, ...Array.from({ length: n }, (_, k) => i * n + k)] }];
		e.values = { i, j, a, di: deg[i], dj: deg[j], y, loops: a, deg: deg[i] };
		return e;
	},
	formula: `${sym('out', '\\hat A')}_{ {{i}},{{j}} } = \\frac{(A+I)_{ {{i}},{{j}} }}{\\sqrt{d_{ {{i}} }\\,d_{ {{j}} }}} = \\frac{ {{a}} }{\\sqrt{ {{di}} \\times {{dj}} }} = {{y}}`,
	narration: {
		what: t('Each member also listens to itself (a 1 is added on the diagonal, A + I), and each friendship i–j gets the weight 1 / √(dᵢ × dⱼ), where d counts a member’s friends plus 1. The result Â has shape {outShape}.', '每个成员也听自己的（对角线上加 1，得到 A + I），每段朋友关系 i–j 得到权重 1 / √(dᵢ × dⱼ)，其中 d 是一个成员的朋友数加 1。结果 Â 的形状是 {outShape}。'),
		how: t('Row i of Â lists how much member i takes from each member; it is 0 for non-friends.', 'Â 的第 i 行列出成员 i 从每个成员那里取多少；不是朋友的地方是 0。'),
		why: t('Without the division a member with many friends would add up a huge sum; dividing by both degrees keeps every row on a similar scale.', '如果不除，朋友多的成员会加出一个很大的和；除以两边的度，让每一行的大小差不多。'),
		look: t('Bright cells sit where two members with few friends meet.', '亮的格子出现在两个朋友都不多的成员之间。'),
		unit: t('Â[{i|int}, {j|int}] = {a|int} / √({di|int} × {dj|int}) = {y}.', 'Â[{i|int}, {j|int}] = {a|int} / √({di|int} × {dj|int}) = {y}。')
	},
	vars: (layer, spec) => baseVars(layer, spec),
	zoom: 'none',
	color: 'sequential',
	parity: { bar: 1e-4 },
	pytorch: () => 'a = A + I; d = a.sum(1); r = d.pow(-0.5); ahat = r[:, None] * a * r[None, :]'
};

export const gcn_conv: OpDef<NoP> = {
	op: 'gcn_conv',
	version: 1,
	attrs: {},
	category: 'graph',
	label: t('Graph convolution', '图卷积'),
	weights: [{ slot: 'weight' }, { slot: 'bias' }],
	subs: [{ name: 'xw', label: t('Every member’s row times W (X Wᵀ)', '每个成员的行乘以 W（X Wᵀ）'), axes: ['n', 'd'] }],
	inferShape(ins, _p, w) {
		const [O, I] = w.weight ?? [];
		square(ins[1], 'gcn_conv: Â');
		if (ins[0].length !== 2 || (typeof ins[0][1] === 'number' && ins[0][1] !== I)) throw new Error(`gcn_conv: H must be [n, ${I}]`);
		return [ins[0][0], O];
	},
	axes: () => ['n', 'd'],
	forward([h, ah], _p, ctx) {
		const W = ctx.w('weight');
		const b = ctx.w('bias').data;
		const [O, I] = W.shape;
		const n = h.shape[0];
		const xw = outArray(n * O, h, W);
		for (let k = 0; k < n * O; k++) xw[k] = rowDot(h.data, Math.floor(k / O), W.data, k % O, I);
		ctx.emit('xw', { shape: [n, O], data: xw });
		const out = outArray(n * O, h, W);
		for (let k = 0; k < n * O; k++) {
			let s = 0;
			for (let j = 0; j < n; j++) s += ah.data[Math.floor(k / O) * n + j] * xw[j * O + (k % O)];
			out[k] = s + b[k % O];
		}
		return { shape: [n, O], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'argmax' },
	explain(q): UnitExplanation {
		const [i, o] = q.unit;
		const [h, ah] = q.ins;
		const W = q.w('weight');
		const [O, I] = W.shape;
		const n = h.shape[0];
		const b = q.w('bias').data[o];
		const terms: NonNullable<UnitExplanation['terms']> = [];
		let s = 0;
		for (let j = 0; j < n; j++) {
			const a = ah.data[i * n + j];
			const x = store(rowDot(h.data, j, W.data, o, I), q.out);
			if (a !== 0) terms.push({ x, w: a, p: a * x, xAt: j * O + o, wAt: i * n + j }), (s += a * x);
		}
		const y = store(s + b, q.out);
		const e = unitBase('weighted-sum', q.layer, q.unit, y, q.out.data[flatIndex(q.out.shape, q.unit)]);
		e.reads = [
			{ tensor: `${q.layer.id}#xw`, cells: terms.map((x) => x.xAt!) },
			{ tensor: q.layer.in[1], cells: terms.map((x) => x.wAt!) },
			{ tensor: weightName(q.layer, 'weight'), cells: Array.from({ length: I }, (_, k) => o * I + k) }
		];
		e.terms = terms;
		e.bias = b;
		e.values = { n: terms.length, s, b, y, node: i, out: o, xw: store(rowDot(h.data, i, W.data, o, I), q.out) }; // xw: the sub-step gc#xw
		return e;
	},
	formula: `${sym('out', 'y')}_{ {{node}},{{out}} } = \\sum_j ${sym('in', '\\hat A')}_{ {{node}},j }\\,(${sym('in', 'X')}${sym('param', 'W')}^\\top)_{j,{{out}} } + ${sym('param', 'b')}_{ {{out}} } = {{s}} + {{b}} = {{y}}`,
	narration: {
		what: t('First every member’s row is multiplied by the same learned weights W, giving {outFeatures|int} numbers per member. Then each member adds up the rows of its friends and itself, each scaled by its weight in Â, and adds a bias b. The result has shape {outShape}.', '先把每个成员的行乘以同一组学来的权重 W，每个成员得到 {outFeatures|int} 个数。然后每个成员把朋友和自己的这些行加起来，每一行先乘上它在 Â 里的权重，再加上偏置 b。结果的形状是 {outShape}。'),
		how: t('Each friend j sends the message Âᵢⱼ × (its row); member i adds up the messages.', '每个朋友 j 发出消息 Âᵢⱼ ×（它的行）；成员 i 把收到的消息加起来。'),
		why: t('Friends tend to be alike, so a member’s numbers become a blend of its neighbourhood.', '朋友之间往往相似，所以一个成员的数变成它周围一圈人的混合。'),
		look: t('Members in the same tight group end up with similar rows.', '同一个紧密小圈子里的成员，最后得到相似的行。'),
		unit: t('{n|int} messages add up to {s}; plus the bias {b} gives {y}.', '{n|int} 条消息加起来是 {s}；加上偏置 {b} 得到 {y}。')
	},
	vars(layer, spec) {
		const w = spec.tensors.find((x) => x.name === weightName(layer, 'weight'))?.shape ?? [];
		return { ...baseVars(layer, spec), outFeatures: w[0] ?? '?', inFeatures: w[1] ?? '?' };
	},
	zoom: 'weighted-sum',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'ahat @ (h @ self.weight.T) + self.bias',
	terms: ['weight', 'bias']
};
