// Attention ops (design §4.3), same loops as tinygpt.ts:
//   attention_scores  q [Tq, D], k [Tk, D] -> [H, Tq, Tk];  s = Σ_c q[i, h·dH + c] · k[j, h·dH + c] (f64), score = s · scale.
//                     causal=true writes −∞ above the diagonal directly (TinyGPT keeps a separate causal_mask layer).
//   causal_mask       [.., Tq, Tk]: −∞ where j > i, else unchanged.
//   attention_mix     w [H, Tq, Tk], v [Tk, D] -> [Tq, D];  out[i, h·dH + c] = Σ_j w[h, i, j] · v[j, h·dH + c].
//                     Masked weights are exactly 0, so summing every j equals tinygpt.ts's j ≤ i loop bit for bit.
import type { Dim, OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { baseVars, int, outArray, store, t, unitBase, words } from './common.ts';

export type ScoresP = { nHead: number; dHead: number; scale?: number; causal: boolean };
export type MixP = { nHead: number; dHead: number };
export type MaskP = { fill?: string };

const scaleOf = (p: ScoresP) => (p.scale !== undefined ? p.scale : 1 / Math.sqrt(p.dHead));

function checkHeads(D: Dim, p: { nHead: number; dHead: number }, what: string): void {
	if (typeof D === 'number' && D !== p.nHead * p.dHead) throw new Error(`${what}: ${D} columns but nHead·dHead = ${p.nHead * p.dHead}`);
}

export const attention_scores: OpDef<ScoresP> = {
	op: 'attention_scores',
	version: 1,
	attrs: {
		nHead: { type: 'int', required: true },
		dHead: { type: 'int', required: true },
		scale: { type: 'float', doc: 'defaults to 1/√dHead' },
		causal: { type: 'bool', default: false, doc: 'write −∞ above the diagonal (fused causal mask)' }
	},
	category: 'attention',
	label: t('Attention scores', '注意力分数'),
	inferShape(ins, p) {
		const [q, k] = ins;
		if (!q || !k || q.length !== 2 || k.length !== 2) throw new Error('attention_scores: q and k must be [T, D]');
		checkHeads(q[1], p, 'attention_scores q');
		checkHeads(k[1], p, 'attention_scores k');
		return [p.nHead, q[0], k[0]];
	},
	axes: () => ['h', 'tq', 'tk'],
	forward(xs, p) {
		const [Q, K] = xs;
		const q = Q.data;
		const k = K.data;
		const H = p.nHead;
		const hd = p.dHead;
		const D = H * hd;
		const Tq = Q.shape[0];
		const Tk = K.shape[0];
		if (Q.shape[1] !== D || K.shape[1] !== D) throw new Error(`attention_scores: q [${Q.shape}] / k [${K.shape}] need ${D} columns`);
		const scale = scaleOf(p);
		const causal = !!p.causal;
		const out = outArray(H * Tq * Tk, Q, K);
		for (let hh = 0; hh < H; hh++) {
			for (let i = 0; i < Tq; i++) {
				for (let j = 0; j < Tk; j++) {
					let s = 0;
					for (let c = 0; c < hd; c++) s += q[i * D + hh * hd + c] * k[j * D + hh * hd + c];
					out[(hh * Tq + i) * Tk + j] = causal && j > i ? -Infinity : s * scale;
				}
			}
		}
		return { shape: [H, Tq, Tk], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'last' },
	explain(q): UnitExplanation {
		const [h, i, j] = q.unit;
		const f = flatIndex(q.out.shape, q.unit);
		const [Q, K] = q.ins;
		const hd = q.p.dHead;
		const D = q.p.nHead * hd;
		let s = 0;
		const terms: NonNullable<UnitExplanation['terms']> = [];
		const qc: number[] = [];
		const kc: number[] = [];
		for (let c = 0; c < hd; c++) {
			const qa = i * D + h * hd + c;
			const ka = j * D + h * hd + c;
			const pr = Q.data[qa] * K.data[ka];
			s += pr;
			terms.push({ x: Q.data[qa], w: K.data[ka], p: pr, xAt: qa, wAt: ka });
			qc.push(qa);
			kc.push(ka);
		}
		const scale = scaleOf(q.p);
		const masked = !!q.p.causal && j > i;
		const y = masked ? -Infinity : store(s * scale, q.out);
		const e = unitBase('weighted-sum', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [
			{ tensor: q.layer.in[0], cells: qc },
			{ tensor: q.layer.in[1], cells: kc }
		];
		e.terms = terms;
		e.values = { s, scale, y, n: hd, head: h, i, j, headNo: h + 1, qNo: i + 1, kNo: j + 1, masked: masked ? 1 : 0 };
		return e;
	},
	formula: `${sym('out', 's')}_{h,i,j} = \\frac{${sym('q', 'q')}_i \\cdot ${sym('k', 'k')}_j}{\\sqrt{d}} = {{s}} \\times {{scale}} = {{y}}`,
	narration: {
		// {scaleWhy}: "(that is, divided by √dHead)" when scale = 1/√dHead; {causalNote}: the fused causal mask (causal=true).
		// No {outShape} here: on a scan step the planner can only show the symbolic [H×T×T].
		what: t(
			'Attention scores measure how well each query matches each key. In every head, the score of query i and key j is the dot product of their {dHead|int} numbers, multiplied by {scale}{scaleWhy}.',
			'注意力分数衡量每个查询和每个键有多匹配。在每个头里，查询 i 和键 j 的分数是它们 {dHead|int} 个数的点积，再乘以 {scale}{scaleWhy}。'
		),
		how: t(
			'There are {nHead|int} heads, each using its own {dHead|int} columns of Q and K, so the result is {nHead|int} tables: one row per query and one column per key.{causalNote}',
			'共有 {nHead|int} 个头，每个头使用 Q 和 K 中属于自己的 {dHead|int} 列，所以结果是 {nHead|int} 张表：每个查询一行，每个键一列。{causalNote}'
		),
		why: t(
			'A large score means this query wants to read from that key’s position. Dividing by the square root of the head size keeps the scores from growing with the head size.',
			'分数大，表示这个查询想从那个键的位置读取信息。除以头大小的平方根，可以避免分数随着头变大而变得过大。'
		),
		look: t('One row at a time: the largest scores mark the keys this query will attend to most.', '一行一行地看：分数最大的位置，就是这个查询最关注的键。'),
		unit: t(
			'Head {head|int}, query {i|int}, key {j|int} (counting from 0): the {n|int} products add up to {s}, and × {scale} gives {y}.',
			'头 {head|int}、查询 {i|int}、键 {j|int}（从 0 数起）：{n|int} 个乘积相加得 {s}，乘以 {scale} 得 {y}。'
		)
	},
	vars(layer, spec) {
		const p = (layer.params ?? {}) as Partial<ScoresP>;
		const dHead = p.dHead ?? 0;
		const scale = p.scale ?? (dHead ? 1 / Math.sqrt(dHead) : NaN);
		const isRoot = dHead > 0 && Math.abs(scale * Math.sqrt(dHead) - 1) < 1e-9;
		return {
			...baseVars(layer, spec),
			nHead: p.nHead ?? '?',
			dHead: dHead || '?',
			scale: Number.isFinite(scale) ? scale : '?',
			scaleWhy: isRoot ? words(` (that is, divided by √${dHead})`, `（也就是除以 √${dHead}）`) : words('', ''),
			causalNote: p.causal
				? words(' Keys after the query get the score −∞ (a causal mask), so the query cannot read the future.', '位于查询之后的键得到分数 −∞（因果掩码），所以查询读不到“未来”。')
				: words('', '')
		};
	},
	distinct: (p, prev) => !!p.causal !== !!prev.causal,
	zoom: 'weighted-sum',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: (layer) => {
		const p = (layer.params ?? {}) as Partial<ScoresP>;
		return `(q_h @ k_h.transpose(-2, -1)) * ${p.scale ?? `${p.dHead ?? 'd'} ** -0.5`}`;
	},
	terms: ['attention', 'query-key-value', 'attention-head']
};

export const causal_mask: OpDef<MaskP> = {
	op: 'causal_mask',
	version: 1,
	attrs: { fill: { type: 'string', default: '-Infinity', doc: 'documentation only: the masked entries are always −∞' } },
	category: 'attention',
	label: t('Causal mask', '因果掩码'),
	inferShape: (ins) => {
		if (ins[0].length < 2) throw new Error('causal_mask: input must be [.., Tq, Tk]');
		return ins[0].slice();
	},
	axes: (ins) => ins[0].slice(),
	forward(xs) {
		const x = xs[0];
		const r = x.shape.length;
		const Tq = x.shape[r - 2];
		const Tk = x.shape[r - 1];
		const out = outArray(x.data.length, x);
		const blocks = x.data.length / (Tq * Tk);
		for (let b = 0; b < blocks; b++) {
			for (let i = 0; i < Tq; i++) {
				for (let j = 0; j < Tk; j++) {
					const o = (b * Tq + i) * Tk + j;
					out[o] = j > i ? -Infinity : x.data[o];
				}
			}
		}
		return { shape: x.shape.slice(), data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'last' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const r = q.unit.length;
		const i = q.unit[r - 2];
		const j = q.unit[r - 1];
		const xv = q.ins[0].data[f];
		const masked = j > i;
		const y = masked ? -Infinity : store(xv, q.out);
		const e = unitBase('index-map', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells: [f] }];
		e.values = { x: xv, y, masked: masked ? 1 : 0, i, j, qNo: i + 1, kNo: j + 1 };
		return e;
	},
	formula: `${sym('out', 'y')}_{i,j} = \\begin{cases} ${sym('in', 'x')}_{i,j} & j \\le i \\\\ -\\infty & j > i \\end{cases} = {{y}}`,
	narration: {
		what: t(
			'The causal mask hides the future: for query i, every key after position i gets the score −∞, so after the softmax its weight is exactly 0.',
			'因果掩码把“未来”遮住：对查询 i，位置 i 之后的每个键的分数都变成 −∞，所以经过 softmax 之后它的权重正好是 0。'
		),
		how: t('Scores on or below the diagonal pass unchanged; scores above the diagonal are replaced. Nothing is learned here.', '对角线上及下方的分数保持不变；对角线上方的分数被替换。这里没有要学习的东西。'),
		why: t(
			'A model that predicts the next token must not peek at it, so each position may only read itself and earlier positions.',
			'预测下一个词元的模型不能偷看答案，所以每个位置只能读取自己和之前的位置。'
		),
		look: t('The upper-right triangle of every head’s table is now −∞ (blank).', '每个头的表格右上角三角形现在都是 −∞（空白）。'),
		unit: t('The score {x} becomes {y}.', '分数 {x} 变成 {y}。')
	},
	vars: (layer, spec) => baseVars(layer, spec),
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => "scores.masked_fill(torch.ones(T, T).triu(1).bool(), float('-inf'))",
	terms: ['causal-mask', 'attention']
};

export const attention_mix: OpDef<MixP> = {
	op: 'attention_mix',
	version: 1,
	attrs: { nHead: { type: 'int', required: true }, dHead: { type: 'int', required: true } },
	category: 'attention',
	label: t('Weighted sum of values', '值的加权和'),
	inferShape(ins, p) {
		const [w, v] = ins;
		if (!w || !v || w.length !== 3 || v.length !== 2) throw new Error('attention_mix: weights must be [H, Tq, Tk] and v [Tk, D]');
		if (typeof w[0] === 'number' && w[0] !== p.nHead) throw new Error(`attention_mix: weights have ${w[0]} heads, nHead = ${p.nHead}`);
		checkHeads(v[1], p, 'attention_mix v');
		return [w[1], p.nHead * p.dHead];
	},
	axes: () => ['t', 'd'],
	forward(xs, p) {
		const [Wt, V] = xs;
		const att = Wt.data;
		const v = V.data;
		const H = p.nHead;
		const hd = p.dHead;
		const D = H * hd;
		const Tq = Wt.shape[1];
		const Tk = Wt.shape[2];
		if (V.shape[0] !== Tk || V.shape[1] !== D) throw new Error(`attention_mix: v [${V.shape}] does not fit weights [${Wt.shape}] with ${D} columns`);
		const out = outArray(Tq * D, Wt, V);
		for (let hh = 0; hh < H; hh++) {
			for (let i = 0; i < Tq; i++) {
				for (let c = 0; c < hd; c++) {
					let s = 0;
					for (let j = 0; j < Tk; j++) s += att[(hh * Tq + i) * Tk + j] * v[j * D + hh * hd + c];
					out[i * D + hh * hd + c] = s;
				}
			}
		}
		return { shape: [Tq, D], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'last' },
	explain(q): UnitExplanation {
		const [i, d] = q.unit;
		const f = flatIndex(q.out.shape, q.unit);
		const [Wt, V] = q.ins;
		const hd = int(q.p, 'dHead');
		const D = int(q.p, 'nHead') * hd;
		const h = Math.floor(d / hd);
		const Tq = Wt.shape[1];
		const Tk = Wt.shape[2];
		let s = 0;
		const terms: NonNullable<UnitExplanation['terms']> = [];
		const wc: number[] = [];
		const vc: number[] = [];
		for (let j = 0; j < Tk; j++) {
			const wa = (h * Tq + i) * Tk + j;
			const va = j * D + d;
			const pr = Wt.data[wa] * V.data[va];
			s += pr;
			terms.push({ x: V.data[va], w: Wt.data[wa], p: pr, xAt: va, wAt: wa });
			wc.push(wa);
			vc.push(va);
		}
		const y = store(s, q.out);
		const e = unitBase('weighted-sum', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [
			{ tensor: q.layer.in[0], cells: wc },
			{ tensor: q.layer.in[1], cells: vc }
		];
		e.terms = terms;
		e.values = { y, s, n: Tk, head: h, headNo: h + 1, i, qNo: i + 1, c: d - h * hd };
		return e;
	},
	formula: `${sym('out', 'o')}_{i,c} = \\sum_j ${sym('prob', 'a')}_{i,j}\\,${sym('v', 'v')}_{j,c} = {{y}}`,
	narration: {
		what: t(
			'Each position now gathers information from the positions it attends to. Its output is a weighted sum of their value vectors (the rows of V), with the attention weights as the weights; each of the {nHead|int} heads does this with its own {dHead|int} columns, and the heads’ results are placed side by side.',
			'每个位置现在从它关注的位置收集信息。它的输出是这些位置的值向量（V 的行）的加权和，权重就是注意力权重；{nHead|int} 个头各用自己的 {dHead|int} 列分别计算，再把结果并排拼在一起。'
		),
		how: t(
			'For one output number, each key’s value is multiplied by that key’s attention weight, and the products are added up.',
			'对于一个输出数：每个键的值乘以这个键的注意力权重，再把乘积相加。'
		),
		why: t(
			'Because each row of weights adds up to 1, the output is a blend of the values that leans toward the positions with the largest weights.',
			'因为每一行权重之和为 1，输出就是各个值的混合，偏向权重最大的位置。'
		),
		look: t('Positions with weight 0, such as the masked future, contribute nothing.', '权重为 0 的位置（例如被遮住的未来）没有任何贡献。'),
		unit: t('Head {head|int}, position {i|int} (counting from 0): the {n|int} weighted values add up to {y}.', '头 {head|int}、位置 {i|int}（从 0 数起）：{n|int} 个加权的值加起来是 {y}。')
	},
	vars(layer, spec) {
		const p = (layer.params ?? {}) as Partial<MixP>;
		return { ...baseVars(layer, spec), nHead: p.nHead ?? '?', dHead: p.dHead ?? '?' };
	},
	zoom: 'weighted-sum',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => '(att @ v_h).transpose(0, 1).reshape(T, n_head * d_head)',
	terms: ['attention', 'query-key-value', 'attention-head']
};

export default attention_scores;
