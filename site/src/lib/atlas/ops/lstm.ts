// lstm: one-layer, unidirectional, unbatched LSTM over x [T, In] -> h [T, H] (PyTorch nn.LSTM semantics, h0 = c0 = 0).
//   z = W_ih x_t + W_hh h_{t−1} + b_ih + b_hh   (f64: s = 0; += W_ih·x; += W_hh·h; + b_ih; + b_hh; both biases kept)
//   gate order i, f, g, o (PyTorch): i = σ(z_i), f = σ(z_f), g = tanh(z_g), o = σ(z_o)
//   c_t = f·c_{t−1} + i·g;  h_t = o·tanh(c_t)    (c_t and h_t are stored with the output precision, as PyTorch fp32 does)
// Sub-steps `${id}#i|f|g|o|c|h`, each [T, H]. explain([t, j]) gives every gate of unit j at step t; asking the executor
// for a sub id ('lstm1#f') returns the same explanation with that gate as the result.
import type { OpDef, Tensor, UnitExplanation } from '../types/index.ts';
import { flatIndex } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { baseVars, needRank, optW, outArray, specWeightShape, t, unitBase, weightName } from './common.ts';

export type LstmP = { hidden: number; inputSize?: number };

const sig = (v: number) => 1 / (1 + Math.exp(-v));

interface LstmWeights {
	Wih: Tensor;
	Whh: Tensor;
	bih: Tensor | null;
	bhh: Tensor | null;
}

/** Pre-activations z[0..4H) of step t from x row t and h_{t−1} (same order everywhere). */
function gates(w: LstmWeights, x: ArrayLike<number>, xo: number, hPrev: ArrayLike<number>, H: number, In: number, z: Float64Array): void {
	const wi = w.Wih.data;
	const wh = w.Whh.data;
	for (let r = 0; r < 4 * H; r++) {
		let s = 0;
		const bi = r * In;
		for (let i = 0; i < In; i++) s += wi[bi + i] * x[xo + i];
		const bh = r * H;
		for (let k = 0; k < H; k++) s += wh[bh + k] * hPrev[k];
		if (w.bih) s += w.bih.data[r];
		if (w.bhh) s += w.bhh.data[r];
		z[r] = s;
	}
}

export const lstm: OpDef<LstmP> = {
	op: 'lstm',
	version: 1,
	attrs: {
		hidden: { type: 'int', required: true, doc: 'H: size of the hidden and cell state' },
		inputSize: { type: 'int', doc: 'informational; must equal weight_ih.shape[1]' }
	},
	category: 'recurrent',
	label: t('LSTM', 'LSTM（长短期记忆）'),
	weights: [{ slot: 'weight_ih' }, { slot: 'weight_hh' }, { slot: 'bias_ih', optional: true }, { slot: 'bias_hh', optional: true }],
	subs: [
		{ name: 'i', label: t('input gate i', '输入门 i'), axes: ['t', 'd'] },
		{ name: 'f', label: t('forget gate f', '遗忘门 f'), axes: ['t', 'd'] },
		{ name: 'g', label: t('candidate g', '候选值 g'), axes: ['t', 'd'] },
		{ name: 'o', label: t('output gate o', '输出门 o'), axes: ['t', 'd'] },
		{ name: 'c', label: t('cell state c', '细胞状态 c'), axes: ['t', 'd'] },
		{ name: 'h', label: t('hidden state h', '隐藏状态 h'), axes: ['t', 'd'] }
	],
	inferShape(ins, p, w) {
		const x = ins[0];
		if (x.length !== 2) throw new Error(`lstm: input must be [T, In], got [${x}]`);
		const H = p.hidden;
		const wi = w.weight_ih;
		const wh = w.weight_hh;
		if (!wi || wi.length !== 2 || wi[0] !== 4 * H) throw new Error(`lstm: weight_ih must be [${4 * H}, In]`);
		if (!wh || wh.length !== 2 || wh[0] !== 4 * H || wh[1] !== H) throw new Error(`lstm: weight_hh must be [${4 * H}, ${H}]`);
		if (typeof x[1] === 'number' && x[1] !== wi[1]) throw new Error(`lstm: input has ${x[1]} features, weight_ih expects ${wi[1]}`);
		if (p.inputSize !== undefined && p.inputSize !== wi[1]) throw new Error(`lstm: inputSize=${p.inputSize} but weight_ih has ${wi[1]}`);
		for (const b of ['bias_ih', 'bias_hh']) if (w[b] && (w[b].length !== 1 || w[b][0] !== 4 * H)) throw new Error(`lstm: ${b} must be [${4 * H}]`);
		return [x[0], H];
	},
	axes: () => ['t', 'd'],
	forward(xs, p, ctx) {
		const src = xs[0];
		needRank(src, 2, 'lstm input');
		const w: LstmWeights = { Wih: ctx.w('weight_ih'), Whh: ctx.w('weight_hh'), bih: optW(ctx, 'bias_ih'), bhh: optW(ctx, 'bias_hh') };
		const H = p.hidden;
		const [T, In] = src.shape;
		if (w.Wih.shape[0] !== 4 * H || w.Wih.shape[1] !== In) throw new Error(`lstm ${ctx.layer.id}: weight_ih [${w.Wih.shape}] does not fit input [${src.shape}] and hidden ${H}`);
		const mk = () => outArray(T * H, src, w.Wih);
		const I = mk();
		const F = mk();
		const G = mk();
		const O = mk();
		const C = mk();
		const Hs = mk();
		const f64 = Hs instanceof Float64Array;
		const rd = f64 ? (v: number) => v : Math.fround;
		const z = new Float64Array(4 * H);
		let hPrev: ArrayLike<number> = new Float64Array(H);
		let cPrev: ArrayLike<number> = new Float64Array(H);
		for (let tt = 0; tt < T; tt++) {
			gates(w, src.data, tt * In, hPrev, H, In, z);
			const o0 = tt * H;
			for (let j = 0; j < H; j++) {
				const ig = sig(z[j]);
				const fg = sig(z[H + j]);
				const gg = Math.tanh(z[2 * H + j]);
				const og = sig(z[3 * H + j]);
				const c = rd(fg * cPrev[j] + ig * gg);
				I[o0 + j] = ig;
				F[o0 + j] = fg;
				G[o0 + j] = gg;
				O[o0 + j] = og;
				C[o0 + j] = c;
				Hs[o0 + j] = og * Math.tanh(c);
			}
			hPrev = Hs.subarray(o0, o0 + H);
			cPrev = C.subarray(o0, o0 + H);
		}
		const sh = [T, H];
		ctx.emit('i', { shape: sh.slice(), data: I });
		ctx.emit('f', { shape: sh.slice(), data: F });
		ctx.emit('g', { shape: sh.slice(), data: G });
		ctx.emit('o', { shape: sh.slice(), data: O });
		ctx.emit('c', { shape: sh.slice(), data: C });
		ctx.emit('h', { shape: sh.slice(), data: Hs.slice() });
		return { shape: sh, data: Hs };
	},
	unit: { axes: (out) => out.slice(), scan: 'time', pick: 'last' },
	explain(q): UnitExplanation {
		const [tt, j] = q.unit;
		const f = flatIndex(q.out.shape, q.unit);
		const src = q.ins[0];
		const H = q.p.hidden;
		const In = src.shape[1];
		const w: LstmWeights = { Wih: q.w('weight_ih'), Whh: q.w('weight_hh'), bih: optW(q, 'bias_ih'), bhh: optW(q, 'bias_hh') };
		const f64 = q.out.data instanceof Float64Array;
		const rd = f64 ? (v: number) => v : Math.fround;
		// h_{t−1} is the traced output row t−1; c_{t−1} the traced `#c` row, else recomputed from the start.
		let hPrev: ArrayLike<number> = new Float64Array(H);
		let cPrev: ArrayLike<number> = new Float64Array(H);
		const cT = q.trace.t[`${q.layer.id}#c`];
		const z = new Float64Array(4 * H);
		if (tt > 0) {
			if (cT) {
				hPrev = q.out.data.subarray((tt - 1) * H, tt * H);
				cPrev = cT.data.subarray((tt - 1) * H, tt * H);
			} else {
				const h = new Float64Array(H);
				const c = new Float64Array(H);
				for (let s = 0; s < tt; s++) {
					gates(w, src.data, s * In, h, H, In, z);
					for (let k = 0; k < H; k++) {
						const cc = rd(sig(z[H + k]) * c[k] + sig(z[k]) * Math.tanh(z[2 * H + k]));
						c[k] = cc;
						z[k] = rd(sig(z[3 * H + k]) * Math.tanh(cc)); // reuse z[k] as scratch for h (gate k already consumed)
					}
					for (let k = 0; k < H; k++) h[k] = z[k];
				}
				hPrev = h;
				cPrev = c;
			}
		}
		gates(w, src.data, tt * In, hPrev, H, In, z);
		const ig = sig(z[j]);
		const fg = sig(z[H + j]);
		const gg = Math.tanh(z[2 * H + j]);
		const og = sig(z[3 * H + j]);
		const c = rd(fg * cPrev[j] + ig * gg);
		const h = rd(og * Math.tanh(c));
		const e = unitBase('gate', q.layer, q.unit, h, q.out.data[f]);
		e.reads = [
			{ tensor: q.layer.in[0], cells: Array.from({ length: In }, (_, i) => tt * In + i) },
			{ tensor: weightName(q.layer, 'weight_ih'), cells: [j, H + j, 2 * H + j, 3 * H + j].flatMap((r) => Array.from({ length: In }, (_, i) => r * In + i)) }
		];
		if (tt > 0) e.reads.push({ tensor: q.layer.id, cells: Array.from({ length: H }, (_, k) => (tt - 1) * H + k) });
		e.steps = [
			{ label: t('forget gate f = σ(z_f)', '遗忘门 f = σ(z_f)'), value: rd(fg), tex: 'f' },
			{ label: t('input gate i = σ(z_i)', '输入门 i = σ(z_i)'), value: rd(ig), tex: 'i' },
			{ label: t('candidate g = tanh(z_g)', '候选值 g = tanh(z_g)'), value: rd(gg), tex: 'g' },
			{ label: t('new cell c = f·c_prev + i·g', '新的细胞值 c = f·c_prev + i·g'), value: c, tex: 'c_t' },
			{ label: t('output gate o = σ(z_o)', '输出门 o = σ(z_o)'), value: rd(og), tex: 'o' },
			{ label: t('output h = o·tanh(c)', '输出 h = o·tanh(c)'), value: h, tex: 'h_t' }
		];
		e.values = {
			i: rd(ig),
			f: rd(fg),
			g: rd(gg),
			o: rd(og),
			c,
			h,
			cPrev: cPrev[j],
			hPrev: tt > 0 ? hPrev[j] : 0,
			zi: z[j],
			zf: z[H + j],
			zg: z[2 * H + j],
			zo: z[3 * H + j],
			t: tt,
			j,
			tNo: tt + 1,
			jNo: j + 1,
			y: h
		};
		return e;
	},
	formula: `${sym('act', 'c')}_t = ${sym('act', 'f')}\\,c_{t-1} + ${sym('act', 'i')}\\,${sym('act', 'g')} = {{f}} \\cdot {{cPrev}} + {{i}} \\cdot {{g}} = {{c}};\\quad ${sym('out', 'h')}_t = ${sym('act', 'o')}\\tanh(c_t) = {{h}}`,
	narration: {
		what: t(
			'An LSTM reads a sequence one step at a time and keeps a memory, the cell state c, of {hidden|int} numbers. At each step three gates and a candidate decide what to forget, what to write and what to show.',
			'LSTM 一步一步地读序列，并保留一份记忆，即由 {hidden|int} 个数组成的细胞状态 c。每一步，三个门和一个候选值决定忘掉什么、写入什么、输出什么。'
		),
		how: t(
			'From the input and the previous output h it computes the forget gate f, the input gate i, the output gate o (sigmoids, between 0 and 1) and the candidate g (a tanh). Then the memory becomes c = f·c_prev + i·g and the output h = o·tanh(c).',
			'根据当前输入和上一步的输出 h，它算出遗忘门 f、输入门 i、输出门 o（sigmoid，介于 0 和 1 之间）和候选值 g（tanh）。然后记忆变成 c = f·c_prev + i·g，输出是 h = o·tanh(c)。'
		),
		why: t(
			'Because c changes only by gentle multiplying and adding, information can survive many steps; the gates learn when to keep it and when to drop it.',
			'因为 c 只通过温和的乘法和加法改变，信息可以保留很多步；这些门学会了什么时候保留它、什么时候丢掉它。'
		),
		look: t('Follow one unit over time: while its forget gate stays near 1, its cell value is carried forward almost unchanged.', '跟踪一个单元随时间的变化：只要它的遗忘门接近 1，它的细胞值就几乎原样传到下一步。'),
		unit: t(
			'At time step {t|int}, unit {j|int} (counting from 0): f = {f}, i = {i}, g = {g}, o = {o}. The cell becomes {f} × {cPrev} + {i} × {g} = {c}, and the output is {o} × tanh({c}) = {h}.',
			'时间步 {t|int}、单元 {j|int}（从 0 数起）：f = {f}，i = {i}，g = {g}，o = {o}。细胞值变成 {f} × {cPrev} + {i} × {g} = {c}，输出是 {o} × tanh({c}) = {h}。'
		)
	},
	vars(layer, spec) {
		const p = (layer.params ?? {}) as Partial<LstmP>;
		const wi = specWeightShape(layer, spec, 'weight_ih');
		return { ...baseVars(layer, spec), hidden: p.hidden ?? '?', inputSize: p.inputSize ?? wi?.[1] ?? '?' };
	},
	zoom: 'gate',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: (layer) => {
		const p = (layer.params ?? {}) as Partial<LstmP>;
		return `nn.LSTM(${p.inputSize ?? 'input_size'}, ${p.hidden ?? 'hidden_size'})`;
	},
	terms: ['activation-function', 'parameter']
};

export default lstm;
