// tinyseq2seq local op 'gru': one GRU layer reads x [T, In] left to right -> the states h [T, H] (nn.GRU semantics,
// h_0 = 0, both biases kept). Sub-steps #r (reset gate), #z (update gate), #n (candidate), each [T, H]. The arithmetic
// is ../seq.ts gruStep (float64 sums, float32 storage); pipeline/tinyseq2seq/ops_gru.py is the PyTorch mirror.
import type { OpDef } from '../../../atlas/types/index.ts';
import { needRank, outArray, t } from '../../../atlas/ops/common.ts';
import { encode, gruW } from '../seq.ts';

type P = { hidden: number };

export const gru: OpDef<P> = {
	op: 'gru',
	version: 1,
	attrs: { hidden: { type: 'int', required: true, doc: 'H: size of the hidden state' } },
	category: 'recurrent',
	label: t('GRU', 'GRU（门控循环单元）'),
	weights: [{ slot: 'weight_ih' }, { slot: 'weight_hh' }, { slot: 'bias_ih' }, { slot: 'bias_hh' }],
	subs: [
		{ name: 'r', label: t('reset gate r', '重置门 r'), axes: ['t', 'd'] },
		{ name: 'z', label: t('update gate z', '更新门 z'), axes: ['t', 'd'] },
		{ name: 'n', label: t('candidate n', '候选值 n'), axes: ['t', 'd'] }
	],
	inferShape(ins, p, w) {
		if (ins[0].length !== 2) throw new Error(`gru: input must be [T, In], got [${ins[0]}]`);
		if (w.weight_hh?.[0] !== 3 * p.hidden) throw new Error(`gru: weight_hh must be [${3 * p.hidden}, ${p.hidden}]`);
		return [ins[0][0], p.hidden];
	},
	axes: () => ['t', 'd'],
	forward(xs, p, ctx) {
		const x = xs[0];
		needRank(x, 2, 'gru input');
		const w = gruW((n) => ctx.w(n).data, (n) => ctx.w(n).shape, '');
		if (w.H !== p.hidden || w.In !== x.shape[1]) throw new Error(`gru ${ctx.layer.id}: weights [${3 * w.H}, ${w.In}] do not fit input [${x.shape}]`);
		const T = x.shape[0];
		const g = encode(w, x.data, T, Math.fround);
		const mk = (a: Float64Array) => {
			const o = outArray(a.length, x);
			o.set(a);
			return { shape: [T, w.H], data: o };
		};
		ctx.emit('r', mk(g.r));
		ctx.emit('z', mk(g.z));
		ctx.emit('n', mk(g.n));
		return mk(g.h);
	},
	unit: { axes: (out) => out.slice(), scan: 'time', pick: 'last' },
	narration: {
		what: t(
			'A GRU (gated recurrent unit) reads a sequence one step at a time and keeps a state of {hidden|int} numbers. After each step the new state is a blend of the old state and a candidate made from the input.',
			'GRU（门控循环单元）一步一步地读序列，并保留一个由 {hidden|int} 个数组成的状态。每读一步，新状态是旧状态和一个由输入算出的候选值的混合。'
		),
		how: t(
			'The update gate z (between 0 and 1) decides, number by number, how much of the old state to keep: h = (1 − z)·n + z·h_prev. The reset gate r decides how much of the old state the candidate n may use.',
			'更新门 z（介于 0 和 1 之间）逐个决定保留多少旧状态：h = (1 − z)·n + z·h_prev。重置门 r 决定候选值 n 能用到多少旧状态。'
		),
		look: t('Read the states column by column: each column is the state after one more character.', '一列一列地看这些状态：每一列是多读一个字符之后的状态。')
	},
	vars: (layer) => ({ hidden: Number(layer.params?.hidden ?? NaN) }),
	zoom: 'none',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: (layer) => `nn.GRU(input_size, ${layer.params?.hidden ?? 'hidden_size'}, batch_first=True)`,
	terms: ['recurrent-network', 'hidden-state']
};

export default gru;
