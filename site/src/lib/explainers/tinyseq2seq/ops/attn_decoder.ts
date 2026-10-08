// tinyseq2seq local op 'attn_decoder': a GRU decoder with Bahdanau (additive) attention, decoding greedily for `steps`
// steps from the encoder states [T, H]: s starts as the last encoder state, y as the start symbol `bos`. Output: the
// logits [steps, V] of every step. Sub-steps: #keys [T, A] (once per input), #query [steps, A], #scores and #weights
// [steps, T], #context and #state [steps, H], #next [steps] (the chosen symbol). The arithmetic is ../seq.ts (float64
// sums, float32 storage); pipeline/tinyseq2seq/ops_attn_decoder.py is the PyTorch mirror.
import type { OpDef, Tensor } from '../../../atlas/types/index.ts';
import { needRank, outArray, t } from '../../../atlas/ops/common.ts';
import { decW, greedy } from '../seq.ts';

type P = { steps: number; bos: number };

const SUBS = ['keys', 'query', 'scores', 'weights', 'context', 'state', 'next'] as const;

export const attnDecoder: OpDef<P> = {
	op: 'attn_decoder',
	version: 1,
	attrs: { steps: { type: 'int', required: true, doc: 'decoder steps (always all of them)' }, bos: { type: 'int', default: 0, doc: 'id of the start symbol' } },
	category: 'attention',
	label: t('Decoder with attention', '带注意力的解码器'),
	weights: ['emb', 'wq', 'wk', 'bk', 'v', 'weight_ih', 'weight_hh', 'bias_ih', 'bias_hh', 'wo', 'bo'].map((slot) => ({ slot })),
	subs: [
		{ name: 'keys', label: t('keys W_k h_j + b_k', '键 W_k h_j + b_k'), axes: ['t', 'd'] },
		{ name: 'query', label: t('queries W_q s', '查询 W_q s'), axes: ['t', 'd'] },
		{ name: 'scores', label: t('attention scores', '注意力分数'), axes: ['tq', 'tk'] },
		{ name: 'weights', label: t('attention weights', '注意力权重'), axes: ['tq', 'tk'] },
		{ name: 'context', label: t('context vectors', '上下文向量'), axes: ['t', 'd'] },
		{ name: 'state', label: t('decoder states', '解码器状态'), axes: ['t', 'd'] },
		{ name: 'next', label: t('chosen symbols', '选出的符号'), axes: ['t'] }
	],
	inferShape(ins, p, w) {
		if (ins[0].length !== 2) throw new Error(`attn_decoder: input must be the encoder states [T, H], got [${ins[0]}]`);
		return [p.steps, w.wo[0]];
	},
	axes: () => ['t', 'vocab'],
	forward(xs, p, ctx) {
		const enc = xs[0];
		needRank(enc, 2, 'attn_decoder input');
		const w = decW((n) => ctx.w(n).data, (n) => ctx.w(n).shape, '');
		const [T, H] = enc.shape;
		const L = p.steps;
		const mk = (n: number, shape: number[]): Tensor => ({ shape, data: outArray(n, enc) });
		const o = { keys: mk(T * w.A, [T, w.A]), query: mk(L * w.A, [L, w.A]), scores: mk(L * T, [L, T]), weights: mk(L * T, [L, T]), context: mk(L * H, [L, H]), state: mk(L * H, [L, H]), next: mk(L, [L]) };
		const logits = mk(L * w.V, [L, w.V]);
		let i = 0;
		for (const r of greedy(w, enc.data, T, Math.fround, L)) {
			if (i === 0) o.keys.data.set(r.K);
			o.query.data.set(r.q, i * w.A);
			o.scores.data.set(r.e, i * T);
			o.weights.data.set(r.a, i * T);
			o.context.data.set(r.c, i * H);
			o.state.data.set(r.s, i * H);
			o.next.data[i] = r.y;
			logits.data.set(r.logits, i * w.V);
			i++;
		}
		for (const k of SUBS) ctx.emit(k, o[k]);
		return logits;
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	narration: {
		what: t(
			'The decoder writes the answer one symbol at a time, {steps|int} steps in all. At every step it scores each position of the input against its own state (attention), mixes the encoder states with those weights into a context vector, and updates its state from the context and the symbol it wrote last.',
			'解码器一次写一个符号，一共 {steps|int} 步。每一步它都拿自己的状态给输入的每个位置打分（注意力），按这些权重把编码器状态混合成一个上下文向量，再根据上下文向量和上一步写下的符号更新自己的状态。'
		),
		look: t('Each row is one decoder step; its scores and weights cover every input character.', '每一行是解码器的一步；它的分数和权重覆盖输入的每一个字符。')
	},
	vars: (layer) => ({ steps: Number(layer.params?.steps ?? NaN) }),
	zoom: 'none',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'for t in range(steps): e = v(tanh(W_q(s) + W_k(h))); a = softmax(e); c = a @ h; s = gru_cell(cat(emb(y), c), s); y = out(cat(s, c)).argmax()',
	terms: ['attention', 'softmax']
};

export default attnDecoder;
