// tiny-encdec local op 'greedy': greedy decoding with the decoder block, from the encoder's memory [T, D]. The decoder
// runs once per symbol on <s> and everything written so far and writes the symbol with the largest score; the output
// is the decoder input [<s>, y_0 .. y_{steps-2}] (as numbers), which the kit layers after it (tgt.emb ... probs) read
// in one pass. The arithmetic is ../tf.ts (float64 sums, float32 storage); pipeline/tiny-encdec/ops_greedy.py is the
// PyTorch mirror. The weights are the decoder's own (the same tensors as the kit layers'; slots in tf.ts SLOTS).
import type { OpDef } from '../../../atlas/types/index.ts';
import { needRank, outArray, t } from '../../../atlas/ops/common.ts';
import { decW, greedy as run, SLOTS } from '../tf.ts';

type P = { steps: number; bos: number; nHead: number };

export const greedy: OpDef<P> = {
	op: 'greedy',
	version: 1,
	attrs: {
		steps: { type: 'int', required: true, doc: 'decoder positions: <s> and steps − 1 symbols' },
		bos: { type: 'int', default: 0, doc: 'id of the start symbol' },
		nHead: { type: 'int', required: true, doc: 'attention heads of the decoder block' }
	},
	category: 'attention',
	label: t('Greedy decoding', '贪心解码'),
	weights: SLOTS.map((slot) => ({ slot })),
	inferShape(ins, p) {
		if (ins[0].length !== 2) throw new Error(`greedy: input must be the encoder memory [T, D], got [${ins[0]}]`);
		return [p.steps];
	},
	axes: () => ['t'],
	forward(xs, p, ctx) {
		const mem = xs[0];
		needRank(mem, 2, 'greedy input');
		const w = decW((s) => ctx.w(s), p.nHead);
		const out = outArray(p.steps, mem);
		out.set(run(w, mem.data, mem.shape[0], Math.fround, p.steps).ids);
		return { shape: [p.steps], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'last' },
	narration: {
		what: t(
			'Greedy decoding: the decoder block runs once per symbol, {steps|int} times, on the start symbol and everything written so far, and each time writes the symbol with the largest score. Its output is what the decoder reads: the start symbol, then the symbols it wrote.',
			'贪心解码：解码器块每写一个符号就运行一次，一共 {steps|int} 次，每次读起始符号和已经写下的全部符号，并写下分数最大的那个符号。它的输出就是解码器要读的东西：起始符号，然后是写下的符号。'
		),
		look: t('One number per decoder position: the id of the symbol it reads.', '每个解码器位置一个数：它读入的符号的 id。')
	},
	vars: (layer) => ({ steps: Number(layer.params?.steps ?? NaN) }),
	zoom: 'none',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'ids = [bos]; for t in range(steps - 1): ids.append(decoder(ids, memory)[-1].argmax())',
	terms: ['greedy-decoding']
};

export default greedy;
