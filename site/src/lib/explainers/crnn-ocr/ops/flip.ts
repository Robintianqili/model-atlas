// crnn-ocr local op 'flip': [T, D] -> [T, D] with the rows in reverse order (torch.flip(x, [0])). The backward LSTM of
// the bidirectional pair reads the columns flipped, and its outputs are flipped back, so row t again belongs to column t.
// Values are copied, never computed, so parity is exact. Python mirror: pipeline/crnn-ocr/ops_flip.py.
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, t, unitBase } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;

export const flip: OpDef<NoP> = {
	op: 'flip',
	version: 1,
	attrs: {},
	category: 'shape',
	label: t('Reverse the order', '倒过来排'),
	inferShape(ins) {
		if (ins[0].length !== 2) throw new Error(`flip: input must be [T, D], got [${ins[0]}]`);
		return ins[0].slice();
	},
	axes: () => ['t', 'd'],
	forward(xs) {
		const x = xs[0];
		const [T, D] = x.shape;
		const out = outArray(x.data.length, x);
		for (let r = 0; r < T; r++) for (let d = 0; d < D; d++) out[r * D + d] = x.data[(T - 1 - r) * D + d];
		return { shape: [T, D], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const [T, D] = q.ins[0].shape;
		const [r, d] = q.unit;
		const src = (T - 1 - r) * D + d;
		const e = unitBase('index-map', q.layer, q.unit, q.ins[0].data[src], q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells: [src] }];
		e.values = { r, d, from: T - 1 - r, v: q.ins[0].data[src] };
		return e;
	},
	formula: `${sym('out', 'y')}[{{r}}, {{d}}] = ${sym('in', 'x')}[{{from}}, {{d}}] = {{v}}`,
	narration: {
		what: t('The {rows|int} rows are put in reverse order: the last becomes the first. No number changes.', '这 {rows|int} 行倒过来排：最后一行变成第一行。没有任何数改变。'),
		how: t('Row t of the output is row T − 1 − t of the input.', '输出的第 t 行是输入的第 T − 1 − t 行。'),
		why: t('An LSTM always reads from the first row on; reversing the rows makes it read right to left.', 'LSTM 总是从第一行读起；把行倒过来，它就从右往左读了。'),
		look: t('The pattern of the table is mirrored top to bottom.', '表的图案上下翻转了。'),
		unit: t('Row {r|int} (counting from 0) is row {from|int} of the input: {v}.', '第 {r|int} 行（从 0 数起）是输入的第 {from|int} 行：{v}。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), rows: Number(layer.outShape[0]) }),
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'torch.flip(x, [0])'
};

export default flip;
