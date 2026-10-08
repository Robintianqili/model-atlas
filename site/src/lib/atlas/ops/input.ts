// input: reads a declared input (spec.inputs) and stores it as a float tensor (layer 0 of the trace).
// legacy.ts turns TinyCNN's explicit `input` layer into this op with in: ['input'].
import type { OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { baseVars, outArray, t, unitBase } from './common.ts';

export type InputP = { name?: string };

export const input: OpDef<InputP> = {
	op: 'input',
	version: 1,
	attrs: { name: { type: 'string', doc: "the declared input to read when the layer's `in` is empty (default: the first declared input)" } },
	category: 'io',
	label: t('Input', '输入'),
	inferShape: (ins) => {
		if (!ins.length) throw new Error('input: no declared input to read');
		return ins[0].slice();
	},
	axes: (ins) => (ins[0] ?? []).slice(),
	forward(xs) {
		const x = xs[0];
		const out = outArray(x.data.length, x);
		out.set(x.data);
		return { shape: x.shape.slice(), data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const src = q.ins[0];
		const x = src ? src.data[f] : q.out.data[f];
		const y = q.out.data instanceof Float64Array ? x : Math.fround(x);
		const e = unitBase('none', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0] ?? q.layer.id, cells: [f] }];
		e.values = { x: y, y, index: f };
		return e;
	},
	receptive: (box) => ({ ...box }),
	formula: `${sym('out', 'y')} = ${sym('in', 'x')} = {{y}}`,
	narration: {
		what: t(
			'This is where your input enters the model, as a tensor of shape {outShape}. The first real layer reads exactly these numbers.',
			'这里是你的输入进入模型的地方，是一个形状为 {outShape} 的张量。第一层真正的计算读到的就是这些数。'
		),
		how: t('Nothing is computed here. The input layer only hands the prepared numbers to the next layer.', '这里不做任何计算。输入层只是把准备好的数交给下一层。'),
		look: t('Each cell is one number; its colour shows how large it is.', '每个格子是一个数，颜色表示它的大小。'),
		unit: t('This cell holds {y}.', '这个格子的值是 {y}。')
	},
	vars: (layer, spec) => baseVars(layer, spec),
	zoom: 'none',
	color: 'sequential',
	parity: { bar: 1e-4 },
	pytorch: () => 'x = torch.as_tensor(prepared_input)',
	terms: ['tensor', 'shape']
};

export default input;
