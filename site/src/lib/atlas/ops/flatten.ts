// flatten: lays any tensor out as one row-major list (channel by channel, row by row). Values are copied unchanged.
import type { OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex, numel, unravel } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { baseVars, outArray, t, unitBase } from './common.ts';

type NoP = Record<string, never>;

export const flatten: OpDef<NoP> = {
	op: 'flatten',
	version: 1,
	attrs: {},
	category: 'shape',
	label: t('Flatten', '展平'),
	inferShape(ins) {
		const x = ins[0];
		if (x.every((d) => typeof d === 'number')) return [numel(x as number[])];
		return [x.join('*')];
	},
	axes: () => ['f'],
	forward(xs) {
		const x = xs[0];
		const out = outArray(x.data.length, x);
		out.set(x.data);
		return { shape: [x.data.length], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const src = q.ins[0];
		const v = src.data[f];
		const y = q.out.data instanceof Float64Array ? v : Math.fround(v);
		const e = unitBase('index-map', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells: [f] }];
		const from = unravel(src.shape, f);
		e.values = { index: f, y };
		// name the source position: (c, row, col) for a [C, H, W] map, else axis0, axis1, …
		if (src.shape.length === 3) Object.assign(e.values, { c: from[0], row: from[1], col: from[2] });
		else from.forEach((v2, i) => (e.values[`axis${i}`] = v2));
		return e;
	},
	formula: `${sym('out', 'y')} = ${sym('in', 'x')} = {{y}}`,
	narration: {
		what: t(
			'Flatten lays the {inShape} grid out as one long list of {count|int} numbers, channel by channel and row by row. No number changes.',
			'展平把形状为 {inShape} 的网格按通道、按行排成一长列，共 {count|int} 个数。数值本身不变。'
		),
		how: t('Only the arrangement changes: each grid position moves to one place in the list.', '只改变排列方式：网格中的每个位置被放到列表中的一个位置上。'),
		why: t('The next layer is a weighted sum over a plain list, so the grid has to become a list first.', '下一层是对一个普通列表做加权和，所以网格必须先变成列表。'),
		look: t('Each stretch of the list is one channel of the maps before.', '列表里的每一段对应前面的一个通道。'),
		unit: t('Index {index|int} holds {y}, copied unchanged from the input.', '第 {index|int} 个位置是 {y}，从输入原样复制过来。')
	},
	vars(layer, spec) {
		const n = layer.outShape.length === 1 ? layer.outShape[0] : layer.outShape.join('×');
		return { ...baseVars(layer, spec), count: n };
	},
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'torch.flatten(x)',
	terms: ['flatten', 'shape']
};

export default flatten;
