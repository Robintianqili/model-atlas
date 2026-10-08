// vae-2d local op 'to_image': [S·S] -> [1, S, S], the decoder's 784 scores laid out as a 28 × 28 picture, row by row.
// Values are copied, never computed, so parity is exact. Python mirror: pipeline/vae-2d/ops_to_image.py (reshape).
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, t, unitBase } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;

function side(n: number): number {
	const s = Math.round(Math.sqrt(n));
	if (s * s !== n) throw new Error(`to_image: ${n} numbers do not make a square picture`);
	return s;
}

export const toImage: OpDef<NoP> = {
	op: 'to_image',
	version: 1,
	attrs: {},
	category: 'shape',
	label: t('Lay out as a picture', '排成一幅图'),
	inferShape(ins) {
		const n = ins[0];
		if (n.length !== 1 || typeof n[0] !== 'number') throw new Error('to_image: input must be a vector of known length');
		const s = side(n[0]);
		return [1, s, s];
	},
	axes: () => ['c', 'y', 'x'],
	forward(xs) {
		const x = xs[0];
		const s = side(x.data.length);
		const out = outArray(x.data.length, x);
		out.set(x.data);
		return { shape: [1, s, s], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const v = q.ins[0].data[f];
		const y = q.out.data instanceof Float64Array ? v : Math.fround(v);
		const e = unitBase('index-map', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells: [f] }];
		e.values = { index: f, row: q.unit[1], col: q.unit[2], y };
		return e;
	},
	formula: `${sym('out', 'y')}_{ {{row}},\\,{{col}} } = ${sym('in', 'x')}_{ {{index}} } = {{y}}`,
	narration: {
		what: t(
			'The list of {count|int} numbers is laid out again as a {outShape} picture, row by row from the top left. No number changes.',
			'这一列 {count|int} 个数被重新排成 {outShape} 的图，从左上角开始逐行排。没有任何数改变。'
		),
		how: t('Number k goes to row k ÷ 28 (rounded down) and column k mod 28.', '第 k 个数放到第 k ÷ 28（向下取整）行、第 k mod 28 列。'),
		why: t('The decoder works on a plain list; as a picture its output can be compared with your drawing pixel by pixel.', '解码器处理的是普通的一列数；排成图以后，它的输出就能和你的画逐个像素比较。'),
		look: t('Each row of the picture is a stretch of 28 numbers of the list.', '图的每一行就是列表里连续的 28 个数。'),
		unit: t('Row {row|int}, column {col|int} (counting from 0) holds {y}, number {index|int} of the list.', '第 {row|int} 行第 {col|int} 列（从 0 数起）是 {y}，即列表里的第 {index|int} 个数。')
	},
	vars(layer, spec) {
		const n = layer.outShape.reduce((a: number, b) => a * (typeof b === 'number' ? b : NaN), 1);
		return { ...baseVars(layer, spec), count: n };
	},
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'x.view(-1, 1, 28, 28)'
};

export default toImage;
