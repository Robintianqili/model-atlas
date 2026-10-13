// crnn-ocr local op 'columns': [C, H, W] -> [W, C·H], the feature maps cut into their W columns, left to right. Row w of
// the output holds every channel's values in column w: feature c·H + h is channel c, row h (model.py columns()). The
// columns become the time steps of the LSTMs. Values are copied, never computed, so parity is exact.
// Python mirror: pipeline/crnn-ocr/ops_columns.py.
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, t, unitBase } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;

export const columns: OpDef<NoP> = {
	op: 'columns',
	version: 1,
	attrs: {},
	category: 'shape',
	label: t('Cut into columns', '切成一列一列'),
	inferShape(ins) {
		if (ins[0].length !== 3) throw new Error(`columns: input must be [C, H, W], got [${ins[0]}]`);
		const [C, H, W] = ins[0];
		return [W, typeof C === 'number' && typeof H === 'number' ? C * H : `${C}*${H}`];
	},
	axes: () => ['t', 'f'],
	forward(xs) {
		const x = xs[0];
		const [C, H, W] = x.shape;
		const out = outArray(x.data.length, x);
		for (let c = 0; c < C; c++) for (let h = 0; h < H; h++) for (let w = 0; w < W; w++) out[w * C * H + c * H + h] = x.data[(c * H + h) * W + w];
		return { shape: [W, C * H], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const [, H, W] = q.ins[0].shape;
		const [col, feat] = q.unit;
		const [c, row] = [Math.floor(feat / H), feat % H];
		const src = (c * H + row) * W + col;
		const e = unitBase('index-map', q.layer, q.unit, q.ins[0].data[src], q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells: [src] }];
		e.values = { col, feat, c, row, v: q.ins[0].data[src] };
		return e;
	},
	formula: `${sym('out', 'y')}[{{col}}, {{feat}}] = ${sym('in', 'x')}[{{c}}, {{row}}, {{col}}] = {{v}}`,
	narration: {
		what: t(
			'The {inShape} feature maps are cut into {cols|int} columns, left to right; each column becomes one list of {feats|int} numbers. No number changes.',
			'形状为 {inShape} 的特征图从左到右切成 {cols|int} 列；每一列变成一组 {feats|int} 个数。没有任何数改变。'
		),
		how: t('Column w collects every channel’s values in that column, channel by channel and row by row.', '第 w 列按通道、按行收集所有通道在这一列里的数。'),
		why: t('A recurrent network reads a sequence, so the picture is turned into a sequence of columns.', '循环网络读的是序列，所以图要先变成一列接一列的序列。'),
		look: t('Each row of the table is one column of the maps.', '表中的每一行就是特征图的一列。'),
		unit: t('Column {col|int}, feature {feat|int} (counting from 0) is channel {c|int}, row {row|int} of the maps: {v}.', '第 {col|int} 列的第 {feat|int} 个特征（从 0 数起）是特征图第 {c|int} 个通道第 {row|int} 行的数：{v}。')
	},
	vars(layer, spec) {
		const [cols, feats] = layer.outShape;
		return { ...baseVars(layer, spec), cols: Number(cols), feats: Number(feats) };
	},
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'x.permute(2, 0, 1).reshape(W, C * H)'
};

export default columns;
