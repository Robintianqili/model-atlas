// tinyvit local ops (design §7): the three steps between the patch grid and the Transformer blocks that no core op does.
//   to_tokens     [C, H, W] -> [H·W, C]: patch t = row-major (y, x) becomes row t (a pure rearrangement)
//   cls_token     [T, D] + learned row c [D] -> [T+1, D]: c in front, the rows moved down by one
//   select_token  [T, D] -> [D]: row `index` (0: the [CLS] row), which the class head reads
// Python mirrors: pipeline/tinyvit/ops_tokens.py. Values are copied, never computed, so parity is exact.
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, int, outArray, store, t, unitBase, weightName } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;
const num = (d: unknown) => (typeof d === 'number' ? d : '?');

export const to_tokens: OpDef<NoP> = {
	op: 'to_tokens',
	version: 1,
	attrs: {},
	category: 'shape',
	label: t('Patches become rows', '图块变成行'),
	inferShape(ins) {
		const [C, H, W] = ins[0];
		if (ins[0].length !== 3) throw new Error('to_tokens: input must be [C, H, W]');
		return [typeof H === 'number' && typeof W === 'number' ? H * W : `${H}*${W}`, C];
	},
	axes: () => ['t', 'd'],
	forward(xs) {
		const [C, H, W] = xs[0].shape;
		const n = H * W;
		const out = outArray(C * n, xs[0]);
		for (let p = 0; p < n; p++) for (let c = 0; c < C; c++) out[p * C + c] = xs[0].data[c * n + p];
		return { shape: [n, C], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const [p, c] = q.unit;
		const [, H, W] = q.ins[0].shape;
		const src = c * H * W + p;
		const y = store(q.ins[0].data[src], q.out);
		const e = unitBase('index-map', q.layer, q.unit, y, q.out.data[flatIndex(q.out.shape, q.unit)]);
		e.reads = [{ tensor: q.layer.in[0], cells: [src] }];
		e.values = { t: p, c, row: Math.floor(p / W), col: p % W, y };
		return e;
	},
	formula: `${sym('out', 'y')}_{ {{t}},\\,{{c}} } = ${sym('in', 'x')}_{ {{c}},\\,{{row}},\\,{{col}} } = {{y}}`,
	narration: {
		what: t(
			'This step only rearranges: the {inShape} grid becomes {outShape}, one row of {d|int} numbers for each of the {n|int} patches, read row by row from the top left. No number changes.',
			'这一步只是重新排列：{inShape} 的网格变成 {outShape}，{n|int} 个图块各占一行、每行 {d|int} 个数，从左上角开始逐行读取。没有任何数改变。'
		),
		how: t('Row t holds the numbers of patch t, one from each channel.', '第 t 行装着第 t 个图块的数，每个通道一个。'),
		why: t('A Transformer works on a list of rows (tokens), so each patch becomes one token.', 'Transformer 处理的是一列行（词元），所以每个图块变成一个词元。'),
		look: t('Each row is one patch.', '每一行就是一个图块。'),
		unit: t('Row {t|int}, column {c|int} (counting from 0) is {y}, copied from channel {c|int} at patch row {row|int}, patch column {col|int}.', '第 {t|int} 行第 {c|int} 列（从 0 数起）是 {y}，复制自通道 {c|int} 在图块第 {row|int} 行、第 {col|int} 列的值。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), n: num(layer.outShape[0]), d: num(layer.outShape[1]) }),
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'x.flatten(1).transpose(0, 1)   # [48, 4, 4] -> [16, 48]',
	terms: ['token']
};

export const cls_token: OpDef<NoP> = {
	op: 'cls_token',
	version: 1,
	attrs: {},
	category: 'embedding',
	label: t('Put the [CLS] row in front', '在前面放上 [CLS] 行'),
	weights: [{ slot: 'weight' }],
	inferShape(ins) {
		const [T, D] = ins[0];
		if (ins[0].length !== 2) throw new Error('cls_token: input must be [T, D]');
		return [typeof T === 'number' ? T + 1 : `${T}+1`, D];
	},
	axes: () => ['t', 'd'],
	forward(xs, _p, ctx) {
		const c = ctx.w('weight');
		const [T, D] = xs[0].shape;
		if (c.data.length !== D) throw new Error(`cls_token: the [CLS] row has ${c.data.length} numbers, the rows ${D}`);
		const out = outArray((T + 1) * D, xs[0]);
		out.set(c.data, 0);
		out.set(xs[0].data, D);
		return { shape: [T + 1, D], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'first' },
	explain(q): UnitExplanation {
		const [r, d] = q.unit;
		const D = q.out.shape[1];
		const fromCls = r === 0;
		const v = fromCls ? q.w('weight').data[d] : q.ins[0].data[(r - 1) * D + d];
		const y = store(v, q.out);
		const e = unitBase('lookup', q.layer, q.unit, y, q.out.data[flatIndex(q.out.shape, q.unit)]);
		e.reads = [fromCls ? { tensor: weightName(q.layer, 'weight'), cells: [d] } : { tensor: q.layer.in[0], cells: [(r - 1) * D + d] }];
		e.values = { t: r, d, y, fromCls: fromCls ? 1 : 0 };
		return e;
	},
	formula: `${sym('out', 'y')}_{0,d} = ${sym('param', 'c')}_d,\\quad ${sym('out', 'y')}_{t,d} = ${sym('in', 'x')}_{t-1,d} = {{y}}`,
	narration: {
		what: t(
			'A learned row of {d|int} numbers, the [CLS] row, is put in front of the {n|int} patch rows, giving {outShape}. It is the same for every drawing: it holds no pixels, and at the end the class scores are read from it.',
			'一行学来的 {d|int} 个数，即 [CLS] 行，被放在 {n|int} 个图块行的前面，得到 {outShape}。它对每幅画都一样：它不含任何像素，最后的类别分数就从它读出。'
		),
		how: t('Row 0 is the [CLS] row; patch t moves to row t + 1.', '第 0 行是 [CLS] 行；第 t 个图块移到第 t + 1 行。'),
		why: t('The blocks let every row read the others, so this empty row can gather a summary of the whole drawing.', '各个块让每一行都能读取其他行，所以这个空行可以收集整幅画的概要。'),
		look: t('Row 0 does not depend on your drawing.', '第 0 行和你的画无关。'),
		unit: t('Row {t|int}, column {d|int} (counting from 0) is {y}.', '第 {t|int} 行第 {d|int} 列（从 0 数起）是 {y}。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), n: typeof layer.outShape[0] === 'number' ? layer.outShape[0] - 1 : '?', d: num(layer.outShape[1]) }),
	zoom: 'lookup',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'torch.cat([cls[None], x], 0)   # cls: nn.Parameter of 48 numbers',
	terms: ['token', 'embedding']
};

type SelP = { index: number };

export const select_token: OpDef<SelP> = {
	op: 'select_token',
	version: 1,
	attrs: { index: { type: 'int', default: 0, doc: 'the row to read (0: the [CLS] row)' } },
	category: 'shape',
	label: t('Read the [CLS] row', '读出 [CLS] 行'),
	inferShape(ins) {
		if (ins[0].length !== 2) throw new Error('select_token: input must be [T, D]');
		return [ins[0][1]];
	},
	axes: () => ['d'],
	forward(xs, p) {
		const D = xs[0].shape[1];
		const i = int(p, 'index', 0);
		if (i < 0 || i >= xs[0].shape[0]) throw new Error(`select_token: row ${i} is outside 0..${xs[0].shape[0] - 1}`);
		const out = outArray(D, xs[0]);
		out.set(xs[0].data.subarray(i * D, i * D + D));
		return { shape: [D], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const [d] = q.unit;
		const i = int(q.p, 'index', 0);
		const src = i * q.ins[0].shape[1] + d;
		const y = store(q.ins[0].data[src], q.out);
		const e = unitBase('index-map', q.layer, q.unit, y, q.out.data[d]);
		e.reads = [{ tensor: q.layer.in[0], cells: [src] }];
		e.values = { row: i, d, y };
		return e;
	},
	formula: `${sym('out', 'y')}_d = ${sym('in', 'x')}_{0,d} = {{y}}`,
	narration: {
		what: t(
			'Only row {index|int} (counting from 0), the [CLS] row, goes on: {d|int} numbers. The patch rows are dropped here; whatever the class head needs must already be in this row.',
			'只有第 {index|int} 行（从 0 数起），也就是 [CLS] 行，继续往下走：{d|int} 个数。图块行到这里就丢掉了；类别头需要的一切，必须已经在这一行里。'
		),
		how: t('The numbers are copied unchanged.', '这些数原样复制。'),
		look: t('These {d|int} numbers are all the class head sees.', '类别头看到的只有这 {d|int} 个数。'),
		unit: t('Entry {d|int} is {y}, copied from row {row|int}.', '第 {d|int} 个数是 {y}，复制自第 {row|int} 行。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), index: (layer.params?.index as number | undefined) ?? 0, d: num(layer.outShape[0]) }),
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'x[0]   # the [CLS] row',
	terms: ['token']
};
