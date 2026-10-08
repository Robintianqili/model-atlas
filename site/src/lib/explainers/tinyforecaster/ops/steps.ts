// tinyforecaster local ops: the two steps around the LSTM that no core op does (model.ts toSpec).
//   relative    [T, D] -> [T, D]: every row minus the last row, y_t = x_t − x_{T−1} (the window as differences from its
//               last value; the last row becomes 0). Same arithmetic as the trainer: one f64 subtraction.
//   last_step   [T, D] -> [D]: the last row (the LSTM's last h for the head; the window's last value for the add).
// PyTorch: x - x[-1:] and x[-1]. Values are subtracted or copied, so the trace equals the autograd forward exactly.
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, store, t, unitBase } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;
const num = (d: unknown) => (typeof d === 'number' ? d : '?');

export const relative: OpDef<NoP> = {
	op: 'relative',
	version: 1,
	attrs: {},
	category: 'shape',
	label: t('Minus the last value', '减去最后一个值'),
	inferShape(ins) {
		if (ins[0].length !== 2) throw new Error('relative: input must be [T, D]');
		return ins[0].slice();
	},
	axes: () => ['t', 'd'],
	forward(xs) {
		const [T, D] = xs[0].shape;
		const x = xs[0].data;
		const out = outArray(T * D, xs[0]);
		for (let r = 0; r < T; r++) for (let d = 0; d < D; d++) out[r * D + d] = x[r * D + d] - x[(T - 1) * D + d];
		return { shape: [T, D], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'first' },
	explain(q): UnitExplanation {
		const [r, d] = q.unit;
		const [T, D] = q.ins[0].shape;
		const a = q.ins[0].data[r * D + d];
		const last = q.ins[0].data[(T - 1) * D + d];
		const y = store(a - last, q.out);
		const e = unitBase('elementwise', q.layer, q.unit, y, q.out.data[flatIndex(q.out.shape, q.unit)]);
		e.reads = [{ tensor: q.layer.in[0], cells: [r * D + d, (T - 1) * D + d] }];
		e.terms = [
			{ x: a, w: 1, p: a, xAt: r * D + d },
			{ x: last, w: -1, p: -last, xAt: (T - 1) * D + d }
		];
		e.values = { a, b: -last, last, y, t: r };
		return e;
	},
	formula: `${sym('out', 'y')}_t = ${sym('in', 'x')}_t - ${sym('in', 'x')}_{\\text{last}} = {{a}} - {{last}} = {{y}}`,
	narration: {
		what: t(
			'Every value of the window has the window’s last value subtracted, giving {outShape}. The network then sees the shape of the window, not how high it sits; the last row becomes 0.',
			'窗口里的每个值都减去窗口的最后一个值，得到 {outShape}。这样网络看到的是窗口的形状，而不是它所在的高度；最后一行变成 0。'
		),
		how: t('y_t = x_t − x_last for every time step t.', '对每个时间步 t：y_t = x_t − x_last。'),
		why: t('A window that is the same shape one unit higher gives the same numbers, so what is learned on low values also works on high ones.', '形状相同、只是整体高出一截的窗口，得到的数完全一样；所以在低处学到的规律，在高处也能用。'),
		unit: t('Step {t|int} (counting from 0): {a} − {last} = {y}.', '第 {t|int} 步（从 0 数起）：{a} − {last} = {y}。')
	},
	vars: (layer, spec) => baseVars(layer, spec),
	zoom: 'elementwise',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'x - x[-1:]   # every value minus the last one',
	terms: []
};

export const last_step: OpDef<NoP> = {
	op: 'last_step',
	version: 1,
	attrs: {},
	category: 'shape',
	label: t('The last row', '最后一行'),
	inferShape(ins) {
		if (ins[0].length !== 2) throw new Error('last_step: input must be [T, D]');
		return [ins[0][1]];
	},
	axes: () => ['d'],
	forward(xs) {
		const [T, D] = xs[0].shape;
		if (T < 1) throw new Error('last_step: the input has no rows');
		const out = outArray(D, xs[0]);
		out.set(xs[0].data.subarray((T - 1) * D, T * D));
		return { shape: [D], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const [d] = q.unit;
		const [T, D] = q.ins[0].shape;
		const src = (T - 1) * D + d;
		const y = store(q.ins[0].data[src], q.out);
		const e = unitBase('index-map', q.layer, q.unit, y, q.out.data[d]);
		e.reads = [{ tensor: q.layer.in[0], cells: [src] }];
		e.values = { row: T - 1, d, y };
		return e;
	},
	formula: `${sym('out', 'y')}_d = ${sym('in', 'x')}_{ {{row}},\\,d } = {{y}}`,
	narration: {
		what: t('Only the last row goes on: {d|int} numbers. Everything the earlier steps found must already be in it.', '只有最后一行继续往下走：{d|int} 个数。前面各步发现的一切，必须已经装在这一行里。'),
		how: t('The numbers are copied unchanged.', '这些数原样复制。'),
		unit: t('Entry {d|int} is {y}, copied from row {row|int}.', '第 {d|int} 个数是 {y}，复制自第 {row|int} 行。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), d: num(layer.outShape[0]) }),
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'x[-1]   # the last row',
	terms: []
};
