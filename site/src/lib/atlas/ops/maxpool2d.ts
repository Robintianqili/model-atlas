// maxpool2d: the largest value of each kernel×kernel block, per channel. Same loops as tinycnn.ts maxpool()
// (m starts at −∞; `if (v > m) m = v`, so the first maximum wins ties). stride defaults to kernel.
// explain(): a reduce. terms[i] = { x: window value, w: 1 for the winner else 0, p: x·w }, so Σp is the max
// (max pooling read as a one-hot selection); values.argmax is the winner's index in the window.
import type { Dim, OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { baseVars, needRank, outArray, store, t, unitBase } from './common.ts';

export type MaxPoolP = { kernel: number; stride?: number };

const size = (n: Dim, k: number, s: number): Dim => {
	if (typeof n !== 'number') return n;
	const o = Math.floor((n - k) / s) + 1;
	if (o < 1) throw new Error(`maxpool2d: input size ${n} is smaller than kernel ${k}`);
	return o;
};

export const maxpool2d: OpDef<MaxPoolP> = {
	op: 'maxpool2d',
	version: 1,
	attrs: {
		kernel: { type: 'int', default: 2 },
		stride: { type: 'int', doc: 'defaults to kernel' }
	},
	category: 'pool',
	label: t('Max pooling', '最大池化'),
	inferShape(ins, p) {
		const x = ins[0];
		if (x.length !== 3) throw new Error(`maxpool2d: input must be [C, H, W], got [${x}]`);
		const k = p.kernel ?? 2;
		const s = p.stride ?? k;
		return [x[0], size(x[1], k, s), size(x[2], k, s)];
	},
	axes: () => ['c', 'y', 'x'],
	forward(xs, p) {
		const src = xs[0];
		needRank(src, 3, 'maxpool2d input');
		const x = src.data;
		const [C, H, W] = src.shape;
		const k = p.kernel ?? 2;
		const stride = p.stride ?? k;
		const OH = Math.floor((H - k) / stride) + 1;
		const OW = Math.floor((W - k) / stride) + 1;
		const out = outArray(C * OH * OW, src);
		for (let c = 0; c < C; c++) {
			for (let oy = 0; oy < OH; oy++) {
				for (let ox = 0; ox < OW; ox++) {
					let m = -Infinity;
					for (let ky = 0; ky < k; ky++) {
						for (let kx = 0; kx < k; kx++) {
							const v = x[(c * H + oy * stride + ky) * W + ox * stride + kx];
							if (v > m) m = v;
						}
					}
					out[(c * OH + oy) * OW + ox] = m;
				}
			}
		}
		return { shape: [C, OH, OW], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'raster', pick: 'argmax' },
	explain(q): UnitExplanation {
		const [c, oy, ox] = q.unit;
		const f = flatIndex(q.out.shape, q.unit);
		const src = q.ins[0];
		const [, H, W] = src.shape;
		const k = q.p.kernel ?? 2;
		const stride = q.p.stride ?? k;
		let m = -Infinity;
		let best = 0;
		const cells: number[] = [];
		const vals: number[] = [];
		for (let ky = 0; ky < k; ky++) {
			for (let kx = 0; kx < k; kx++) {
				const at = (c * H + oy * stride + ky) * W + ox * stride + kx;
				const v = src.data[at];
				if (v > m) {
					m = v;
					best = ky * k + kx;
				}
				cells.push(at);
				vals.push(v);
			}
		}
		const y = store(m, q.out);
		const e = unitBase('reduce', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells, box: { y0: oy * stride, x0: ox * stride, y1: oy * stride + k - 1, x1: ox * stride + k - 1 } }];
		e.terms = vals.map((v, i) => ({ x: v, w: i === best ? 1 : 0, p: i === best ? v : 0, xAt: cells[i] }));
		e.layout = { rows: k, cols: k };
		e.values = { n: k * k, y, max: m, argmax: best, ch: c, row: oy, col: ox };
		return e;
	},
	receptive(box, p) {
		const k = p.kernel ?? 2;
		const s = p.stride ?? k;
		return { y0: box.y0 * s, x0: box.x0 * s, y1: box.y1 * s + k - 1, x1: box.x1 * s + k - 1 };
	},
	formula: `${sym('out', 'y')} = \\max_{i,j}\\ ${sym('in', 'x')}_{i,j} = {{y}}`,
	narration: {
		what: t(
			'Max pooling looks at each {kernel|int}×{kernel|int} block of a map and keeps only the largest of its {windowCount|int} numbers. Moving {stride|int} cells at a time, it shrinks the maps from {inShape} to {outShape}.',
			'最大池化查看每张图里每个 {kernel|int}×{kernel|int} 的小块，只保留其中 {windowCount|int} 个数里最大的那个。它每次移动 {stride|int} 格，把图从 {inShape} 缩小到 {outShape}。'
		),
		how: t('Each channel is pooled on its own. Nothing is learned here: there are no weights.', '每个通道单独池化。这里没有要学习的东西，也没有权重。'),
		why: t(
			'Keeping the strongest response of each block makes the maps smaller and lets later layers care less about small shifts of the input.',
			'在每个小块里只保留最强的响应，让图变小，也让后面的层不太在意输入的小幅移动。'
		),
		look: t('Each output cell equals the largest cell of its block in the map before.', '每个输出格子等于前一张图中对应小块里最大的那个格子。'),
		unit: t('The largest of the {n|int} values in this block is {y}.', '这个小块的 {n|int} 个值中最大的是 {y}。')
	},
	vars(layer, spec) {
		const p = (layer.params ?? {}) as Partial<MaxPoolP>;
		const k = p.kernel ?? 2;
		return { ...baseVars(layer, spec), kernel: k, stride: p.stride ?? k, windowCount: k * k };
	},
	zoom: 'reduce',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch(layer) {
		const p = (layer.params ?? {}) as Partial<MaxPoolP>;
		const k = p.kernel ?? 2;
		return p.stride !== undefined && p.stride !== k ? `nn.MaxPool2d(${k}, stride=${p.stride})` : `nn.MaxPool2d(${k})`;
	},
	terms: ['pooling', 'receptive-field', 'feature-map']
};

export default maxpool2d;
