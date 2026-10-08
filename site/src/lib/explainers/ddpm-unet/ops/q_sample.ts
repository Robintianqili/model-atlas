// ddpm-unet local op 'q_sample' (the forward process in one jump): x_t = √ᾱ_t · (2x₀ − 1) + √(1 − ᾱ_t) · ε, cell by cell:
// the drawing's 0 (paper) … 1 (ink) is first stretched to −1 … 1, centred like the noise. Inputs [x0 [1, 28, 28], eps [1, 28, 28], t [1]];
// attribute abar [T]: ᾱ_t at index t − 1, the float32 values of the PyTorch buffer (the schedule, not learned).
// Python mirror: pipeline/ddpm-unet/ops_q_sample.py (model.py Noised.forward).
import type { OpDef, Tensor, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, store, t, unitBase } from '../../../atlas/ops/common.ts';

type P = { abar: number[] };

/** √ᾱ_t and √(1 − ᾱ_t) of the timestep t (the table holds ᾱ_t at index t − 1) */
export function mixOf(abar: ArrayLike<number>, tv: number) {
	const i = Math.round(tv) - 1;
	if (!(i >= 0 && i < abar.length)) throw new Error(`q_sample: t = ${tv} is outside 1..${abar.length}`);
	const ab = abar[i];
	return { abar: ab, a: Math.sqrt(ab), b: Math.sqrt(1 - ab) };
}

export const qSample: OpDef<P> = {
	op: 'q_sample',
	version: 1,
	attrs: { abar: { type: 'float[]', required: true, doc: 'ᾱ_t for t = 1 … T (index t − 1): the noise schedule' } },
	category: 'merge',
	label: t('Add noise: x_t = √ᾱ_t·(2x₀ − 1) + √(1 − ᾱ_t)·ε', '加噪：x_t = √ᾱ_t·(2x₀ − 1) + √(1 − ᾱ_t)·ε'),
	inferShape(ins) {
		if (ins.length !== 3 || ins[2].join() !== '1') throw new Error('q_sample: needs [x0, eps, t] with t of shape [1]');
		if (ins[0].join() !== ins[1].join()) throw new Error('q_sample: x0 and eps must have the same shape');
		return ins[0].slice();
	},
	axes: () => ['c', 'y', 'x'],
	forward(xs, p) {
		const [x0, eps, tt] = xs;
		const { a, b } = mixOf(p.abar, tt.data[0]);
		const y = outArray(x0.data.length, x0, eps);
		for (let i = 0; i < y.length; i++) y[i] = a * (2 * x0.data[i] - 1) + b * eps.data[i];
		return { shape: x0.shape.slice(), data: y };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const [x0, eps, tt] = (q.ins as Tensor[]).map((x, k) => x.data[k === 2 ? 0 : f]);
		const m = mixOf(q.p.abar, tt);
		const s = 2 * x0 - 1, y = store(m.a * s + m.b * eps, q.out);
		const e = unitBase('elementwise', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[0], cells: [f] }, { tensor: q.layer.in[1], cells: [f] }];
		e.terms = [{ x: s, w: m.a, p: m.a * s }, { x: eps, w: m.b, p: m.b * eps }];
		e.values = { t: tt, abar: m.abar, a: m.a, b: m.b, x0, s, eps, y };
		return e;
	},
	formula: `${sym('out', 'x_t')} = \\sqrt{\\bar\\alpha_t}\\,(2${sym('in', 'x_0')} - 1) + \\sqrt{1-\\bar\\alpha_t}\\,${sym('in', '\\varepsilon')} = {{a}} \\times {{s}} + {{b}} \\times {{eps}} = {{y}}`,
	narration: {
		what: t(
			'Noise is mixed into the drawing in one jump: the drawing’s pixels are stretched from 0 … 1 to −1 … 1, then every pixel of x_t is √ᾱ_t times that plus √(1 − ᾱ_t) times a random number ε from a bell curve. ᾱ_t is read from the noise schedule at step t.',
			'噪声一步就混进画里：先把画的像素从 0 … 1 拉伸到 −1 … 1，然后 x_t 的每个像素等于 √ᾱ_t 乘以它，再加上 √(1 − ᾱ_t) 乘以一个从钟形曲线里抽出的随机数 ε。ᾱ_t 是在第 t 步从噪声调度表里查出来的。'
		),
		how: t('The two weights √ᾱ_t and √(1 − ᾱ_t) are the same for every pixel; their squares add up to 1.', '√ᾱ_t 和 √(1 − ᾱ_t) 这两个权重对所有像素都一样；它们的平方加起来等于 1。'),
		why: t('Training needs noisy drawings at any step t; this shortcut makes one without running t small steps.', '训练要用任意第 t 步的加噪画；这条捷径不必真的走 t 小步就能造出一幅。'),
		look: t('The larger t, the less of the drawing is left and the more noise.', 't 越大，画剩下的越少，噪声越多。'),
		unit: t('{a} × (2 × {x0} − 1) + {b} × {eps} = {y}.', '{a} × (2 × {x0} − 1) + {b} × {eps} = {y}。')
	},
	vars: (layer, spec) => baseVars(layer, spec),
	zoom: 'none',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'xt = torch.sqrt(abar[t - 1]) * (2 * x0 - 1) + torch.sqrt(1 - abar[t - 1]) * eps',
	terms: ['forward-process', 'noise-schedule']
};

export default qSample;
