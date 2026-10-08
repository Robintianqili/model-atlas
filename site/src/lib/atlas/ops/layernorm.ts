// layernorm over the last axis, biased variance (PyTorch nn.LayerNorm). Same arithmetic as tinygpt.ts layerNorm():
//   mean = Σx / D; v = Σ(x − mean)² / D; inv = 1 / sqrt(v + eps); y = (x − mean) · inv · γ + β   (all f64, stored f32)
// γ and β are optional (elementwise_affine=False gives γ = 1, β = 0).
import type { OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { baseVars, optW, outArray, store, t, unitBase, weightName } from './common.ts';

export type LayerNormP = { eps: number };

/** ε as a plain decimal for narration ("0.00001", not "1.00e−5"); very small values keep the exponent form. */
function epsText(eps: number): number | string {
	if (!(eps > 0) || eps < 1e-9) return eps;
	const digits = Math.min(20, Math.ceil(-Math.log10(eps)) + 3);
	return eps.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
}

export const layernorm: OpDef<LayerNormP> = {
	op: 'layernorm',
	version: 1,
	attrs: { eps: { type: 'float', default: 1e-5, doc: 'added to the variance before the square root' } },
	category: 'norm',
	label: t('Layer normalization', '层归一化'),
	weights: [
		{ slot: 'weight', optional: true },
		{ slot: 'bias', optional: true }
	],
	inferShape(ins, _p, w) {
		const x = ins[0];
		const D = x[x.length - 1];
		for (const s of ['weight', 'bias']) {
			const ws = w[s];
			if (ws && (ws.length !== 1 || (typeof D === 'number' && ws[0] !== D))) throw new Error(`layernorm: ${s} must be [${D}]`);
		}
		return x.slice();
	},
	axes: (ins) => ins[0].slice(),
	forward(xs, p, ctx) {
		const src = xs[0];
		const x = src.data;
		const D = src.shape[src.shape.length - 1];
		const G = optW(ctx, 'weight');
		const B = optW(ctx, 'bias');
		const g = G ? G.data : null;
		const b = B ? B.data : null;
		const eps = p.eps ?? 1e-5;
		const T = x.length / D;
		const y = outArray(x.length, src);
		for (let r = 0; r < T; r++) {
			const o = r * D;
			let mean = 0;
			for (let i = 0; i < D; i++) mean += x[o + i];
			mean /= D;
			let v = 0;
			for (let i = 0; i < D; i++) v += (x[o + i] - mean) ** 2;
			v /= D;
			const inv = 1 / Math.sqrt(v + eps);
			if (g && b) for (let i = 0; i < D; i++) y[o + i] = (x[o + i] - mean) * inv * g[i] + b[i];
			else for (let i = 0; i < D; i++) y[o + i] = (x[o + i] - mean) * inv * (g ? g[i] : 1) + (b ? b[i] : 0);
		}
		return { shape: src.shape.slice(), data: y };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const src = q.ins[0];
		const x = src.data;
		const D = src.shape[src.shape.length - 1];
		const i = q.unit[q.unit.length - 1];
		const o = f - i;
		const G = optW(q, 'weight');
		const B = optW(q, 'bias');
		const eps = q.p.eps ?? 1e-5;
		let mean = 0;
		for (let k = 0; k < D; k++) mean += x[o + k];
		mean /= D;
		let v = 0;
		for (let k = 0; k < D; k++) v += (x[o + k] - mean) ** 2;
		v /= D;
		const inv = 1 / Math.sqrt(v + eps);
		const gamma = G ? G.data[i] : 1;
		const beta = B ? B.data[i] : 0;
		const xv = x[o + i];
		const xhat = (xv - mean) * inv;
		const y = store(G && B ? (xv - mean) * inv * G.data[i] + B.data[i] : (xv - mean) * inv * gamma + beta, q.out);
		const e = unitBase('normalize', q.layer, q.unit, y, q.out.data[f]);
		const row = Array.from({ length: D }, (_, k) => o + k);
		e.reads = [{ tensor: q.layer.in[0], cells: row }];
		if (G) e.reads.push({ tensor: weightName(q.layer, 'weight'), cells: [i] });
		if (B) e.reads.push({ tensor: weightName(q.layer, 'bias'), cells: [i] });
		const std = Math.sqrt(v + eps);
		e.steps = [
			{ label: t('mean of the row', '这一行的平均值'), value: mean, tex: '\\mu' },
			{ label: t('variance of the row', '这一行的方差'), value: v, tex: '\\sigma^2' },
			{ label: t('standard deviation √(variance + ε)', '标准差 √(方差 + ε)'), value: std, tex: '\\sqrt{\\sigma^2+\\varepsilon}' },
			{ label: t('normalized value (x − mean) / std', '归一化后的值 (x − 平均值) / 标准差'), value: xhat, tex: '\\hat{x}' },
			{ label: t('times the learned scale γ', '乘以学来的缩放 γ'), value: xhat * gamma, tex: '\\hat{x}\\,\\gamma' },
			{ label: t('plus the learned shift β', '加上学来的偏移 β'), value: y, tex: '\\hat{x}\\,\\gamma + \\beta' }
		];
		e.values = { x: xv, mean, var: v, eps, std, xhat, gamma, beta, y, n: D };
		return e;
	},
	formula: `${sym('out', 'y')} = \\frac{${sym('in', 'x')} - \\mu}{\\sqrt{\\sigma^2 + \\varepsilon}}\\,${sym('param', '\\gamma')} + ${sym('param', '\\beta')} = \\frac{ {{x}} - {{mean}} }{ {{std}} } \\cdot {{gamma}} + {{beta}} = {{y}}`,
	narration: {
		what: t(
			'Layer normalization rescales each row of {features|int} numbers so that its mean is 0 and its spread (standard deviation) is 1. Then every feature is multiplied by a learned scale γ and gets a learned shift β added.',
			'层归一化把每一行 {features|int} 个数重新缩放，使它的平均值为 0、离散程度（标准差）为 1。然后每个特征乘上学来的缩放 γ，再加上学来的偏移 β。'
		),
		how: t(
			'For one row: subtract the mean, then divide by the standard deviation, computed as the square root of the variance plus a tiny ε = {epsilon} so that a flat row never divides by 0.',
			'对一行：先减去平均值，再除以标准差；标准差按“方差加上一个很小的 ε = {epsilon} 再开平方”来算，这样完全平的一行也不会除以 0。'
		),
		why: t('It keeps the numbers in a steady range from layer to layer, which helps deep stacks train reliably.', '它让各层的数保持在稳定的范围内，帮助很深的网络可靠地训练。'),
		look: t('After normalization every row has a similar spread, even if the rows before had very different sizes.', '归一化之后每一行的离散程度都差不多，即使之前各行的大小相差很大。'),
		unit: t(
			'This row has mean {mean} and standard deviation {std}. The value {x} becomes ({x} − {mean}) / {std} = {xhat}, then × {gamma} + {beta} = {y}.',
			'这一行的平均值是 {mean}，标准差是 {std}。值 {x} 变成 ({x} − {mean}) / {std} = {xhat}，再 × {gamma} + {beta} = {y}。'
		)
	},
	vars(layer, spec) {
		const p = (layer.params ?? {}) as Partial<LayerNormP>;
		return { ...baseVars(layer, spec), features: layer.outShape[layer.outShape.length - 1], epsilon: epsText(p.eps ?? 1e-5) };
	},
	zoom: 'normalize',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch(layer) {
		const p = (layer.params ?? {}) as Partial<LayerNormP>;
		const D = layer.outShape[layer.outShape.length - 1];
		return `nn.LayerNorm(${D}${p.eps !== undefined && p.eps !== 1e-5 ? `, eps=${p.eps}` : ''})`;
	},
	terms: ['layer-norm']
};

export default layernorm;
