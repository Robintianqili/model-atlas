// conv2d: a kernel slides over a [C, H, W] grid. Loop order and arithmetic copy tinycnn.ts conv2dValid
// (for each output cell: sum over c, ky, kx in f64 starting at 0, then + bias), so the kit is bit-identical
// with padding 0. padding > 0 skips the out-of-range taps (they read zeros). accumulate: 'bias-first' starts at the bias.
import type { AtlasLayer, AtlasSpec, Box, Dim, OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { accMode, baseVars, biasWords, hasBias, intText, needRank, optW, outArray, specWeightShape, store, t, unitBase, weightName, words, type StructuralTerms } from './common.ts';

export type Conv2dP = {
	kernel: number;
	stride: number;
	padding: number;
	inChannels?: number;
	outChannels?: number;
	accumulate: 'bias-last' | 'bias-first';
};

function outSize(n: Dim, k: number, s: number, pad: number): Dim {
	if (typeof n !== 'number') return n; // symbolic spatial dims are passed through (validated at the dim extremes)
	const o = Math.floor((n + 2 * pad - k) / s) + 1;
	if (o < 1) throw new Error(`conv2d: input size ${n} is smaller than kernel ${k} (padding ${pad})`);
	return o;
}

function channels(layer: AtlasLayer, spec: AtlasSpec): { C: number | string; O: number | string; k: number } {
	const p = (layer.params ?? {}) as Partial<Conv2dP>;
	const ws = specWeightShape(layer, spec, 'weight');
	return {
		O: p.outChannels ?? ws?.[0] ?? layer.outShape[0],
		C: p.inChannels ?? ws?.[1] ?? '?',
		k: p.kernel ?? ws?.[2] ?? 0
	};
}

export const conv2d: OpDef<Conv2dP> & StructuralTerms = {
	op: 'conv2d',
	version: 1,
	attrs: {
		kernel: { type: 'int', required: true, doc: 'kernel height and width (square)' },
		stride: { type: 'int', default: 1 },
		padding: { type: 'int', default: 0, doc: 'zeros added on every side' },
		inChannels: { type: 'int', doc: 'informational; must equal weight.shape[1]' },
		outChannels: { type: 'int', doc: 'informational; must equal weight.shape[0]' },
		accumulate: { type: 'string', default: 'bias-last', doc: "'bias-last' (tinycnn.ts) or 'bias-first'" }
	},
	category: 'conv',
	label: t('Convolution', '卷积'),
	weights: [{ slot: 'weight' }, { slot: 'bias', optional: true }],
	inferShape(ins, p, w) {
		const x = ins[0];
		if (x.length !== 3) throw new Error(`conv2d: input must be [C, H, W], got [${x}]`);
		const ws = w.weight;
		if (!ws || ws.length !== 4) throw new Error('conv2d: weight must be [O, C, k, k]');
		const [O, C, kh, kw] = ws;
		if (kh !== kw || kh !== p.kernel) throw new Error(`conv2d: weight kernel ${kh}×${kw} does not match kernel=${p.kernel}`);
		if (typeof x[0] === 'number' && x[0] !== C) throw new Error(`conv2d: input has ${x[0]} channels, weight expects ${C}`);
		if (p.inChannels !== undefined && p.inChannels !== C) throw new Error(`conv2d: inChannels=${p.inChannels} but weight has ${C}`);
		if (p.outChannels !== undefined && p.outChannels !== O) throw new Error(`conv2d: outChannels=${p.outChannels} but weight has ${O}`);
		if (w.bias && (w.bias.length !== 1 || w.bias[0] !== O)) throw new Error(`conv2d: bias must be [${O}]`);
		const s = p.stride ?? 1;
		const pad = p.padding ?? 0;
		return [O, outSize(x[1], p.kernel, s, pad), outSize(x[2], p.kernel, s, pad)];
	},
	axes: () => ['c', 'y', 'x'],
	forward(xs, p, ctx) {
		const src = xs[0];
		needRank(src, 3, 'conv2d input');
		const W = ctx.w('weight');
		const B = optW(ctx, 'bias');
		const x = src.data;
		const w = W.data;
		const b = B ? B.data : null;
		const [C, H, Wd] = src.shape;
		const O = W.shape[0];
		const k = W.shape[2];
		const stride = p.stride ?? 1;
		const pad = p.padding ?? 0;
		const first = accMode(p) === 'bias-first';
		if (W.shape[1] !== C) throw new Error(`conv2d ${ctx.layer.id}: input has ${C} channels, weight expects ${W.shape[1]}`);
		const OH = Math.floor((H + 2 * pad - k) / stride) + 1;
		const OW = Math.floor((Wd + 2 * pad - k) / stride) + 1;
		const out = outArray(O * OH * OW, src, W);
		for (let o = 0; o < O; o++) {
			for (let oy = 0; oy < OH; oy++) {
				for (let ox = 0; ox < OW; ox++) {
					let s = first && b ? b[o] : 0;
					if (pad === 0) {
						for (let c = 0; c < C; c++) {
							const wBase = (o * C + c) * k * k;
							const xBase = c * H * Wd;
							for (let ky = 0; ky < k; ky++) {
								const row = xBase + (oy * stride + ky) * Wd + ox * stride;
								const wRow = wBase + ky * k;
								for (let kx = 0; kx < k; kx++) s += x[row + kx] * w[wRow + kx];
							}
						}
					} else {
						for (let c = 0; c < C; c++) {
							const wBase = (o * C + c) * k * k;
							const xBase = c * H * Wd;
							for (let ky = 0; ky < k; ky++) {
								const iy = oy * stride + ky - pad;
								if (iy < 0 || iy >= H) continue;
								for (let kx = 0; kx < k; kx++) {
									const ix = ox * stride + kx - pad;
									if (ix < 0 || ix >= Wd) continue;
									s += x[xBase + iy * Wd + ix] * w[wBase + ky * k + kx];
								}
							}
						}
					}
					out[(o * OH + oy) * OW + ox] = !first && b ? s + b[o] : s;
				}
			}
		}
		return { shape: [O, OH, OW], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'raster', pick: 'strongestChannel' },
	explain(q): UnitExplanation {
		const [o, oy, ox] = q.unit;
		const f = flatIndex(q.out.shape, q.unit);
		const src = q.ins[0];
		const W = q.w('weight');
		const B = optW(q, 'bias');
		const [C, H, Wd] = src.shape;
		const k = W.shape[2];
		const stride = q.p.stride ?? 1;
		const pad = q.p.padding ?? 0;
		const first = accMode(q.p) === 'bias-first';
		const bias = B ? B.data[o] : 0;
		let s = first && B ? bias : 0;
		let sumProducts = 0;
		const terms: NonNullable<UnitExplanation['terms']> = [];
		const groups: NonNullable<UnitExplanation['groups']> = [];
		const xCells: number[] = [];
		const wCells: number[] = [];
		for (let c = 0; c < C; c++) {
			const from = terms.length;
			let cs = 0;
			for (let ky = 0; ky < k; ky++) {
				for (let kx = 0; kx < k; kx++) {
					const iy = oy * stride + ky - pad;
					const ix = ox * stride + kx - pad;
					const wAt = ((o * C + c) * k + ky) * k + kx;
					const inside = iy >= 0 && iy < H && ix >= 0 && ix < Wd;
					const xAt = inside ? (c * H + iy) * Wd + ix : -1;
					const xv = inside ? src.data[xAt] : 0;
					const wv = W.data[wAt];
					const pr = xv * wv;
					if (inside) {
						s += pr; // same order as forward (padded taps are skipped there too)
						xCells.push(xAt);
					}
					sumProducts += pr;
					cs += pr;
					wCells.push(wAt);
					terms.push(inside ? { x: xv, w: wv, p: pr, xAt, wAt } : { x: 0, w: wv, p: pr, wAt });
				}
			}
			groups.push({ label: t(`input channel ${c}`, `输入通道 ${c}`), from, to: terms.length, sum: cs });
		}
		const y = store(!first && B ? s + bias : s, q.out);
		const e = unitBase('weighted-sum', q.layer, q.unit, y, q.out.data[f]);
		const y0 = oy * stride - pad;
		const x0 = ox * stride - pad;
		const box: Box = { y0: Math.max(0, y0), x0: Math.max(0, x0), y1: Math.min(H - 1, y0 + k - 1), x1: Math.min(Wd - 1, x0 + k - 1) };
		e.reads = [
			{ tensor: q.layer.in[0], cells: xCells, box },
			{ tensor: weightName(q.layer, 'weight'), cells: wCells }
		];
		e.terms = terms;
		e.groups = groups;
		e.layout = { rows: k, cols: k };
		if (B) e.bias = bias;
		e.values = { n: terms.length, s: sumProducts, b: bias, y, ch: o, row: oy, col: ox, k, c: C };
		return e;
	},
	receptive(box, p) {
		const s = p.stride ?? 1;
		const pad = p.padding ?? 0;
		const k = p.kernel;
		return { y0: box.y0 * s - pad, x0: box.x0 * s - pad, y1: box.y1 * s - pad + k - 1, x1: box.x1 * s - pad + k - 1 };
	},
	formula: `${sym('out', 'y')} = \\sum_{c,i,j} ${sym('in', 'x')}_{c,i,j}\\,${sym('param', 'w')}_{c,i,j} + ${sym('param', 'b')} = {{s}} + {{b}} = {{y}}`,
	narration: {
		// what + how = 4 sentences (AK407). {kernelIs}, {slideHow} and {biasClause} are phrases chosen by vars() from the
		// layer: one input channel or several, stride and padding, a bias or none.
		what: t(
			'A kernel is {kernelIs}. It slides over the input {slideHow}; at each stop the numbers under it are multiplied one by one by the kernel’s numbers, and the {kernelCount|int} products are summed{biasClause}. That sum is one cell of the output.',
			'卷积核{kernelIs}。它在输入上滑动，{slideHow}；每到一处，它下面的数和卷积核的数逐个相乘，{kernelCount|int} 个乘积相加{biasClause}。这个和就是输出里的一个格子。'
		),
		how: t(
			'This layer has {outChannels|int} kernels, one per output channel, so the output has shape {outShape}.',
			'这一层有 {outChannels|int} 个卷积核，每个对应一个输出通道，所以输出的形状是 {outShape}。'
		),
		why: t(
			'The same kernel is used at every position, so a pattern it detects is found wherever it appears in the input.',
			'同一个卷积核在每个位置都使用，所以它识别的图案无论出现在输入的哪里都能被找到。'
		),
		look: t(
			'Large positive cells show where the window matches the kernel’s pattern; large negative cells show where it matches the opposite pattern.',
			'数值大的正格子表示窗口和卷积核的图案相符；数值大的负格子表示和相反的图案相符。'
		),
		// the short step of a distinct repeat (op.distinct: the first conv that reads more than one channel)
		repeat: t(
			'This convolution reads {inChannels|int} channels, not one, so each kernel is {inChannels|int} stacked {kernel|int}×{kernel|int} grids. Each output cell sums {kernelCount|int} products, the same {kernel|int}×{kernel|int} window in each of the {inChannels|int} channels{biasClause}.',
			'这一层卷积读入 {inChannels|int} 个通道，而不是 1 个，所以每个卷积核是 {inChannels|int} 个叠在一起的 {kernel|int}×{kernel|int} 网格。每个输出格子把 {kernelCount|int} 个乘积相加：{inChannels|int} 个通道里同一个 {kernel|int}×{kernel|int} 窗口{biasClause}。'
		),
		unit: t(
			'Here the {n|int} products add up to {s}; {biasIs} {b}. The output is {s} + ({b}) = {y}.',
			'这里 {n|int} 个乘积加起来是 {s}；{biasIs} {b}。输出是 {s} + ({b}) = {y}。'
		)
	},
	vars(layer, spec) {
		const { C, O, k } = channels(layer, spec);
		const p = (layer.params ?? {}) as Partial<Conv2dP>;
		const stride = p.stride ?? 1;
		const pad = p.padding ?? 0;
		const v: Record<string, string | number> = {
			...baseVars(layer, spec),
			kernel: k,
			inChannels: C,
			outChannels: O,
			stride,
			padding: pad
		};
		v.kernelCount = typeof C === 'number' ? C * k * k : `${C}×${k}×${k}`;
		v.weightCount = typeof C === 'number' && typeof O === 'number' ? O * C * k * k + O : '?';
		const n = intText(v.kernelCount);
		v.kernelIs =
			C === 1
				? words(`a ${k}×${k} grid of ${n} learned numbers`, `是一个 ${k}×${k} 的网格，共 ${n} 个学来的数`)
				: typeof C === 'number'
					? words(`${n} learned numbers: a ${k}×${k} grid for each of the ${C} input channels`, `是 ${n} 个学来的数：${C} 个输入通道各有一个 ${k}×${k} 的网格`)
					: words(`a stack of ${k}×${k} grids of learned numbers, one grid for each input channel`, `是一叠 ${k}×${k} 的网格，每个输入通道一个，里面都是学来的数`);
		const stepEn = stride === 1 ? 'one cell at a time' : `${stride} cells at a time`;
		const stepZh = stride === 1 ? '每次移动一格' : `每次移动 ${stride} 格`;
		v.slideHow =
			pad > 0
				? words(`${stepEn}, with ${pad === 1 ? 'one ring' : `${pad} rings`} of zeros added around the input first`, `${stepZh}（输入四周先补上 ${pad} 圈 0）`)
				: words(stepEn, stepZh);
		Object.assign(v, biasWords(hasBias(layer, spec)));
		return v;
	},
	distinct: (p, prev) => (prev.inChannels ?? 1) === 1 && (p.inChannels ?? 1) > 1,
	zoom: 'weighted-sum',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch(layer) {
		const p = (layer.params ?? {}) as Partial<Conv2dP>;
		const C = p.inChannels ?? 'in_channels';
		const O = p.outChannels ?? (typeof layer.outShape[0] === 'number' ? layer.outShape[0] : 'out_channels');
		const extra = `${p.stride && p.stride !== 1 ? `, stride=${p.stride}` : ''}${p.padding ? `, padding=${p.padding}` : ''}`;
		return `nn.Conv2d(${C}, ${O}, kernel_size=${p.kernel ?? '?'}${extra})`;
	},
	terms: ['convolution', 'kernel', 'stride', 'padding', 'feature-map', 'receptive-field'],
	// 'stride' and 'padding' only where the layer uses them
	termsFor: (layer) => {
		const p = (layer.params ?? {}) as Partial<Conv2dP>;
		return ['convolution', 'kernel', ...((p.stride ?? 1) > 1 ? ['stride'] : []), ...((p.padding ?? 0) > 0 ? ['padding'] : []), 'feature-map', 'receptive-field'];
	}
};

export default conv2d;

/** Output size helper for other ops (pooling uses the same arithmetic). */
export { outSize as convOutSize };
