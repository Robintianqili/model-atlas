// fno local ops: the four steps of a Fourier layer's spectral path (Python mirrors: pipeline/fno/ops_fno.py, the same
// arithmetic as model.py). A field is [x, c] (64 points, c channels); a spectrum is [k, c, 2] (mode k, channel c,
// real and imaginary part), as torch.view_as_real(torch.fft.rfft(v, dim=0, norm="forward")) lays it out.
//   rfft          [n, c] -> [n/2 + 1, c, 2]   X_k = (1/n) sum_x u_x (cos - i sin)(2 pi k x / n)
//   keep_modes    [m, c, 2], keep [1] -> [modes, c, 2]: the lowest `modes` modes; those at or above keep[0] become 0
//   spectral_mix  [K, i, 2] -> [K, o, 2]: Y_k,o = sum_i X_k,i (wr + i wi)_k,i,o, one complex weight per mode and channel pair
//   irfft         [K, c, 2] -> [n, c]      u_x = sum_k a_k (Re Y_k cos - Im Y_k sin)(2 pi k x / n), a_0 = 1, a_k = 2
// f64 accumulation, f32 storage (atlas/ops/common.ts conventions); explain() lists the products of one output number.
import type { OpDef, UnitExplanation, UnitQuery } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, int, outArray, store, t, unitBase, weightName } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;
type Term = NonNullable<UnitExplanation['terms']>[number];
const TAU = 2 * Math.PI;
const ang = (k: number, x: number, n: number) => (TAU * ((k * x) % n)) / n;
/** the weight of mode k in the inverse transform of a real curve of n points */
const twice = (k: number, n: number) => (k === 0 || 2 * k === n ? 1 : 2);
const base = { version: 1, color: 'diverging' as const, parity: { bar: 1e-4 } };
/** the explanation of one output number that is a weighted sum of terms (x · w = p) */
function sumOf<P>(q: UnitQuery<P>, terms: Term[], values: Record<string, number>): UnitExplanation {
	const y = store(terms.reduce((a, v) => a + v.p, 0), q.out), e = unitBase('weighted-sum', q.layer, q.unit, y, q.out.data[flatIndex(q.out.shape, q.unit)]);
	e.reads = [{ tensor: q.layer.in[0], cells: terms.map((v) => v.xAt!) }];
	return Object.assign(e, { terms, values: { ...values, y } });
}

export const rfft: OpDef<NoP> = {
	op: 'rfft',
	...base,
	attrs: {},
	category: 'linear',
	label: t('Fourier transform (FFT)', '傅里叶变换（FFT）'),
	inferShape(ins) {
		const [n, c] = ins[0];
		if (ins[0].length !== 2 || typeof n !== 'number') throw new Error('rfft: input must be [n, c]');
		return [Math.floor(n / 2) + 1, c, 2];
	},
	axes: () => ['k', 'c', 'xy'],
	forward([x]) {
		const [n, C] = x.shape, m = Math.floor(n / 2) + 1, out = outArray(m * C * 2, x);
		for (let k = 0; k < m; k++)
			for (let c = 0; c < C; c++) {
				let re = 0, im = 0;
				for (let j = 0; j < n; j++) (re += x.data[j * C + c] * Math.cos(ang(k, j, n))), (im -= x.data[j * C + c] * Math.sin(ang(k, j, n)));
				out[(k * C + c) * 2] = re / n;
				out[(k * C + c) * 2 + 1] = im / n;
			}
		return { shape: [m, C, 2], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const [k, c, p] = q.unit, [n, C] = q.ins[0].shape;
		const terms = Array.from({ length: n }, (_, j) => {
			const x = q.ins[0].data[j * C + c], w = (p === 0 ? Math.cos(ang(k, j, n)) : -Math.sin(ang(k, j, n))) / n;
			return { x, w, p: x * w, xAt: j * C + c };
		});
		return sumOf(q, terms, { k, c, part: p, n });
	},
	formula: `${sym('out', 'X')}_{ {{k}} } = \\tfrac{1}{N}\\sum_{x} ${sym('in', 'u')}_x \\big(\\cos - i\\sin\\big)\\tfrac{2\\pi\\,{{k}}\\,x}{N} \\;\\Rightarrow\\; {{y}}`,
	narration: {
		what: t('The Fourier transform rewrites each channel’s {n|int} heights as {m|int} waves (modes): mode k goes k times around the ring. Each mode is two numbers, how much cosine and how much sine of that wave the curve holds; the result has shape {outShape}.', '傅里叶变换把每个通道的 {n|int} 个高度改写成 {m|int} 个波（模式）：模式 k 绕环 k 圈。每个模式是两个数，表示曲线里含有多少这个波的余弦和多少正弦；结果的形状是 {outShape}。'),
		how: t('One number of mode k is a weighted sum of all the heights, the weights being a cosine (or minus a sine) wave divided by N.', '模式 k 的一个数是全部高度的加权和，权重是一条余弦波（或负的正弦波）再除以 N。'),
		why: t('Heat spreads each wave on its own: in this form every mode can be treated separately.', '热量对每个波各自起作用：写成这种形式后，每个模式都可以单独处理。'),
		unit: t('Mode {k|int}, channel {c|int} (counting from 0): the sum is {y}.', '模式 {k|int}、通道 {c|int}（从 0 数起）：总和是 {y}。')
	},
	vars: (layer, spec) => {
		const v = baseVars(layer, spec), n = Number(layer.outShape[0]);
		return { ...v, m: n, n: (n - 1) * 2 };
	},
	zoom: 'weighted-sum',
	pytorch: () => 'torch.view_as_real(torch.fft.rfft(v, dim=0, norm="forward"))',
};

export type KeepP = { modes: number };
export const keep_modes: OpDef<KeepP> = {
	op: 'keep_modes',
	...base,
	attrs: { modes: { type: 'int', required: true, doc: 'how many of the lowest modes the layer has weights for' } },
	category: 'shape',
	label: t('Keep the lowest modes', '只保留最低的模式'),
	inferShape(ins, p) {
		if (ins[0].length !== 3) throw new Error('keep_modes: input must be [m, c, 2]');
		return [int(p, 'modes'), ins[0][1], 2];
	},
	axes: () => ['k', 'c', 'xy'],
	forward([x, keep], p) {
		const K = int(p, 'modes'), [m, C] = x.shape, cut = Math.min(K, m, Math.max(0, Math.round(keep.data[0]))), out = outArray(K * C * 2, x);
		out.set(x.data.subarray(0, cut * C * 2));
		return { shape: [K, C, 2], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'first' },
	explain(q): UnitExplanation {
		const [k, c, p] = q.unit, C = q.ins[0].shape[1], cut = Math.min(int(q.p, 'modes'), q.ins[0].shape[0], Math.max(0, Math.round(q.ins[1].data[0])));
		const src = (k * C + c) * 2 + p, kept = k < cut ? 1 : 0, y = store(kept ? q.ins[0].data[src] : 0, q.out);
		const e = unitBase('index-map', q.layer, q.unit, y, q.out.data[flatIndex(q.out.shape, q.unit)]);
		e.reads = [{ tensor: q.layer.in[0], cells: [src] }];
		e.values = { k, c, kept, cut, x: q.ins[0].data[src], y };
		return e;
	},
	formula: `${sym('out', 'X')}_{ {{k}} } = ${sym('in', 'X')}_{ {{k}} } \\text{ if } {{k}} < {{cut}} \\text{, else } 0 \\;\\Rightarrow\\; {{y}}`,
	narration: {
		what: t('Only the lowest {modes|int} modes go on; every higher mode is dropped. The result has shape {outShape}.', '只有最低的 {modes|int} 个模式继续往下走；更高的模式全部丢掉。结果的形状是 {outShape}。'),
		how: t('A kept mode is copied unchanged; a mode at or above the cut becomes 0.', '保留的模式原样复制；在截断处及以上的模式变成 0。'),
		unit: t('Mode {k|int}: {y}.', '模式 {k|int}：{y}。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), modes: Number(layer.outShape[0]) }),
	zoom: 'index-map',
	pytorch: (l) => `X[:${l.outShape[0]}] * (torch.arange(${l.outShape[0]}) < keep)[:, None, None]`,
};

export const spectral_mix: OpDef<NoP> = {
	op: 'spectral_mix',
	...base,
	attrs: {},
	category: 'linear',
	label: t('Multiply by learned complex weights', '乘以学来的复数权重'),
	weights: [{ slot: 'wr' }, { slot: 'wi' }],
	inferShape(ins, _p, w) {
		const [K, I, O] = w.wr ?? [];
		if (ins[0].length !== 3 || ins[0][0] !== K || ins[0][1] !== I) throw new Error(`spectral_mix: input must be [${K}, ${I}, 2]`);
		return [K, O, 2];
	},
	axes: () => ['k', 'c', 'xy'],
	forward([x], _p, ctx) {
		const wr = ctx.w('wr').data, wi = ctx.w('wi').data, [K, I, O] = ctx.w('wr').shape, out = outArray(K * O * 2, x);
		for (let k = 0; k < K; k++)
			for (let o = 0; o < O; o++) {
				let re = 0, im = 0;
				for (let i = 0; i < I; i++) {
					const a = x.data[(k * I + i) * 2], b = x.data[(k * I + i) * 2 + 1], w = (k * I + i) * O + o;
					re += a * wr[w] - b * wi[w];
					im += a * wi[w] + b * wr[w];
				}
				out[(k * O + o) * 2] = re;
				out[(k * O + o) * 2 + 1] = im;
			}
		return { shape: [K, O, 2], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const [k, o, p] = q.unit, WR = q.w('wr'), wr = WR.data, wi = q.w('wi').data, [, I, O] = WR.shape, x = q.ins[0].data;
		const terms: Term[] = [];
		for (let i = 0; i < I; i++) {
			const at = (k * I + i) * 2, w = (k * I + i) * O + o;
			// real part: a·wr − b·wi; imaginary part: a·wi + b·wr (a, b: the input's real and imaginary parts)
			const pairs: [number, number, number][] = p === 0 ? [[x[at], wr[w], at], [x[at + 1], -wi[w], at + 1]] : [[x[at], wi[w], at], [x[at + 1], wr[w], at + 1]];
			for (const [xv, wv, xAt] of pairs) terms.push({ x: xv, w: wv, p: xv * wv, xAt, wAt: w });
		}
		const e = sumOf(q, terms, { k, o, part: p, n: terms.length }), cells = Array.from({ length: I }, (_, i) => (k * I + i) * O + o);
		e.reads.push({ tensor: weightName(q.layer, 'wr'), cells }, { tensor: weightName(q.layer, 'wi'), cells });
		return e;
	},
	formula: `${sym('out', 'Y')}_{ {{k}},{{o}} } = \\sum_i ${sym('in', 'X')}_{ {{k}},i }\\,(${sym('param', 'w^{re}')} + i\\,${sym('param', 'w^{im}')})_{ {{k}},i,{{o}} } \\;\\Rightarrow\\; {{y}}`,
	narration: {
		what: t('Each kept mode is multiplied by learned complex weights: for every mode, output channel o adds up the input channels’ two numbers times a learned pair (a real and an imaginary weight). The result has shape {outShape}.', '每个保留的模式都乘以学来的复数权重：对每个模式，输出通道 o 把各输入通道的两个数乘上一对学来的数（实部权重和虚部权重）再相加。结果的形状是 {outShape}。'),
		how: t('(a + bi)(c + di) = (ac − bd) + (ad + bc)i: a real part needs a·c and −b·d.', '(a + bi)(c + di) = (ac − bd) + (ad + bc)i：实部需要 a·c 和 −b·d。'),
		why: t('Scaling a mode changes how much of that wave survives; turning it (the imaginary part) shifts the wave around the ring.', '把一个模式放大或缩小，改变这个波留下多少；把它旋转（虚部）则让这个波沿环移动。'),
		unit: t('Mode {k|int}, output channel {o|int} (counting from 0): {n|int} products add up to {y}.', '模式 {k|int}、输出通道 {o|int}（从 0 数起）：{n|int} 个乘积加起来是 {y}。')
	},
	vars: (layer, spec) => baseVars(layer, spec),
	zoom: 'weighted-sum',
	pytorch: () => 'torch.einsum("ki,kio->ko", X, torch.complex(self.wr, self.wi))',
	terms: ['weight']
};

export type IrfftP = { n: number };
export const irfft: OpDef<IrfftP> = {
	op: 'irfft',
	...base,
	attrs: { n: { type: 'int', required: true, doc: 'the number of points of the curve' } },
	category: 'linear',
	label: t('Inverse Fourier transform', '傅里叶逆变换'),
	inferShape(ins, p) {
		if (ins[0].length !== 3) throw new Error('irfft: input must be [K, c, 2]');
		return [int(p, 'n'), ins[0][1]];
	},
	axes: () => ['x', 'c'],
	forward([y], p) {
		const n = int(p, 'n'), [K, C] = y.shape, out = outArray(n * C, y);
		for (let j = 0; j < n; j++)
			for (let c = 0; c < C; c++) {
				let s = 0;
				for (let k = 0; k < Math.min(K, Math.floor(n / 2) + 1); k++) s += twice(k, n) * (y.data[(k * C + c) * 2] * Math.cos(ang(k, j, n)) - y.data[(k * C + c) * 2 + 1] * Math.sin(ang(k, j, n)));
				out[j * C + c] = s;
			}
		return { shape: [n, C], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const [j, c] = q.unit, n = int(q.p, 'n'), [K, C] = q.ins[0].shape, d = q.ins[0].data;
		const terms: Term[] = [];
		for (let k = 0; k < Math.min(K, Math.floor(n / 2) + 1); k++) {
			const at = (k * C + c) * 2, a = twice(k, n);
			terms.push({ x: d[at], w: a * Math.cos(ang(k, j, n)), p: d[at] * a * Math.cos(ang(k, j, n)), xAt: at });
			terms.push({ x: d[at + 1], w: -a * Math.sin(ang(k, j, n)), p: -d[at + 1] * a * Math.sin(ang(k, j, n)), xAt: at + 1 });
		}
		return sumOf(q, terms, { x: j, c, modes: K });
	},
	formula: `${sym('out', 'u')}_{ {{x}} } = \\sum_{k} a_k\\big(\\mathrm{Re}\\,${sym('in', 'Y')}_k \\cos - \\mathrm{Im}\\,${sym('in', 'Y')}_k \\sin\\big)\\tfrac{2\\pi k\\,{{x}}}{N} = {{y}}`,
	narration: {
		what: t('The inverse transform adds the waves back up into {n|int} heights per channel, shape {outShape}. Only the kept modes contribute, so the result is a smooth curve.', '逆变换把这些波重新加起来，每个通道得到 {n|int} 个高度，形状是 {outShape}。只有保留的模式参与，所以结果是一条平滑的曲线。'),
		how: t('Each height is a weighted sum over the modes: cosine and sine of mode k at that point, counted twice for k > 0 (the wave and its mirror).', '每个高度都是对各模式的加权和：模式 k 在该点的余弦和正弦；k > 0 时计两次（这个波和它的镜像）。'),
		unit: t('Point {x|int}, channel {c|int} (counting from 0): {y}.', '第 {x|int} 个点、通道 {c|int}（从 0 数起）：{y}。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), n: Number(layer.outShape[0]) }),
	zoom: 'weighted-sum',
	pytorch: (l) => `torch.fft.irfft(torch.view_as_complex(Y), n=${l.outShape[0]}, dim=0, norm="forward")`,
};

