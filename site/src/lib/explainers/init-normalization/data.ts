// init-normalization: the toy's setup and the plain arithmetic shared by the processes, the widget, the views and the
// claim tests. The toy is a deep MLP: 256 examples of 64 random numbers (standard normal, seeded) pass through 10
// hidden layers of 64 units (tanh or ReLU) and one output; the loss is the mean squared error against 256 random
// targets. Every weight starts as a random number with spread σ (the learner's setting); every bias starts at 0.
// No third-party data.
import { fmtSetting } from '../../atlas/core/format.ts';

export const WIDTH = 64;
export const DEPTH = 10;
export const BATCH = 256;
/** Histogram bins of the views (one histogram per layer). */
export const BINS = 40;
/** A tanh output at or beyond this size counts as saturated (the curve is flat there: its slope 1 − a² is below 0.02). */
export const SAT = 0.99;
/** The small tensor of the BatchNorm and LayerNorm steps: the first NB examples and NF units of layer 1. */
export const NB = 4;
export const NF = 6;
/** The ε both normalizations add to the variance before the square root (PyTorch's default). */
export const EPS = 1e-5;
/** Weights: 10 hidden layers of 64 × 64 and an output of 64; biases: 64 per hidden layer and 1. */
export const N_WEIGHTS = DEPTH * WIDTH * WIDTH + WIDTH;
export const N_BIASES = DEPTH * WIDTH + 1;

export type Act = 'tanh' | 'relu';
export type Norm = 'none' | 'batch' | 'layer';
export type InitId = 'zero' | 'small' | 'xavier' | 'he' | 'large';

export interface Setup {
	act: Act;
	/** the spread (standard deviation) of every weight at the start */
	std: number;
	/** a normalization of every hidden layer's weighted sums, before the activation function */
	norm: Norm;
}

export const INITS: InitId[] = ['zero', 'small', 'xavier', 'he', 'large'];
/** σ of each named initialization for this network (every hidden layer has fan-in = fan-out = 64). */
export const INIT_STD: Record<InitId, number> = {
	zero: 0,
	small: 0.01,
	xavier: Math.sqrt(2 / (WIDTH + WIDTH)),
	he: Math.sqrt(2 / WIDTH),
	large: 1
};
export const INIT_NAME: Record<InitId, { en: string; zh: string }> = {
	zero: { en: 'zero', zh: '全零' },
	small: { en: 'too small', zh: '过小' },
	xavier: { en: 'Xavier', zh: 'Xavier' },
	he: { en: 'He', zh: 'He' },
	large: { en: 'too large', zh: '过大' }
};
export const ACT_NAME: Record<Act, { en: string; zh: string }> = { tanh: { en: 'tanh', zh: 'tanh' }, relu: { en: 'ReLU', zh: 'ReLU' } };
export const NORM_NAME: Record<Norm, { en: string; zh: string }> = {
	none: { en: 'no normalization', zh: '不归一化' },
	batch: { en: 'BatchNorm', zh: 'BatchNorm' },
	layer: { en: 'LayerNorm', zh: 'LayerNorm' }
};
/** The slider's range of σ (log scale); σ = 0 only from the zero button. */
export const STD_RANGE: [number, number] = [0.001, 2];

export const START: Setup = { act: 'tanh', std: INIT_STD.xavier, norm: 'none' };

const ACTS: Act[] = ['tanh', 'relu'];
const NORMS: Norm[] = ['none', 'batch', 'layer'];

/** Any raw input -> a setup: a known activation and normalization, σ = 0 or inside the slider's range. */
export function setupOf(raw: unknown): Setup {
	const o = (raw && typeof raw === 'object' ? raw : {}) as Partial<Record<keyof Setup, unknown>>;
	const act = ACTS.includes(o.act as Act) ? (o.act as Act) : START.act;
	const norm = NORMS.includes(o.norm as Norm) ? (o.norm as Norm) : 'none';
	let std = typeof o.std === 'number' && Number.isFinite(o.std) ? o.std : START.std;
	std = std <= 0 ? 0 : Math.min(STD_RANGE[1], Math.max(STD_RANGE[0], std));
	return { act, std, norm };
}

/** The named initialization a σ is, or null. */
export function initOf(std: number): InitId | null {
	for (const id of INITS) if (Math.abs(std - INIT_STD[id]) <= 1e-12) return id;
	return null;
}
export const actCode = (a: Act) => (a === 'relu' ? 1 : 0);
export const normCode = (n: Norm) => NORMS.indexOf(n);
export const actOfCode = (c: number): Act => (c === 1 ? 'relu' : 'tanh');
export const normOfCode = (c: number): Norm => NORMS[c] ?? 'none';

/** The mean and spread (standard deviation, biased as the normalizations use it) of n numbers of a, from `from`, every
 *  `stride`-th. */
export function meanStd(a: ArrayLike<number>, from = 0, n = a.length - from, stride = 1): { mean: number; std: number } {
	let m = 0;
	for (let i = 0; i < n; i++) m += a[from + i * stride];
	m /= n || 1;
	let v = 0;
	for (let i = 0; i < n; i++) v += (a[from + i * stride] - m) ** 2;
	return { mean: m, std: Math.sqrt(v / (n || 1)) };
}

/** Counts of the values in `bins` equal bins over [lo, hi]; values outside go to the end bins. */
export function histogram(a: ArrayLike<number>, lo: number, hi: number, bins = BINS): Float64Array {
	const h = new Float64Array(bins);
	const w = (hi - lo) / bins || 1;
	for (let i = 0; i < a.length; i++) {
		const k = Math.floor((a[i] - lo) / w);
		h[k < 0 ? 0 : k >= bins ? bins - 1 : k]++;
	}
	return h;
}

/** BatchNorm of a [rows, cols] tensor (rows = examples), without its learned gain and shift: each column becomes
 *  (x − its mean) / sqrt(its variance + ε). */
export function batchNorm(x: ArrayLike<number>, rows: number, cols: number, eps = EPS): Float64Array {
	const y = new Float64Array(rows * cols);
	for (let j = 0; j < cols; j++) {
		const { mean, std } = meanStd(x, j, rows, cols);
		const d = Math.sqrt(std * std + eps);
		for (let i = 0; i < rows; i++) y[i * cols + j] = (x[i * cols + j] - mean) / d;
	}
	return y;
}

/** LayerNorm of a [rows, cols] tensor, without its learned gain and shift: each row becomes (x − its mean) /
 *  sqrt(its variance + ε). */
export function layerNorm(x: ArrayLike<number>, rows: number, cols: number, eps = EPS): Float64Array {
	const y = new Float64Array(rows * cols);
	for (let i = 0; i < rows; i++) {
		const { mean, std } = meanStd(x, i * cols, cols);
		const d = Math.sqrt(std * std + eps);
		for (let j = 0; j < cols; j++) y[i * cols + j] = (x[i * cols + j] - mean) / d;
	}
	return y;
}

/** σ as the narration prints it: as set (0.01, not 0.0100; 1; 0), else 3 significant digits (He's 0.177). */
export const stdText = (s: number): string => fmtSetting(s);

/** Narration helper: a frame's metrics. */
export const metricsOf = (at: { frame?: { metrics?: Record<string, number> } }): Record<string, number> => at.frame?.metrics ?? {};

/** The probe of the TinyGPT panel: TinyGPT's replay probe 0, "Once upon a time, there was a little", as its 9 token ids
 *  and their texts (static/models/tinygpt/tokenizer.json; the claim tests check both). */
export const PROBE_IDS = [428, 437, 258, 397, 44, 400, 282, 258, 390];
export const PROBE_TOKENS = ['Once', ' upon', ' a', ' time', ',', ' there', ' was', ' a', ' little'];
/** The tensor of the TinyGPT panel: the residual stream that enters block 2's LayerNorm (layer L1.ln1), i.e. the
 *  output of block 1 (L0.resid2). */
export const GPT_LAYER = 'L0.resid2';
export const GPT_LN = 'L1.ln1';
