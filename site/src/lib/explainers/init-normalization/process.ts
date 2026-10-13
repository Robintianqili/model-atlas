// init-normalization: the processes (kind C). Every number comes from one forward and one backward pass of the kit's
// autograd (net.ts) or from TinyGPT's own forward pass (the kit's executor on static/models/tinygpt); nothing is trained.
//   mlp     the toy network with one setup (σ, activation, normalization): frame 'forward' (the activations of every
//           hidden layer as histograms, their spreads, and the 4 × 6 corner of layer 1's weighted sums for the
//           BatchNorm / LayerNorm steps), then frame 'backward' (adds the gradient dL/dz that flows back into every
//           layer, as histograms and spreads, and how many weights get a gradient at all)
//   inits   the same network under the five named initializations (zero, too small, Xavier, He, too large) with one
//           activation and normalization: one frame 'compare' with the spreads of every layer for each
//   gpt     TinyGPT on its replay probe "Once upon a time, there was a little" (9 tokens): the residual stream that enters
//           block 2's LayerNorm, [9, 76], normalized per token (LayerNorm, as TinyGPT does) and per dimension over the
//           tokens (BatchNorm, as TinyGPT does not), and the largest gap between our LayerNorm (with TinyGPT's γ and β)
//           and TinyGPT's own layer L1.ln1. Its input is TinyGPT's spec and weights, copied from the trace source 'gpt'.
//   npm run atlas -- run init-normalization small --frames 0..1
import type { AtlasSpec, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { CORE_OPS } from '../../atlas/ops/index.ts';
import {
	BINS, DEPTH, EPS, GPT_LAYER, GPT_LN, INITS, INIT_STD, NB, NF, PROBE_IDS, SAT, WIDTH,
	actCode, batchNorm, histogram, layerNorm, meanStd, normCode, setupOf, type Setup
} from './data.ts';
import { deepRun, type DeepRun } from './net.ts';

const T = (shape: number[], data: ArrayLike<number>): Tensor => ({ shape, data: Float64Array.from(data) });

export type MlpState = { ha: Tensor; sa: Tensor; t: Tensor; hg: Tensor; sg: Tensor };
export type MlpOpt = Record<string, never>;

/** The row and column of the largest |value| of a [rows, cols] tensor (ties to the first). */
export function argmaxCell(a: ArrayLike<number>, cols: number): [number, number] {
	let k = 0;
	for (let i = 1; i < a.length; i++) if (Math.abs(a[i]) > Math.abs(a[k])) k = i;
	return [Math.floor(k / cols), k % cols];
}

/** The corner [NB, NF] of layer 1's weighted sums: the first NB examples and NF units. */
export function cornerOf(r: DeepRun): Float64Array {
	const t = new Float64Array(NB * NF);
	for (let i = 0; i < NB; i++) for (let j = 0; j < NF; j++) t[i * NF + j] = r.z[0][i * WIDTH + j];
	return t;
}

/** Everything the frames of one setup hold, from one forward and backward pass. */
export function mlpFrames(s: Setup, seed: number) {
	const r = deepRun(s, seed);
	const sa = r.h.map((a) => meanStd(a).std);
	const sg = r.gz.map((a) => meanStd(a).std);
	const aLo = s.act === 'tanh' ? -1 : 0;
	const aHi = s.act === 'tanh' ? 1 : 4 * Math.max(...sa) || 1;
	const gHi = 4 * Math.max(...sg) || 1;
	const ha = new Float64Array(DEPTH * BINS);
	const hg = new Float64Array(DEPTH * BINS);
	r.h.forEach((a, l) => ha.set(histogram(a, aLo, aHi), l * BINS));
	r.gz.forEach((a, l) => hg.set(histogram(a, -gHi, gHi), l * BINS));
	const last = r.h[DEPTH - 1];
	let sat = 0;
	for (let i = 0; i < last.length; i++) if (Math.abs(last[i]) >= SAT) sat++;
	let nz = 0;
	for (const g of r.gW) for (let i = 0; i < g.length; i++) if (g[i] !== 0) nz++;
	for (let i = 0; i < r.gOut.length; i++) if (r.gOut[i] !== 0) nz++;
	let nzb = r.gOutBias !== 0 ? 1 : 0;
	for (const g of r.gB) for (let i = 0; i < g.length; i++) if (g[i] !== 0) nzb++;
	const t = cornerOf(r);
	const [cr, cc] = argmaxCell(t, NF);
	const fwd: Record<string, number> = {
		std: s.std, act: actCode(s.act), norm: normCode(s.norm), s0: meanStd(r.x).std, zs1: meanStd(r.z[0]).std,
		sat: sat / last.length, aLo, aHi, cr, cc, f: (sa[DEPTH - 1] / sa[0]) ** (1 / (DEPTH - 1)), ratio: sa[DEPTH - 1] / sa[0]
	};
	sa.forEach((v, l) => (fwd[`s${l + 1}`] = v));
	const bwd: Record<string, number> = { ...fwd, gLo: -gHi, gHi, loss: r.loss, nz, nzb, gb: r.gOutBias, gf: (sg[0] / sg[DEPTH - 1]) ** (1 / (DEPTH - 1)) };
	sg.forEach((v, l) => (bwd[`g${l + 1}`] = v));
	return {
		forward: { state: { ha: T([DEPTH, BINS], ha), sa: T([DEPTH], sa), t: T([NB, NF], t) }, metrics: fwd },
		backward: { state: { ha: T([DEPTH, BINS], ha), sa: T([DEPTH], sa), t: T([NB, NF], t), hg: T([DEPTH, BINS], hg), sg: T([DEPTH], sg) }, metrics: bwd }
	};
}

const SCHEMA_MLP = {
	ha: { kind: 'tensor', label: { en: 'Activations of each hidden layer, as a histogram', zh: '每个隐藏层的激活值（直方图）' }, shape: [DEPTH, BINS], scale: 'sequential', role: 'activation', tol: 'exact' },
	sa: { kind: 'tensor', label: { en: 'Spread (standard deviation) of each layer\'s activations', zh: '每层激活值的离散程度（标准差）' }, shape: [DEPTH], role: 'activation', tol: 1e-9 },
	t: { kind: 'tensor', label: { en: 'Layer 1\'s weighted sums: 4 examples × 6 units', zh: '第 1 层的加权和：4 个样例 × 6 个单元' }, shape: [NB, NF], scale: 'diverging', role: 'activation', tol: 1e-9 },
	hg: { kind: 'tensor', label: { en: 'Gradient dL/dz reaching each layer, as a histogram', zh: '传回每层的梯度 dL/dz（直方图）' }, shape: [DEPTH, BINS], scale: 'sequential', role: 'grad', tol: 'exact' },
	sg: { kind: 'tensor', label: { en: 'Spread of the gradient reaching each layer', zh: '传回每层的梯度的离散程度' }, shape: [DEPTH], role: 'grad', tol: 1e-9 }
} as const;

export const mlp: ProcessDef<unknown, MlpOpt, MlpState> = {
	id: 'mlp',
	kind: 'inference',
	title: { en: 'A 10-layer network, forward and backward once', zh: '一个 10 层网络，前向和反向各一次' },
	phases: {
		forward: {
			label: { en: 'Forward: the activations', zh: '前向：激活值' },
			kind: 'forward',
			narration: { what: { en: 'The activations have spread {frame.s1} at layer 1 and {frame.s10} at layer 10.', zh: '激活值的标准差在第 1 层是 {frame.s1}，在第 10 层是 {frame.s10}。' } }
		},
		backward: {
			label: { en: 'Backward: the gradients', zh: '反向：梯度' },
			kind: 'backward',
			narration: { what: { en: 'The gradient has spread {frame.g10} at layer 10 and {frame.g1} at layer 1.', zh: '梯度的标准差在第 10 层是 {frame.g10}，在第 1 层是 {frame.g1}。' } }
		}
	},
	schema: SCHEMA_MLP as unknown as ProcessDef['schema'],
	defaults: {},
	*run(raw, ctx): ProcessRun<MlpState, MlpOpt> {
		const f = mlpFrames(setupOf(raw), ctx.rng.seed);
		yield { phase: 'forward', step: 0, state: f.forward.state, metrics: f.forward.metrics };
		yield { phase: 'backward', step: 0, state: f.backward.state, metrics: f.backward.metrics };
	}
};

export type InitsState = { sa: Tensor; sg: Tensor };

export const inits: ProcessDef<unknown, MlpOpt, InitsState> = {
	id: 'inits',
	kind: 'inference',
	title: { en: 'Five initializations of the same network', zh: '同一个网络的五种初始化' },
	phases: {
		compare: {
			label: { en: 'Five initializations', zh: '五种初始化' },
			kind: 'observe',
			narration: { what: { en: 'Layer 10\'s spread: {frame.small10} (too small), {frame.xavier10} (Xavier), {frame.he10} (He), {frame.large10} (too large).', zh: '第 10 层的标准差：{frame.small10}（过小）、{frame.xavier10}（Xavier）、{frame.he10}（He）、{frame.large10}（过大）。' } }
		}
	},
	schema: {
		sa: { kind: 'tensor', label: { en: 'Activation spread per layer, one row per initialization', zh: '每层激活值的标准差，每种初始化一行' }, shape: [INITS.length, DEPTH], role: 'activation', tol: 1e-9 },
		sg: { kind: 'tensor', label: { en: 'Gradient spread per layer, one row per initialization', zh: '每层梯度的标准差，每种初始化一行' }, shape: [INITS.length, DEPTH], role: 'grad', tol: 1e-9 }
	},
	defaults: {},
	*run(raw, ctx): ProcessRun<InitsState, MlpOpt> {
		const base = setupOf(raw);
		const sa = new Float64Array(INITS.length * DEPTH);
		const sg = new Float64Array(INITS.length * DEPTH);
		const m: Record<string, number> = { act: actCode(base.act), norm: normCode(base.norm) };
		INITS.forEach((id, k) => {
			const r = deepRun({ ...base, std: INIT_STD[id] }, ctx.rng.seed);
			r.h.forEach((a, l) => (sa[k * DEPTH + l] = meanStd(a).std));
			r.gz.forEach((a, l) => (sg[k * DEPTH + l] = meanStd(a).std));
			m[`${id}1`] = sa[k * DEPTH];
			m[`${id}10`] = sa[k * DEPTH + DEPTH - 1];
			m[`${id}G1`] = sg[k * DEPTH];
			m[`${id}G10`] = sg[k * DEPTH + DEPTH - 1];
			m[`${id}Std`] = INIT_STD[id];
		});
		yield { phase: 'compare', step: 0, state: { sa: T([INITS.length, DEPTH], sa), sg: T([INITS.length, DEPTH], sg) }, metrics: m };
	}
};

export type GptInput = { spec: AtlasSpec; weights: Record<string, Tensor> } | null;
export type GptState = { x: Tensor; ln: Tensor; bn: Tensor };

export const gpt: ProcessDef<GptInput, MlpOpt, GptState> = {
	id: 'gpt',
	kind: 'inference',
	title: { en: 'TinyGPT\'s residual stream, normalized two ways', zh: 'TinyGPT 的残差流，用两种方式归一化' },
	phases: {
		stream: {
			label: { en: 'Block 1\'s output, per token and per dimension', zh: '第 1 块的输出：按词元和按维度' },
			kind: 'forward',
			narration: { what: { en: 'Token {frame.row}\'s 76 numbers have mean {frame.rowMean} and spread {frame.rowStd}.', zh: '第 {frame.row} 个词元的 76 个数均值为 {frame.rowMean}，标准差为 {frame.rowStd}。' } }
		}
	},
	schema: {
		x: { kind: 'tensor', label: { en: 'Block 1\'s output (tokens × dimensions)', zh: '第 1 块的输出（词元 × 维度）' }, shape: [PROBE_IDS.length, 'd'], scale: 'diverging', role: 'activation', tol: 1e-5 },
		ln: { kind: 'tensor', label: { en: 'LayerNorm: each token\'s row to mean 0, spread 1', zh: 'LayerNorm：每个词元的一行变成均值 0、标准差 1' }, shape: [PROBE_IDS.length, 'd'], scale: 'diverging', role: 'activation', tol: 1e-5 },
		bn: { kind: 'tensor', label: { en: 'BatchNorm over the tokens: each dimension\'s column to mean 0, spread 1', zh: '在词元上做 BatchNorm：每个维度的一列变成均值 0、标准差 1' }, shape: [PROBE_IDS.length, 'd'], scale: 'diverging', role: 'activation', tol: 1e-5 }
	},
	defaults: {},
	metricTol: 1e-5,
	*run(input): ProcessRun<GptState, MlpOpt> {
		if (!input || !input.weights) return;
		const m = fromParts(input.spec, input.weights, CORE_OPS);
		const tr = m.forward({ shape: [PROBE_IDS.length], data: Int32Array.from(PROBE_IDS), dtype: 'i32' }, { record: 'layers' }).t;
		const xt = tr[GPT_LAYER];
		const [rows, cols] = xt.shape as [number, number];
		const x = Float64Array.from(xt.data as ArrayLike<number>);
		const ln = layerNorm(x, rows, cols, EPS);
		const bn = batchNorm(x, rows, cols, EPS);
		// our LayerNorm with TinyGPT's learned gain γ and shift β, against TinyGPT's own layer
		const g = input.weights[`${GPT_LN}.weight`].data;
		const b = input.weights[`${GPT_LN}.bias`].data;
		const own = tr[GPT_LN].data as ArrayLike<number>;
		let diff = 0;
		for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) diff = Math.max(diff, Math.abs(ln[i * cols + j] * g[j] + b[j] - own[i * cols + j]));
		// the token the lesson follows (the last) and the dimension whose mean over the tokens is largest in size
		const row = rows - 1;
		let col = 0;
		const colMeans = Array.from({ length: cols }, (_, j) => meanStd(x, j, rows, cols).mean);
		for (let j = 1; j < cols; j++) if (Math.abs(colMeans[j]) > Math.abs(colMeans[col])) col = j;
		const rs = meanStd(x, row * cols, cols);
		const cs = meanStd(x, col, rows, cols);
		yield {
			phase: 'stream',
			step: 0,
			state: { x: T([rows, cols], x), ln: T([rows, cols], ln), bn: T([rows, cols], bn) },
			metrics: {
				tokens: rows, dims: cols, row, col, rowMean: rs.mean, rowStd: rs.std, colMean: cs.mean, colStd: cs.std,
				x: x[row * cols + col], ln: ln[row * cols + col], bn: bn[row * cols + col], diff
			}
		};
	}
};
