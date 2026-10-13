// number-formats: the processes (kind C).
//   formats  one number in every format of fmt.ts: one frame; state 'codes' holds each format's code (its bit pattern as
//            an unsigned number, FMT_IDS order), metrics x and x32 (x rounded to fp32). The views and the narration
//            decode the codes. Its input is x, or the learner's raw input { x }.
//   sweep    TinyGPT (static/models/tinygpt, the kit's own forward pass) with its 24 linear weight matrices cast to a
//            format (CASTS, fmt.ts castArray): frame 'fp32' (as trained, with the fate of its weights in each float
//            format and a histogram of log2 |w|) and one 'cast' frame per setting. Each frame scores the four evaluation
//            windows of evalset.ts (256 next tokens): perplexity, mean KL(fp32 ‖ cast) per position, top-1 agreement,
//            the largest logit change and the probe's top tokens; plus the size of the weights in bytes.
//   sum      mixed precision: a weight w0 gets n updates of size u, stored and added in fp32, bf16 and fp16 (each sum is
//            computed in fp32 and rounded to the format), and a bf16 copy of an fp32 master weight. Frame k: after k updates;
//            metrics d32, dbf, d16, dmix: how far each has moved from w0 (the chart of step "mixed").
//   npm run atlas -- run number-formats sum --frames 0..5
import type { AtlasSpec, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { EVAL_IDS, PROBE_IDS } from './evalset.ts';
import { cast, castArray, encode, fateOf, FLOAT_IDS, FMT_IDS, FMTS, nbits, rawOf, type CastCfg } from './fmt.ts';

// ---------------------------------------------------------------- formats

export type FormatsState = { codes: Tensor };
export const formats: ProcessDef<unknown, Record<string, never>, FormatsState> = {
	id: 'formats',
	kind: 'algorithm',
	title: { en: 'One number in nine formats', zh: '一个数的九种格式' },
	phases: { store: { label: { en: 'Store x', zh: '存下 x' }, kind: 'forward', narration: { what: { en: 'The number x = {frame.x}.', zh: '数 x = {frame.x}。' } } } },
	schema: { codes: { kind: 'tensor', label: { en: 'The code of x in each format (its bits as a number)', zh: 'x 在每种格式里的编码（各位组成的数）' }, shape: [FMT_IDS.length], role: 'output', tol: 'exact' } },
	defaults: {},
	*run(input): ProcessRun<FormatsState, Record<string, never>> {
		const x = rawOf(input).x;
		const codes = Float64Array.from(FMT_IDS.map((id) => encode(id, x)));
		yield { phase: 'store', step: 0, state: { codes: { shape: [codes.length], data: codes } }, metrics: { x, x32: Math.fround(x) } };
	}
};

// ---------------------------------------------------------------- sweep

export type GptInput = { spec: AtlasSpec; weights: Record<string, Tensor> } | null;
export type SweepState = { top: Tensor; p0: Tensor; p: Tensor; hist: Tensor };

/** The settings of the sweep, in frame order after 'fp32'. */
export const CASTS: CastCfg[] = [
	{ id: 'bf16', fmt: 'bf16', scale: 'none' },
	{ id: 'fp16', fmt: 'fp16', scale: 'none' },
	{ id: 'e5m2', fmt: 'e5m2', scale: 'none' },
	{ id: 'e4m3', fmt: 'e4m3', scale: 'none' },
	{ id: 'e4m3-t', fmt: 'e4m3', scale: 'tensor' },
	{ id: 'fp4', fmt: 'fp4', scale: 'none' },
	{ id: 'fp4-b', fmt: 'fp4', scale: 'block', block: 32 },
	{ id: 'int4-b', fmt: 'int4', scale: 'block', block: 64 },
	{ id: 'nf4-b', fmt: 'nf4', scale: 'block', block: 64 }
];
export const castIndex = (id: string) => CASTS.findIndex((c) => c.id === id);
export const TOPK = 5;
/** The histogram of log2 |w|: one bin per power of 2 from 2^HIST_LO up to 2^HIST_HI. */
export const HIST_LO = -26;
export const HIST_HI = 18;

/** The linear layers' weight matrices (24 in TinyGPT: q, k, v, proj, fc and proj of each block), in the weights' order. */
export const isMatrix = (name: string) => /^L\d+\.(attn\.(q|k|v|proj)|mlp\.(fc|proj))\.weight$/.test(name);
export const matrixNames = (weights: Record<string, Tensor>) => Object.keys(weights).filter(isMatrix);

function logSoftmax(row: ArrayLike<number>): Float64Array {
	let m = -Infinity;
	for (let i = 0; i < row.length; i++) m = Math.max(m, row[i]);
	let s = 0;
	for (let i = 0; i < row.length; i++) s += Math.exp(row[i] - m);
	const lse = m + Math.log(s);
	const out = new Float64Array(row.length);
	for (let i = 0; i < row.length; i++) out[i] = row[i] - lse;
	return out;
}
const argmax = (a: ArrayLike<number>) => {
	let k = 0;
	for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i;
	return k;
};

/** The logits of every window (inputs = the first 64 ids) and of the probe, as rows of [T, V]. */
function logitsOf(spec: AtlasSpec, weights: Record<string, Tensor>): { rows: Float32Array[][]; probe: Float32Array } {
	const m = fromParts(spec, weights, CORE_OPS);
	const run = (ids: number[]) => {
		const t = m.forward({ shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' }, { record: 'layers' }).t.logits;
		const V = t.shape[1];
		return Array.from({ length: t.shape[0] }, (_, i) => Float32Array.from((t.data as Float32Array).subarray(i * V, (i + 1) * V)));
	};
	const rows = EVAL_IDS.map((w) => run(w.slice(0, -1)));
	const p = run(PROBE_IDS);
	return { rows, probe: p[p.length - 1] };
}

/** Bits per weight of a setting: the code, plus one fp32 scale per group (per tensor: tiny, but counted). */
export function bitsPerWeight(c: CastCfg, n: number, groups: number): number {
	return nbits(FMTS[c.fmt]) + (c.scale === 'none' ? 0 : (32 * groups) / n);
}

export const sweep: ProcessDef<GptInput, Record<string, never>, SweepState> = {
	id: 'sweep',
	kind: 'inference',
	title: { en: 'TinyGPT with its weights in other formats', zh: '权重换成其他格式的 TinyGPT' },
	phases: {
		fp32: { label: { en: 'The model as trained (fp32)', zh: '训练好的原模型（fp32）' }, kind: 'forward', narration: { what: { en: 'In fp32, TinyGPT scores a perplexity of {frame.ppl} on the evaluation text.', zh: '用 fp32 时，TinyGPT 在评测文本上的困惑度是 {frame.ppl}。' } } },
		cast: { label: { en: 'Weights cast to a format', zh: '权重换成一种格式' }, kind: 'forward', narration: { what: { en: 'With the weight matrices cast, the perplexity is {frame.ppl}.', zh: '权重矩阵换成这种格式后，困惑度是 {frame.ppl}。' } } }
	},
	schema: {
		top: { kind: 'tensor', label: { en: 'The probe’s top tokens in fp32 (ids)', zh: '探针句在 fp32 下的前几名词元（编号）' }, shape: [TOPK], role: 'output', tol: 'exact' },
		p0: { kind: 'tensor', label: { en: 'Their probabilities in fp32', zh: '它们在 fp32 下的概率' }, shape: [TOPK], scale: 'sequential', role: 'output', tol: 1e-4 },
		p: { kind: 'tensor', label: { en: 'Their probabilities after the cast', zh: '它们在换格式后的概率' }, shape: [TOPK], scale: 'sequential', role: 'output', tol: 1e-4 },
		hist: { kind: 'tensor', label: { en: 'Weights per power of 2 of their size', zh: '按大小（2 的幂）统计的权重个数' }, shape: [HIST_HI - HIST_LO], role: 'data', tol: 'exact' }
	},
	defaults: {},
	metricTol: { ppl: 1e-4, ppl0: 1e-4, kl: 2e-3, agree: 0.02, dmax: 1e-3, probeP: 1e-4, probeP0: 1e-4, wRel: 1e-5 },
	*run(input): ProcessRun<SweepState, Record<string, never>> {
		if (!input || !input.weights) return;
		const { spec, weights } = input;
		const names = matrixNames(weights);
		const n = names.reduce((a, k) => a + weights[k].data.length, 0);
		const params = Object.values(weights).reduce((a, t) => a + t.data.length, 0);
		const base = logitsOf(spec, weights);
		const lp0 = base.rows.map((w) => w.map(logSoftmax));
		const probeLp0 = logSoftmax(base.probe);
		const top = Array.from(probeLp0)
			.map((v, i) => [v, i])
			.sort((a, b) => b[0] - a[0] || a[1] - b[1])
			.slice(0, TOPK)
			.map((x) => x[1]);
		const tokens = EVAL_IDS.reduce((a, w) => a + w.length - 1, 0);
		const pick = (lp: Float64Array[][]) => {
			let nll = 0;
			EVAL_IDS.forEach((w, k) => lp[k].forEach((row, t) => (nll -= row[w[t + 1]])));
			return Math.exp(nll / tokens);
		};
		const ppl0 = pick(lp0);

		// the matrices' weights in one array, their histogram of log2 |w| and their fate in each float format
		const all = new Float32Array(n);
		let o = 0;
		for (const k of names) {
			all.set(weights[k].data as Float32Array, o);
			o += weights[k].data.length;
		}
		const hist = new Float64Array(HIST_HI - HIST_LO);
		const sorted = Float64Array.from(all, Math.abs).sort();
		for (let i = 0; i < n; i++) {
			const a = Math.abs(all[i]);
			if (a > 0) hist[Math.min(hist.length - 1, Math.max(0, Math.floor(Math.log2(a)) - HIST_LO))]++;
		}
		const stats = { n, nm: names.length, params, amax: sorted[n - 1], wMed: (sorted[(n - 1) >> 1] + sorted[n >> 1]) / 2, wMin: sorted.find((a) => a > 0) ?? 0 };
		const fates: Record<string, number> = {};
		for (const id of FLOAT_IDS) {
			const f = fateOf(all, id);
			Object.assign(fates, { [`sub_${id}`]: f.sub, [`zero_${id}`]: f.zero, [`over_${id}`]: f.over });
		}

		const frameOf = (w: Record<string, Tensor>, out: { rows: Float32Array[][]; probe: Float32Array }) => {
			const lp = out.rows.map((win) => win.map(logSoftmax));
			let kl = 0;
			let agree = 0;
			let dmax = 0;
			lp.forEach((win, k) =>
				win.forEach((row, t) => {
					const r0 = lp0[k][t];
					let s = 0;
					for (let j = 0; j < row.length; j++) s += Math.exp(r0[j]) * (r0[j] - row[j]);
					kl += s;
					if (argmax(row) === argmax(r0)) agree++;
					const a = out.rows[k][t];
					const b = base.rows[k][t];
					for (let j = 0; j < a.length; j++) dmax = Math.max(dmax, Math.abs(a[j] - b[j]));
				})
			);
			let se = 0;
			let ss = 0;
			let zeros = 0;
			for (const k of names) {
				const q = w[k].data;
				const r = weights[k].data;
				for (let i = 0; i < q.length; i++) {
					se += (q[i] - r[i]) ** 2;
					ss += r[i] ** 2;
					if (q[i] === 0 && r[i] !== 0) zeros++;
				}
			}
			const plp = logSoftmax(out.probe);
			return {
				state: {
					top: { shape: [TOPK], data: Float64Array.from(top) },
					p0: { shape: [TOPK], data: Float64Array.from(top.map((i) => Math.exp(probeLp0[i]))) },
					p: { shape: [TOPK], data: Float64Array.from(top.map((i) => Math.exp(plp[i]))) },
					hist: { shape: [hist.length], data: hist }
				},
				metrics: {
					...stats,
					ppl0,
					ppl: pick(lp),
					kl: kl / tokens,
					agree: agree / tokens,
					dmax,
					tokens,
					probeTop: argmax(out.probe),
					probeTop0: top[0],
					probeP: Math.exp(plp[top[0]]),
					probeP0: Math.exp(probeLp0[top[0]]),
					wRel: Math.sqrt(se / ss),
					zeroShare: zeros / n
				}
			};
		};
		const f0 = frameOf(weights, base);
		yield { phase: 'fp32', step: 0, state: f0.state, metrics: { ...f0.metrics, ...fates, k: -1, bits: 32, bpw: 32, bytes: 4 * params, bytes0: 4 * params } };

		for (let k = 0; k < CASTS.length; k++) {
			const c = CASTS[k];
			const w = { ...weights };
			let groups = 0;
			for (const name of names) {
				const t = weights[name];
				const cols = t.shape[1];
				w[name] = { shape: t.shape.slice(), data: castArray(t.data, c, cols) };
				groups += c.scale === 'tensor' ? 1 : c.scale === 'block' ? t.shape[0] * Math.ceil(cols / (c.block ?? 32)) : 0;
			}
			const f = frameOf(w, logitsOf(spec, w));
			const bpw = bitsPerWeight(c, n, groups);
			yield { phase: 'cast', step: k, state: f.state, metrics: { ...f.metrics, k, bits: nbits(FMTS[c.fmt]), bpw, bytes: 4 * (params - n) + (n * bpw) / 8, bytes0: 4 * params } };
		}
	}
};

// ---------------------------------------------------------------- sum

export type SumOpt = { n: number; u: number; w0: number };
export type SumState = { w: Tensor };

export const sum: ProcessDef<unknown, SumOpt, SumState> = {
	id: 'sum',
	kind: 'algorithm',
	title: { en: 'Small updates to one weight', zh: '对一个权重做许多次小更新' },
	phases: {
		update: {
			label: { en: 'Add one update', zh: '加一次更新' },
			kind: 'update',
			narration: { what: { en: 'After {frame.step} updates: fp32 {frame.fp32}, bf16 {frame.bf16}.', zh: '更新 {frame.step} 次之后：fp32 为 {frame.fp32}，bf16 为 {frame.bf16}。' } }
		}
	},
	schema: { w: { kind: 'tensor', label: { en: 'The weight in fp32, bf16, fp16, and the bf16 copy of an fp32 master', zh: '权重在 fp32、bf16、fp16 下的值，以及 fp32 主权重的 bf16 副本' }, shape: [4], role: 'output', tol: 'exact' } },
	defaults: { n: 100, u: 0.001, w0: 1 },
	*run(_input, ctx): ProcessRun<SumState, SumOpt> {
		const o = { ...sum.defaults, ...ctx.opt };
		const add = (id: 'fp32' | 'bf16' | 'fp16', w: number) => cast(id, Math.fround(w + cast(id, o.u)));
		let [w32, wbf, w16] = [cast('fp32', o.w0), cast('bf16', o.w0), cast('fp16', o.w0)];
		for (let k = 0; k <= o.n; k++) {
			if (k > 0) {
				w32 = add('fp32', w32);
				wbf = add('bf16', wbf);
				w16 = add('fp16', w16);
			}
			const mixed = cast('bf16', w32); // the bf16 copy of the fp32 master weight
			yield { phase: 'update', step: k, state: { w: { shape: [4], data: Float64Array.of(w32, wbf, w16, mixed) } }, metrics: { step: k, fp32: w32, bf16: wbf, fp16: w16, mixed, exact: o.w0 + k * o.u, d32: w32 - o.w0, dbf: wbf - o.w0, d16: w16 - o.w0, dmix: mixed - o.w0 } };
		}
	}
};
