// quantization: the processes (kind C).
//   ledger   the rounding ledger: eight numbers rounded to int8 or int4 with one scale per block (quant.ts ledger()).
//            Frames: 'scale' (the scale s and zero point z of block 0), one 'round' frame per value (step i, counting from
//            0: x / s + z, its code q, the value it stands for and the error) and 'done' (the errors, the bits per value
//            and, when opt.alt is set, the same numbers for a second setting, as alt* metrics). Its input is the eight
//            numbers, or the learner's raw input { x, vals }.
//   sweep    TinyGPT (static/models/tinygpt, the kit's own forward pass) with its weight matrices quantized: frame 'fp32'
//            (the model as trained), four 'layer' frames (only L0.mlp.fc.weight quantized: int8 per tensor, int4 per
//            tensor, per channel, per block of 32) and six 'all' frames (every linear weight matrix: int8 and int4, per
//            tensor, channel and block). Each frame scores the four evaluation windows of evalset.ts (256 next tokens):
//            perplexity, mean KL(fp32 ‖ quantized) per position, top-1 agreement with fp32, the largest logit change, and
//            the probe prompt's top tokens; plus the model's size in bytes. Its input is TinyGPT's spec and weights,
//            copied from the trace source 'gpt' (explainer.ts), so it runs in the package worker.
//   npm run atlas -- run quantization int4 --frames 0..9
import type { AtlasSpec, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { fromParts } from '../../atlas/graph/executor.ts';
import { CORE_OPS } from '../../atlas/ops/index.ts';
import { EVAL_IDS, PROBE_IDS } from './evalset.ts';
import { biggest, fp32Bits, groupCount, ledger as ledgerOf, medianAbs, N, quantizeMatrix, rawOf, type Gran, type LedgerOpt } from './quant.ts';

// ---------------------------------------------------------------- formats

export type FormatsState = { bits: Tensor };
/** One number in the formats of quant.ts formatsOf(): one frame; its metric x is the number (the views and the narration
 *  recompute the formats from it). Its input is x, or the learner's raw input. */
export const formats: ProcessDef<unknown, Record<string, never>, FormatsState> = {
	id: 'formats',
	kind: 'algorithm',
	title: { en: 'One number in five formats', zh: '一个数的五种格式' },
	phases: { formats: { label: { en: 'Store x', zh: '存下 x' }, kind: 'forward', narration: { what: { en: 'The number x = {frame.x}.', zh: '数 x = {frame.x}。' } } } },
	schema: { bits: { kind: 'tensor', label: { en: 'The fp32 bits of x (sign first)', zh: 'x 的 fp32 各位（符号位在前）' }, shape: [32], role: 'data', tol: 'exact' } },
	defaults: {},
	*run(input): ProcessRun<FormatsState, Record<string, never>> {
		const x = typeof input === 'number' ? rawOf({ x: input }).x : rawOf(input).x;
		const b = fp32Bits(Math.fround(x));
		const bits = Float64Array.from({ length: 32 }, (_, i) => (b >>> (31 - i)) & 1);
		yield { phase: 'formats', step: 0, state: { bits: { shape: [32], data: bits } }, metrics: { x, x32: Math.fround(x) } };
	}
};

// ---------------------------------------------------------------- ledger

export type LedgerProcOpt = LedgerOpt & { alt: Partial<LedgerOpt> | null };
export type LedgerState = { x: Tensor; q: Tensor; xh: Tensor; err: Tensor };

/** The eight numbers of a ledger source's input: an array, or the learner's raw input. */
export const valsOf = (input: unknown): number[] => (Array.isArray(input) ? rawOf({ vals: input }).vals : rawOf(input).vals);
const vec = (a: number[]): Tensor => ({ shape: [a.length], data: Float64Array.from(a) });
const fmtOpt = (o: LedgerOpt) => ({ bits: o.bits, levels: 2 ** o.bits, block: o.block, affine: o.scheme === 'affine' ? 1 : 0, clip: o.clip === 'second' ? 1 : 0 });

export const ledger: ProcessDef<unknown, LedgerProcOpt, LedgerState> = {
	id: 'ledger',
	kind: 'algorithm',
	title: { en: 'The rounding ledger', zh: '舍入账本' },
	phases: {
		scale: {
			label: { en: 'Pick the scale', zh: '确定缩放比例' },
			kind: 'input',
			formula: { tex: 's = \\frac{\\max |x|}{q_{\\max}} = \\frac{ {{amax}} }{ {{qmax}} } = {{s}}', values: (f) => ({ amax: f.metrics!.amax, qmax: f.metrics!.qmax, s: f.metrics!.s }) },
			narration: { what: { en: 'One scale s = {frame.s} for the block: the step between two neighbouring codes.', zh: '这一块共用一个缩放比例 s = {frame.s}：相邻两个编码之间的步长。' } }
		},
		round: {
			label: { en: 'Round one value', zh: '舍入一个数' },
			kind: 'forward',
			formula: {
				tex: '\\begin{aligned} q &= \\operatorname{round}(x / s + z) = \\operatorname{round}({{r}}) = {{q}} \\\\ \\hat{x} &= (q - z)\\, s = {{xh}}, \\quad \\hat{x} - x = {{err}} \\end{aligned}',
				values: (f) => ({ r: f.metrics!.r, q: f.metrics!.q, xh: f.metrics!.xh, err: f.metrics!.err })
			},
			narration: { what: { en: 'Value {frame.i} is {frame.x}: its code is {frame.q}, which stands for {frame.xh}.', zh: '第 {frame.i} 个值是 {frame.x}：它的编码是 {frame.q}，代表 {frame.xh}。' } }
		},
		done: {
			label: { en: 'The errors', zh: '误差' },
			kind: 'output',
			narration: { what: { en: 'The largest rounding error is {frame.maxErr}.', zh: '最大的舍入误差是 {frame.maxErr}。' } }
		}
	},
	schema: {
		x: { kind: 'tensor', label: { en: 'The numbers x', zh: '原始数值 x' }, shape: [N], role: 'data', tol: 1e-12 },
		q: { kind: 'tensor', label: { en: 'Their codes q', zh: '它们的编码 q' }, shape: [N], role: 'output', tol: 'exact' },
		xh: { kind: 'tensor', label: { en: 'What the codes stand for', zh: '编码代表的数值' }, shape: [N], role: 'output', tol: 1e-12 },
		err: { kind: 'tensor', label: { en: 'Rounding errors', zh: '舍入误差' }, shape: [N], role: 'output', tol: 1e-12 }
	},
	defaults: { bits: 8, scheme: 'absmax', clip: 'none', block: 8, alt: null },
	*run(input, ctx): ProcessRun<LedgerState, LedgerProcOpt> {
		const o = { ...ledger.defaults, ...ctx.opt };
		const vals = valsOf(input);
		const L = ledgerOf(vals, o);
		const state = { x: vec(L.x), q: vec(L.q), xh: vec(L.xh), err: vec(L.err) };
		const b0 = L.blocks[0];
		const b1 = L.blocks[1] ?? b0;
		const k = biggest(vals);
		yield {
			phase: 'scale',
			step: 0,
			state,
			metrics: { ...fmtOpt(L.opt), s: b0.s, z: b0.z, amax: Math.max(Math.abs(b0.lo), Math.abs(b0.hi)), lo: b0.lo, hi: b0.hi, qmax: b0.qhi, qlo: b0.qlo, nBlocks: L.blocks.length, s1: b1.s }
		};
		for (let i = 0; i < vals.length; i++) {
			const b = L.blocks[L.blockOf[i]];
			yield { phase: 'round', step: i, state, focus: [{ field: 'x', index: [i] }], metrics: { i, x: L.x[i], s: b.s, z: b.z, r: L.r[i], q: L.q[i], xh: L.xh[i], err: L.err[i] } };
		}
		const rest = L.err.filter((_, i) => i !== k).map(Math.abs);
		const m: Record<string, number> = {
			...fmtOpt(L.opt),
			maxErr: L.maxErr,
			meanErr: L.meanErr,
			zeros: L.zeros,
			nClipped: L.clipped.length,
			bitsPerValue: L.bitsPerValue,
			big: k,
			bigX: vals[k],
			bigErr: Math.abs(L.err[k]),
			restMean: rest.reduce((a, b) => a + b, 0) / rest.length,
			restMax: Math.max(...rest),
			med: medianAbs(vals),
			s: b0.s,
			s1: b1.s,
			nBlocks: L.blocks.length
		};
		if (o.alt) {
			const A = ledgerOf(vals, { ...o, ...o.alt });
			const ar = A.err.filter((_, i) => i !== k).map(Math.abs);
			Object.assign(m, {
				altMaxErr: A.maxErr,
				altMeanErr: A.meanErr,
				altZeros: A.zeros,
				altBits: A.bitsPerValue,
				altS: A.blocks[0].s,
				altBigErr: Math.abs(A.err[k]),
				altRestMean: ar.reduce((a, b) => a + b, 0) / ar.length,
				altRestMax: Math.max(...ar)
			});
		}
		yield { phase: 'done', step: 0, state, metrics: m };
	}
};

// ---------------------------------------------------------------- sweep

export type GptInput = { spec: AtlasSpec; weights: Record<string, Tensor> } | null;
export type SweepOpt = { focus: string };
export type SweepState = { top: Tensor; p0: Tensor; p: Tensor; l0: Tensor; l: Tensor };

/** The layer the lesson quantizes alone, and the configs of the sweep (in frame order). */
export const FOCUS = 'L0.mlp.fc.weight';
export const LAYER_CONFIGS: [number, Gran][] = [
	[8, 'tensor'],
	[4, 'tensor'],
	[4, 'channel'],
	[4, 'block']
];
export const ALL_CONFIGS: [number, Gran][] = [
	[8, 'tensor'],
	[8, 'channel'],
	[8, 'block'],
	[4, 'tensor'],
	[4, 'channel'],
	[4, 'block']
];
export const GRAN_CODE: Record<Gran, number> = { tensor: 0, channel: 1, block: 2 };
export const TOPK = 5;

/** The linear layers' weight matrices (24 in TinyGPT: q, k, v, proj, fc and proj of each block), in the weights' order. */
export const isMatrix = (name: string) => /^L\d+\.(attn\.(q|k|v|proj)|mlp\.(fc|proj))\.weight$/.test(name);
export const matrixNames = (weights: Record<string, Tensor>) => Object.keys(weights).filter(isMatrix);

/** Log-softmax of one row of logits, in float64. */
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

export const sweep: ProcessDef<GptInput, SweepOpt, SweepState> = {
	id: 'sweep',
	kind: 'inference',
	title: { en: 'TinyGPT with quantized weights', zh: '权重量化后的 TinyGPT' },
	phases: {
		fp32: { label: { en: 'The model as trained (fp32)', zh: '训练好的原模型（fp32）' }, kind: 'forward', narration: { what: { en: 'In fp32, TinyGPT scores a perplexity of {frame.ppl} on the evaluation text.', zh: '用 fp32 时，TinyGPT 在评测文本上的困惑度是 {frame.ppl}。' } } },
		layer: { label: { en: 'One layer quantized', zh: '只量化一层' }, kind: 'forward', narration: { what: { en: 'With one layer at int{frame.bits}, the perplexity is {frame.ppl}.', zh: '一层用 int{frame.bits} 时，困惑度是 {frame.ppl}。' } } },
		all: { label: { en: 'Every weight matrix quantized', zh: '量化所有权重矩阵' }, kind: 'forward', narration: { what: { en: 'With every weight matrix at int{frame.bits}, the perplexity is {frame.ppl}.', zh: '所有权重矩阵都用 int{frame.bits} 时，困惑度是 {frame.ppl}。' } } }
	},
	schema: {
		top: { kind: 'tensor', label: { en: 'The probe’s top tokens in fp32 (ids)', zh: '探针句在 fp32 下的前几名词元（编号）' }, shape: [TOPK], role: 'output', tol: 'exact' },
		p0: { kind: 'tensor', label: { en: 'Their probabilities in fp32', zh: '它们在 fp32 下的概率' }, shape: [TOPK], scale: 'sequential', role: 'output', tol: 1e-4 },
		p: { kind: 'tensor', label: { en: 'Their probabilities, quantized', zh: '它们在量化后的概率' }, shape: [TOPK], scale: 'sequential', role: 'output', tol: 1e-4 },
		l0: { kind: 'tensor', label: { en: 'Their logits in fp32', zh: '它们在 fp32 下的 logit' }, shape: [TOPK], role: 'output', tol: 1e-4 },
		l: { kind: 'tensor', label: { en: 'Their logits, quantized', zh: '它们在量化后的 logit' }, shape: [TOPK], role: 'output', tol: 1e-4 }
	},
	defaults: { focus: FOCUS },
	metricTol: { ppl: 1e-4, ppl0: 1e-4, kl: 2e-3, agree: 0.02, dmax: 1e-3, probeP: 1e-4, probeP0: 1e-4, probePq: 1e-4, wRms: 1e-5, wMax: 1e-5 },
	*run(input, ctx): ProcessRun<SweepState, SweepOpt> {
		if (!input || !input.weights) return;
		const { spec, weights } = input;
		const focus = ctx.opt.focus ?? FOCUS;
		const names = matrixNames(weights);
		const params = Object.values(weights).reduce((a, t) => a + t.data.length, 0);
		const base = logitsOf(spec, weights);
		const lp0 = base.rows.map((w) => w.map(logSoftmax));
		const probeLp0 = logSoftmax(base.probe);
		const top = Array.from(probeLp0).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, TOPK).map((x) => x[1]);
		const top0 = top[0];
		const tokens = EVAL_IDS.reduce((a, w) => a + w.length - 1, 0);
		const pick = (lp: Float64Array[][]) => {
			let nll = 0;
			EVAL_IDS.forEach((w, k) => lp[k].forEach((row, t) => (nll -= row[w[t + 1]])));
			return Math.exp(nll / tokens);
		};
		const ppl0 = pick(lp0);
		const fw = weights[focus];
		const [rows, cols] = fw.shape;

		const frameOf = (lp: Float64Array[][], logits: Float32Array[][], probe: Float32Array, w: Record<string, Tensor>, bits: number, gran: number, nq: number, groups: number) => {
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
					const a = logits[k][t];
					const b = base.rows[k][t];
					for (let j = 0; j < a.length; j++) dmax = Math.max(dmax, Math.abs(a[j] - b[j]));
				})
			);
			const plp = logSoftmax(probe);
			const q = w[focus].data;
			let se = 0;
			let me = 0;
			for (let i = 0; i < q.length; i++) {
				const e = q[i] - fw.data[i];
				se += e * e;
				me = Math.max(me, Math.abs(e));
			}
			const bytes = 4 * (params - nq) + (nq * bits) / 8 + 4 * groups;
			return {
				state: {
					top: { shape: [TOPK], data: Float64Array.from(top) },
					p0: { shape: [TOPK], data: Float64Array.from(top.map((i) => Math.exp(probeLp0[i]))) },
					p: { shape: [TOPK], data: Float64Array.from(top.map((i) => Math.exp(plp[i]))) },
					l0: { shape: [TOPK], data: Float64Array.from(top.map((i) => base.probe[i])) },
					l: { shape: [TOPK], data: Float64Array.from(top.map((i) => probe[i])) }
				},
				metrics: {
					bits,
					gran,
					ppl0,
					ppl: pick(lp),
					kl: kl / tokens,
					agree: agree / tokens,
					dmax,
					tokens,
					probeTop: argmax(probe),
					probeTop0: top0,
					probeP: Math.exp(plp[top0]),
					probePq: Math.exp(plp[argmax(probe)]),
					probeP0: Math.exp(probeLp0[top0]),
					bytes,
					bytes0: 4 * params,
					params,
					nq,
					groups,
					wRms: Math.sqrt(se / q.length),
					wMax: me,
					fRows: rows,
					fCols: cols
				}
			};
		};
		const f0 = frameOf(lp0, base.rows, base.probe, weights, 32, -1, 0, 0);
		yield { phase: 'fp32', step: 0, state: f0.state, metrics: { ...f0.metrics, bytes: 4 * params } };

		const runWith = (targets: string[], bits: number, gran: Gran) => {
			const w = { ...weights };
			let groups = 0;
			let nq = 0;
			for (const name of targets) {
				const t = weights[name];
				w[name] = { shape: t.shape.slice(), data: quantizeMatrix(t.data, t.shape[0], t.shape[1], bits, gran) };
				groups += groupCount(t.shape[0], t.shape[1], gran);
				nq += t.data.length;
			}
			const out = logitsOf(spec, w);
			return frameOf(out.rows.map((win) => win.map(logSoftmax)), out.rows, out.probe, w, bits, GRAN_CODE[gran], nq, groups);
		};
		for (let k = 0; k < LAYER_CONFIGS.length; k++) {
			const [bits, gran] = LAYER_CONFIGS[k];
			const f = runWith([focus], bits, gran);
			yield { phase: 'layer', step: k, state: f.state, metrics: f.metrics };
		}
		for (let k = 0; k < ALL_CONFIGS.length; k++) {
			const [bits, gran] = ALL_CONFIGS[k];
			const f = runWith(names, bits, gran);
			yield { phase: 'all', step: k, state: f.state, metrics: f.metrics };
		}
	}
};

