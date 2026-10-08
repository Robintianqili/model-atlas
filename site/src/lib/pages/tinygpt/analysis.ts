/**
 * Small pure helpers for the TinyGPT page: head summaries, the sampling pick, parameter counts.
 */
import type { ForwardResult, TinyGPTSpec } from '../../models/tinygpt';

export type HeadMetric = 'prev' | 'first' | 'self';

/**
 * Per layer × head averages over query positions i ≥ 1 of the attention weight on
 * the previous token (i−1), on the first token (0) and on itself (i). NaN when T = 1.
 * Arrays are [L × H], index l * H + h.
 */
export function headMetrics(f: ForwardResult, L: number, H: number): Record<HeadMetric, Float32Array> {
	const T = f.ids.length;
	const prev = new Float32Array(L * H);
	const first = new Float32Array(L * H);
	const self = new Float32Array(L * H);
	for (let l = 0; l < L; l++) {
		const w = f.t[`L${l}.attn.weights`].data;
		for (let h = 0; h < H; h++) {
			let sp = 0;
			let sf = 0;
			let ss = 0;
			for (let i = 1; i < T; i++) {
				const o = (h * T + i) * T;
				sp += w[o + i - 1];
				sf += w[o];
				ss += w[o + i];
			}
			const n = T - 1;
			prev[l * H + h] = n ? sp / n : NaN;
			first[l * H + h] = n ? sf / n : NaN;
			self[l * H + h] = n ? ss / n : NaN;
		}
	}
	return { prev, first, self };
}

/**
 * The sampling step of tinygpt.ts generate(): walk the distribution in token-id order and stop
 * when the running sum passes u. Returns the token and the interval [lo, hi) of u that picks it.
 */
export function pickToken(dist: ArrayLike<number>, u: number): { token: number; lo: number; hi: number } {
	let token = -1;
	let cum = 0;
	let lo = 0;
	for (let i = 0; i < dist.length; i++) {
		if (dist[i] === 0) continue;
		lo = cum;
		cum += dist[i];
		token = i;
		if (cum > u) break;
	}
	return { token, lo, hi: cum };
}

/** What the last "Generate" click did (shown under the buttons). */
export type Pick = {
	token: number;
	/** the random number used; null for greedy */
	u: number | null;
	lo: number;
	hi: number;
	/** probability under the sampling distribution, and under the plain model */
	p: number;
	pModel: number;
	/** which draw of the seeded generator (1 = first) */
	draw: number;
	seed: number;
};

export function argmax(a: ArrayLike<number>): number {
	let b = 0;
	for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i;
	return b;
}

/** Indices of the k largest values (ties: smaller index first). */
export function topIndices(a: ArrayLike<number>, k: number): number[] {
	const idx = Array.from({ length: a.length }, (_, i) => i);
	idx.sort((x, y) => a[y] - a[x] || x - y);
	return idx.slice(0, k);
}

/** Entropy in nats of a probability vector. */
export function entropy(p: ArrayLike<number>): number {
	let h = 0;
	for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log(p[i]);
	return h;
}

/** Row t of a [T, n] tensor (no copy). */
export function row(data: Float32Array, t: number, n: number): Float32Array {
	return data.subarray(t * n, (t + 1) * n);
}

/** Head h's [T, dHead] slice of a [T, D] tensor (copied). */
export function headSlice(data: Float32Array, T: number, D: number, h: number, dh: number): Float32Array {
	const out = new Float32Array(T * dh);
	for (let t = 0; t < T; t++) for (let c = 0; c < dh; c++) out[t * dh + c] = data[t * D + h * dh + c];
	return out;
}

/** Head h's [T, T] matrix of a [H, T, T] tensor (no copy). */
export function headMatrix(data: Float32Array, T: number, h: number): Float32Array {
	return data.subarray(h * T * T, (h + 1) * T * T);
}

export function mean(a: ArrayLike<number>): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i];
	return a.length ? s / a.length : 0;
}

/** Standard deviation (biased, as LayerNorm uses). */
export function std(a: ArrayLike<number>): number {
	const m = mean(a);
	let v = 0;
	for (let i = 0; i < a.length; i++) v += (a[i] - m) ** 2;
	return a.length ? Math.sqrt(v / a.length) : 0;
}

export function norm(a: ArrayLike<number>): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * a[i];
	return Math.sqrt(s);
}

export type ParamRow = {
	key: string;
	label: { en: string; zh: string };
	formula: string;
	/** counted from the real tensor table of spec.json */
	count: number;
	/** how many times it occurs (blocks) */
	times: number;
};

/** Parameter breakdown counted from spec.tensors (the real weights), with the formula for each group. */
export function paramBreakdown(spec: TinyGPTSpec): { rows: ParamRow[]; total: number; formulaTotal: number } {
	const d = spec.dModel;
	const V = spec.vocabSize;
	const C = spec.context;
	const L = spec.nLayer;
	const len = (re: RegExp) => spec.tensors.filter((t) => re.test(t.name)).reduce((a, t) => a + t.length, 0);
	const rows: ParamRow[] = [
		{ key: 'tok', label: { en: 'Token embedding E (also the output head, tied)', zh: '词元嵌入 E（同时作为输出头，权重共享）' }, formula: `V·d = ${V}·${d}`, count: len(/^tok_emb\./), times: 1 },
		{ key: 'pos', label: { en: 'Position embedding', zh: '位置嵌入' }, formula: `C·d = ${C}·${d}`, count: len(/^pos_emb\./), times: 1 },
		{ key: 'ln1', label: { en: 'LayerNorm 1 (γ, β)', zh: '层归一化 1（γ, β）' }, formula: `2d`, count: len(/^L0\.ln1\./), times: L },
		{ key: 'qkv', label: { en: 'Q, K, V projections (W + b)', zh: 'Q、K、V 投影（W + b）' }, formula: `3(d² + d)`, count: len(/^L0\.attn\.[qkv]\./), times: L },
		{ key: 'proj', label: { en: 'Attention output projection', zh: '注意力输出投影' }, formula: `d² + d`, count: len(/^L0\.attn\.proj\./), times: L },
		{ key: 'ln2', label: { en: 'LayerNorm 2 (γ, β)', zh: '层归一化 2（γ, β）' }, formula: `2d`, count: len(/^L0\.ln2\./), times: L },
		{ key: 'fc', label: { en: 'MLP expand 76 → 304', zh: 'MLP 扩展 76 → 304' }, formula: `d·4d + 4d`, count: len(/^L0\.mlp\.fc\./), times: L },
		{ key: 'mproj', label: { en: 'MLP project 304 → 76', zh: 'MLP 投影 304 → 76' }, formula: `4d·d + d`, count: len(/^L0\.mlp\.proj\./), times: L },
		{ key: 'lnf', label: { en: 'Final LayerNorm (γ, β)', zh: '最终层归一化（γ, β）' }, formula: `2d`, count: len(/^ln_f\./), times: 1 },
		{ key: 'head', label: { en: 'Output head (tied to E: no extra numbers)', zh: '输出头（与 E 共享：不增加参数）' }, formula: `0`, count: 0, times: 1 }
	];
	const total = spec.tensors.reduce((a, t) => a + t.length, 0);
	const formulaTotal = V * d + C * d + L * (12 * d * d + 13 * d) + 2 * d;
	return { rows, total, formulaTotal };
}

/** GELU (tanh approximation), as in the model. */
export function gelu(x: number): number {
	return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}
