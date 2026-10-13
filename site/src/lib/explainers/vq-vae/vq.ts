// vq-vae helpers (pure TS, shared by the views, the lesson and the claim tests):
//   the learner's raw input may carry two extras besides the drawing (drawing28's { strokes } / { b64 } / { ink }):
//     cell   the patch the learner outlined (0 … 15, row by row; default: the patch with the most ink)
//     swap   { cell, code }: the code the learner put into one cell (the swap step)
//   structure read from the spec (dims), the trace of the learner's drawing (z_e, distances, code indices, redrawing),
//   the decoder alone on any grid of codes (decodeZq / decodeIdx: the spec's layers from dec1 on, the same weights and
//   ops as the trace), what each code draws among blank neighbours (tilesOf), and the decoder's gradient at z_q (gradAt).
import type { ExplainerRuntime, GraphModel, Tensor } from '../../atlas/types/index.ts';
import { fromParts } from '../../atlas/graph/executor.ts';

type RT = ExplainerRuntime;
export interface Extra { cell?: number; swap?: { cell: number; code: number } }
export interface Usage { counts: number[]; patches: number; blank: number }

export const extraOf = (raw: unknown): Extra => (raw && typeof raw === 'object' && !Array.isArray(raw) && !ArrayBuffer.isView(raw) ? (raw as Extra) : {});

/** the learner's raw input with some extras changed (the drawing itself is kept as it is) */
export function withExtra(raw: unknown, extra: Extra): Record<string, unknown> {
	if (raw === null || raw === undefined) return { strokes: [], ...extra };
	if (ArrayBuffer.isView(raw) || (Array.isArray(raw) && typeof raw[0] === 'number')) return { ink: raw, ...extra };
	if (Array.isArray(raw)) return { strokes: raw, ...extra };
	return { ...(raw as Record<string, unknown>), ...extra };
}

export const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
export const usageOf = (rt: RT) => (rt as RT & { assets?: Record<string, unknown> }).assets?.['usage.json'] as Usage | undefined;

/** structure, read from the spec (never typed into the text): codes K, numbers per code D, grid side G, patch side P */
export function dims(rt: RT) {
	const L = (id: string) => rt.spec?.layers.find((l) => l.id === id);
	const cb = rt.spec?.tensors.find((x) => x.name === 'zq.codebook')?.shape ?? [NaN, NaN];
	const K = Number(cb[0]), D = Number(cb[1]), G = Number(L('ze')?.outShape[1] ?? NaN), P = Number((L('img')?.params as { factor?: number } | undefined)?.factor ?? NaN);
	return { K, D, G, P, cells: G * G, h: Number(L('enc1')?.outShape[0] ?? NaN), n: (G * P) ** 2, block: P * P, bits: G * G * Math.log2(K), bitsPer: Math.log2(K) };
}

/** the code index of every cell, row by row */
export const codesOf = (rt: RT): number[] => Array.from(T(rt, 'zq#idx')?.data ?? []);
/** ink of each patch (sum of its pixels), row by row */
export function inkOf(x: ArrayLike<number>, G: number, P: number): number[] {
	const s = new Array(G * G).fill(0), W = G * P;
	for (let i = 0; i < x.length; i++) s[Math.floor(Math.floor(i / W) / P) * G + Math.floor((i % W) / P)] += x[i];
	return s;
}
/** the outlined cell: the learner's pick, else the patch with the most ink (ties: the smaller index) */
export function cellOf(rt: RT): number {
	const pick = extraOf(rt.input).cell, x = T(rt, 'input')?.data, G = T(rt, 'ze')?.shape[2] ?? 0;
	if (typeof pick === 'number' && pick >= 0 && pick < G * G) return pick;
	if (!x || !G) return 0;
	const P = Math.round(Math.sqrt(x.length)) / G;
	const ink = inkOf(x, G, P);
	return ink.reduce((b, v, i) => (v > ink[b] ? i : b), 0);
}
/** the D numbers of one cell of a [D, G, G] tensor */
export function cellVec(t: Tensor | undefined, c: number): number[] {
	if (!t) return [];
	const n = t.shape[1] * t.shape[2];
	return Array.from({ length: t.shape[0] }, (_, d) => t.data[d * n + c]);
}
/** the squared distance of cell c to every code (the vq layer's sub-step 'dist') */
export const distOf = (rt: RT, c: number): number[] => cellVec(T(rt, 'zq#dist'), c);
/** codes by distance, nearest first: [{ k, d }] */
export const rankOf = (dist: number[]) => dist.map((d, k) => ({ k, d })).sort((a, b) => a.d - b.d || a.k - b.k);

const DEC = new WeakMap<GraphModel, GraphModel>();
/** the decoder alone: the spec's layers from dec1 on, fed a declared input zq [D, G, G] */
export function decoderOf(m: GraphModel): GraphModel {
	let d = DEC.get(m);
	if (!d) {
		const layers = m.spec.layers.slice(m.spec.layers.findIndex((l) => l.id === 'dec1'));
		const shape = m.spec.layers.find((l) => l.id === 'zq')!.outShape as number[];
		d = fromParts({ ...m.spec, layers, inputs: [{ name: 'zq', adapter: 'codes', kind: 'tensor', shape, dtype: 'f32' }], outputs: ['out'] }, m.weights, m.ops);
		DEC.set(m, d);
	}
	return d;
}
/** decode any z_q [D, G, G]: the scores (before the sigmoid) and the picture */
export function decodeZq(m: GraphModel, zq: Float32Array) {
	const shape = m.spec.layers.find((l) => l.id === 'zq')!.outShape as number[];
	const tr = decoderOf(m).forward({ zq: { shape, data: zq } }, { record: 'layers' });
	return { img: tr.t.img.data, out: tr.t.out.data };
}
/** z_q of a grid of code indices: each cell gets its code's row of the codebook */
export function zqOf(m: GraphModel, idx: number[]): Float32Array {
	const cb = m.weights['zq.codebook'], D = cb.shape[1], n = idx.length, z = new Float32Array(D * n);
	idx.forEach((k, c) => {
		for (let d = 0; d < D; d++) z[d * n + c] = cb.data[k * D + d];
	});
	return z;
}
export const decodeIdx = (m: GraphModel, idx: number[]) => decodeZq(m, zqOf(m, idx)).out;

/** grid side G and patch side P of a model (its spec) */
export const gridOf = (m: GraphModel) => ({ G: Number(m.spec.layers.find((l) => l.id === 'ze')!.outShape[1]), P: Number((m.spec.layers.find((l) => l.id === 'img')!.params as { factor: number }).factor) });

const BLANK = new WeakMap<GraphModel, number>();
/** the code an empty patch snaps to: the encoder on a blank drawing */
export function blankOf(m: GraphModel): number {
	let b = BLANK.get(m);
	if (b === undefined) {
		const shape = m.spec.inputs![0].shape as number[];
		b = m.forward({ input: { shape, data: new Float32Array(shape.reduce((a, v) => a * v, 1)) } }, { record: 'all' }).t['zq#idx'].data[0];
		BLANK.set(m, b);
	}
	return b;
}

const TILES = new WeakMap<GraphModel, Float32Array[]>();
/** what each code draws among blank neighbours: the P × P block of cell (1, 1) of a grid of blank codes with code k there */
export function tilesOf(m: GraphModel): Float32Array[] {
	let tiles = TILES.get(m);
	if (!tiles) {
		const K = m.weights['zq.codebook'].shape[0], blank = blankOf(m), { G, P } = gridOf(m), at = G + 1, W = G * P;
		tiles = Array.from({ length: K }, (_, k) => {
			const idx = new Array(G * G).fill(blank);
			idx[at] = k;
			const out = decodeIdx(m, idx), tile = new Float32Array(P * P);
			for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) tile[a * P + b] = out[(P + a) * W + P + b];
			return tile;
		});
		TILES.set(m, tiles);
	}
	return tiles;
}

/** pixels above half ink inside cell c's block of a 28 × 28 picture */
export function blockInk(a: ArrayLike<number>, c: number, G = 4, P = 7): number {
	let s = 0;
	const r0 = Math.floor(c / G) * P, c0 = (c % G) * P;
	for (let a1 = 0; a1 < P; a1++) for (let b = 0; b < P; b++) if (a[(r0 + a1) * G * P + c0 + b] > 0.5) s++;
	return s;
}

/** the learner's swap: the redrawing before and after one cell's code changed, and what changed where */
export function swapOf(rt: RT) {
	const sw = extraOf(rt.input).swap, m = rt.model, codes = codesOf(rt), before = T(rt, 'out')?.data;
	if (!sw || !m || !before || !codes.length || sw.cell < 0 || sw.cell >= codes.length) return null;
	const idx = codes.slice();
	idx[sw.cell] = sw.code;
	const after = decodeIdx(m, idx), { G, P } = dims(rt);
	let inside = 0, outside = 0;
	for (let i = 0; i < after.length; i++) {
		const ch = Math.abs(after[i] - before[i]), c = Math.floor(Math.floor(i / (G * P)) / P) * G + Math.floor((i % (G * P)) / P);
		if (c === sw.cell) inside += ch;
		else outside += ch;
	}
	const ink0 = blockInk(before, sw.cell, G, P), ink1 = blockInk(after, sw.cell, G, P);
	return { cell: sw.cell, from: codes[sw.cell], to: sw.code, before, after, ink0, ink1, added: ink1 - ink0, inside, outside };
}

/** the gradient of this drawing's training loss (BCE averaged over the 784 pixels) with respect to z_q at cell c, by
 *  backpropagation through the decoder (pixel_shuffle, the 1×1 and 3×3 convolutions and the ReLUs), read from the trace */
export function gradAt(rt: RT, c: number): number[] {
	const m = rt.model, x = T(rt, 'input')?.data, out = T(rt, 'out')?.data, h1 = T(rt, 'dec1')?.data, h2 = T(rt, 'dec2')?.data;
	if (!m || !x || !out || !h1 || !h2) return [];
	const { G, P, D } = dims(rt), n = G * G, W = G * P, N = W * W;
	const w1 = m.weights['dec1.weight'], w2 = m.weights['dec2.weight'], w3 = m.weights['dec3.weight'];
	const H = w1.shape[0], B = P * P;
	const g3 = new Float64Array(B * n); // ∂loss/∂scores, laid out as dec3's output [P·P, G, G]
	for (let p = 0; p < N; p++) {
		const r = Math.floor(p / W), q = p % W;
		g3[((r % P) * P + (q % P)) * n + Math.floor(r / P) * G + Math.floor(q / P)] = (out[p] - x[p]) / N;
	}
	/** back through a 1×1 convolution w [O, C] and the ReLU before it (h: its input, pre-activation) */
	const back = (w: Tensor, g: Float64Array, h: ArrayLike<number>) => {
		const [O, C] = w.shape, r = new Float64Array(C * n);
		for (let ch = 0; ch < C; ch++)
			for (let cell = 0; cell < n; cell++) {
				if (!(h[ch * n + cell] > 0)) continue;
				for (let o = 0; o < O; o++) r[ch * n + cell] += w.data[o * C + ch] * g[o * n + cell];
			}
		return r;
	};
	const g1 = back(w2, back(w3, g3, h2), h1);
	const i = Math.floor(c / G), j = c % G;
	return Array.from({ length: D }, (_, d) => {
		let s = 0;
		for (let ch = 0; ch < H; ch++)
			for (let ky = 0; ky < 3; ky++)
				for (let kx = 0; kx < 3; kx++) {
					const oi = i - ky + 1, oj = j - kx + 1;
					if (oi >= 0 && oi < G && oj >= 0 && oj < G) s += w1.data[((ch * D + d) * 3 + ky) * 3 + kx] * g1[ch * n + oi * G + oj];
				}
		return s;
	});
}

/** this drawing's training numbers: BCE averaged over the pixels, and the commitment term averaged over the D·G·G numbers */
export function lossOf(rt: RT) {
	const x = T(rt, 'input')?.data, s = T(rt, 'img')?.data, ze = T(rt, 'ze')?.data, zq = T(rt, 'zq')?.data;
	if (!x || !s || !ze || !zq) return null;
	let rec = 0, com = 0;
	for (let i = 0; i < x.length; i++) rec += Math.max(s[i], 0) + Math.log1p(Math.exp(-Math.abs(s[i]))) - s[i] * x[i];
	for (let i = 0; i < ze.length; i++) com += (ze[i] - zq[i]) ** 2;
	return { rec: rec / x.length, commit: com / ze.length };
}
