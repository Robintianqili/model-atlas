// tinyseq2seq: the model's arithmetic, shared by the local ops (ops/gru.ts, ops/attn_decoder.ts: float64 sums, float32
// storage via rd = Math.fround, checked against PyTorch fp32) and the process (process.ts: float64 throughout, rd = the
// identity, checked against pipeline/tinyseq2seq/reference.py in PyTorch float64). Weight layouts are PyTorch's.
//   GRU (nn.GRU, nn.GRUCell; rows r, z, n):  r = σ(W_ir x + b_ir + W_hr h + b_hr),  z = σ(W_iz x + b_iz + W_hz h + b_hz)
//                                            n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn)),  h' = (1 − z) ⊙ n + z ⊙ h
//   attention (Bahdanau):  key_j = W_k h_j + b_k,  query = W_q s,  e_j = v · tanh(query + key_j),  a = softmax(e),
//                          c = Σ_j a_j h_j;  s' = GRUCell([emb(y); c], s);  logits = W_o [s'; c] + b_o;  y' = argmax
import type { GraphModel, Tensor } from '../../atlas/types/index.ts';
import { BOS, STEPS } from './data.ts';

export type Arr = ArrayLike<number>;
export type Rd = (v: number) => number;
export const f64: Rd = (v) => v;
const sig = (v: number) => 1 / (1 + Math.exp(-v));
/** row `row` of an [*, n] matrix W times v[o .. o + n) */
function dot(W: Arr, row: number, v: Arr, o: number, n: number): number {
	let s = 0;
	for (let k = 0; k < n; k++) s += W[row * n + k] * v[o + k];
	return s;
}

export interface GruW { Wih: Arr; Whh: Arr; bih: Arr; bhh: Arr; In: number; H: number }
export interface DecW { emb: Arr; E: number; wq: Arr; wk: Arr; bk: Arr; v: Arr; A: number; cell: GruW; wo: Arr; bo: Arr; V: number }

/** One GRU step on x[xo .. xo + In) from h. */
export function gruStep(w: GruW, x: Arr, xo: number, h: Arr, rd: Rd) {
	const { In, H, Wih, Whh, bih, bhh } = w;
	const g = { r: new Float64Array(H), z: new Float64Array(H), n: new Float64Array(H), h: new Float64Array(H) };
	for (let j = 0; j < H; j++) {
		g.r[j] = rd(sig(dot(Wih, j, x, xo, In) + bih[j] + dot(Whh, j, h, 0, H) + bhh[j]));
		g.z[j] = rd(sig(dot(Wih, H + j, x, xo, In) + bih[H + j] + dot(Whh, H + j, h, 0, H) + bhh[H + j]));
		g.n[j] = rd(Math.tanh(dot(Wih, 2 * H + j, x, xo, In) + bih[2 * H + j] + g.r[j] * (dot(Whh, 2 * H + j, h, 0, H) + bhh[2 * H + j])));
		g.h[j] = rd((1 - g.z[j]) * g.n[j] + g.z[j] * h[j]);
	}
	return g;
}

/** The encoder over x [T, In]: the states h [T, H] and the gates, row by row. */
export function encode(w: GruW, x: Arr, T: number, rd: Rd) {
	const out = { r: new Float64Array(T * w.H), z: new Float64Array(T * w.H), n: new Float64Array(T * w.H), h: new Float64Array(T * w.H) };
	let h: Arr = new Float64Array(w.H);
	for (let t = 0; t < T; t++) {
		const g = gruStep(w, x, t * w.In, h, rd);
		for (const k of ['r', 'z', 'n', 'h'] as const) out[k].set(g[k], t * w.H);
		h = g.h;
	}
	return out;
}

/** the index of the largest entry (the first one on a tie) */
export function argmax(x: Arr): number { let b = 0; for (let k = 1; k < x.length; k++) if (x[k] > x[b]) b = k; return b; }

export function softmax(x: Arr, rd: Rd = f64): Float64Array {
	const m = Math.max(...Array.from(x));
	const p = Float64Array.from(x, (v) => Math.exp(v - m));
	const s = p.reduce((a, b) => a + b, 0);
	return p.map((v) => rd(v / s));
}

/** One decoder step from state s and the previous symbol y: query, scores, weights, context, the new state, logits.
 *  K holds the keys [T, A] (key_j = W_k h_j + b_k), computed once per input. */
export function decStep(w: DecW, enc: Arr, T: number, K: Arr, s: Arr, y: number, rd: Rd) {
	const { A, E } = w;
	const H = w.cell.H;
	const q = Float64Array.from({ length: A }, (_, a) => rd(dot(w.wq, a, s, 0, H)));
	const e = Float64Array.from({ length: T }, (_, j) => {
		let v = 0;
		for (let a = 0; a < A; a++) v += w.v[a] * Math.tanh(q[a] + K[j * A + a]);
		return rd(v);
	});
	const al = softmax(e, rd);
	const c = Float64Array.from({ length: H }, (_, k) => {
		let v = 0;
		for (let j = 0; j < T; j++) v += al[j] * enc[j * H + k];
		return rd(v);
	});
	const x = new Float64Array(E + H);
	for (let k = 0; k < E; k++) x[k] = w.emb[y * E + k];
	x.set(c, E);
	const g = gruStep(w.cell, x, 0, s, rd);
	const sc = new Float64Array(2 * H);
	sc.set(g.h);
	sc.set(c, H);
	return { q, e, a: al, c, s: g.h, logits: Float64Array.from({ length: w.V }, (_, o) => rd(dot(w.wo, o, sc, 0, 2 * H) + w.bo[o])) };
}

export type Step = ReturnType<typeof decStep> & { y: number; p: Float64Array; K: Float64Array };

/** Greedy decoding for `steps` steps from s = the last encoder state and y = <s> (the trace always takes every step;
 *  the process stops after the end symbol). */
export function* greedy(w: DecW, enc: Arr, T: number, rd: Rd, steps = STEPS): Generator<Step> {
	const H = w.cell.H;
	const K = Float64Array.from({ length: T * w.A }, (_, i) => rd(dot(w.wk, i % w.A, enc, Math.floor(i / w.A) * H, H) + w.bk[i % w.A]));
	let s: Arr = Array.prototype.slice.call(enc, (T - 1) * H, T * H);
	let y = BOS;
	for (let t = 0; t < steps; t++) {
		const r = decStep(w, enc, T, K, s, y, rd);
		y = argmax(r.logits);
		yield { ...r, y, p: softmax(r.logits), K };
		s = r.s;
	}
}

type D = (n: string) => Arr;
type S = (n: string) => number[];
/** GRU weights from tensors named <p>weight_ih, <p>weight_hh, <p>bias_ih, <p>bias_hh. */
export const gruW = (d: D, sh: S, p: string): GruW => ({ Wih: d(`${p}weight_ih`), Whh: d(`${p}weight_hh`), bih: d(`${p}bias_ih`), bhh: d(`${p}bias_hh`), In: sh(`${p}weight_ih`)[1], H: sh(`${p}weight_hh`)[1] });
/** Decoder weights from tensors named <p>emb, <p>wq, <p>wk, <p>bk, <p>v, the cell's, <p>wo, <p>bo. */
export const decW = (d: D, sh: S, p: string): DecW => ({ emb: d(`${p}emb`), E: sh(`${p}emb`)[1], wq: d(`${p}wq`), wk: d(`${p}wk`), bk: d(`${p}bk`), v: d(`${p}v`), A: sh(`${p}wq`)[0], cell: gruW(d, sh, p), wo: d(`${p}wo`), bo: d(`${p}bo`), V: sh(`${p}wo`)[0] });

export type Weights = { src: Arr; Es: number; enc: GruW; dec: DecW };
const memo = new WeakMap<object, Weights>();
/** The exported model's weights by role (tensor names as export.py writes them: emb.weight, enc.*, dec.*). */
export function weightsOf(m: GraphModel): Weights {
	const t = (n: string): Tensor => {
		if (!m.weights[n]) throw new Error(`tinyseq2seq: the model has no tensor '${n}' (export it: npm run atlas -- export tinyseq2seq)`);
		return m.weights[n];
	};
	const [d, sh] = [(n: string) => t(n).data, (n: string) => t(n).shape];
	if (!memo.has(m)) memo.set(m, { src: d('emb.weight'), Es: sh('emb.weight')[1], enc: gruW(d, sh, 'enc.'), dec: decW(d, sh, 'dec.') });
	return memo.get(m)!;
}

/** The whole model in float64 on character ids: encoder states and the greedy steps (the process and the claim tests). */
export function run64(w: Weights, ids: ArrayLike<number>, steps = STEPS) {
	const T = ids.length;
	const x = Float64Array.from({ length: T * w.Es }, (_, i) => w.src[ids[Math.floor(i / w.Es)] * w.Es + (i % w.Es)]);
	const enc = encode(w.enc, x, T, f64);
	return { enc, steps: T ? [...greedy(w.dec, enc.h, T, f64, steps)] : [] };
}
