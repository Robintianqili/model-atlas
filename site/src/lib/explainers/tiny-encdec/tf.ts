// tiny-encdec: the model's arithmetic in TS, shared by the local op 'greedy' (ops/greedy.ts: float64 sums, float32
// storage via rd = Math.fround, checked against PyTorch fp32) and the process 'decode' (process.ts: float64 throughout,
// rd = the identity, checked against pipeline/tiny-encdec/reference.py in PyTorch float64). Weight layouts are
// PyTorch's (a Linear's W is [out, in]); the pre-norm blocks are model.py's:
//   encoder  x = emb(c) + pos;  x += SelfAttn(LN1 x);  x += W_down relu(W_up LN2 x);  memory = LN x
//   decoder  y = emb(s) + pos;  y += SelfAttn_causal(LN1 y);  y += CrossAttn(LN2 y, memory);  y += FF(LN3 y);
//            logits = W_head LN y
//   attention (H heads of D/H numbers): s = q·k / √(D/H), w = softmax(s), out = W_o (Σ_j w_j v_j)
import type { GraphModel } from '../../atlas/types/index.ts';
import { BOS, STEPS } from './data.ts';

export type Arr = ArrayLike<number>;
export type Rd = (v: number) => number;
export const f64: Rd = (v) => v;
const EPS = 1e-5;

/** The decoder's weights by slot (the op's weight slots; the tensors export.py registers as greedy.<slot>). */
export const SLOTS = ['emb', 'pos', 'ln1_w', 'ln1_b', 'sq', 'sk', 'sv', 'so', 'ln2_w', 'ln2_b', 'cq', 'ck', 'cv', 'co', 'ln3_w', 'ln3_b', 'up', 'down', 'ln_w', 'ln_b', 'head'] as const;
export type DecW = Record<(typeof SLOTS)[number], Arr> & { D: number; H: number; F: number; V: number };
type Blk = { ln1_w: Arr; ln1_b: Arr; q: Arr; k: Arr; v: Arr; o: Arr; ln2_w: Arr; ln2_b: Arr; up: Arr; down: Arr };
export type EncW = Blk & { emb: Arr; pos: Arr; ln_w: Arr; ln_b: Arr };

/** x [n, In] times Wᵀ (W [Out, In]) -> [n, Out] */
function lin(x: Arr, n: number, W: Arr, In: number, Out: number, rd: Rd): Float64Array {
	const y = new Float64Array(n * Out);
	for (let i = 0; i < n; i++)
		for (let o = 0; o < Out; o++) {
			let s = 0;
			for (let k = 0; k < In; k++) s += x[i * In + k] * W[o * In + k];
			y[i * Out + o] = rd(s);
		}
	return y;
}
function norm(x: Arr, n: number, D: number, w: Arr, b: Arr, rd: Rd): Float64Array {
	const y = new Float64Array(n * D);
	for (let i = 0; i < n; i++) {
		let m = 0, v = 0;
		for (let d = 0; d < D; d++) m += x[i * D + d];
		m /= D;
		for (let d = 0; d < D; d++) v += (x[i * D + d] - m) ** 2;
		const r = 1 / Math.sqrt(v / D + EPS);
		for (let d = 0; d < D; d++) y[i * D + d] = rd((x[i * D + d] - m) * r * w[d] + b[d]);
	}
	return y;
}
const add = (a: Arr, b: Arr, rd: Rd) => Float64Array.from(a, (v, i) => rd(v + b[i]));
const look = (tab: Arr, ids: ArrayLike<number>, pos: Arr, D: number, rd: Rd) => Float64Array.from({ length: ids.length * D }, (_, i) => rd(tab[ids[Math.floor(i / D)] * D + (i % D)] + pos[i]));

export function softmax(x: Arr, rd: Rd = f64): Float64Array {
	const m = Math.max(...Array.from(x));
	const p = Float64Array.from(x, (v) => Math.exp(v - m));
	const s = p.reduce((a, b) => a + b, 0);
	return p.map((v) => rd(v / s));
}
/** the index of the largest entry (the first one on a tie) */
export function argmax(x: Arr): number { let b = 0; for (let k = 1; k < x.length; k++) if (x[k] > x[b]) b = k; return b; }

/** Multi-head attention of xq [Tq, D] over xkv [Tk, D]: the output [Tq, D] and the weights [H, Tq, Tk]. */
function attend(xq: Arr, Tq: number, xkv: Arr, Tk: number, W: { q: Arr; k: Arr; v: Arr; o: Arr }, D: number, H: number, causal: boolean, rd: Rd) {
	const [q, k, v] = [lin(xq, Tq, W.q, D, D, rd), lin(xkv, Tk, W.k, D, D, rd), lin(xkv, Tk, W.v, D, D, rd)];
	const dh = D / H, sc = 1 / Math.sqrt(dh);
	const w = new Float64Array(H * Tq * Tk);
	const mix = new Float64Array(Tq * D);
	for (let h = 0; h < H; h++)
		for (let i = 0; i < Tq; i++) {
			const s = Float64Array.from({ length: Tk }, (_, j) => {
				let t = 0;
				for (let c = 0; c < dh; c++) t += q[i * D + h * dh + c] * k[j * D + h * dh + c];
				return causal && j > i ? -Infinity : rd(t * sc);
			});
			const a = softmax(s, rd);
			w.set(a, (h * Tq + i) * Tk);
			for (let c = 0; c < dh; c++) {
				let t = 0;
				for (let j = 0; j < Tk; j++) t += a[j] * v[j * D + h * dh + c];
				mix[i * D + h * dh + c] = rd(t);
			}
		}
	return { out: lin(mix, Tq, W.o, D, D, rd), w };
}
const ff = (x: Arr, n: number, up: Arr, down: Arr, D: number, F: number, rd: Rd) => lin(lin(x, n, up, D, F, rd).map((v) => Math.max(0, v)), n, down, F, D, rd);

/** The encoder on character ids: the memory [T, D] and its self-attention weights [H, T, T]. */
export function encode(w: EncW, ids: ArrayLike<number>, D: number, H: number, F: number, rd: Rd) {
	const T = ids.length;
	let x = look(w.emb, ids, w.pos, D, rd);
	const h = norm(x, T, D, w.ln1_w, w.ln1_b, rd);
	const a = attend(h, T, h, T, w, D, H, false, rd);
	x = add(x, a.out, rd);
	x = add(x, ff(norm(x, T, D, w.ln2_w, w.ln2_b, rd), T, w.up, w.down, D, F, rd), rd);
	return { mem: norm(x, T, D, w.ln_w, w.ln_b, rd), self: a.w };
}

/** The decoder on symbols ids [L] (<s> first) over the memory [T, D]: logits [L, V], the causal self-attention
 *  weights [H, L, L] and the cross-attention weights [H, L, T]. Row t depends only on ids[0..t] (the causal mask). */
export function decode(w: DecW, ids: ArrayLike<number>, mem: Arr, T: number, rd: Rd) {
	const { D, H, F, V } = w;
	const L = ids.length;
	let y = look(w.emb, ids, w.pos, D, rd);
	const h = norm(y, L, D, w.ln1_w, w.ln1_b, rd);
	const s = attend(h, L, h, L, { q: w.sq, k: w.sk, v: w.sv, o: w.so }, D, H, true, rd);
	y = add(y, s.out, rd);
	const c = attend(norm(y, L, D, w.ln2_w, w.ln2_b, rd), L, mem, T, { q: w.cq, k: w.ck, v: w.cv, o: w.co }, D, H, false, rd);
	y = add(y, c.out, rd);
	y = add(y, ff(norm(y, L, D, w.ln3_w, w.ln3_b, rd), L, w.up, w.down, D, F, rd), rd);
	return { logits: lin(norm(y, L, D, w.ln_w, w.ln_b, rd), L, w.head, D, V, rd), self: s.w, cross: c.w };
}

/** Greedy decoding: the decoder runs once per symbol on <s> and everything written so far; returns the decoder input
 *  [<s>, y_0 .. y_{steps-2}] (the op's output) and every step's result. */
export function greedy(w: DecW, mem: Arr, T: number, rd: Rd, steps = STEPS) {
	const ids = [BOS];
	const runs: ReturnType<typeof decode>[] = [];
	for (let t = 0; t < steps; t++) {
		runs.push(decode(w, ids, mem, T, rd));
		if (t < steps - 1) ids.push(argmax(runs[t].logits.subarray(t * w.V, (t + 1) * w.V)));
	}
	return { ids, runs };
}

/** The decoder's weights from a getter of its slot tensors (the op's ctx.w, or the model's greedy.<slot>). */
export function decW(get: (slot: string) => { data: Arr; shape: number[] }, H: number): DecW {
	const w = { D: get('emb').shape[1], V: get('emb').shape[0], F: get('up').shape[0], H } as Record<string, unknown>;
	for (const s of SLOTS) w[s] = get(s).data;
	return w as DecW;
}

export type Weights = { enc: EncW; dec: DecW };
const memo = new WeakMap<object, Weights>();
/** The exported model's weights by role (tensor names as export.py writes them). */
export function weightsOf(m: GraphModel): Weights {
	if (memo.has(m)) return memo.get(m)!;
	const t = (n: string) => {
		if (!m.weights[n]) throw new Error(`tiny-encdec: the model has no tensor '${n}' (export it: npm run atlas -- export tiny-encdec)`);
		return m.weights[n];
	};
	const d = (n: string) => t(n).data;
	const dec = decW((s) => t(`greedy.${s}`), Number(m.spec.layers.find((l) => l.id === 'greedy')?.params?.nHead ?? NaN));
	const enc: EncW = { emb: d('src.emb.weight'), pos: d('src.pos.weight'), ln1_w: d('enc.ln1.weight'), ln1_b: d('enc.ln1.bias'), q: d('enc.q.weight'), k: d('enc.k.weight'), v: d('enc.v.weight'), o: d('enc.o.weight'), ln2_w: d('enc.ln2.weight'), ln2_b: d('enc.ln2.bias'), up: d('enc.up.weight'), down: d('enc.down.weight'), ln_w: d('mem.weight'), ln_b: d('mem.bias') };
	memo.set(m, { enc, dec });
	return memo.get(m)!;
}

/** The whole model in float64 on character ids (the process and the claim tests). */
export function run64(w: Weights, ids: ArrayLike<number>, steps = STEPS) {
	const { D, H, F } = w.dec;
	const e = encode(w.enc, ids, D, H, F, f64);
	return { ...e, ...greedy(w.dec, e.mem, ids.length, f64, steps) };
}
