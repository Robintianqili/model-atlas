// char-rnn: the LSTM worked out one character at a time, in float64, from the exported float32 weights. The `unroll`
// and `sample` processes (process.ts) use it; pipeline/char-rnn/reference.py does the same in PyTorch float64
// (model.double()), so `atlas parity` can compare their frames to 1e-12. Gate order i, f, g, o as in nn.LSTM:
//   z = W_ih x + W_hh h_prev + b_ih + b_hh;  i = σ(z_i), f = σ(z_f), g = tanh(z_g), o = σ(z_o)
//   c = f·c_prev + i·g;  h = o·tanh(c);  scores = W_head h + b_head;  chances = softmax(scores)
import type { GraphModel, Tensor } from '../../atlas/types/index.ts';

export interface Weights { E: Tensor; Wih: Tensor; Whh: Tensor; bih: Tensor; bhh: Tensor; Wo: Tensor; bo: Tensor; V: number; D: number; H: number }
export interface Step { i: Float64Array; f: Float64Array; g: Float64Array; o: Float64Array; c: Float64Array; h: Float64Array }

const memo = new WeakMap<object, Weights>();
export function weightsOf(m: GraphModel): Weights {
	const hit = memo.get(m);
	if (hit) return hit;
	const w = (n: string) => {
		const t = m.weights[n];
		if (!t) throw new Error(`char-rnn: the model has no tensor '${n}' (export it: npm run atlas -- export char-rnn)`);
		return t;
	};
	const E = w('emb.weight');
	const Whh = w('lstm.weight_hh');
	const r: Weights = { E, Wih: w('lstm.weight_ih'), Whh, bih: w('lstm.bias_ih'), bhh: w('lstm.bias_hh'), Wo: w('head.weight'), bo: w('head.bias'), V: E.shape[0], D: E.shape[1], H: Whh.shape[1] };
	memo.set(m, r);
	return r;
}

const sig = (v: number) => 1 / (1 + Math.exp(-v));

/** One LSTM step on character `id`, from the previous h and c. */
export function lstmStep(w: Weights, id: number, hPrev: ArrayLike<number>, cPrev: ArrayLike<number>): Step {
	const { D, H } = w;
	const s: Step = { i: new Float64Array(H), f: new Float64Array(H), g: new Float64Array(H), o: new Float64Array(H), c: new Float64Array(H), h: new Float64Array(H) };
	const z = new Float64Array(4 * H);
	for (let r = 0; r < 4 * H; r++) {
		let a = 0;
		for (let k = 0; k < D; k++) a += w.Wih.data[r * D + k] * w.E.data[id * D + k];
		for (let k = 0; k < H; k++) a += w.Whh.data[r * H + k] * hPrev[k];
		z[r] = a + w.bih.data[r] + w.bhh.data[r];
	}
	for (let j = 0; j < H; j++) {
		s.i[j] = sig(z[j]);
		s.f[j] = sig(z[H + j]);
		s.g[j] = Math.tanh(z[2 * H + j]);
		s.o[j] = sig(z[3 * H + j]);
		s.c[j] = s.f[j] * cPrev[j] + s.i[j] * s.g[j];
		s.h[j] = s.o[j] * Math.tanh(s.c[j]);
	}
	return s;
}

/** The scores (logits) for the next character from h. */
export function scores(w: Weights, h: ArrayLike<number>): Float64Array {
	const { V, H } = w;
	const out = new Float64Array(V);
	for (let v = 0; v < V; v++) {
		let a = 0;
		for (let k = 0; k < H; k++) a += w.Wo.data[v * H + k] * h[k];
		out[v] = a + w.bo.data[v];
	}
	return out;
}

export function softmax(x: ArrayLike<number>): Float64Array {
	let m = -Infinity;
	for (let k = 0; k < x.length; k++) m = Math.max(m, x[k]);
	const p = Float64Array.from(x, (v) => Math.exp(v - m));
	let s = 0;
	for (let k = 0; k < p.length; k++) s += p[k];
	for (let k = 0; k < p.length; k++) p[k] /= s;
	return p;
}

/** The scores after reading `ids` from an empty memory (no ids: h = 0, so the scores are the head's bias). */
export function scoresAfter(w: Weights, ids: ArrayLike<number>): Float64Array {
	let h: ArrayLike<number> = new Float64Array(w.H);
	let c: ArrayLike<number> = new Float64Array(w.H);
	for (let t = 0; t < ids.length; t++) ({ h, c } = lstmStep(w, ids[t], h, c));
	return scores(w, h);
}

/** The process input: the learner's character ids and the weights. The weights are typed arrays, so they travel to the
 *  package worker; JSON (atlas parity passes the input to the Python reference as JSON) carries only the ids, and the
 *  reference loads the same weights itself. */
export type RnnInput = { ids: number[]; w?: Weights | null };
export function rnnInput(rt: { pre: { tensors: Record<string, { data: ArrayLike<number> }> } | null; model: GraphModel | null }): RnnInput {
	const o: RnnInput = { ids: Array.from(rt.pre?.tensors.input?.data ?? []), w: rt.model ? weightsOf(rt.model) : null };
	Object.defineProperty(o, 'toJSON', { value: () => ({ ids: o.ids }), enumerable: false });
	return o;
}
