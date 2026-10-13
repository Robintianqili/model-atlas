// tiny-tcn: the network. A temporal convolutional network (TCN) reads the changes c of the series and gives, at every time
// t at once, the predicted next change; the forecast of y_{t+1} is y_t + that change.
//   l1      causal conv, kernel 2, dilation 1, 1 -> C channels, ReLU:          h1 = relu(conv(c))
//   l2..l4  residual blocks, kernel 2, dilations 2, 4, 8, C -> C channels:     h_l = h_{l−1} + relu(conv(h_{l−1}))
//   head    1×1 conv C -> 1:                                                    out_t = predicted c_{t+1}
// Causal: zeros are padded on the left only (d·(k − 1) of them), so the output at t never reads a change after t; with
// kernel 2 a layer of dilation d reads its input at t − d and t. PyTorch twin: pipeline/tiny-tcn/twin.py (nn.Conv1d after
// F.pad, the same state_dict names l1.weight [C, 1, 2] … head.bias [1]).
// `forwardPlain` is the same network on plain numbers for the views and the narration (every node of the receptive-field
// tree); a claim test proves it equals the autograd forward (≤ 1e-12).
import type { Module, Rec, Rng, Tensor } from '../../atlas/types/index.ts';
import { Conv1d, namedParams } from '../../atlas/autograd/nn.ts';
import * as op from '../../atlas/autograd/ops.ts';
import type { Var } from '../../atlas/autograd/var.ts';
import { DILATIONS, KERNEL } from './data.ts';

export const CH = 8;
export const LAYERS = ['l1', 'l2', 'l3', 'l4'];

export class TCN implements Module {
	readonly layers: Conv1d[];
	readonly head: Conv1d;
	constructor(rng: Rng) {
		this.layers = DILATIONS.map((d, i) => new Conv1d(LAYERS[i], i ? CH : 1, CH, KERNEL, rng, { dilation: d, causal: true }));
		this.head = new Conv1d('head', CH, 1, 1, rng);
	}
	/** x [N, 1, T] changes -> [N, 1, T] predicted next changes. rec: 'l1' … 'l4' (h after each layer). */
	forward(x: Var, rec?: Rec): Var {
		let h = x;
		this.layers.forEach((l, i) => {
			const a = op.relu(l.forward(h));
			h = i ? op.add(h, a) : a;
			rec?.(LAYERS[i], h);
		});
		return this.head.forward(h);
	}
	named() {
		return namedParams(...this.layers, this.head);
	}
	params() {
		return Object.values(this.named());
	}
}

export type Params = Record<string, Tensor>;

/** Every number of the network on the changes c (length T): per layer the conv sums (pre), the ReLU branch (a) and the
 *  output h (channel-major: index ch·T + t), and the predicted next change at every t. */
export interface Plain {
	T: number;
	pre: Float64Array[];
	a: Float64Array[];
	h: Float64Array[];
	out: Float64Array;
}

export function forwardPlain(P: Params, c: ArrayLike<number>): Plain {
	const T = c.length;
	let x = Float64Array.from(c);
	let cin = 1;
	const pre: Float64Array[] = [];
	const a: Float64Array[] = [];
	const h: Float64Array[] = [];
	DILATIONS.forEach((d, l) => {
		const w = P[`${LAYERS[l]}.weight`].data;
		const b = P[`${LAYERS[l]}.bias`].data;
		const p = new Float64Array(CH * T);
		for (let o = 0; o < CH; o++)
			for (let t = 0; t < T; t++) {
				let s = b[o];
				for (let i = 0; i < cin; i++) {
					const base = (o * cin + i) * KERNEL;
					if (t - d >= 0) s += w[base] * x[i * T + t - d];
					s += w[base + 1] * x[i * T + t];
				}
				p[o * T + t] = s;
			}
		const r = p.map((v) => Math.max(0, v));
		const out = l ? x.map((v, k) => v + r[k]) : r;
		pre.push(p);
		a.push(r);
		h.push(out);
		x = out;
		cin = CH;
	});
	const hw = P['head.weight'].data;
	const out = new Float64Array(T);
	for (let t = 0; t < T; t++) {
		let s = P['head.bias'].data[0];
		for (let i = 0; i < CH; i++) s += hw[i] * x[i * T + t];
		out[t] = s;
	}
	return { T, pre, a, h, out };
}

/** The predicted next change after the last of `c` (only the last 16 changes matter). */
export function nextChange(P: Params, c: ArrayLike<number>): number {
	const p = forwardPlain(P, c);
	return p.out[p.T - 1];
}

/** One layer's inputs to node (layer l, time t, channel o): the two taps (t − d, t) of every input channel, their weights
 *  and products, the bias, the sum, the ReLU and (for l ≥ 1) the skip value. Times before 0 read the zero padding. */
export function nodeTerms(P: Params, pl: Plain, c: ArrayLike<number>, l: number, t: number, o: number) {
	const d = DILATIONS[l];
	const cin = l ? CH : 1;
	const x = l ? pl.h[l - 1] : Float64Array.from(c);
	const w = P[`${LAYERS[l]}.weight`].data;
	const terms: { ch: number; time: number; x: number; w: number }[] = [];
	for (let i = 0; i < cin; i++)
		for (let k = 0; k < KERNEL; k++) {
			const time = t - d * (KERNEL - 1 - k);
			terms.push({ ch: i, time, x: time >= 0 ? x[i * pl.T + time] : 0, w: w[(o * cin + i) * KERNEL + k] });
		}
	const T = pl.T;
	return { dil: d, terms, bias: P[`${LAYERS[l]}.bias`].data[o], pre: pl.pre[l][o * T + t], a: pl.a[l][o * T + t], skip: l ? x[o * T + t] : NaN, h: pl.h[l][o * T + t] };
}
