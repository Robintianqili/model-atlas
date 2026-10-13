// regularization: the network, 2 -> width -> width -> 2 with tanh, PyTorch naming (net.0, net.2, net.4), built from
// the kit's autograd Linear layers so that dropout can sit after each hidden layer: forward(x, masks) multiplies hidden
// layer i by masks[i] (0 for a dropped unit, 1/(1 − p) for a kept one: inverted dropout, as torch.nn.Dropout in train
// mode); without masks (testing, and every evaluation) nothing is dropped or scaled. Plus plain float64 forward passes
// from recorded weights (no autograd) for the views: the field P(class 1) over the plane and each point's answer.
import type { Rng, Tensor } from '../../atlas/types/index.ts';
import { Linear, namedParams } from '../../atlas/autograd/nn.ts';
import * as op from '../../atlas/autograd/ops.ts';
import type { Var } from '../../atlas/autograd/var.ts';

export class RegNet {
	readonly layers: Linear[];
	constructor(width: number, rng: Rng) {
		this.layers = [new Linear('net.0', 2, width, rng), new Linear('net.2', width, width, rng), new Linear('net.4', width, 2, rng)];
	}
	forward(x: Var, masks?: Var[] | null): Var {
		let h = x;
		for (let i = 0; i < 2; i++) {
			h = op.tanh(this.layers[i].forward(h));
			if (masks) h = op.mul(h, masks[i]);
		}
		return this.layers[2].forward(h);
	}
	named(): Record<string, Var> {
		return namedParams(...this.layers);
	}
	/** the weight matrices (the L2 penalty leaves the biases alone, as weight decay usually does) */
	weights(): Var[] {
		return this.layers.map((l) => l.weight);
	}
}

export type Params = Record<string, Tensor>;

/** The two class scores for P rows of x [P·2], from recorded weights (no dropout: the network as it is used). */
export function scores(params: Params, x: ArrayLike<number>, P: number): Float64Array {
	let h = Float64Array.from(x);
	let I = 2;
	for (let li = 0; li < 3; li++) {
		const w = params[`net.${2 * li}.weight`];
		const b = params[`net.${2 * li}.bias`];
		const O = w.shape[0];
		const y = new Float64Array(P * O);
		for (let p = 0; p < P; p++)
			for (let o = 0; o < O; o++) {
				let s = b.data[o];
				for (let i = 0; i < I; i++) s += h[p * I + i] * w.data[o * I + i];
				y[p * O + o] = li < 2 ? Math.tanh(s) : s;
			}
		h = y;
		I = O;
	}
	return h;
}

/** P(class 1) for each row (softmax of the two scores). */
export function p1Of(params: Params, x: ArrayLike<number>, P: number): Float64Array {
	const z = scores(params, x, P);
	return Float64Array.from({ length: P }, (_, p) => 1 / (1 + Math.exp(z[2 * p] - z[2 * p + 1])));
}

/** The plane every view draws (data units) and its grid of RES_X × RES_Y cells, row 0 at the top. */
export const BOUNDS = { x0: -1.6, x1: 1.6, y0: -1.3, y1: 1.3 };
export const RES_X = 48;
export const RES_Y = 39;

/** P(class 1) at every cell centre of the plane, row 0 at the top. */
export function fieldOf(params: Params): Float64Array {
	const xy = new Float64Array(RES_X * RES_Y * 2);
	for (let r = 0; r < RES_Y; r++)
		for (let c = 0; c < RES_X; c++) {
			const i = r * RES_X + c;
			xy[2 * i] = BOUNDS.x0 + ((c + 0.5) / RES_X) * (BOUNDS.x1 - BOUNDS.x0);
			xy[2 * i + 1] = BOUNDS.y1 - ((r + 0.5) / RES_Y) * (BOUNDS.y1 - BOUNDS.y0);
		}
	return p1Of(params, xy, RES_X * RES_Y);
}
