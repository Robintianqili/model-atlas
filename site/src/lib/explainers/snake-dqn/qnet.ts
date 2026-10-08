// snake-dqn: the Q-network in float64 from the exported float32 weights (pipeline/snake-dqn/model.py), for the processes.
// conv 3→16 (3×3, padding 1), ReLU, conv 16→32, ReLU, flatten, linear 2,592→64, ReLU, linear 64→3. Every sum runs in
// the kit's order (for each output: the taps in order, starting at 0, then the bias), so pipeline/snake-dqn/reference.py
// gets the same bits; the trace (source fwd) is the float32 forward pass that atlas parity checks against PyTorch.
import type { GraphModel } from '../../atlas/types/index.ts';
import { CELLS, SIZE } from './rules.ts';

const NAMES = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'out.weight', 'out.bias'] as const;
export type QW = Float32Array[];
const memo = new WeakMap<object, QW>();
export function weightsOf(m: GraphModel | null | undefined): QW | null {
	if (!m) return null;
	let w = memo.get(m);
	if (!w) {
		w = NAMES.map((n) => {
			const t = m.weights[n];
			if (!t) throw new Error(`snake-dqn: no tensor ${n} (run npm run atlas -- export snake-dqn)`);
			return t.data as Float32Array;
		});
		memo.set(m, w);
	}
	return w;
}
function conv(x: ArrayLike<number>, C: number, W: Float32Array, b: Float32Array): Float64Array {
	const O = b.length, y = new Float64Array(O * CELLS);
	for (let o = 0; o < O; o++)
		for (let i = 0; i < SIZE; i++)
			for (let j = 0; j < SIZE; j++) {
				let a = 0;
				for (let c = 0; c < C; c++)
					for (let ky = 0; ky < 3; ky++)
						for (let kx = 0; kx < 3; kx++) {
							const yy = i + ky - 1, xx = j + kx - 1;
							if (yy >= 0 && yy < SIZE && xx >= 0 && xx < SIZE) a += W[((o * C + c) * 3 + ky) * 3 + kx] * x[c * CELLS + yy * SIZE + xx];
						}
				a += b[o];
				y[o * CELLS + i * SIZE + j] = a < 0 ? 0 : a;
			}
	return y;
}
function dense(x: ArrayLike<number>, W: Float32Array, b: Float32Array, relu: boolean): Float64Array {
	const n = b.length, k = x.length, y = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		let a = 0;
		for (let j = 0; j < k; j++) a += W[i * k + j] * x[j];
		a += b[i];
		y[i] = relu && a < 0 ? 0 : a;
	}
	return y;
}
/** Q(s, ·) for the three actions (straight on, left, right), from the encoded board (rules.ts encode). */
export function qvalues(w: QW, x: ArrayLike<number>): number[] {
	const h = dense(conv(conv(x, 3, w[0], w[1]), w[1].length, w[2], w[3]), w[4], w[5], true);
	return Array.from(dense(h, w[6], w[7], false));
}
