// adversarial-tinycnn: TinyCNN's forward pass on the kit's autograd core (float64), so the gradient of the loss with
// respect to the INPUT pixels comes from reverse mode: grad(loss, x). The weights are TinyCNN's shipped ones
// (static/models/tinycnn, loaded by the kit through legacy.ts) and enter as constants: only x gets a gradient.
// The core has no 2-D convolution or max-pool, so both are built from its ops:
//   conv 3×3 (valid)  gather each output position's C×3×3 window (an im2col table), then linear with W as [O, C·9]
//   max-pool 2×2      gather each window's largest entry (the first one on a tie, as PyTorch scans it), so the
//                     gradient flows to that entry only, as in PyTorch
// pipeline/adversarial-tinycnn/attack.py is the same network in PyTorch; reference.py proves the frames equal.
import { grad, input, noGrad, raw, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';

/** TinyCNN's tensors by state_dict name ('conv1_1.weight' [8, 1, 3, 3], ..., 'dense.bias' [10]) */
export type Weights = Record<string, { shape: number[]; data: ArrayLike<number> }>;
export const SIDE = 28;
export const N = SIDE * SIDE;

const tables = new Map<string, Int32Array>();
/** im2col: for each output position (oy, ox) of a 3×3 valid convolution over [C, H, H], its C·9 input indices (c, ky, kx) */
function im2col(C: number, H: number): Int32Array {
	const key = `${C},${H}`;
	let t = tables.get(key);
	if (t) return t;
	const O = H - 2;
	t = new Int32Array(O * O * C * 9);
	let k = 0;
	for (let oy = 0; oy < O; oy++)
		for (let ox = 0; ox < O; ox++)
			for (let c = 0; c < C; c++) for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) t[k++] = c * H * H + (oy + ky) * H + ox + kx;
	tables.set(key, t);
	return t;
}

const consts = new WeakMap<object, Record<string, Var>>();
/** the weights as float64 constants; a conv weight [O, C, 3, 3] as [O, C·9] (the same order as an im2col row) */
function constants(w: Weights): Record<string, Var> {
	let c = consts.get(w);
	if (c) return c;
	c = {};
	for (const [k, t] of Object.entries(w)) c[k] = raw(Float64Array.from(t.data), t.shape.length === 4 ? [t.shape[0], t.shape[1] * 9] : t.shape.slice());
	consts.set(w, c);
	return c;
}

/** conv 3×3 over a flat [C·H·H] map: returns the flat [O·(H−2)·(H−2)] map (channel, row, column order) */
function conv(x: Var, c: Record<string, Var>, name: string, C: number, H: number): Var {
	const W = c[`${name}.weight`];
	const O = H - 2;
	const cols = op.reshape(op.gather(x, 0, im2col(C, H)), [O * O, C * 9]);
	return op.reshape(op.transpose(op.linear(cols, W, c[`${name}.bias`])), [W.shape[0] * O * O]);
}

/** max-pool 2×2, stride 2, over a flat [C·H·H] map */
function pool(x: Var, C: number, H: number): Var {
	const h = H / 2, d = x.data, idx = new Int32Array(C * h * h);
	let k = 0;
	for (let ch = 0; ch < C; ch++)
		for (let oy = 0; oy < h; oy++)
			for (let ox = 0; ox < h; ox++) {
				const a = ch * H * H + 2 * oy * H + 2 * ox;
				let best = a;
				for (const j of [a + 1, a + H, a + H + 1]) if (d[j] > d[best]) best = j;
				idx[k++] = best;
			}
	return op.gather(x, 0, idx);
}

/** the 10 logits for a flat [784] input (conv 8, conv 8, pool, conv 16, conv 16, pool, dense 256 → 10) */
export function logitsVar(x: Var, w: Weights): Var {
	const c = constants(w);
	let h = op.relu(conv(x, c, 'conv1_1', 1, 28)); // 8 × 26 × 26
	h = pool(op.relu(conv(h, c, 'conv1_2', 8, 26)), 8, 24); // 8 × 12 × 12
	h = op.relu(conv(h, c, 'conv2_1', 8, 12)); // 16 × 10 × 10
	h = pool(op.relu(conv(h, c, 'conv2_2', 16, 10)), 16, 8); // 16 × 4 × 4 = 256, flattened as PyTorch does
	return op.linear(h, c['dense.weight'], c['dense.bias']);
}

export function logitsOf(x: ArrayLike<number>, w: Weights): Float64Array {
	return noGrad(() => logitsVar(input({ shape: [N], data: x }), w).data.slice());
}

/** the logits, the loss L = cross-entropy with class y, and dL/dx for all 784 pixels */
export function lossGrad(x: ArrayLike<number>, y: number, w: Weights): { z: Float64Array; loss: number; g: Float64Array } {
	const xv = input({ shape: [N], data: x }, true);
	const z = logitsVar(xv, w);
	const L = crossEntropy(z, [y]);
	return { z: z.data.slice(), loss: L.item(), g: grad(L, xv).data.slice() };
}

export function softmax(z: ArrayLike<number>): Float64Array {
	let m = -Infinity;
	for (let i = 0; i < z.length; i++) m = Math.max(m, z[i]);
	const e = Float64Array.from(z, (v) => Math.exp(v - m));
	const s = e.reduce((a, b) => a + b, 0);
	return e.map((v) => v / s);
}

/** the index of the largest entry, leaving out `skip` (the smallest index on a tie) */
export function argmax(a: ArrayLike<number>, skip = -1): number {
	let b = skip === 0 ? 1 : 0;
	for (let i = 0; i < a.length; i++) if (i !== skip && a[i] > a[b]) b = i;
	return b;
}
