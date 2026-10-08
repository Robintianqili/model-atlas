// tinymamba: the model worked out in float64 from the exported float32 weights (pure TS), for the 'scan' process and
// its Python twin (pipeline/tinymamba/reference.py: the same arithmetic on model_final.pt in float64). Weight names are
// export.py's: emb.weight, B<i>.norm, B<i>.in, B<i>.dt, B<i>.B, B<i>.C (.weight, .bias), B<i>.scan.A_log, B<i>.scan.D, B<i>.out.
import type { Tensor } from '../../atlas/types/index.ts';
import { softplus } from './ops/scan.ts';

export type Weights = Record<string, Tensor>;

export function sizes(w: Weights) {
	const [V, d] = w['emb.weight'].shape;
	const [D, N] = w['B1.scan.A_log'].shape;
	let L = 0;
	while (w[`B${L + 1}.scan.A_log`]) L++;
	return { V, d, D, N, L };
}

/** y = W x + b for every row of x [T, I] (W [O, I]); float64. */
function linear(x: Float64Array[], W: Tensor, b?: Tensor): Float64Array[] {
	const [O, I] = W.shape;
	return x.map((r) => {
		const y = new Float64Array(O);
		for (let o = 0; o < O; o++) {
			let s = 0;
			for (let i = 0; i < I; i++) s += W.data[o * I + i] * r[i];
			y[o] = s + (b ? b.data[o] : 0);
		}
		return y;
	});
}

/** LayerNorm of one row (biased variance, eps 1e-5, as nn.LayerNorm). */
function norm(r: Float64Array, g: Tensor, b: Tensor): Float64Array {
	const n = r.length;
	let mean = 0;
	for (const v of r) mean += v / n;
	let vr = 0;
	for (const v of r) vr += ((v - mean) * (v - mean)) / n;
	const s = Math.sqrt(vr + 1e-5);
	return r.map((v, i) => ((v - mean) / s) * g.data[i] + b.data[i]);
}

export type BlockRun = { x: Float64Array[]; u: Float64Array[]; delta: Float64Array[]; B: Float64Array[]; C: Float64Array[]; A: Float64Array; Dk: ArrayLike<number> };

/** Every block's scan inputs, in float64: x (the block's input), u, delta, B, C, A = −exp(A_log) and D. */
export function blocks(w: Weights, ids: number[]): BlockRun[] {
	const { d, N, L } = sizes(w);
	let x = ids.map((id) => Float64Array.from(w['emb.weight'].data.subarray(id * d, (id + 1) * d)));
	const out: BlockRun[] = [];
	for (let b = 1; b <= L; b++) {
		const p = `B${b}.`;
		const z = x.map((r) => norm(r, w[p + 'norm.weight'], w[p + 'norm.bias']));
		const u = linear(z, w[p + 'in.weight'], w[p + 'in.bias']).map((r) => r.map((v) => v / (1 + Math.exp(-v))));
		const delta = linear(u, w[p + 'dt.weight'], w[p + 'dt.bias']).map((r) => r.map(softplus));
		const run: BlockRun = { x, u, delta, B: linear(u, w[p + 'B.weight']), C: linear(u, w[p + 'C.weight']), A: Float64Array.from(w[p + 'scan.A_log'].data, (v) => -Math.exp(v)), Dk: w[p + 'scan.D'].data };
		out.push(run);
		const y = scanAll(run, N).y;
		const o = linear(y, w[p + 'out.weight'], w[p + 'out.bias']);
		x = x.map((r, t) => r.map((v, i) => v + o[t][i]));
	}
	return out;
}

/** One scan step for every channel and slot: h ← exp(Δ·A)·h + Δ·B·u, then y = Σ C·h + D·u. Changes h in place. */
export function step(r: BlockRun, t: number, h: Float64Array, N: number): Float64Array {
	const D = r.u[t].length;
	const y = new Float64Array(D);
	for (let c = 0; c < D; c++) {
		const dl = r.delta[t][c];
		let s = 0;
		for (let n = 0; n < N; n++) {
			const k = c * N + n;
			h[k] = Math.exp(dl * r.A[k]) * h[k] + dl * r.B[t][n] * r.u[t][c];
			s += r.C[t][n] * h[k];
		}
		y[c] = s + r.Dk[c] * r.u[t][c];
	}
	return y;
}

export function scanAll(r: BlockRun, N: number) {
	const h = new Float64Array(r.u[0]?.length * N || 0);
	const y = r.u.map((_, t) => step(r, t, h, N));
	return { y, h };
}
