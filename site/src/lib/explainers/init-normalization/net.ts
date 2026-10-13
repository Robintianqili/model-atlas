// init-normalization: the toy network, run once forward and once backward by the kit's autograd (atlas/autograd, tested
// against PyTorch in the kit's own tests). 256 examples x of 64 standard-normal numbers and 256 standard-normal targets
// y are drawn from the run's seed (fork 'data'); every weight is σ times a standard-normal number drawn from fork
// 'weights', so two settings of σ run the SAME network scaled differently; every bias starts at 0. Ten hidden layers
//   z = h·Wᵀ + b,  z ← norm(z) (none, BatchNorm over the 256 examples, or LayerNorm over the 64 units),  h = act(z)
// then one output ŷ = h·wᵀ + b and the loss L = mean((ŷ − y)²). grad() gives dL/dz of every layer (the gradient that
// flows back into its weighted sums, before the normalization) and dL/dW, dL/db of every layer.
import { grad, input, param, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { BATCH, DEPTH, EPS, WIDTH, type Setup } from './data.ts';

export interface DeepRun {
	/** the input [BATCH, WIDTH] */
	x: Float64Array;
	/** per hidden layer (index l is layer l + 1): the weighted sums z = h·Wᵀ + b [BATCH, WIDTH], before the normalization */
	z: Float64Array[];
	/** per hidden layer: the activations h = act(norm(z)) [BATCH, WIDTH] */
	h: Float64Array[];
	/** per hidden layer: dL/dz, the gradient that flows back into its weighted sums [BATCH, WIDTH] */
	gz: Float64Array[];
	/** per hidden layer: dL/dW [WIDTH, WIDTH] and dL/db [WIDTH] */
	gW: Float64Array[];
	gB: Float64Array[];
	/** the output layer's dL/dw [WIDTH] and dL/db */
	gOut: Float64Array;
	gOutBias: number;
	loss: number;
}

/** The standard-normal numbers of a seed: the data and the unscaled weights (the same for every σ). */
export function draws(seed: number): { x: Float64Array; y: Float64Array; w: Float64Array[]; wOut: Float64Array } {
	const rd = createRng(seed).fork('data');
	const rw = createRng(seed).fork('weights');
	const fill = (n: number, r: { normal(): number }) => {
		const a = new Float64Array(n);
		for (let i = 0; i < n; i++) a[i] = r.normal();
		return a;
	};
	const x = fill(BATCH * WIDTH, rd);
	const y = fill(BATCH, rd);
	const w = Array.from({ length: DEPTH }, () => fill(WIDTH * WIDTH, rw));
	const wOut = fill(WIDTH, rw);
	return { x, y, w, wOut };
}

const cache = new Map<number, ReturnType<typeof draws>>();
function drawsOf(seed: number) {
	let d = cache.get(seed);
	if (!d) {
		d = draws(seed);
		if (cache.size > 4) cache.clear();
		cache.set(seed, d);
	}
	return d;
}

/** BatchNorm without gain and shift: every column (unit) over the rows (examples). */
function batchNormVar(z: Var): Var {
	const mu = op.mean(z, 0, true);
	const zc = op.sub(z, mu);
	const v = op.mean(op.mul(zc, zc), 0, true);
	return op.mul(zc, op.div(1, op.sqrt(op.add(v, EPS))));
}

/** One forward and one backward pass of the toy network with the setup's σ, activation and normalization. */
export function deepRun(s: Setup, seed: number): DeepRun {
	const d = drawsOf(seed);
	const scaled = (a: Float64Array) => {
		const o = new Float64Array(a.length);
		for (let i = 0; i < a.length; i++) o[i] = s.std * a[i];
		return o;
	};
	let h: Var = input({ shape: [BATCH, WIDTH], data: d.x });
	const Ws: Var[] = [];
	const Bs: Var[] = [];
	const zs: Var[] = [];
	const hs: Float64Array[] = [];
	for (let l = 0; l < DEPTH; l++) {
		const W = param(`L${l + 1}.weight`, { shape: [WIDTH, WIDTH], data: scaled(d.w[l]) });
		const b = param(`L${l + 1}.bias`, { shape: [WIDTH], data: new Float64Array(WIDTH) });
		Ws.push(W);
		Bs.push(b);
		const z = op.linear(h, W, b);
		zs.push(z);
		const n = s.norm === 'batch' ? batchNormVar(z) : s.norm === 'layer' ? op.layerNorm(z, null, null, EPS) : z;
		h = s.act === 'relu' ? op.relu(n) : op.tanh(n);
		hs.push(Float64Array.from(h.data));
	}
	const wOut = param('out.weight', { shape: [1, WIDTH], data: scaled(d.wOut) });
	const bOut = param('out.bias', { shape: [1], data: new Float64Array(1) });
	const yhat = op.linear(h, wOut, bOut);
	const loss = op.mean(op.square(op.sub(yhat, input({ shape: [BATCH, 1], data: d.y }))));
	const g = grad(loss, [...zs, ...Ws, ...Bs, wOut, bOut]);
	const cp = (v: Var) => Float64Array.from(v.data);
	return {
		x: d.x,
		z: zs.map(cp),
		h: hs,
		gz: g.slice(0, DEPTH).map(cp),
		gW: g.slice(DEPTH, 2 * DEPTH).map(cp),
		gB: g.slice(2 * DEPTH, 3 * DEPTH).map(cp),
		gOut: cp(g[3 * DEPTH]),
		gOutBias: g[3 * DEPTH + 1].data[0],
		loss: loss.item()
	};
}
