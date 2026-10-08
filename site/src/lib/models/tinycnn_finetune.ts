// Transfer learning with TinyCNN: train a new dense "head" on frozen flatten features, live in the browser.
// Pure TypeScript: no DOM, no Svelte. Deterministic (seeded init, full-batch gradient descent).
//
//   featuresOf(model, input)                       -> Float32Array (256 flatten features)
//   trainHead(features, labels, { epochs, lr, seed }) -> Head { W, b, lossHistory, accHistory, ... }
//   createHeadTrainer(...)                         -> step-by-step version for animating training
//   predictHead(head, features)                    -> { logits, probs, top }
//   projectPCA(basis, features)                    -> [x, y] in the map from pca.json
import type { TinyCNN } from './tinycnn.ts';

/** Flatten features of one input (the 256 numbers the dense layer reads). */
export function featuresOf(model: TinyCNN, input: ArrayLike<number>): Float32Array {
	return model.features(input);
}

export interface HeadOptions {
	/** Full passes over the training set (each pass = one gradient step). Default 200. */
	epochs?: number;
	/** Learning rate on scale-normalised features. Default 0.5. */
	lr?: number;
	/** Seed for the small random initial weights. Default 0. */
	seed?: number;
	/** L2 penalty on W (weight decay). Default 1e-4. */
	l2?: number;
	/** Number of classes. Default max(label) + 1. */
	numClasses?: number;
	/** Initial weight scale (uniform in [-s, s]). Default 0.01. */
	initScale?: number;
}

export interface Head {
	/** Weights [K x D], row-major: W[k * D + d]. They act on features / scale. */
	W: Float32Array;
	/** Biases [K]. */
	b: Float32Array;
	K: number;
	D: number;
	/** Features are divided by this before the head (root-mean-square feature norm of the training set). */
	scale: number;
	/** Mean cross-entropy loss: index 0 = before training, then after each epoch. */
	lossHistory: number[];
	/** Training-set accuracy: index 0 = before training, then after each epoch. */
	accHistory: number[];
	epochs: number;
	/** Trainable parameter count: K * D + K. */
	trainableParams: number;
}

export interface HeadTrainer {
	/** Run n more epochs (default 1); returns the head so far. */
	step(n?: number): Head;
	/** Current head (shares the arrays; copy if you keep it). */
	head(): Head;
	epoch(): number;
}

/** Small deterministic PRNG (mulberry32). */
function mulberry32(seed: number): () => number {
	let s = seed >>> 0;
	return () => {
		s = (s + 0x6d2b79f5) | 0;
		let t = s;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/**
 * Step-by-step head trainer: softmax cross-entropy, full-batch gradient descent.
 * Use this to animate training (call step() from an animation frame); trainHead() runs it to the end.
 */
export function createHeadTrainer(features: ArrayLike<number>[], labels: number[], opts: HeadOptions = {}): HeadTrainer {
	const n = features.length;
	if (n === 0 || n !== labels.length) throw new Error('trainHead: need the same, non-zero number of features and labels');
	const D = features[0].length;
	const K = opts.numClasses ?? Math.max(...labels) + 1;
	const lr = opts.lr ?? 0.5;
	const l2 = opts.l2 ?? 1e-4;
	const rand = mulberry32(opts.seed ?? 0);
	const init = opts.initScale ?? 0.01;

	// scale-normalise: divide by the RMS feature norm so one learning rate works for any feature scale
	let sq = 0;
	for (const f of features) for (let d = 0; d < D; d++) sq += f[d] * f[d];
	const scale = Math.sqrt(sq / n) || 1;
	const X = new Float64Array(n * D);
	for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) X[i * D + d] = features[i][d] / scale;

	const W = new Float64Array(K * D);
	const b = new Float64Array(K);
	for (let i = 0; i < W.length; i++) W[i] = (rand() * 2 - 1) * init;
	const gW = new Float64Array(K * D);
	const gb = new Float64Array(K);
	const z = new Float64Array(K);
	const lossHistory: number[] = [];
	const accHistory: number[] = [];
	let ep = 0;

	// one pass: forward on all examples; if `update`, accumulate gradients and take a step
	const pass = (update: boolean) => {
		let loss = 0;
		let correct = 0;
		gW.fill(0);
		gb.fill(0);
		for (let i = 0; i < n; i++) {
			let m = -Infinity;
			for (let k = 0; k < K; k++) {
				let s = b[k];
				const wk = k * D;
				const xi = i * D;
				for (let d = 0; d < D; d++) s += W[wk + d] * X[xi + d];
				z[k] = s;
				if (s > m) m = s;
			}
			let se = 0;
			// the predicted class from the logits, before z is overwritten with exp() below
			// (comparing a logit with an already exponentiated entry misreported accHistory)
			let top = 0;
			for (let k = 1; k < K; k++) if (z[k] > z[top]) top = k;
			for (let k = 0; k < K; k++) {
				z[k] = Math.exp(z[k] - m);
				se += z[k];
			}
			const y = labels[i];
			loss += -Math.log(z[y] / se);
			if (top === y) correct++;
			if (update) {
				for (let k = 0; k < K; k++) {
					const g = z[k] / se - (k === y ? 1 : 0);
					gb[k] += g;
					const wk = k * D;
					const xi = i * D;
					for (let d = 0; d < D; d++) gW[wk + d] += g * X[xi + d];
				}
			}
		}
		if (update) {
			for (let j = 0; j < W.length; j++) W[j] -= lr * (gW[j] / n + l2 * W[j]);
			for (let k = 0; k < K; k++) b[k] -= lr * (gb[k] / n);
		}
		return { loss: loss / n, acc: correct / n };
	};

	const record = () => {
		const r = pass(false);
		lossHistory.push(r.loss);
		accHistory.push(r.acc);
	};
	record();

	const head = (): Head => ({
		W: Float32Array.from(W),
		b: Float32Array.from(b),
		K,
		D,
		scale,
		lossHistory,
		accHistory,
		epochs: ep,
		trainableParams: K * D + K
	});

	return {
		step(k = 1) {
			for (let s = 0; s < k; s++) {
				pass(true);
				ep++;
				record();
			}
			return head();
		},
		head,
		epoch: () => ep
	};
}

/** Train a new softmax head on frozen features (full-batch gradient descent, deterministic). */
export function trainHead(features: ArrayLike<number>[], labels: number[], opts: HeadOptions = {}): Head {
	const t = createHeadTrainer(features, labels, opts);
	return t.step(opts.epochs ?? 200);
}

/** Apply a trained head to one feature vector. */
export function predictHead(head: Head, features: ArrayLike<number>): { logits: Float32Array; probs: Float32Array; top: number } {
	const { K, D, W, b, scale } = head;
	const logits = new Float32Array(K);
	let m = -Infinity;
	let top = 0;
	for (let k = 0; k < K; k++) {
		let s = b[k];
		for (let d = 0; d < D; d++) s += W[k * D + d] * (features[d] / scale);
		logits[k] = s;
		if (s > m) {
			m = s;
			top = k;
		}
	}
	const probs = new Float32Array(K);
	let se = 0;
	for (let k = 0; k < K; k++) se += Math.exp(logits[k] - m);
	for (let k = 0; k < K; k++) probs[k] = Math.exp(logits[k] - m) / se;
	return { logits, probs, top };
}

/** Fraction of examples the head classifies correctly. */
export function headAccuracy(head: Head, features: ArrayLike<number>[], labels: number[]): number {
	let ok = 0;
	for (let i = 0; i < features.length; i++) if (predictHead(head, features[i]).top === labels[i]) ok++;
	return features.length ? ok / features.length : 0;
}

/** A 2D PCA basis as stored in pca.json (bases.all or bases.heldOut). */
export interface PCABasis {
	mean: number[];
	components: number[][];
}

/** Project a feature vector into the 2D PCA map: (features - mean) . components. */
export function projectPCA(basis: PCABasis, features: ArrayLike<number>): [number, number] {
	const [c0, c1] = basis.components;
	let x = 0;
	let y = 0;
	for (let d = 0; d < basis.mean.length; d++) {
		const v = features[d] - basis.mean[d];
		x += v * c0[d];
		y += v * c1[d];
	}
	return [x, y];
}
