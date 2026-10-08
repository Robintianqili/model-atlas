/**
 * Demo data for the kit gallery. Everything here is ILLUSTRATIVE: a hand-designed filter bank
 * (edge detectors and a blur, not a trained model) and a random dense layer. The arithmetic is
 * real, the "model" is not, so every panel that shows these numbers is badged Illustrative.
 */
import { seededRandom, softmax } from '$lib/kit/stats';
import type { TensorLike } from '$lib/kit/types';

export const DEMO_KERNELS: { name: string; k: number[][] }[] = [
	{ name: 'horizontal edge', k: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]] },
	{ name: 'vertical edge', k: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] },
	{ name: 'diagonal edge', k: [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]] },
	{ name: 'blur', k: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]] }
];
export const DEMO_BIAS = [0, 0, 0, -0.1];
export const DEMO_CLASSES = ['cat', 'house', 'cup', 'tree', 'fish', 'key', 'cloud', 'star', 'door', 'clock'];

/** A 12×12 "7" so the gallery has something to show before anyone draws. */
export function defaultInput(): Float32Array {
	const x = new Float32Array(144);
	const on = (r: number, c: number, v = 1) => (x[r * 12 + c] = Math.max(x[r * 12 + c], v));
	for (let c = 2; c <= 9; c++) on(2, c);
	for (let c = 2; c <= 9; c++) on(3, c, 0.5);
	for (let r = 3; r <= 10; r++) {
		const c = Math.round(9 - (r - 3) * 0.55);
		on(r, c);
		on(r, c - 1, 0.45);
	}
	return x;
}

/** Area-average an N×N ink image down to M×M. */
export function downsample(ink: Float32Array, n: number, m: number): Float32Array {
	const out = new Float32Array(m * m);
	const s = n / m;
	for (let r = 0; r < m; r++)
		for (let c = 0; c < m; c++) {
			let sum = 0;
			let cnt = 0;
			for (let y = Math.floor(r * s); y < Math.floor((r + 1) * s); y++)
				for (let x = Math.floor(c * s); x < Math.floor((c + 1) * s); x++) {
					sum += ink[y * n + x];
					cnt++;
				}
			out[r * m + c] = cnt ? Math.min(1, (sum / cnt) * 1.6) : 0;
		}
	return out;
}

/** Valid 3×3 convolution of one [H, W] map with each kernel: [K, H-2, W-2]. */
export function conv3x3(x: Float32Array, h: number, w: number): Float32Array {
	const oh = h - 2;
	const ow = w - 2;
	const out = new Float32Array(DEMO_KERNELS.length * oh * ow);
	DEMO_KERNELS.forEach(({ k }, ki) => {
		for (let r = 0; r < oh; r++)
			for (let c = 0; c < ow; c++) {
				let s = DEMO_BIAS[ki];
				for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) s += k[u][v] * x[(r + u) * w + (c + v)];
				out[ki * oh * ow + r * ow + c] = s;
			}
	});
	return out;
}

export function relu(x: Float32Array): Float32Array {
	return x.map((v) => (v > 0 ? v : 0));
}

export function maxpool2(x: Float32Array, ch: number, h: number, w: number): Float32Array {
	const oh = Math.floor(h / 2);
	const ow = Math.floor(w / 2);
	const out = new Float32Array(ch * oh * ow);
	for (let c = 0; c < ch; c++)
		for (let r = 0; r < oh; r++)
			for (let q = 0; q < ow; q++) {
				let m = -Infinity;
				for (let u = 0; u < 2; u++)
					for (let v = 0; v < 2; v++) m = Math.max(m, x[c * h * w + (2 * r + u) * w + (2 * q + v)]);
				out[c * oh * ow + r * ow + q] = m;
			}
	return out;
}

const rand = seededRandom(7);
const DENSE_W = Float32Array.from({ length: 10 * 4 * 5 * 5 }, () => (rand() - 0.5) * 0.9);

export function dense(x: Float32Array): Float32Array {
	const out = new Float32Array(10);
	for (let o = 0; o < 10; o++) {
		let s = 0;
		for (let i = 0; i < x.length; i++) s += DENSE_W[o * x.length + i] * x[i];
		out[o] = s;
	}
	return out;
}

export interface DemoTrace {
	order: string[];
	t: Record<string, TensorLike & { data: Float32Array }>;
}

/** The whole illustrative forward pass on a 12×12 input. */
export function demoForward(input12: Float32Array): DemoTrace {
	const conv = conv3x3(input12, 12, 12);
	const act = relu(conv);
	const pool = maxpool2(act, 4, 10, 10);
	const logits = dense(pool);
	const probs = softmax(logits);
	return {
		order: ['input', 'conv', 'relu', 'pool', 'logits', 'probs'],
		t: {
			input: { shape: [1, 12, 12], data: input12 },
			conv: { shape: [4, 10, 10], data: conv },
			relu: { shape: [4, 10, 10], data: act },
			pool: { shape: [4, 5, 5], data: pool },
			logits: { shape: [10], data: logits },
			probs: { shape: [10], data: probs }
		}
	};
}

export const DEMO_LAYERS = [
	{ id: 'input', label: { en: 'Input', zh: '输入' }, op: 'input', outShape: [1, 12, 12] },
	{ id: 'conv', label: { en: 'Conv 3×3', zh: '卷积 3×3' }, op: 'conv2d', outShape: [4, 10, 10] },
	{ id: 'relu', label: { en: 'ReLU', zh: 'ReLU' }, op: 'relu', outShape: [4, 10, 10] },
	{ id: 'pool', label: { en: 'Max pool', zh: '最大池化' }, op: 'maxpool2d', outShape: [4, 5, 5] },
	{ id: 'logits', label: { en: 'Dense', zh: '全连接' }, op: 'linear', outShape: [10] },
	{ id: 'probs', label: { en: 'Softmax', zh: 'Softmax' }, op: 'softmax', outShape: [10] }
];

/** Random attention weights with a causal mask, rows summing to 1 (illustrative). */
export function demoAttention(tokens: number, seed = 3): Float32Array {
	const r = seededRandom(seed);
	const out = new Float32Array(tokens * tokens);
	for (let i = 0; i < tokens; i++) {
		const scores = Array.from({ length: i + 1 }, (_, j) => r() * 3 + (j === i ? 1.2 : 0) + (j === 0 ? 0.8 : 0));
		const p = softmax(scores);
		for (let j = 0; j <= i; j++) out[i * tokens + j] = p[j];
	}
	return out;
}

/** A 768-long "embedding" for the strip demo. */
export function demoVector(n = 768, seed = 11): Float32Array {
	const r = seededRandom(seed);
	return Float32Array.from({ length: n }, () => {
		const u = r() + r() + r() - 1.5;
		return u * 0.9;
	});
}
