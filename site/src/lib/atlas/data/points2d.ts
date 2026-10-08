// Seeded 2-D point sets (W1-B1). TS is the source of truth for synthetic data: Python twins read the points TS wrote.
// Every generator draws only from the Rng it is given (pass core/rng.ts's rng, or a fork of a run's rng).
// Labelled sets alternate classes (point i has class i % classes), so every prefix is balanced. Coordinates lie
// roughly in [−1, 1]², a good range for tanh networks.
import type { Rng, Tensor } from '../types/index.ts';

export interface Points2D {
	name: string;
	/** [n·2] interleaved x, y */
	xy: Float64Array;
	/** class per point (all 0 for unlabelled sets) */
	label: Int32Array;
	n: number;
	classes: number;
}

function make(name: string, n: number, classes: number, f: (i: number, cls: number) => [number, number]): Points2D {
	const xy = new Float64Array(n * 2);
	const label = new Int32Array(n);
	for (let i = 0; i < n; i++) {
		const cls = classes > 1 ? i % classes : 0;
		const [x, y] = f(i, cls);
		xy[2 * i] = x;
		xy[2 * i + 1] = y;
		label[i] = cls;
	}
	return { name, xy, label, n, classes };
}

/** Two interleaving half circles. */
export function moons(n: number, rng: Rng, o: { noise?: number } = {}): Points2D {
	const noise = o.noise ?? 0.1;
	return make('moons', n, 2, (_, cls) => {
		const a = Math.PI * rng.next();
		const x = cls === 0 ? Math.cos(a) : 1 - Math.cos(a);
		const y = cls === 0 ? Math.sin(a) : 0.5 - Math.sin(a);
		return [0.8 * (x - 0.5 + noise * rng.normal()), 0.8 * (y - 0.25 + noise * rng.normal())];
	});
}

/** A disc (class 1) inside a ring (class 0). */
export function circles(n: number, rng: Rng, o: { noise?: number; factor?: number } = {}): Points2D {
	const noise = o.noise ?? 0.05;
	const factor = o.factor ?? 0.5;
	return make('circles', n, 2, (_, cls) => {
		const a = 2 * Math.PI * rng.next();
		const r = cls === 0 ? 0.9 : 0.9 * factor;
		return [r * Math.cos(a) + noise * rng.normal(), r * Math.sin(a) + noise * rng.normal()];
	});
}

/** Two interleaved spiral arms (hard for small networks). */
export function spiral(n: number, rng: Rng, o: { noise?: number; turns?: number } = {}): Points2D {
	const noise = o.noise ?? 0.03;
	const turns = o.turns ?? 1.5;
	return make('spiral', n, 2, (_, cls) => {
		const t = rng.next();
		const r = 0.05 + 0.9 * t;
		const a = 2 * Math.PI * turns * t + cls * Math.PI;
		return [r * Math.cos(a) + noise * rng.normal(), r * Math.sin(a) + noise * rng.normal()];
	});
}

/** XOR quadrants: class 0 where x and y have the same sign. Points keep `margin` away from the axes. */
export function xor(n: number, rng: Rng, o: { margin?: number } = {}): Points2D {
	const margin = o.margin ?? 0.05;
	return make('xor', n, 2, (_, cls) => {
		const sx = rng.next() < 0.5 ? -1 : 1;
		const sy = cls === 0 ? sx : -sx;
		const x = sx * (margin + (1 - margin) * rng.next());
		const y = sy * (margin + (1 - margin) * rng.next());
		return [x, y];
	});
}

/** Two Gaussian blobs (an easy, linearly separable set). */
export function gaussians(n: number, rng: Rng, o: { spread?: number } = {}): Points2D {
	const s = o.spread ?? 0.25;
	return make('gaussians', n, 2, (_, cls) => {
		const c = cls === 0 ? -0.45 : 0.45;
		return [c + s * rng.normal(), c + s * rng.normal()];
	});
}

/** An unlabelled ring (flow-2d preset). */
export function ring(n: number, rng: Rng, o: { radius?: number; noise?: number } = {}): Points2D {
	const radius = o.radius ?? 0.7;
	const noise = o.noise ?? 0.04;
	return make('ring', n, 1, () => {
		const a = 2 * Math.PI * rng.next();
		return [radius * Math.cos(a) + noise * rng.normal(), radius * Math.sin(a) + noise * rng.normal()];
	});
}

/** An unlabelled smiley face: outline (50 %), two eyes (10 % each), smile (30 %) (flow-2d preset). */
export function smiley(n: number, rng: Rng, o: { noise?: number } = {}): Points2D {
	const noise = o.noise ?? 0.025;
	return make('smiley', n, 1, () => {
		const u = rng.next();
		let x: number;
		let y: number;
		if (u < 0.5) {
			const a = 2 * Math.PI * rng.next();
			x = 0.8 * Math.cos(a);
			y = 0.8 * Math.sin(a);
		} else if (u < 0.7) {
			const a = 2 * Math.PI * rng.next();
			const r = 0.08 * Math.sqrt(rng.next());
			x = (u < 0.6 ? -0.28 : 0.28) + r * Math.cos(a);
			y = 0.25 + r * Math.sin(a);
		} else {
			const a = Math.PI * (1.15 + 0.7 * rng.next());
			x = 0.45 * Math.cos(a);
			y = -0.05 + 0.45 * Math.sin(a);
		}
		return [x + noise * rng.normal(), y + noise * rng.normal()];
	});
}

export const POINTS2D = { moons, circles, spiral, xor, gaussians, ring, smiley };
export type Points2DName = keyof typeof POINTS2D;

/** The points as tensors: x [n, 2], y [n] (class indices as float64). */
export function pointsTensors(p: Points2D): { x: Tensor; y: Tensor } {
	return { x: { shape: [p.n, 2], data: p.xy.slice() }, y: { shape: [p.n], data: Float64Array.from(p.label) } };
}

/** A batch of `size` points drawn with replacement (rng.int). */
export function sampleBatch(p: Points2D, size: number, rng: Rng): { x: Tensor; label: Int32Array; index: Int32Array } {
	const index = new Int32Array(size);
	const xy = new Float64Array(size * 2);
	const label = new Int32Array(size);
	for (let i = 0; i < size; i++) {
		const j = rng.int(p.n);
		index[i] = j;
		xy[2 * i] = p.xy[2 * j];
		xy[2 * i + 1] = p.xy[2 * j + 1];
		label[i] = p.label[j];
	}
	return { x: { shape: [size, 2], data: xy }, label, index };
}
