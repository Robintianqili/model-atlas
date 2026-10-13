// pointnet: the five shapes as loose points along their outlines, and what the model sees of a cloud. The page's presets
// and the training data (pipeline/pointnet/make_data.ts) both come from here, so PyTorch trains on the page's bytes.
//   cloudOf(raw)      the learner's points (painted, finite), at most MAX_N
//   normalize(xy)     centre on the middle of the box around the points (not their average: a drawing has more
//                     points where the pen moved slowly), scale so the farthest point is at distance 1 (float32, as the model)
//   sample(k, n, rng) n points along shape k's outline: any rotation, a little stretch, jitter and uneven spacing
//   withStray(xy)     the same points plus one stray point (the honest-limits step)
import type { I18n, Rng } from '../../atlas/types/index.ts';
import { createRng, fnv1a32 } from '../../atlas/core/rng.ts';

export const CLASSES = ['circle', 'square', 'triangle', 'star', 'ring'] as const;
export type ShapeName = (typeof CLASSES)[number];
export const LABELS: Record<ShapeName, I18n> = {
	circle: { en: 'circle', zh: '圆' },
	square: { en: 'square', zh: '正方形' },
	triangle: { en: 'triangle', zh: '三角形' },
	star: { en: 'star', zh: '五角星' },
	ring: { en: 'ring', zh: '圆环' }
};
/** the most points the model reads (spec dim N); later points are left out */
export const MAX_N = 1024;
/** the stray point of the limits step: this many radii from the centre, along +x (in the model's units) */
export const STRAY = 3;

type Pt = { x: number; y: number };
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** the learner's points (painted, finite), in order, at most MAX_N */
export function cloudOf(raw: unknown): Pt[] {
	const ps = ((raw && typeof raw === 'object' ? (raw as { points?: unknown[] }).points : null) ?? []) as Pt[];
	return ps.filter((q) => finite(q?.x) && finite(q?.y)).slice(0, MAX_N).map((q) => ({ x: q.x, y: q.y }));
}

/** centre (the middle of the box around the points) and scale: returns the model's input [n, 2] (float32) and the centre and scale used */
export function normalize(xy: ArrayLike<number>) {
	const n = Math.floor(xy.length / 2);
	let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
	for (let i = 0; i < n; i++) {
		(x0 = Math.min(x0, xy[2 * i])), (x1 = Math.max(x1, xy[2 * i]));
		(y0 = Math.min(y0, xy[2 * i + 1])), (y1 = Math.max(y1, xy[2 * i + 1]));
	}
	const cx = n ? (x0 + x1) / 2 : 0, cy = n ? (y0 + y1) / 2 : 0;
	let r = 0;
	for (let i = 0; i < n; i++) r = Math.max(r, Math.hypot(xy[2 * i] - cx, xy[2 * i + 1] - cy));
	const s = r > 1e-9 ? 1 / r : 1;
	const out = new Float32Array(2 * n);
	for (let i = 0; i < n; i++) (out[2 * i] = (xy[2 * i] - cx) * s), (out[2 * i + 1] = (xy[2 * i + 1] - cy) * s);
	return { xy: out, n, cx, cy, scale: s, radius: r };
}

/** the model's input plus one stray point STRAY radii to the right of the centre, normalized again */
export function withStray(xy: ArrayLike<number>) {
	return normalize([...Array.from(xy), STRAY, 0]);
}

/** a seeded order of n points (Fisher–Yates), the same everywhere */
export function shuffled(n: number, seed = fnv1a32('pointnet shuffle')): number[] {
	const rng = createRng(seed);
	const p = Array.from({ length: n }, (_, i) => i);
	for (let i = n - 1; i > 0; i--) {
		const j = rng.int(i + 1);
		[p[i], p[j]] = [p[j], p[i]];
	}
	return p;
}

/** closed outlines of a shape around (0, 0) at radius 1; `inner` is the star's inner radius or the ring's inner circle */
function loops(k: ShapeName, inner: number): number[][][] {
	const poly = (m: number, r: (i: number) => number, a0 = Math.PI / 2) =>
		Array.from({ length: m }, (_, i) => [r(i) * Math.cos(a0 + (2 * Math.PI * i) / m), r(i) * Math.sin(a0 + (2 * Math.PI * i) / m)]);
	if (k === 'circle') return [poly(48, () => 1)];
	if (k === 'square') return [poly(4, () => 1, Math.PI / 4)];
	if (k === 'triangle') return [poly(3, () => 1)];
	if (k === 'star') return [poly(10, (i) => (i % 2 ? inner : 1))];
	return [poly(48, () => 1), poly(48, () => inner)];
}

export interface SampleOpts { rot?: number; stretch?: number; noise?: number; inner?: number; gap?: number; warp?: number }

/** n points along shape k's outline, in drawing order (walked along each loop), radius about 1 around (0, 0) */
export function sample(k: ShapeName, n: number, rng: Rng, o: SampleOpts = {}): Float64Array {
	const inner = o.inner ?? (k === 'star' ? 0.35 + 0.15 * rng.next() : 0.35 + 0.2 * rng.next());
	const rot = o.rot ?? 2 * Math.PI * rng.next();
	const st = o.stretch ?? 0.85 + 0.3 * rng.next();
	const noise = o.noise ?? 0.04 * rng.next();
	const gap = o.gap ?? (rng.next() < 0.3 ? 0.12 * rng.next() : 0);
	const warp = o.warp ?? 0.6 * rng.next();
	const phase = 2 * Math.PI * rng.next();
	const segs: { a: number[]; b: number[]; len: number }[] = [];
	for (const L of loops(k, inner)) for (let i = 0; i < L.length; i++) {
		const a = L[i], b = L[(i + 1) % L.length];
		segs.push({ a, b, len: Math.hypot(b[0] - a[0], b[1] - a[1]) });
	}
	const total = segs.reduce((s, q) => s + q.len, 0);
	const g0 = rng.next() * total;
	const out = new Float64Array(2 * n);
	const c = Math.cos(rot), s = Math.sin(rot);
	for (let i = 0; i < n; i++) {
		// even steps along the outline (a gap of `gap` is skipped), warped so some stretches get more points
		const u = (i + 0.5) / n;
		const w = u + (warp / (2 * Math.PI)) * Math.sin(2 * Math.PI * u + phase) - (warp / (2 * Math.PI)) * Math.sin(phase);
		let d = (g0 + gap * total + w * (1 - gap) * total) % total;
		if (d < 0) d += total;
		let j = 0;
		while (j < segs.length - 1 && d > segs[j].len) d -= segs[j++].len;
		const q = segs[j], f = Math.min(1, d / Math.max(q.len, 1e-12));
		const x = q.a[0] + f * (q.b[0] - q.a[0]) + noise * rng.normal();
		const y = q.a[1] + f * (q.b[1] - q.a[1]) + noise * rng.normal();
		out[2 * i] = (c * x - s * y) * st;
		out[2 * i + 1] = (s * x + c * y) / st;
	}
	return out;
}

/** the page's presets: one clean drawing of each shape, about 0.85 across the editor's plane */
export function presetPoints(k: ShapeName): Pt[] {
	const rng = createRng(fnv1a32(`pointnet preset ${k}`));
	const n = k === 'ring' ? 96 : 72;
	const xy = sample(k, n, rng, { rot: k === 'square' ? 0.2 : 0.1, stretch: 1, noise: 0.012, gap: 0, warp: 0.2, inner: k === 'ring' ? 0.5 : 0.42 });
	return Array.from({ length: n }, (_, i) => ({ x: Math.round(xy[2 * i] * 850) / 1000, y: Math.round(xy[2 * i + 1] * 850) / 1000 }));
}
