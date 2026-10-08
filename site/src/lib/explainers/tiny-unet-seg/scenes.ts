// tiny-unet-seg: the synthetic world. A 32×32 grey picture of filled circles, squares and triangles on a noisy
// background, and the true class of every pixel. render() makes every picture the model ever sees: the training set
// (pipeline/tiny-unet-seg/make_data.ts draws the scenes and writes render()'s bytes, train.py checks their CRC-32), the
// presets and the learner's own scenes. The noise uses only + and × (four uniforms, Irwin–Hall), so it is the same number in every JS engine.
import { createRng } from '../../atlas/core/rng.ts';

export const S = 32;
export const CLASSES = [
	{ id: 'background', en: 'background', zh: '背景' },
	{ id: 'circle', en: 'circle', zh: '圆' },
	{ id: 'square', en: 'square', zh: '正方形' },
	{ id: 'triangle', en: 'triangle', zh: '三角形' }
];
/** k: 1 circle, 2 square, 3 triangle; centre (x, y) and size r in pixels (a circle's radius); rot in radians; v brightness;
 *  hollow: only a ring 1.5 pixels wide (an outline, the limits set) */
export type Shape = { k: number; x: number; y: number; r: number; rot?: number; v?: number; hollow?: boolean };
/** paint: pixel index × 4 + class, applied in order after the shapes (class 0 erases to background) */
export type SegRaw = { shapes?: Shape[]; paint?: number[]; seed?: number; bg?: number; noise?: number; preset?: string };
/** how the learner's scenes look: background, shape brightness, noise */
export const LOOK = { bg: 0.1, v: 0.8, noise: 0.1 };
/** the sizes the training set used (r from 3 to 8 pixels) */
export const TRAIN_R = [3, 8];
const SQ3 = Math.sqrt(3);

function inShape(k: number, r: number, u: number, w: number): boolean {
	if (k === 1) return u * u + w * w <= r * r;
	if (k === 2) return Math.abs(u) <= 0.85 * r && Math.abs(w) <= 0.85 * r;
	const R = 1.35 * r; // a triangle pointing up with circumradius R
	return w <= R / 2 && Math.abs(u) * SQ3 <= w + R;
}
/** Is the point (px, py) inside the shape? */
export function inside(sh: Shape, px: number, py: number): boolean {
	const dx = px - sh.x, dy = py - sh.y;
	const c = sh.rot ? Math.cos(sh.rot) : 1, s = sh.rot ? Math.sin(sh.rot) : 0;
	const u = c * dx + s * dy, w = c * dy - s * dx;
	return inShape(sh.k, sh.r, u, w) && !(sh.hollow && inShape(sh.k, sh.r - 1.5, u, w));
}

/** The picture the model reads ([1, 32, 32] numbers) and the true class of every pixel. */
export function render(raw: SegRaw | null | undefined): { image: Float32Array; label: Uint8Array } {
	const bg = raw?.bg ?? LOOK.bg, sd = raw?.noise ?? LOOK.noise;
	const val = new Float64Array(S * S).fill(bg), label = new Uint8Array(S * S);
	for (const sh of raw?.shapes ?? []) {
		for (let i = 0; i < S * S; i++) {
			if (inside(sh, (i % S) + 0.5, Math.floor(i / S) + 0.5)) (label[i] = sh.k), (val[i] = sh.v ?? LOOK.v);
		}
	}
	for (const e of raw?.paint ?? []) {
		const i = e >> 2, k = e & 3;
		if (i >= 0 && i < S * S) (label[i] = k), (val[i] = k ? LOOK.v : bg);
	}
	const rng = createRng(raw?.seed ?? 0);
	const image = new Float32Array(S * S);
	for (let i = 0; i < S * S; i++) image[i] = val[i] + sd * (rng.next() + rng.next() + rng.next() + rng.next() - 2) * SQ3;
	return { image, label };
}

const sh = (k: number, x: number, y: number, r: number): Shape => ({ k, x, y, r });
/** The presets: fixed scenes (the first is the page's start input). */
export const PRESETS: { id: string; label: { en: string; zh: string }; raw: SegRaw }[] = [
	{ id: 'three', label: { en: 'one of each', zh: '每种一个' }, raw: { shapes: [sh(1, 9, 10, 6), sh(2, 23, 9, 5.5), sh(3, 16, 23, 6.5)], seed: 11 } },
	{ id: 'small', label: { en: 'small shapes', zh: '小形状' }, raw: { shapes: [sh(3, 7, 7, 3.2), sh(1, 24, 8, 3), sh(2, 9, 24, 3), sh(1, 23, 23, 3.4)], seed: 12 } },
	{ id: 'big', label: { en: 'two big ones', zh: '两个大的' }, raw: { shapes: [sh(2, 10, 11, 7.5), sh(1, 22, 22, 7.5)], seed: 13 } },
	{ id: 'touch', label: { en: 'touching', zh: '挨在一起' }, raw: { shapes: [sh(1, 11, 16, 6), sh(3, 21.5, 17, 6)], seed: 14 } }
];
export const presetRaw = (id: string): SegRaw => ({ ...structuredClone(PRESETS.find((p) => p.id === id)!.raw), preset: id });
