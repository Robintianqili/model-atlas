/**
 * TinyCNN guided lesson: small pure helpers (no Svelte, no DOM).
 */

/** Angles of the thin test strokes, in image coordinates (y down): —, \, |, / */
export const STROKE_ANGLES = [0, 45, 90, 135] as const;
export const STROKE_GLYPHS = ['—', '\\', '|', '/'] as const;
export const STROKE_NAMES = [
	{ en: 'horizontal', zh: '横' },
	{ en: 'diagonal \\', zh: '斜线 \\' },
	{ en: 'vertical', zh: '竖' },
	{ en: 'diagonal /', zh: '斜线 /' }
] as const;

/**
 * How strongly a 3×3 kernel responds to a thin straight stroke at each of STROKE_ANGLES:
 * the largest output (window · kernel + bias) anywhere on a 15×15 image with one anti-aliased
 * line of width `width` px through its centre. The default width is the stroke width the drawings
 * are rendered at (1.6 px). Computed from the kernel's own 9 weights, so it is live, not a label.
 */
export function strokeResponses(kernel: ArrayLike<number>, bias: number, width = 1.6): number[] {
	const N = 15;
	const c0 = (N - 1) / 2;
	return STROKE_ANGLES.map((deg) => {
		const a = (deg * Math.PI) / 180;
		const nx = -Math.sin(a);
		const ny = Math.cos(a);
		const img = new Float32Array(N * N);
		for (let y = 0; y < N; y++)
			for (let x = 0; x < N; x++) {
				const d = Math.abs((x - c0) * nx + (y - c0) * ny);
				img[y * N + x] = Math.max(0, Math.min(1, width / 2 + 0.5 - d));
			}
		let best = -Infinity;
		for (let y = 0; y + 2 < N; y++)
			for (let x = 0; x + 2 < N; x++) {
				let s = bias;
				for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) s += kernel[u * 3 + v] * img[(y + u) * N + x + v];
				if (s > best) best = s;
			}
		return best;
	});
}

/** Index of the smallest value in channel c of a [C, H, W] tensor, as a cell. */
export function argminCell(data: ArrayLike<number>, c: number, h: number, w: number): { y: number; x: number } {
	let best = Infinity;
	let bi = 0;
	const o = c * h * w;
	for (let i = 0; i < h * w; i++)
		if (data[o + i] < best) {
			best = data[o + i];
			bi = i;
		}
	return { y: Math.floor(bi / w), x: bi % w };
}

/** Fraction of negative values in channel c of a [C, H, W] tensor. */
export function negativeShare(data: ArrayLike<number>, c: number, hw: number): number {
	let n = 0;
	for (let i = 0; i < hw; i++) if (data[c * hw + i] < 0) n++;
	return n / hw;
}

/** Conv 2-2 channels whose precomputed top test drawings show one clear part (checked in channels.json). */
export const PART_CHANNELS = [
	{ c: 2, name: { en: 'round parts (wheels)', zh: '圆形部件（车轮）' } },
	{ c: 6, name: { en: 'roof-like corners', zh: '屋顶似的尖角' } },
	{ c: 14, name: { en: 'sun rays', zh: '太阳光芒' } }
] as const;
