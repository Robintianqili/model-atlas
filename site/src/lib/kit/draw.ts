/**
 * Stroke model shared by DrawPad and the model pages.
 *
 * Coordinates are normalized to the square pad: x, y in [0, 1] (0,0 = top-left).
 * `size` is the brush diameter as a fraction of the pad side, so strokes render the same
 * at any resolution. Pressure (pen) is optional and scales the width (0.5–1.5×).
 *
 * The pad shows dark ink on light paper. Exported images are ALWAYS black ink (0) on a
 * white background (255), independent of the theme. `toInk()` gives ink intensity in [0, 1]
 * (1 = full ink), which is what most drawing models want after their own preprocessing.
 */

export interface StrokePoint {
	x: number;
	y: number;
	/** pen pressure 0..1 (0.5 for mouse/touch) */
	p?: number;
}

export interface Stroke {
	points: StrokePoint[];
	/** brush diameter as a fraction of the pad side */
	size: number;
}

/** Draw strokes onto a 2D context of a `px` × `px` square. Does not clear. */
export function paintStrokes(
	ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
	strokes: readonly Stroke[],
	px: number,
	ink = '#000'
): void {
	ctx.save();
	ctx.lineCap = 'round';
	ctx.lineJoin = 'round';
	ctx.strokeStyle = ink;
	ctx.fillStyle = ink;
	for (const s of strokes) {
		const pts = s.points;
		if (pts.length === 0) continue;
		if (pts.length === 1) {
			const r = (s.size * px * widthFactor(pts[0].p)) / 2;
			ctx.beginPath();
			ctx.arc(pts[0].x * px, pts[0].y * px, r, 0, Math.PI * 2);
			ctx.fill();
			continue;
		}
		for (let i = 1; i < pts.length; i++) {
			const a = pts[i - 1];
			const b = pts[i];
			ctx.lineWidth = s.size * px * widthFactor(((a.p ?? 0.5) + (b.p ?? 0.5)) / 2);
			ctx.beginPath();
			ctx.moveTo(a.x * px, a.y * px);
			ctx.lineTo(b.x * px, b.y * px);
			ctx.stroke();
		}
	}
	ctx.restore();
}

function widthFactor(p: number | undefined): number {
	return 0.5 + (p ?? 0.5);
}

/** Render strokes into a fresh `size` × `size` ImageData (black ink on white). Browser only. */
export function rasterize(strokes: readonly Stroke[], size: number): ImageData {
	const canvas =
		typeof OffscreenCanvas !== 'undefined'
			? new OffscreenCanvas(size, size)
			: Object.assign(document.createElement('canvas'), { width: size, height: size });
	const ctx = canvas.getContext('2d', { willReadFrequently: true }) as
		| CanvasRenderingContext2D
		| OffscreenCanvasRenderingContext2D;
	ctx.fillStyle = '#fff';
	ctx.fillRect(0, 0, size, size);
	paintStrokes(ctx, strokes, size, '#000');
	return ctx.getImageData(0, 0, size, size);
}

/** Ink intensity per pixel in [0, 1] from an ImageData (1 - luminance), row-major [H, W]. */
export function toInk(img: ImageData): Float32Array {
	const out = new Float32Array(img.width * img.height);
	const d = img.data;
	for (let i = 0; i < out.length; i++) {
		const r = d[i * 4];
		const g = d[i * 4 + 1];
		const b = d[i * 4 + 2];
		out[i] = 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 255;
	}
	return out;
}

/** Bounding box of all stroke points in normalized coordinates (null if empty). */
export function strokeBounds(strokes: readonly Stroke[]): { x0: number; y0: number; x1: number; y1: number } | null {
	let x0 = Infinity;
	let y0 = Infinity;
	let x1 = -Infinity;
	let y1 = -Infinity;
	for (const s of strokes)
		for (const p of s.points) {
			x0 = Math.min(x0, p.x);
			y0 = Math.min(y0, p.y);
			x1 = Math.max(x1, p.x);
			y1 = Math.max(y1, p.y);
		}
	return x0 === Infinity ? null : { x0, y0, x1, y1 };
}

/** Build strokes from simple polylines given in a 0..100 coordinate box (handy for presets). */
export function polylines(lines: number[][], size = 0.06): Stroke[] {
	return lines.map((flat) => {
		const points: StrokePoint[] = [];
		for (let i = 0; i + 1 < flat.length; i += 2) points.push({ x: flat[i] / 100, y: flat[i + 1] / 100 });
		return { points, size };
	});
}
