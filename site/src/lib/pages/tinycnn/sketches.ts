/**
 * Hand-made vector sketches for the TinyCNN page (our own drawings, not dataset data).
 * Coordinates are normalized pad units (0..1, y down), the format DrawPad uses.
 * The page runs them through the same preprocessing as anything the learner draws.
 */
import type { Stroke, StrokePoint } from '$lib/kit/draw';

const BRUSH = 0.055;

/** A slightly wobbly circle so it looks drawn by hand (deterministic). */
function circle(cx: number, cy: number, r: number, n = 44, wob = 0.025, from = 0, to = 2 * Math.PI): StrokePoint[] {
	const pts: StrokePoint[] = [];
	for (let i = 0; i <= n; i++) {
		const a = from + ((to - from) * i) / n - Math.PI / 2;
		const rr = r * (1 + wob * Math.sin(3 * a + 0.7) + wob * 0.6 * Math.cos(5 * a));
		pts.push({ x: cx + rr * Math.cos(a), y: cy + rr * Math.sin(a), p: 0.5 });
	}
	return pts;
}

function line(...xy: number[]): StrokePoint[] {
	const pts: StrokePoint[] = [];
	for (let i = 0; i + 3 < xy.length; i += 2) {
		const [x0, y0, x1, y1] = [xy[i], xy[i + 1], xy[i + 2], xy[i + 3]];
		const n = Math.max(2, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 0.02));
		for (let k = i === 0 ? 0 : 1; k <= n; k++) pts.push({ x: x0 + ((x1 - x0) * k) / n, y: y0 + ((y1 - y0) * k) / n, p: 0.5 });
	}
	return pts;
}

const S = (points: StrokePoint[]): Stroke => ({ points, size: BRUSH });

// ---- the sun -> tree experiment (one geometry, three stages)
const SUN_C = { x: 0.5, y: 0.3, r: 0.14 };
const sunDisc = S(circle(SUN_C.x, SUN_C.y, SUN_C.r));
const sunRays: Stroke[] = Array.from({ length: 8 }, (_, i) => {
	const a = (i / 8) * 2 * Math.PI + 0.1;
	const r0 = 0.19;
	const r1 = 0.27;
	return S(line(SUN_C.x + r0 * Math.cos(a), SUN_C.y + r0 * Math.sin(a), SUN_C.x + r1 * Math.cos(a), SUN_C.y + r1 * Math.sin(a)));
});
const trunk: Stroke[] = [S(line(0.465, 0.44, 0.46, 0.9)), S(line(0.535, 0.44, 0.54, 0.9))];

export const SUN: Stroke[] = [sunDisc, ...sunRays];
export const SUN_TRUNK: Stroke[] = [sunDisc, ...sunRays, ...trunk];
export const DISC_TRUNK: Stroke[] = [sunDisc, ...trunk];

// ---- a few starting drawings
export const HOUSE: Stroke[] = [
	S(line(0.2, 0.86, 0.2, 0.47, 0.8, 0.47, 0.8, 0.86, 0.2, 0.86)),
	S(line(0.15, 0.49, 0.5, 0.16, 0.85, 0.49)),
	S(line(0.44, 0.86, 0.44, 0.66, 0.56, 0.66, 0.56, 0.86))
];

export const FISH: Stroke[] = [
	S(circle(0.43, 0.5, 0.26, 40, 0.02, 0.25 * Math.PI + Math.PI / 2, 1.75 * Math.PI + Math.PI / 2).map((p) => ({ ...p, y: 0.5 + (p.y - 0.5) * 0.62 }))),
	S(line(0.62, 0.38, 0.86, 0.24, 0.86, 0.76, 0.62, 0.62)),
	S(circle(0.3, 0.45, 0.022, 10, 0))
];

/** Copy strokes (so the pad never mutates the module constants). */
export function cloneStrokes(s: readonly Stroke[]): Stroke[] {
	return s.map((st) => ({ size: st.size, points: st.points.map((p) => ({ ...p })) }));
}

/** Shrink strokes into a corner: the "draw very small" experiment. */
export function shrinkToCorner(s: readonly Stroke[], scale = 0.22, at = { x: 0.06, y: 0.06 }): Stroke[] {
	let x0 = Infinity;
	let y0 = Infinity;
	for (const st of s)
		for (const p of st.points) {
			if (p.x < x0) x0 = p.x;
			if (p.y < y0) y0 = p.y;
		}
	if (x0 === Infinity) return [];
	return s.map((st) => ({
		size: st.size,
		points: st.points.map((p) => ({ ...p, x: at.x + (p.x - x0) * scale, y: at.y + (p.y - y0) * scale }))
	}));
}
