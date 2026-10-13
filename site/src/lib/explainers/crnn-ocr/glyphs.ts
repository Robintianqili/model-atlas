// crnn-ocr: handwriting-like digits as pen strokes, and the renderer that turns any strokes (the learner's, a preset's,
// a training word's) into the model's input, a strip of 16 × 64 ink values. One code path for the page and for the
// training data (pipeline/crnn-ocr/make_data.ts writes what render() returns), so PyTorch trains on the page's bytes.
//   word(text, rng)   strokes for a string of digits (and the letters A E H X, for the limits preset): each character
//                     from a template with jittered shape, size, slant, tilt and spacing; y points down, height ≈ 1
//   render(strokes)   crop to the ink, scale so the ink is 13 pixels tall (or the word fits 61 pixels across), centre,
//                     draw every segment with a round pen of width LW, 4 × 4 samples per pixel: ink = hits / 16
import type { Rng } from '../../atlas/types/index.ts';

export const H = 16;
export const W = 64;
/** the page's pen width in pixels (training words use 1.2 to 2.2) */
export const LW = 1.7;
/** the model's symbols: class 0 is the CTC blank, class d + 1 is digit d */
export const SYMBOLS = ['–', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
export type Pt = { x: number; y: number };
type J = (v: number, a?: number) => number;

const P = (...xy: number[]): Pt[] => Array.from({ length: xy.length / 2 }, (_, i) => ({ x: xy[2 * i], y: xy[2 * i + 1] }));
/** an elliptic arc from angle a0 to a1 (degrees; 0 = right, 90 = down) */
function arc(cx: number, cy: number, rx: number, ry: number, a0: number, a1: number): Pt[] {
	const n = Math.max(6, Math.ceil(Math.abs(a1 - a0) / 12));
	return Array.from({ length: n + 1 }, (_, i) => {
		const a = ((a0 + ((a1 - a0) * i) / n) * Math.PI) / 180;
		return { x: cx + rx * Math.cos(a), y: cy + ry * Math.sin(a) };
	});
}

/** templates: strokes in a box of height 1 and the given width; v picks a variant (0 to 2) */
const GLYPH: Record<string, { w: number; s: (j: J, v: number) => Pt[][] }> = {
	'0': { w: 0.56, s: (j) => [arc(0.28, 0.5, j(0.26), j(0.49, 0.02), j(-95, 15), j(275, 20))] },
	'1': {
		w: 0.34,
		s: (j, v) => [
			[...(v > 0 ? P(j(0.02), j(0.24)) : []), ...P(j(0.2, 0.03), 0, j(0.17, 0.03), 1)],
			...(v > 1 ? [P(j(0), 1, j(0.34), 1)] : [])
		]
	},
	'2': { w: 0.58, s: (j) => [[...arc(0.28, j(0.27), j(0.25), j(0.25), j(-165, 15), j(25, 15)), ...P(j(0.02), 1, j(0.58), j(1, 0.03))]] },
	'3': { w: 0.56, s: (j) => [[...arc(0.27, 0.25, j(0.24), 0.24, j(-150, 15), 90), ...arc(0.27, 0.74, j(0.28), 0.25, -90, j(150, 15))]] },
	'4': {
		w: 0.62,
		s: (j, v) =>
			v > 0
				? [P(j(0.1), 0, j(0.02), j(0.62), j(0.62), 0.62), P(j(0.44), j(0.25), j(0.44), 1)]
				: [P(j(0.44), 1, 0.44, 0, j(0), j(0.66), j(0.62), 0.66)]
	},
	'5': {
		w: 0.58,
		s: (j, v) => {
			const body = [...P(j(0.1), 0, j(0.06), j(0.43)), ...arc(0.28, 0.69, j(0.28), j(0.3, 0.02), -125, j(145, 15))];
			return v > 0 ? [body, P(j(0.1), 0, j(0.56), j(0))] : [[...P(j(0.56), j(0)), ...body]];
		}
	},
	'6': { w: 0.56, s: (j) => [[...P(j(0.46), 0, j(0.2), j(0.25)), ...arc(0.29, 0.72, j(0.25), j(0.26, 0.02), 180, j(-150, 15))]] },
	'7': { w: 0.6, s: (j, v) => [P(j(0.02), j(0.02), j(0.6), 0, j(0.22), 1), ...(v > 1 ? [P(j(0.16), j(0.52), j(0.5), j(0.5))] : [])] },
	'8': {
		w: 0.56,
		s: (j) => [arc(0.28, 0.25, j(0.21), j(0.24, 0.02), j(-90, 10), j(270, 10)), arc(0.28, 0.74, j(0.26), j(0.25, 0.02), j(-90, 10), j(270, 10))]
	},
	'9': { w: 0.56, s: (j) => [[...arc(0.28, 0.28, j(0.25), j(0.27, 0.02), j(10, 10), j(-355, 10)), ...P(j(0.53), 0.3, j(0.44), 1)]] },
	// letters, never trained on: only for the limits preset
	A: { w: 0.56, s: () => [P(0, 1, 0.28, 0, 0.56, 1), P(0.12, 0.62, 0.44, 0.62)] },
	E: { w: 0.5, s: () => [P(0.5, 0, 0, 0, 0, 1, 0.5, 1), P(0, 0.5, 0.4, 0.5)] },
	H: { w: 0.54, s: () => [P(0, 0, 0, 1), P(0.54, 0, 0.54, 1), P(0, 0.5, 0.54, 0.5)] },
	X: { w: 0.56, s: () => [P(0, 0, 0.56, 1), P(0.56, 0, 0, 1)] }
};
export const KNOWN = Object.keys(GLYPH).join('');

/** Strokes for `text`, every shape choice drawn from `rng` (the same rng state gives the same strokes); `wild` scales
 *  every random change (1 for the presets; training words use 1 to 2.2). */
export function word(text: string, rng: Rng, wild = 1): Pt[][] {
	const u = (a: number, b: number) => a + (b - a) * rng.next();
	const j: J = (v, a = 0.04) => v + (rng.next() * 2 - 1) * a * wild;
	const out: Pt[][] = [];
	let x = 0;
	for (const ch of text) {
		const g = GLYPH[ch];
		if (!g) throw new Error(`crnn-ocr: no template for '${ch}' (known: ${KNOWN})`);
		const strokes = g.s(j, rng.int(3));
		const hs = 1 - u(0, 0.2) * wild;
		const ws = hs * u(0.85, 1.2);
		const slant = 0.09 + u(-0.21, 0.21) * wild;
		const tilt = u(-0.07, 0.07) * wild;
		const dy = u(-0.05, 0.05) * wild;
		const [c, s] = [Math.cos(tilt), Math.sin(tilt)];
		for (const st of strokes)
			out.push(
				st.map((p) => {
					const px = (p.x - g.w / 2) * ws + slant * (0.5 - p.y) * hs;
					const py = (p.y - 0.5) * hs;
					return { x: x + (g.w * ws) / 2 + c * px - s * py + j(0, 0.012), y: 0.5 + dy + s * px + c * py + j(0, 0.012) };
				})
			);
		x += g.w * ws + u(-0.02, 0.4);
	}
	return out;
}

function dist2(qx: number, qy: number, ax: number, ay: number, bx: number, by: number): number {
	const dx = bx - ax;
	const dy = by - ay;
	const l2 = dx * dx + dy * dy;
	const t = l2 > 0 ? Math.max(0, Math.min(1, ((qx - ax) * dx + (qy - ay) * dy) / l2)) : 0;
	const ex = ax + t * dx - qx;
	const ey = ay + t * dy - qy;
	return ex * ex + ey * ey;
}

export type Rendered = { ink: Float32Array; empty: boolean; scale: number; inkCells: number };
/** The [16, 64] ink strip of any strokes (row-major; 0 = paper, 1 = full ink; every value is a multiple of 1/16). */
export function render(strokes: Pt[][], lw = LW): Rendered {
	const ink = new Float32Array(H * W);
	const lines = strokes.filter((s) => s.length > 0);
	if (!lines.length) return { ink, empty: true, scale: NaN, inkCells: 0 };
	let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
	for (const s of lines) for (const p of s) [x0, y0, x1, y1] = [Math.min(x0, p.x), Math.min(y0, p.y), Math.max(x1, p.x), Math.max(y1, p.y)];
	const sc = Math.min((H - 3 - lw) / Math.max(y1 - y0, 1e-6), (W - 3 - lw) / Math.max(x1 - x0, 1e-6));
	const ox = W / 2 - ((x0 + x1) / 2) * sc;
	const oy = H / 2 - ((y0 + y1) / 2) * sc;
	const S = 4;
	const [GW, GH] = [W * S, H * S];
	const hit = new Uint8Array(GW * GH);
	const r = lw / 2;
	for (const s of lines)
		for (let i = 0; i < Math.max(1, s.length - 1); i++) {
			const a = s[i];
			const b = s[Math.min(i + 1, s.length - 1)];
			const [ax, ay, bx, by] = [a.x * sc + ox, a.y * sc + oy, b.x * sc + ox, b.y * sc + oy];
			const gx0 = Math.max(0, Math.floor((Math.min(ax, bx) - r) * S));
			const gx1 = Math.min(GW - 1, Math.ceil((Math.max(ax, bx) + r) * S));
			const gy0 = Math.max(0, Math.floor((Math.min(ay, by) - r) * S));
			const gy1 = Math.min(GH - 1, Math.ceil((Math.max(ay, by) + r) * S));
			for (let gy = gy0; gy <= gy1; gy++)
				for (let gx = gx0; gx <= gx1; gx++) {
					const k = gy * GW + gx;
					if (!hit[k] && dist2((gx + 0.5) / S, (gy + 0.5) / S, ax, ay, bx, by) <= r * r) hit[k] = 1;
				}
		}
	let inkCells = 0;
	for (let py = 0; py < H; py++)
		for (let px = 0; px < W; px++) {
			let n = 0;
			for (let sy = 0; sy < S; sy++) for (let sx = 0; sx < S; sx++) n += hit[(py * S + sy) * GW + px * S + sx];
			ink[py * W + px] = n / (S * S);
			if (n) inkCells++;
		}
	return { ink, empty: false, scale: sc, inkCells };
}

/** Strokes fitted into the square drawing pad (0..1), so a preset shows on the pad and can be drawn over. */
export function toPad(strokes: Pt[][]): { points: Pt[] }[] {
	let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
	for (const s of strokes) for (const p of s) [x0, y0, x1, y1] = [Math.min(x0, p.x), Math.min(y0, p.y), Math.max(x1, p.x), Math.max(y1, p.y)];
	const sc = Math.min(0.84 / Math.max(x1 - x0, 1e-6), 0.4 / Math.max(y1 - y0, 1e-6));
	const r4 = (v: number) => Math.round(v * 1e4) / 1e4;
	return strokes.map((s) => ({ points: s.map((p) => ({ x: r4(0.5 + (p.x - (x0 + x1) / 2) * sc), y: r4(0.5 + (p.y - (y0 + y1) / 2) * sc) })) }));
}
