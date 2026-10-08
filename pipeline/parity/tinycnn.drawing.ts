// Drawing-path check for TinyCNN: does preprocessDrawing keep accuracy when a Quick, Draw! test drawing
// arrives the way a drawing pad would deliver it (a 280x280 canvas, brush-like soft strokes, any size/position)?
// Run from the repo root: node pipeline/parity/tinycnn.drawing.ts
// Needs pipeline/tinycnn/out/base_test_sample.{bin,json} (python dump_samples.py).
// Writes the result into spec.training.metrics.drawingPath.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { load, preprocessDrawing, preprocessStrokes, preprocessPhoto } from '../../site/src/lib/models/tinycnn.ts';
import type { FetchLike, TinyCNN } from '../../site/src/lib/models/tinycnn.ts';

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, '..', '..', 'site', 'static', 'models', 'tinycnn');
const fsFetch: FetchLike = async (url: string) => {
	const buf = readFileSync(url);
	return {
		ok: true,
		json: async () => JSON.parse(buf.toString('utf8')),
		arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
	};
};
const model: TinyCNN = await load(fsFetch, dir);
const bin = readFileSync(join(here, '..', 'tinycnn', 'out', 'base_test_sample.bin'));
const meta = JSON.parse(readFileSync(join(here, '..', 'tinycnn', 'out', 'base_test_sample.json'), 'utf8'));
const labels: number[] = meta.labels;
const N = labels.length;

// deterministic PRNG
let seed = 12345;
const rand = () => {
	seed = (seed + 0x6d2b79f5) | 0;
	let t = seed;
	t = Math.imul(t ^ (t >>> 15), t | 1);
	t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
	return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

let seed2 = 777;
const rand2 = () => {
	seed2 = (seed2 + 0x6d2b79f5) | 0;
	let t = seed2;
	t = Math.imul(t ^ (t >>> 15), t | 1);
	t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
	return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const S = 280;

function gaussBlur(a: Float32Array, w: number, h: number, sigma: number): Float32Array {
	if (sigma <= 0) return a;
	const r = Math.ceil(3 * sigma);
	const k = new Float32Array(2 * r + 1);
	let ks = 0;
	for (let i = -r; i <= r; i++) ks += k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
	for (let i = 0; i < k.length; i++) k[i] /= ks;
	const tmp = new Float32Array(w * h);
	const out = new Float32Array(w * h);
	for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
		let s = 0;
		for (let i = -r; i <= r; i++) { const xx = x + i; if (xx >= 0 && xx < w) s += k[i + r] * a[y * w + xx]; }
		tmp[y * w + x] = s;
	}
	for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
		let s = 0;
		for (let i = -r; i <= r; i++) { const yy = y + i; if (yy >= 0 && yy < h) s += k[i + r] * tmp[yy * w + x]; }
		out[y * w + x] = s;
	}
	return out;
}

function dilate(a: Float32Array, w: number, h: number, r: number): Float32Array {
	if (r <= 0) return a;
	const tmp = new Float32Array(w * h);
	const out = new Float32Array(w * h);
	for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
		let m = 0;
		for (let i = -r; i <= r; i++) { const xx = x + i; if (xx >= 0 && xx < w && a[y * w + xx] > m) m = a[y * w + xx]; }
		tmp[y * w + x] = m;
	}
	for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
		let m = 0;
		for (let i = -r; i <= r; i++) { const yy = y + i; if (yy >= 0 && yy < h && tmp[yy * w + x] > m) m = tmp[yy * w + x]; }
		out[y * w + x] = m;
	}
	return out;
}

/** Render a 28x28 bitmap (0..255) onto an S x S canvas: bilinear upscale to D x D at (ox, oy). */
function renderCanvas(img: Uint8Array, D: number, ox: number, oy: number): Float32Array {
	const c = new Float32Array(S * S);
	const sc = 28 / D;
	const px = (yy: number, xx: number) => (yy < 0 || yy > 27 || xx < 0 || xx > 27 ? 0 : img[yy * 28 + xx] / 255);
	for (let v = Math.max(0, Math.floor(oy)); v < Math.min(S, Math.ceil(oy + D)); v++) {
		for (let u = Math.max(0, Math.floor(ox)); u < Math.min(S, Math.ceil(ox + D)); u++) {
			const sx = (u + 0.5 - ox) * sc - 0.5;
			const sy = (v + 0.5 - oy) * sc - 0.5;
			const x0 = Math.floor(sx), y0 = Math.floor(sy);
			const fx = sx - x0, fy = sy - y0;
			c[v * S + u] = (1 - fy) * ((1 - fx) * px(y0, x0) + fx * px(y0, x0 + 1)) + fy * ((1 - fx) * px(y0 + 1, x0) + fx * px(y0 + 1, x0 + 1));
		}
	}
	return c;
}

// bg/ink: opaque pad colours (RGB). brush: a fixed brush width in canvas px (the stroke is binarised and
// thickened to that width, like a real pad), instead of strokes that scale with the drawing size.
type RGB = [number, number, number];
type Mode = { name: string; dMin: number; dMax: number; blur: number; dil: number; opaqueDark: boolean; bg?: RGB; ink?: RGB; brush?: number };
const modes: Mode[] = [
	{ name: 'soft brush, alpha canvas', dMin: 140, dMax: 260, blur: 1.5, dil: 0, opaqueDark: false },
	{ name: 'dark ink on white (opaque)', dMin: 140, dMax: 260, blur: 1.5, dil: 0, opaqueDark: true },
	{ name: 'thick brush (+8 px)', dMin: 140, dMax: 260, blur: 1.5, dil: 4, opaqueDark: false },
	{ name: 'small drawing (60-100 px, stroke scaled with the drawing)', dMin: 60, dMax: 100, blur: 1.0, dil: 0, opaqueDark: false },
	// added by the verification pass (notes/tinycnn.md "Verification")
	{ name: 'dark-mode pad: light ink on #1e1e1e (opaque)', dMin: 140, dMax: 260, blur: 1.5, dil: 0, opaqueDark: false, bg: [30, 30, 30], ink: [240, 240, 240] },
	{ name: 'cream pad: blue ink on #faf5e1 (opaque)', dMin: 140, dMax: 260, blur: 1.5, dil: 0, opaqueDark: false, bg: [250, 245, 225], ink: [20, 40, 160] },
	{ name: 'fixed 15 px brush, drawing 140-260 px', dMin: 140, dMax: 260, blur: 1.0, dil: 0, opaqueDark: false, brush: 15 },
	{ name: 'fixed 15 px brush, small drawing (60-100 px)', dMin: 60, dMax: 100, blur: 1.0, dil: 0, opaqueDark: false, brush: 15 }
];

function accuracy(inputs: (i: number) => Float32Array): { acc: number; perClass: number[] } {
	let ok = 0;
	const pc = new Array(10).fill(0), pn = new Array(10).fill(0);
	for (let i = 0; i < N; i++) {
		const { top } = model.predict(inputs(i));
		pn[labels[i]]++;
		if (top === labels[i]) { ok++; pc[labels[i]]++; }
	}
	return { acc: ok / N, perClass: pc.map((v, k) => v / pn[k]) };
}

const raw = accuracy((i) => Float32Array.from(bin.subarray(i * 784, i * 784 + 784), (v) => v / 255));
console.log(`raw bitmaps: ${(raw.acc * 100).toFixed(2)}% (n=${N})`);
// identity check: framing a raw bitmap again should change it very little
let reframeDiff = 0;
for (let i = 0; i < 200; i++) {
	const x = Float32Array.from(bin.subarray(i * 784, i * 784 + 784), (v) => v / 255);
	const p = preprocessDrawing({ ink: x, width: 28, height: 28 }, { threshold: 0 }).input;
	let d = 0;
	for (let j = 0; j < 784; j++) d += Math.abs(p[j] - x[j]);
	reframeDiff += d / 784;
}
console.log(`re-framing a 28x28 bitmap: mean |diff| per pixel = ${(reframeDiff / 200).toFixed(4)}`);

const results: Record<string, number> = { raw: Number(raw.acc.toFixed(4)) };
for (const m of modes) {
	const t0 = performance.now();
	const r = accuracy((i) => {
		const img = bin.subarray(i * 784, i * 784 + 784);
		// the added modes draw from their own PRNG so the original modes and the photo check keep their numbers
		const rr = m.bg || m.brush ? rand2 : rand;
		const D = m.dMin + rr() * (m.dMax - m.dMin);
		const ox = rr() * (S - D), oy = rr() * (S - D);
		let c = renderCanvas(img, D, ox, oy);
		if (m.brush) {
			// dataset strokes are ~1.5/26 of the drawing's long side: at D px they are ~1.5*D/26 px wide
			for (let j = 0; j < c.length; j++) c[j] = c[j] > 0.35 ? 1 : 0;
			c = dilate(c, S, S, Math.max(0, Math.round((m.brush - (1.5 * D) / 26) / 2)));
		} else c = dilate(c, S, S, m.dil);
		c = gaussBlur(c, S, S, m.blur);
		const rgba = new Uint8ClampedArray(S * S * 4);
		for (let j = 0; j < S * S; j++) {
			const a = Math.round(Math.min(1, c[j]) * 255);
			if (m.bg && m.ink) {
				for (let k = 0; k < 3; k++) rgba[4 * j + k] = Math.round((m.bg[k] * (255 - a) + m.ink[k] * a) / 255);
				rgba[4 * j + 3] = 255;
			} else if (m.opaqueDark) { rgba[4 * j] = rgba[4 * j + 1] = rgba[4 * j + 2] = 255 - a; rgba[4 * j + 3] = 255; }
			else { rgba[4 * j] = rgba[4 * j + 1] = rgba[4 * j + 2] = 20; rgba[4 * j + 3] = a; }
		}
		return preprocessDrawing({ data: rgba, width: S, height: S }).input;
	});
	const ms = (performance.now() - t0) / N;
	console.log(`${m.name}: ${(r.acc * 100).toFixed(2)}%  (${ms.toFixed(1)} ms per drawing incl. rendering)`);
	results[m.name] = Number(r.acc.toFixed(4));
}

// photo path: pencil-grey strokes on uneven, noisy, warm paper in a 480x360 (non-square) photo -> preprocessPhoto
{
	const PW = 480, PH = 360;
	const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
	const t0 = performance.now();
	let inverted = 0;
	const r = accuracy((i) => {
		const img = bin.subarray(i * 784, i * 784 + 784);
		const D = 150 + rand() * 150;
		const ox = rand() * (PW - D), oy = rand() * (PH - D);
		// render on a PW x PW square, then keep the top PH rows (the drawing fits inside them)
		const sq = new Float32Array(PW * PW);
		const sc = 28 / D;
		const px = (yy: number, xx: number) => (yy < 0 || yy > 27 || xx < 0 || xx > 27 ? 0 : img[yy * 28 + xx] / 255);
		for (let v = Math.floor(oy); v < Math.min(PH, Math.ceil(oy + D)); v++) {
			for (let u = Math.floor(ox); u < Math.min(PW, Math.ceil(ox + D)); u++) {
				const sx = (u + 0.5 - ox) * sc - 0.5, sy = (v + 0.5 - oy) * sc - 0.5;
				const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
				sq[v * PW + u] = (1 - fy) * ((1 - fx) * px(y0, x0) + fx * px(y0, x0 + 1)) + fy * ((1 - fx) * px(y0 + 1, x0) + fx * px(y0 + 1, x0 + 1));
			}
		}
		const c = gaussBlur(sq.subarray(0, PW * PH), PW, PH, 1.5);
		const rgba = new Uint8ClampedArray(PW * PH * 4);
		for (let v = 0; v < PH; v++) {
			for (let u = 0; u < PW; u++) {
				const j = v * PW + u;
				const bg = 0.8 + 0.08 * (u / PW - 0.5) + 0.05 * (v / PH - 0.5) + 0.03 * gauss();
				const lum = bg * (1 - 0.7 * Math.min(1, c[j]));
				rgba[4 * j] = lum * 255 * 1.03;
				rgba[4 * j + 1] = lum * 255;
				rgba[4 * j + 2] = lum * 255 * 0.93;
				rgba[4 * j + 3] = 255;
			}
		}
		const p = preprocessPhoto({ data: rgba, width: PW, height: PH });
		if (p.inverted) inverted++;
		return p.input;
	});
	const ms = (performance.now() - t0) / N;
	const name = 'photo of a sketch on paper (480x360, noisy, uneven light)';
	console.log(`${name}: ${(r.acc * 100).toFixed(2)}%  (${ms.toFixed(1)} ms per photo incl. rendering; inverted ${inverted}/${N})`);
	results[name] = Number(r.acc.toFixed(4));
}

// preprocessStrokes sanity: simple hand-made shapes should land on their class
const circle = (cx: number, cy: number, r: number, n = 40): [number, number][] =>
	Array.from({ length: n + 1 }, (_, i) => [cx + r * Math.cos((2 * Math.PI * i) / n), cy + r * Math.sin((2 * Math.PI * i) / n)]);
const starPts: [number, number][] = Array.from({ length: 6 }, (_, i) => {
	const a = -Math.PI / 2 + (i * 4 * Math.PI) / 5;
	return [150 + 110 * Math.cos(a), 150 + 110 * Math.sin(a)];
});
const sunRays: [number, number][][] = Array.from({ length: 8 }, (_, i) => {
	const a = (i * Math.PI) / 4;
	return [[150 + 70 * Math.cos(a), 150 + 70 * Math.sin(a)], [150 + 120 * Math.cos(a), 150 + 120 * Math.sin(a)]];
});
const shapes: Record<string, [number, number][][]> = {
	star: [starPts],
	sun: [circle(150, 150, 55), ...sunRays],
	house: [[[60, 260], [60, 140], [240, 140], [240, 260], [60, 260]], [[50, 145], [150, 50], [250, 145]], [[130, 260], [130, 200], [170, 200], [170, 260]]],
	clock: [circle(150, 150, 110), [[150, 150], [150, 80]], [[150, 150], [200, 170]]],
	umbrella: [[...circle(150, 150, 110, 40).slice(20, 41), [40, 150]], [[150, 150], [150, 250], [130, 260], [115, 245]]],
	apple: [circle(150, 165, 95), [[150, 70], [160, 30]], [[160, 55], [200, 40], [175, 70], [160, 55]]]
};
const strokeCheck: Record<string, string> = {};
for (const [name, strokes] of Object.entries(shapes)) {
	const a = preprocessStrokes(strokes);
	// the same strokes drawn on a 300x300 raster with a 16 px brush, through preprocessDrawing
	const ras = new Float32Array(300 * 300);
	for (let y = 0; y < 300; y++) for (let x = 0; x < 300; x++) {
		let hit = false;
		for (const s of strokes) {
			for (let i = 1; i < s.length && !hit; i++) {
				const [ax, ay] = s[i - 1], [bx, by] = s[i];
				const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
				let t = L ? ((x + 0.5 - ax) * dx + (y + 0.5 - ay) * dy) / L : 0;
				t = Math.max(0, Math.min(1, t));
				const ex = ax + t * dx - x - 0.5, ey = ay + t * dy - y - 0.5;
				if (ex * ex + ey * ey <= 64) hit = true;
			}
			if (hit) break;
		}
		ras[y * 300 + x] = hit ? 1 : 0;
	}
	const b = preprocessDrawing({ ink: gaussBlur(ras, 300, 300, 1), width: 300, height: 300 });
	const pa = model.predict(a.input), pb = model.predict(b.input);
	let d = 0;
	for (let j = 0; j < 784; j++) d += Math.abs(a.input[j] - b.input[j]);
	const cls = model.spec.classes;
	strokeCheck[name] = `strokes -> ${cls[pa.top].en} ${pa.probs[pa.top].toFixed(2)}; raster -> ${cls[pb.top].en} ${pb.probs[pb.top].toFixed(2)}; mean|diff| ${(d / 784).toFixed(3)}`;
	console.log(`${name}: ${strokeCheck[name]}`);
}

const specPath = join(dir, 'spec.json');
const spec = JSON.parse(readFileSync(specPath, 'utf8'));
spec.training.metrics.drawingPath = {
	note: '1,500 Quick, Draw! test drawings (150 per class) rendered onto a simulated 280x280 drawing pad (bilinear upscale to a random size and position, Gaussian blur, optional dilation) and passed through preprocessDrawing. Unless a mode says "fixed 15 px brush", the stroke width grows with the drawing size; with a real fixed brush, small drawings get thick strokes and accuracy drops (use preprocessStrokes for a drawing pad: it renders at the dataset stroke width). Script: pipeline/parity/tinycnn.drawing.ts',
	accuracy: results,
	checked: new Date().toLocaleDateString('sv-SE')
};
writeFileSync(specPath, JSON.stringify(spec, null, 1));
