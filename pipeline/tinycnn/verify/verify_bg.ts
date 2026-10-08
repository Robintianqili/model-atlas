// Background-colour check for preprocessDrawing on opaque canvases (verification stream).
// 750 Quick, Draw! test drawings rendered onto a 280x280 opaque pad with a non-pure background colour.
// Run from the repo root: node pipeline/tinycnn/verify/verify_bg.ts
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { load, preprocessDrawing } from '../../../site/src/lib/models/tinycnn.ts';
import type { FetchLike } from '../../../site/src/lib/models/tinycnn.ts';

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, '..', '..', '..', 'site', 'static', 'models', 'tinycnn');
const fsFetch: FetchLike = async (url: string) => {
	const buf = readFileSync(url);
	return { ok: true, json: async () => JSON.parse(buf.toString('utf8')), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer };
};
const model = await load(fsFetch, dir);
const bin = readFileSync(join(here, '..', 'out', 'base_test_sample.bin'));
const labels: number[] = JSON.parse(readFileSync(join(here, '..', 'out', 'base_test_sample.json'), 'utf8')).labels;
const S = 280;
let seed = 99;
const rand = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296; };
function render(img: Uint8Array, D: number, ox: number, oy: number): Float32Array {
	const c = new Float32Array(S * S), sc = 28 / D;
	const px = (yy: number, xx: number) => (yy < 0 || yy > 27 || xx < 0 || xx > 27 ? 0 : img[yy * 28 + xx] / 255);
	for (let v = Math.floor(oy); v < Math.min(S, Math.ceil(oy + D)); v++) for (let u = Math.floor(ox); u < Math.min(S, Math.ceil(ox + D)); u++) {
		const sx = (u + 0.5 - ox) * sc - 0.5, sy = (v + 0.5 - oy) * sc - 0.5;
		const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
		c[v * S + u] = (1 - fy) * ((1 - fx) * px(y0, x0) + fx * px(y0, x0 + 1)) + fy * ((1 - fx) * px(y0 + 1, x0) + fx * px(y0 + 1, x0 + 1));
	}
	return c;
}
const pads: { name: string; bg: [number, number, number]; ink: [number, number, number] }[] = [
	{ name: 'pure white bg, black ink', bg: [255, 255, 255], ink: [0, 0, 0] },
	{ name: 'cream bg #faf5e1, blue ink', bg: [250, 245, 225], ink: [20, 40, 160] },
	{ name: 'light grey bg #f0f0f0, dark ink #222', bg: [240, 240, 240], ink: [34, 34, 34] },
	{ name: 'dark-mode bg #1e1e1e, light ink #f0f0f0', bg: [30, 30, 30], ink: [240, 240, 240] },
	{ name: 'dark-mode bg #121826, cyan ink #7dd3fc', bg: [18, 24, 38], ink: [125, 211, 252] }
];
for (const p of pads) {
	let ok = 0, n = 0;
	seed = 99;
	for (let i = 0; i < labels.length; i += 2) {
		const D = 140 + rand() * 120, ox = rand() * (S - D), oy = rand() * (S - D);
		const c = render(bin.subarray(i * 784, (i + 1) * 784), D, ox, oy);
		const d = new Uint8ClampedArray(S * S * 4);
		for (let j = 0; j < S * S; j++) {
			const a = Math.min(1, c[j]);
			for (let k = 0; k < 3; k++) d[4 * j + k] = Math.round(p.bg[k] * (1 - a) + p.ink[k] * a);
			d[4 * j + 3] = 255;
		}
		if (model.predict(preprocessDrawing({ data: d, width: S, height: S }).input).top === labels[i]) ok++;
		n++;
	}
	console.log(`${p.name.padEnd(42)} ${((ok / n) * 100).toFixed(2)}% (n=${n})`);
}
