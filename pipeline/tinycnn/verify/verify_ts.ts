// Independent verification of the TinyCNN TypeScript code (verification stream).
// Run from the repo root: node pipeline/tinycnn/verify/verify_ts.ts
// Needs: python pipeline/tinycnn/verify/verify_assets.py (writes out/extra_ref.*) and
//        python pipeline/tinycnn/dump_samples.py (writes pipeline/tinycnn/out/base_test_sample.*).
// Writes out/shapes.json (28x28 outputs for rendering with render_shapes.py). Read-only on site/.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { load, preprocessDrawing, preprocessStrokes, preprocessPhoto, decodePreset } from '../../../site/src/lib/models/tinycnn.ts';
import type { FetchLike, TinyCNN, Preprocessed } from '../../../site/src/lib/models/tinycnn.ts';
import { featuresOf, trainHead, headAccuracy, predictHead } from '../../../site/src/lib/models/tinycnn_finetune.ts';

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, '..', '..', '..', 'site', 'static', 'models', 'tinycnn');
const fsFetch: FetchLike = async (url: string) => {
	const buf = readFileSync(url);
	return {
		ok: true,
		status: 200,
		json: async () => JSON.parse(buf.toString('utf8')),
		arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
	};
};
const model: TinyCNN = await load(fsFetch, dir);
const cls = model.spec.classes.map((c) => c.en);
const problems: string[] = [];
const check = (ok: boolean, msg: string) => {
	console.log((ok ? 'OK   ' : 'FAIL ') + msg);
	if (!ok) problems.push(msg);
};

// ------------------------------------------------ 1. per-layer parity (reference.json + 40 extra inputs)
{
	const ids = model.spec.layers.map((l) => l.id);
	const perLayer: Record<string, number> = Object.fromEntries(ids.map((i) => [i, 0]));
	const ref = JSON.parse(readFileSync(join(dir, 'reference.json'), 'utf8'));
	for (const r of ref.inputs) {
		const tr = model.forward(r.input);
		for (const id of ids) {
			const w = r.layers[id].data as number[];
			const g = tr.t[id].data;
			if (g.length !== w.length) problems.push(`${id} length`);
			for (let i = 0; i < w.length; i++) perLayer[id] = Math.max(perLayer[id], Math.abs(w[i] - g[i]));
		}
	}
	const meta = JSON.parse(readFileSync(join(here, 'out', 'extra_ref.json'), 'utf8'));
	const b = readFileSync(join(here, 'out', 'extra_ref.bin'));
	const all = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
	const nIn = meta.labels.length;
	let correct = 0;
	const perLayerExtra: Record<string, number> = Object.fromEntries(ids.map((i) => [i, 0]));
	for (let n = 0; n < nIn; n++) {
		const x = all.subarray(n * 784, (n + 1) * 784);
		const tr = model.forward(x);
		for (const id of ids) {
			const L = meta.layers[id];
			const want = all.subarray(L.offset + n * L.per, L.offset + (n + 1) * L.per);
			const g = tr.t[id].data;
			if (g.length !== L.per) problems.push(`${id} size ${g.length} vs ${L.per}`);
			for (let i = 0; i < L.per; i++) perLayerExtra[id] = Math.max(perLayerExtra[id], Math.abs(want[i] - g[i]));
		}
		if (model.predict(x).top === meta.labels[n]) correct++;
	}
	console.log('per-layer max |diff|  reference.json(3) | weights.bin-rebuilt PyTorch on 40 extra test drawings');
	for (const id of ids) {
		console.log(`  ${id.padEnd(9)} ${perLayer[id].toExponential(2)} | ${perLayerExtra[id].toExponential(2)}`);
		check(perLayer[id] <= 1e-4 && perLayerExtra[id] <= 1e-4, `layer ${id} <= 1e-4`);
	}
	const sp = model.spec.parity as { max_abs_diff: number; pass: boolean };
	const measured = Math.max(...Object.values(perLayer));
	check(sp.pass === true && Math.abs(sp.max_abs_diff - measured) <= 0.01 * measured + 1e-9,
		`spec.parity.max_abs_diff ${sp.max_abs_diff} matches measured ${measured.toExponential(3)} (reference.json)`);
	console.log(`  extra inputs classified correctly: ${correct}/${nIn}`);
}

// ------------------------------------------------ 2. presets through decodePreset
{
	const pre = JSON.parse(readFileSync(join(dir, 'presets.json'), 'utf8'));
	let ok = 0, tot = 0;
	for (const g of pre.classes) {
		const k = cls.indexOf(g.id);
		for (const it of g.items) {
			const p = model.predict(decodePreset(it.b64));
			tot++;
			if (p.top === k && p.probs[k] > 0.9) ok++;
		}
	}
	check(ok === tot, `base presets via decodePreset: ${ok}/${tot} predicted right with p > 0.9`);
}

// ------------------------------------------------ 3. dataset framing statistics (what preprocess must mimic)
type Stats = { long: number; cy: number; cx: number; max: number; mass: number; border: number; zeros: number; faint: number };
function stats(x: ArrayLike<number>, thr = 0.05): Stats {
	let mx = 0, mass = 0, border = 0, zeros = 0, faint = 0;
	for (let i = 0; i < 784; i++) { if (x[i] > mx) mx = x[i]; mass += x[i]; if (x[i] === 0) zeros++; if (x[i] > 0 && x[i] < 0.08) faint++; }
	let y0 = 28, x0 = 28, y1 = -1, x1 = -1;
	for (let y = 0; y < 28; y++) for (let xx = 0; xx < 28; xx++) {
		const v = x[y * 28 + xx];
		if (y === 0 || y === 27 || xx === 0 || xx === 27) border = Math.max(border, v);
		if (v > thr * mx) { y0 = Math.min(y0, y); y1 = Math.max(y1, y); x0 = Math.min(x0, xx); x1 = Math.max(x1, xx); }
	}
	return { long: Math.max(y1 - y0 + 1, x1 - x0 + 1), cy: (y0 + y1 + 1) / 2, cx: (x0 + x1 + 1) / 2, max: mx, mass, border, zeros: zeros / 784, faint: faint / 784 };
}
const sampleBin = readFileSync(join(here, '..', 'out', 'base_test_sample.bin'));
const sampleMeta = JSON.parse(readFileSync(join(here, '..', 'out', 'base_test_sample.json'), 'utf8'));
const sampleLabels: number[] = sampleMeta.labels;
const NS = sampleLabels.length;
const sample = (i: number) => Float32Array.from(sampleBin.subarray(i * 784, (i + 1) * 784), (v) => v / 255);
{
	const L: number[] = [], M: number[] = [], CY: number[] = [], CX: number[] = [];
	for (let i = 0; i < NS; i++) {
		const s = stats(sample(i));
		L.push(s.long); M.push(s.mass); CY.push(s.cy); CX.push(s.cx);
	}
	const med = (a: number[]) => [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)];
	const frac = (a: number[], f: (v: number) => boolean) => a.filter(f).length / a.length;
	console.log(`dataset (${NS} test drawings): bbox long side median ${med(L)} (==26: ${(frac(L, (v) => v === 26) * 100).toFixed(1)}%), centre median (${med(CY)}, ${med(CX)}), ink mass median ${med(M).toFixed(1)}`);
}

// ------------------------------------------------ 4. synthetic drawn shapes -> preprocessDrawing / preprocessStrokes
type P = [number, number];
const circ = (cx: number, cy: number, rx: number, ry: number, a0 = 0, a1 = 2 * Math.PI, n = 48): P[] =>
	Array.from({ length: n + 1 }, (_, i) => { const a = a0 + ((a1 - a0) * i) / n; return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)] as P; });
// shapes in a unit box (y down); each a list of strokes
const SHAPES: Record<string, P[][]> = {
	fish: [circ(0.42, 0.5, 0.36, 0.2, 0.25 * Math.PI, 1.75 * Math.PI), [[0.7, 0.36], [1.0, 0.18], [1.0, 0.82], [0.7, 0.64]], circ(0.22, 0.45, 0.025, 0.025)],
	house: [[[0.15, 1], [0.15, 0.45], [0.85, 0.45], [0.85, 1], [0.15, 1]], [[0.05, 0.5], [0.5, 0.05], [0.95, 0.5]], [[0.42, 1], [0.42, 0.72], [0.58, 0.72], [0.58, 1]]],
	umbrella: [[...circ(0.5, 0.45, 0.48, 0.38, Math.PI, 2 * Math.PI), [0.02, 0.45]], [[0.5, 0.45], [0.5, 0.92], [0.44, 0.99], [0.36, 0.94]]],
	sun: [circ(0.5, 0.5, 0.22, 0.22), ...Array.from({ length: 10 }, (_, i) => { const a = (i * 2 * Math.PI) / 10; return [[0.5 + 0.3 * Math.cos(a), 0.5 + 0.3 * Math.sin(a)], [0.5 + 0.48 * Math.cos(a), 0.5 + 0.48 * Math.sin(a)]] as P[]; })],
	tree: [[[0.42, 1], [0.44, 0.6]], [[0.58, 1], [0.56, 0.6]], [...circ(0.5, 0.35, 0.4, 0.3, 0.6 * Math.PI, 2.4 * Math.PI)]],
	envelope: [[[0, 0.15], [1, 0.15], [1, 0.85], [0, 0.85], [0, 0.15]], [[0, 0.15], [0.5, 0.55], [1, 0.15]]],
	ladder: [[[0.25, 0], [0.25, 1]], [[0.75, 0], [0.75, 1]], ...[0.12, 0.3, 0.48, 0.66, 0.84].map((y) => [[0.25, y], [0.75, y]] as P[])]
};
function distSeg2(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
	const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy;
	let t = L > 0 ? ((px - ax) * dx + (py - ay) * dy) / L : 0;
	t = Math.max(0, Math.min(1, t));
	const ex = ax + t * dx - px, ey = ay + t * dy - py;
	return ex * ex + ey * ey;
}
/** Rasterise strokes (already in canvas px) with a round brush of width bw: coverage 0..1 by 3x3 supersampling. */
function rasterise(strokes: P[][], W: number, H: number, bw: number): Float32Array {
	const out = new Float32Array(W * H);
	const r2 = (bw / 2) ** 2;
	const segs: number[] = [];
	for (const s of strokes) for (let i = 1; i < s.length; i++) segs.push(s[i - 1][0], s[i - 1][1], s[i][0], s[i][1]);
	for (let k = 0; k < segs.length; k += 4) {
		const xa = Math.max(0, Math.floor(Math.min(segs[k], segs[k + 2]) - bw)), xb = Math.min(W - 1, Math.ceil(Math.max(segs[k], segs[k + 2]) + bw));
		const ya = Math.max(0, Math.floor(Math.min(segs[k + 1], segs[k + 3]) - bw)), yb = Math.min(H - 1, Math.ceil(Math.max(segs[k + 1], segs[k + 3]) + bw));
		for (let y = ya; y <= yb; y++) for (let x = xa; x <= xb; x++) {
			if (out[y * W + x] === 1) continue;
			let hit = 0;
			for (let sy = 0; sy < 3; sy++) for (let sx = 0; sx < 3; sx++) if (distSeg2(x + (sx + 0.5) / 3, y + (sy + 0.5) / 3, segs[k], segs[k + 1], segs[k + 2], segs[k + 3]) <= r2) hit++;
			out[y * W + x] = Math.max(out[y * W + x], hit / 9);
		}
	}
	return out;
}
const place = (s: P[][], D: number, ox: number, oy: number): P[][] => s.map((st) => st.map(([x, y]) => [ox + x * D, oy + y * D] as P));
function rgba(cov: Float32Array, W: number, H: number, mode: 'alpha' | 'darkOnWhite' | 'whiteOnBlack' | 'blueOnCream'): { data: Uint8ClampedArray; width: number; height: number } {
	const d = new Uint8ClampedArray(W * H * 4);
	for (let i = 0; i < W * H; i++) {
		const a = cov[i];
		if (mode === 'alpha') { d[4 * i] = 30; d[4 * i + 1] = 30; d[4 * i + 2] = 200; d[4 * i + 3] = Math.round(a * 255); }
		else if (mode === 'darkOnWhite') { const v = Math.round(255 * (1 - a)); d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = v; d[4 * i + 3] = 255; }
		else if (mode === 'whiteOnBlack') { const v = Math.round(255 * a); d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = v; d[4 * i + 3] = 255; }
		else { d[4 * i] = Math.round(250 * (1 - a) + 20 * a); d[4 * i + 1] = Math.round(245 * (1 - a) + 40 * a); d[4 * i + 2] = Math.round(225 * (1 - a) + 160 * a); d[4 * i + 3] = 255; }
	}
	return { data: d, width: W, height: H };
}
const dump: { name: string; img: number[]; pred: string; p: number }[] = [];
const keep = (name: string, r: Preprocessed | { input: Float32Array }) => {
	const pr = model.predict(r.input);
	dump.push({ name, img: Array.from(r.input, (v) => Math.round(v * 1000) / 1000), pred: cls[pr.top], p: Number(pr.probs[pr.top].toFixed(3)) });
	return pr;
};
// dataset references (one of each base class from the sample) for the side-by-side picture
for (const k of [4, 5, 9, 7, 8]) {
	const i = sampleLabels.indexOf(k);
	keep(`dataset ${cls[k]}`, { input: sample(i) });
}
const variants: { name: string; W: number; H: number; D: number; ox: number; oy: number; bw: number; mode: 'alpha' | 'darkOnWhite' | 'whiteOnBlack' | 'blueOnCream' }[] = [
	{ name: 'pad 280 alpha, D=220, brush 15', W: 280, H: 280, D: 220, ox: 30, oy: 30, bw: 15, mode: 'alpha' },
	{ name: 'pad 280 dark-on-white, D=220, brush 15', W: 280, H: 280, D: 220, ox: 30, oy: 30, bw: 15, mode: 'darkOnWhite' },
	{ name: 'pad 280 white-on-black, D=220, brush 15', W: 280, H: 280, D: 220, ox: 30, oy: 30, bw: 15, mode: 'whiteOnBlack' },
	{ name: 'pad 280 alpha, small off-centre D=110, brush 15', W: 280, H: 280, D: 110, ox: 15, oy: 140, bw: 15, mode: 'alpha' },
	{ name: 'wide 400x250 blue-on-cream, D=180 at right, brush 8', W: 400, H: 250, D: 180, ox: 200, oy: 40, bw: 8, mode: 'blueOnCream' }
];
const shapeResults: Record<string, Record<string, string>> = {};
let frameFails = 0;
for (const [sname, strokes] of Object.entries(SHAPES)) {
	shapeResults[sname] = {};
	const st = preprocessStrokes(place(strokes, 300, 0, 0));
	const ps = keep(`${sname} | preprocessStrokes`, st);
	shapeResults[sname]['strokes'] = `${cls[ps.top]} ${ps.probs[ps.top].toFixed(2)}`;
	const sStr = stats(st.input);
	for (const v of variants) {
		const s = place(strokes, v.D, v.ox, v.oy);
		// keep the drawing's aspect: shapes are drawn in a unit box, the wide shapes keep their own aspect
		const cov = rasterise(s, v.W, v.H, v.bw);
		const r = preprocessDrawing(rgba(cov, v.W, v.H, v.mode));
		const pr = keep(`${sname} | ${v.name}`, r);
		const t = stats(r.input);
		// faint: share of pixels in (0, 0.08); a pad colour read as ink leaves a faint grey background everywhere
		const framed = Math.abs(t.max - 1) < 1e-6 && t.border < 0.02 && t.faint < 0.15 && t.long >= 25 && t.long <= 27 && Math.abs(t.cy - 14) <= 1 && Math.abs(t.cx - 14) <= 1;
		if (!framed) frameFails++;
		shapeResults[sname][v.name] = `${cls[pr.top]} ${pr.probs[pr.top].toFixed(2)} | long ${t.long} centre (${t.cy},${t.cx}) max ${t.max.toFixed(2)} border ${t.border.toFixed(2)} zeros ${(t.zeros * 100).toFixed(0)}% faint ${(t.faint * 100).toFixed(0)}% mass ${t.mass.toFixed(0)} (strokes mass ${sStr.mass.toFixed(0)})${framed ? '' : '  <-- framing off'}`;
	}
}
for (const [s, r] of Object.entries(shapeResults)) {
	console.log(`shape ${s}:`);
	for (const [k, v] of Object.entries(r)) console.log(`    ${k.padEnd(52)} ${v}`);
}
check(frameFails === 0, `synthetic drawings framed like the dataset (long side 25-27, centred within 1 px, max 1, empty border, < 15% faint background pixels): ${frameFails} failures`);

// ------------------------------------------------ 5. brush size vs drawing size (fixed brush on a pad, dataset drawings)
// Upscale each dataset drawing to D px, binarise, then thicken so the stroke is ~ a fixed 15 px brush.
// Dataset strokes are ~1.5/26 of the drawing's long side, so at D px they are ~1.5*D/26 px wide.
function dilateDisk(a: Float32Array, W: number, H: number, r: number): Float32Array {
	if (r <= 0) return a;
	const out = new Float32Array(W * H);
	const offs: [number, number][] = [];
	for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) if (dx * dx + dy * dy <= r * r + r) offs.push([dy, dx]);
	for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
		if (a[y * W + x] < 0.5) continue;
		for (const [dy, dx] of offs) { const yy = y + dy, xx = x + dx; if (yy >= 0 && yy < H && xx >= 0 && xx < W) out[yy * W + xx] = 1; }
	}
	return out;
}
function upscale(img: Float32Array, D: number, pad: number): { c: Float32Array; S: number } {
	const S = D + 2 * pad;
	const c = new Float32Array(S * S);
	const sc = 28 / D;
	const px = (yy: number, xx: number) => (yy < 0 || yy > 27 || xx < 0 || xx > 27 ? 0 : img[yy * 28 + xx]);
	for (let v = 0; v < S; v++) for (let u = 0; u < S; u++) {
		const sx = (u - pad + 0.5) * sc - 0.5, sy = (v - pad + 0.5) * sc - 0.5;
		const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
		c[v * S + u] = (1 - fy) * ((1 - fx) * px(y0, x0) + fx * px(y0, x0 + 1)) + fy * ((1 - fx) * px(y0 + 1, x0) + fx * px(y0 + 1, x0 + 1));
	}
	return { c, S };
}
const brushRes: Record<string, number> = {};
{
	const step = 2; // every 2nd drawing (750) to keep it quick
	for (const D of [60, 90, 140, 220]) {
		const r = Math.max(0, Math.round((15 - (1.5 * D) / 26) / 2));
		let ok = 0, n = 0;
		for (let i = 0; i < NS; i += step) {
			const { c, S } = upscale(sample(i), D, r + 4);
			const bin = new Float32Array(S * S);
			for (let j = 0; j < S * S; j++) bin[j] = c[j] > 0.35 ? 1 : 0;
			const th = dilateDisk(bin, S, S, r);
			const res = preprocessDrawing({ ink: th, width: S, height: S });
			if (model.predict(res.input).top === sampleLabels[i]) ok++;
			n++;
			if (i === 0 && (D === 60 || D === 220)) keep(`dataset #0 at D=${D}, 15 px brush`, res);
		}
		brushRes[`D=${D}px (dilate r=${r})`] = ok / n;
		console.log(`fixed 15 px brush, drawing ${D} px wide (stroke ~${(((15) * 26) / D).toFixed(1)} px at 28x28): ${((ok / n) * 100).toFixed(2)}% (n=${n})`);
	}
}

// ------------------------------------------------ 6. a synthetic phone photo of a drawn house
{
	const W = 640, H = 480;
	const cov = rasterise(place(SHAPES.house, 260, 200, 120), W, H, 5);
	let s = 7;
	const rnd = () => { s = (s * 1103515245 + 12345) >>> 0; return s / 4294967296; };
	const d = new Uint8ClampedArray(W * H * 4);
	for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
		const i = y * W + x;
		const shade = 0.85 - 0.25 * (x / W) * (y / H) + 0.04 * (rnd() - 0.5); // shadow toward the bottom right
		const v = shade * (1 - 0.75 * cov[i]);
		d[4 * i] = 255 * v; d[4 * i + 1] = 250 * v; d[4 * i + 2] = 235 * v; d[4 * i + 3] = 255;
	}
	const r = preprocessPhoto({ data: d, width: W, height: H });
	const pr = keep('house | photo 640x480 shadow + noise', r);
	const t = stats(r.input);
	console.log(`photo house: ${cls[pr.top]} ${pr.probs[pr.top].toFixed(2)} inverted=${r.inverted} bbox=${JSON.stringify(r.bbox)} long ${t.long} centre (${t.cy},${t.cx})`);
	check(cls[pr.top] === 'house', 'synthetic photo of a house -> house');
}

// ------------------------------------------------ 7. trainHead once on heldout.json
const headRes: Record<string, number> = {};
{
	const held = JSON.parse(readFileSync(join(dir, 'heldout.json'), 'utf8'));
	const tf: Float32Array[] = [], tl: number[] = [];
	const trf: Float32Array[][] = [[], []];
	held.classes.forEach((c: { train: string[]; test: string[] }, k: number) => {
		for (const b of c.test) { tf.push(featuresOf(model, decodePreset(b))); tl.push(k); }
		for (const b of c.train) trf[k].push(featuresOf(model, decodePreset(b)));
	});
	for (const N of [1, 2, 4, 8, 16, 32]) {
		const f = [...trf[0].slice(0, N), ...trf[1].slice(0, N)];
		const l = [...Array(N).fill(0), ...Array(N).fill(1)];
		const t0 = performance.now();
		const head = trainHead(f, l, { epochs: 200, lr: 0.5, seed: 0 });
		const ms = performance.now() - t0;
		const acc = headAccuracy(head, tf, tl);
		headRes[`N=${N}`] = acc;
		const lh = head.lossHistory;
		console.log(`trainHead N=${N}/class: test acc ${(acc * 100).toFixed(1)}% on ${tl.length} heldout.json test drawings; loss ${lh[0].toFixed(3)} -> ${lh[lh.length - 1].toFixed(4)}; train acc ${head.accHistory[head.accHistory.length - 1]}; params ${head.trainableParams}; ${ms.toFixed(1)} ms`);
		if (N === 16) {
			check(acc >= 0.95 && lh[lh.length - 1] < lh[0] && head.trainableParams === 514, 'trainHead N=16: test acc >= 95%, loss decreased, 514 params');
			const p = predictHead(head, tf[0]);
			const sum = p.probs[0] + p.probs[1];
			check(Math.abs(sum - 1) < 1e-5, 'predictHead probs sum to 1');
		}
	}
}

writeFileSync(join(here, 'out', 'shapes.json'), JSON.stringify(dump));
writeFileSync(join(here, 'out', 'verify_ts_result.json'), JSON.stringify({ shapeResults, brushRes, headRes, problems }, null, 1));
console.log(`\nPROBLEMS: ${problems.length}`);
for (const p of problems) console.log(' - ' + p);
