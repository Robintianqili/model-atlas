// crnn-ocr: write the training data with the page's own code (site/src/lib/explainers/crnn-ocr/glyphs.ts), so train.py
// reads exactly the bytes the page computes. Run from the repo root (or give a lane's site with --site):
//
//     node pipeline/crnn-ocr/make_data.ts [--site <site folder>] [--out <folder>]
//
// Writes <out>/<set>.bin (default pipeline/crnn-ocr/runs/data, git-ignored): images as uint8 [count, 16, 64] holding
// ink × 16 (render() gives multiples of 1/16, so the bytes are exact), labels as uint8 [count, 8] (class d + 1 for
// digit d, 0 = padding) and lengths as uint8 [count]; meta.json (counts, CRC-32 of each part) and cases.json (the
// export's reference cases: the page's presets and edge fixtures through the page's own adapter, and 7 digits).
// Sets (seeded forks of one rng): train 80,000 words of 1-7 digits, glyphs.ts wild 1 to 2.2 · val 2,000 (the same) ·
// test 7,000 (1,000 of each length, wild 1, as the presets) · wild 7,000 (wild 2.2, messier writing than any training
// word); a doubled digit follows with chance 0.2; pen width 1.2-2.2 pixels (the page uses 1.7).
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = resolve(import.meta.dirname, '..', '..');
const arg = (k: string) => (process.argv.indexOf(k) > 0 ? process.argv[process.argv.indexOf(k) + 1] : undefined);
const site = resolve(arg('--site') ?? process.env.ATLAS_SITE ?? join(ROOT, 'site'));
const out = resolve(arg('--out') ?? join(ROOT, 'pipeline', 'crnn-ocr', 'runs', 'data'));
const pkg = join(site, 'src', 'lib', 'explainers', 'crnn-ocr');
const G = await import(pathToFileURL(join(pkg, 'glyphs.ts')).href);
const A = (await import(pathToFileURL(join(pkg, 'adapters', 'word-strokes', 'adapter.ts')).href)).default;
const { createRng, fnv1a32 } = await import(pathToFileURL(join(site, 'src', 'lib', 'atlas', 'core', 'rng.ts')).href);

const MAXL = 8;
const LEN_W = [0.1, 0.16, 0.18, 0.18, 0.14, 0.12, 0.12]; // lengths 1..7
const CRC = Array.from({ length: 256 }, (_, n) => {
	let c = n;
	for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
	return c >>> 0;
});
function crc(b: Uint8Array): number {
	let c = 0xffffffff;
	for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8);
	return (c ^ 0xffffffff) >>> 0;
}
mkdirSync(out, { recursive: true });
const root = createRng(fnv1a32('crnn-ocr data v1'));

function text(rng: { next(): number; int(n: number): number }, len?: number): string {
	let L = len ?? 0;
	if (!L) {
		let u = rng.next();
		L = 1;
		while (L < LEN_W.length && u > LEN_W[L - 1]) u -= LEN_W[(L++) - 1];
	}
	let s = '';
	for (let i = 0; i < L; i++) s += i > 0 && rng.next() < 0.2 ? s[i - 1] : String(rng.int(10));
	return s;
}
function write(name: string, n: number, len?: (i: number) => number, wild = (r: { next(): number }) => 1 + 1.2 * r.next()) {
	const X = new Uint8Array(n * G.H * G.W), Y = new Uint8Array(n * MAXL), L = new Uint8Array(n);
	for (let i = 0; i < n; i++) {
		const rng = root.fork(`${name}:${i}`);
		const t = text(rng, len?.(i));
		const lw = 1.2 + rng.next();
		const r = G.render(G.word(t, rng, wild(rng)), lw);
		for (let k = 0; k < r.ink.length; k++) X[i * G.H * G.W + k] = Math.round(r.ink[k] * 16);
		for (let k = 0; k < t.length; k++) Y[i * MAXL + k] = Number(t[k]) + 1;
		L[i] = t.length;
	}
	writeFileSync(join(out, `${name}.bin`), Buffer.concat([X, Y, L].map((p) => Buffer.from(p.buffer, p.byteOffset, p.byteLength))));
	return { n, crc: [X, Y, L].map(crc) };
}
const byLen = (i: number) => 1 + (i % 7);
const sets = { train: write('train', 80000), val: write('val', 2000), test: write('test', 7000, byLen, () => 1), wild: write('wild', 7000, byLen, () => 2.2) };

const cases: { id: string; label: string; edge: boolean; text: string; ink: number[] }[] = [];
const inkOf = (raw: unknown) => Array.from(A.preprocess(raw, null, {}).tensors.input.data as Float32Array);
for (const p of A.presets({})) cases.push({ id: p.id, label: p.label.en, edge: false, text: p.raw.text, ink: inkOf(p.raw) });
for (const f of A.edgeFixtures()) cases.push({ id: f.id, label: f.label.en, edge: true, text: '', ink: inkOf(f.raw) });
writeFileSync(join(out, 'cases.json'), JSON.stringify({ cases }));
writeFileSync(join(out, 'meta.json'), JSON.stringify({ h: G.H, w: G.W, maxl: MAXL, symbols: G.SYMBOLS, lenWeights: LEN_W, sets }, null, 1));
console.log(`crnn-ocr data -> ${out}: ${Object.entries(sets).map(([k, v]) => `${k} ${v.n}`).join(', ')}; ${cases.length} cases`);
