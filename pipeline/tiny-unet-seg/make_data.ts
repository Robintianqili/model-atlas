// tiny-unet-seg: write the training data with the page's own code (site/src/lib/explainers/tiny-unet-seg/scenes.ts), so
// train.py reads exactly the bytes the page computes. Run from the repo root (or a lane's site with --site):
//
//     node pipeline/tiny-unet-seg/make_data.ts [--site site/.work/tiny-unet-seg/site]
//
// Writes pipeline/tiny-unet-seg/runs/data/<set>.bin (git-ignored): n pictures as float32 [n, 32, 32], then their true
// classes as uint8 [n, 32, 32]; and meta.json: per set the count, the CRC-32 of each part (zlib.crc32, checked by
// train.py) and, for the named cases (presets and edge fixtures), their ids. Sets (seeded forks of one rng):
//   train 16,000 · val 1,000 · test 2,000 (sizes 3-8 px) · big 500 (sizes 9-13) · hollow 500 (outlines) · dark 500
//   (dark shapes on a light background) · cases: the page's presets and edge fixtures
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = resolve(import.meta.dirname, '..', '..');
const a = process.argv.indexOf('--site');
const site = resolve(a > 0 ? process.argv[a + 1] : (process.env.ATLAS_SITE ?? join(ROOT, 'site')));
const pkg = join(site, 'src', 'lib', 'explainers', 'tiny-unet-seg');
const { render, PRESETS, S, TRAIN_R } = await import(pathToFileURL(join(pkg, 'scenes.ts')).href);
const { EDGES } = await import(pathToFileURL(join(pkg, 'adapters', 'shapes32', 'adapter.ts')).href);
const { createRng, fnv1a32 } = await import(pathToFileURL(join(site, 'src', 'lib', 'atlas', 'core', 'rng.ts')).href);

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

type Rng = { next(): number; int(n: number): number };
type Shape = { k: number; x: number; y: number; r: number; rot?: number; v?: number; hollow?: boolean };
type SegRaw = { shapes: Shape[]; bg: number; noise: number; seed: number };
/** How far a shape reaches from its centre (for placing shapes without much overlap). */
const reach = (k: number, r: number) => r * (k === 1 ? 1 : k === 2 ? 1.21 : 1.35);

/** One training scene: 1 to 3 shapes (none in 1 of 20), sizes in `rr`, any rotation, brightness above the background. */
function randomScene(rng: Rng, o: { rr?: number[]; hollow?: boolean; dark?: boolean } = {}): SegRaw {
	const [r0, r1] = o.rr ?? TRAIN_R;
	const bg = o.dark ? 0.6 + rng.next() * 0.3 : rng.next() * 0.3;
	const noise = 0.05 + rng.next() * 0.1;
	const n = rng.next() < 0.05 ? 0 : 1 + rng.int(3);
	const shapes: Shape[] = [];
	for (let a = 0; a < 30 && shapes.length < n; a++) {
		const k = 1 + rng.int(3), r = r0 + rng.next() * (r1 - r0);
		const e = Math.min(reach(k, r), S / 2 - 1.5);
		const x = e + 1 + rng.next() * (S - 2 * e - 2), y = e + 1 + rng.next() * (S - 2 * e - 2);
		const rot = k === 2 ? rng.next() * (Math.PI / 2) : k === 3 ? rng.next() * ((2 * Math.PI) / 3) : 0;
		const v = o.dark ? bg - 0.35 - rng.next() * 0.25 : bg + 0.35 + rng.next() * 0.5;
		if (shapes.every((q) => Math.hypot(q.x - x, q.y - y) >= 0.9 * (reach(q.k, q.r) + reach(k, r)))) shapes.push({ k, x, y, r, rot, v, hollow: o.hollow });
	}
	return { shapes, bg, noise, seed: rng.int(2 ** 31) };
}

const out = join(ROOT, 'pipeline', 'tiny-unet-seg', 'runs', 'data');
mkdirSync(out, { recursive: true });
const root = createRng(fnv1a32('tiny-unet-seg data v1'));
const meta: Record<string, unknown> = { format: 'tiny-unet-seg-data/1', size: S, site, sets: {} };

function write(name: string, raws: unknown[], ids?: string[]) {
	const n = raws.length, px = S * S;
	const img = new Float32Array(n * px), lab = new Uint8Array(n * px);
	raws.forEach((r, j) => {
		const { image, label } = render(r);
		img.set(image, j * px);
		lab.set(label, j * px);
	});
	const ib = new Uint8Array(img.buffer), lb = lab;
	const buf = new Uint8Array(ib.length + lb.length);
	buf.set(ib, 0);
	buf.set(lb, ib.length);
	writeFileSync(join(out, `${name}.bin`), buf);
	const counts = [0, 0, 0, 0];
	for (const k of lab) counts[k]++;
	(meta.sets as Record<string, unknown>)[name] = { n, crcImages: crc(ib), crcLabels: crc(lb), classPixels: counts, ...(ids ? { ids } : {}) };
	console.log(`${name}: ${n} pictures, class pixels ${counts.join(' / ')}`);
}

const scenes = (label: string, n: number, o = {}) => {
	const rng = root.fork(label);
	return Array.from({ length: n }, () => randomScene(rng, o));
};
write('train', scenes('train', 16000));
write('val', scenes('val', 1000));
write('test', scenes('test', 2000));
write('big', scenes('big', 500, { rr: [9, 13] }));
write('hollow', scenes('hollow', 500, { rr: [5, 8], hollow: true }));
write('dark', scenes('dark', 500, { dark: true }));
const cases = [...PRESETS, ...EDGES];
write('cases', cases.map((c: { raw: unknown }) => c.raw), cases.map((c: { id: string }) => c.id));
writeFileSync(join(out, 'meta.json'), JSON.stringify(meta, null, 1) + '\n');
console.log(`wrote ${out}`);
