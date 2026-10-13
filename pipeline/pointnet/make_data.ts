// pointnet: write the training data with the page's own code (site/src/lib/explainers/pointnet/shapes.ts), so train.py
// reads exactly the bytes the page computes. Run from the repo root (or give a lane's site with --site):
//
//     node pipeline/pointnet/make_data.ts [--site site/.work/pointnet/site]
//
// Writes pipeline/pointnet/runs/data/<set>.bin (git-ignored): clouds as float32 [count, PAD, 2] (each cloud already
// normalized as the page does it, then repeated cyclically up to PAD points: repeats do not change a max), their true
// classes as uint8 [count] and their point counts as uint16 [count]; and meta.json (counts, CRC-32 of each part, PAD).
// cases.json holds the export's reference cases (the page's presets, edge cases, the largest cloud) unpadded.
// Sets (seeded forks of one rng; 16 to 200 points, spread evenly on a log scale):
//   train 30,000 · val 2,000 · test 5,000 · stray: the test clouds plus one stray point (shapes.ts withStray) ·
//   half: the test clouds with a seeded half of their points removed, normalized again
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = resolve(import.meta.dirname, '..', '..');
const a = process.argv.indexOf('--site');
const site = resolve(a > 0 ? process.argv[a + 1] : (process.env.ATLAS_SITE ?? join(ROOT, 'site')));
const pkg = join(site, 'src', 'lib', 'explainers', 'pointnet');
const S = await import(pathToFileURL(join(pkg, 'shapes.ts')).href);
const A = (await import(pathToFileURL(join(pkg, 'adapters', 'points2d', 'adapter.ts')).href)).default;
const { createRng, fnv1a32 } = await import(pathToFileURL(join(site, 'src', 'lib', 'atlas', 'core', 'rng.ts')).href);

const PAD = 256, NMIN = 16, NMAX = 200;
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

type Item = { xy: ArrayLike<number>; k: number };
const out = join(ROOT, 'pipeline', 'pointnet', 'runs', 'data');
mkdirSync(out, { recursive: true });
const root = createRng(fnv1a32('pointnet data v1'));

/** a raw cloud of set `name`, item i: a class in turn, a point count, then shapes.ts sample() */
function raw(name: string, i: number) {
	const rng = root.fork(`${name}:${i}`);
	const k = i % S.CLASSES.length;
	const n = Math.round(Math.exp(Math.log(NMIN) + rng.next() * (Math.log(NMAX) - Math.log(NMIN))));
	return { xy: S.sample(S.CLASSES[k], n, rng), k, rng };
}

function write(name: string, items: Item[]) {
	const X = new Float32Array(items.length * PAD * 2);
	const y = new Uint8Array(items.length), ns = new Uint16Array(items.length);
	items.forEach((it, j) => {
		const n = it.xy.length / 2;
		for (let p = 0; p < PAD; p++) (X[(j * PAD + p) * 2] = it.xy[2 * (p % n)]), (X[(j * PAD + p) * 2 + 1] = it.xy[2 * (p % n) + 1]);
		y[j] = it.k;
		ns[j] = n;
	});
	const parts = [new Uint8Array(X.buffer), y, new Uint8Array(ns.buffer)];
	writeFileSync(join(out, `${name}.bin`), Buffer.concat(parts.map((p) => Buffer.from(p.buffer, p.byteOffset, p.byteLength))));
	return { n: items.length, crc: parts.map(crc) };
}

const sets: Record<string, { n: number; crc: number[] }> = {};
const norm = (xy: ArrayLike<number>) => S.normalize(xy).xy as Float32Array;
for (const [name, count] of [['train', 30000], ['val', 2000], ['test', 5000]] as const) {
	const items: Item[] = [];
	for (let i = 0; i < count; i++) {
		const r = raw(name, i);
		items.push({ xy: norm(r.xy), k: r.k });
	}
	sets[name] = write(name, items);
	if (name === 'test') {
		sets.stray = write('stray', items.map((it) => ({ xy: S.withStray(it.xy).xy, k: it.k })));
		const half: Item[] = [];
		for (let i = 0; i < count; i++) {
			const r = raw(name, i);
			const n = r.xy.length / 2;
			const keep = S.shuffled(n, fnv1a32(`pointnet half ${i}`)).slice(0, Math.ceil(n / 2)).sort((p: number, q: number) => p - q);
			half.push({ xy: norm(keep.flatMap((p: number) => [r.xy[2 * p], r.xy[2 * p + 1]])), k: r.k });
		}
		sets.half = write('half', half);
	}
}

// reference cases for export.py: the page's presets and edge fixtures through the page's own adapter, plus the largest cloud
const cases: { id: string; label: string; edge: boolean; xy: number[] }[] = [];
const inputOf = (r: unknown) => A.preprocess(r, null, {}).tensors.input;
for (const p of A.presets({})) cases.push({ id: p.id, label: p.label.en, edge: false, xy: Array.from(inputOf(p.raw).data as Float32Array) });
for (const f of A.edgeFixtures()) {
	const t = inputOf(f.raw);
	if (t) cases.push({ id: f.id, label: f.label.en, edge: true, xy: Array.from(t.data as Float32Array) });
}
const big = S.sample('circle', S.MAX_N, createRng(fnv1a32('pointnet big')));
cases.push({ id: 'max-points', label: `${S.MAX_N} points`, edge: true, xy: Array.from(norm(big)) });
writeFileSync(join(out, 'cases.json'), JSON.stringify({ cases }));
writeFileSync(join(out, 'meta.json'), JSON.stringify({ pad: PAD, classes: S.CLASSES, nmin: NMIN, nmax: NMAX, stray: S.STRAY, sets }, null, 1));
console.log(`pointnet data -> ${out}: ${Object.entries(sets).map(([k, v]) => `${k} ${v.n}`).join(', ')}; ${cases.length} cases`);
