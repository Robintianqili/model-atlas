// egnn: write the training data with the page's own code (site/src/lib/explainers/egnn/mol.ts), so train.py reads exactly
// the bytes the page computes. Run from the repo root (or give a lane's site with --site):
//
//     node pipeline/egnn/make_data.ts [--site <lane>/site] [--out <folder>]      (default out: pipeline/egnn/runs/data)
//
// Writes <out>/<set>.bin: positions float32 [count, PAD, 2], kinds uint8 [count, PAD], atom counts uint8 [count], toy
// energies float32 [count] and forces float32 [count, PAD, 2] (mol.ts energy(), computed in float64 on the float32
// positions); meta.json (counts, CRC-32 of each part, PAD); cases.json (the export's reference cases, unpadded: the page's
// presets and edge fixtures through the page's own adapter, the bent molecule turned by 90 degrees and mirrored, and a
// random molecule of MAX_ATOMS atoms). Sets (seeded forks of one rng; NMIN to NMAX atoms, uniform):
//   train 40,000 · val 2,000 · test 5,000
import { mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = resolve(import.meta.dirname, '..', '..');
const arg = (k: string) => (process.argv.indexOf(k) > 0 ? process.argv[process.argv.indexOf(k) + 1] : undefined);
const site = resolve(arg('--site') ?? process.env.ATLAS_SITE ?? join(ROOT, 'site'));
const out = resolve(arg('--out') ?? process.env.EGNN_DATA ?? join(ROOT, 'pipeline', 'egnn', 'runs', 'data'));
const pkg = join(site, 'src', 'lib', 'explainers', 'egnn');
const M = await import(pathToFileURL(join(pkg, 'mol.ts')).href);
const A = (await import(pathToFileURL(join(pkg, 'adapters', 'points2d', 'adapter.ts')).href)).default;
const { createRng, fnv1a32 } = await import(pathToFileURL(join(site, 'src', 'lib', 'atlas', 'core', 'rng.ts')).href);

const PAD = M.NMAX;
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
const root = createRng(fnv1a32('egnn data v1'));
const sets: Record<string, { n: number; crc: number[] }> = {};
for (const [name, count] of [['train', 40000], ['val', 2000], ['test', 5000]] as const) {
	const P = new Float32Array(count * PAD * 2), Kd = new Uint8Array(count * PAD), N = new Uint8Array(count), E = new Float32Array(count), F = new Float32Array(count * PAD * 2);
	for (let i = 0; i < count; i++) {
		const rng = root.fork(`${name}:${i}`);
		const n = M.NMIN + rng.int(M.NMAX - M.NMIN + 1);
		const m = M.sample(n, rng);
		const pos = Float32Array.from(m.pos);
		const e = M.energy(pos, m.kind);
		P.set(pos, i * PAD * 2);
		F.set(Float32Array.from(e.F), i * PAD * 2);
		for (let a = 0; a < n; a++) Kd[i * PAD + a] = m.kind[a];
		(N[i] = n), (E[i] = e.E);
	}
	const parts = [P, Kd, N, E, F].map((p) => new Uint8Array(p.buffer, p.byteOffset, p.byteLength));
	writeFileSync(join(out, `${name}.bin`), Buffer.concat(parts.map((p) => Buffer.from(p.buffer, p.byteOffset, p.byteLength))));
	sets[name] = { n: count, crc: parts.map(crc) };
}

// reference cases for export.py, through the page's own adapter
const cases: { id: string; label: string; edge: boolean; pos: number[]; kind: number[]; E: number; F: number[] }[] = [];
function add(id: string, label: string, raw: unknown, edge: boolean) {
	const t = A.preprocess(raw, null, {}).tensors;
	if (!t.pos) return;
	const n = t.pos.shape[0];
	const kind = Array.from({ length: n }, (_, i) => Array.from(t.kind.data.subarray(i * M.K, (i + 1) * M.K)).indexOf(1));
	const e = M.energy(t.pos.data, kind);
	cases.push({ id, label, edge, pos: Array.from(t.pos.data as Float32Array), kind, E: e.E, F: Array.from(e.F) });
}
for (const p of A.presets({})) add(p.id, p.label.en, p.raw, false);
const bent = A.presets({})[0].raw;
add('bent-turned', 'bent, turned by 90 degrees', { ...bent, turn: 90 }, false);
add('bent-mirrored', 'bent, mirrored', { ...bent, flip: true }, false);
for (const f of A.edgeFixtures()) add(f.id, f.label.en, f.raw, true);
const big = M.sample(M.MAX_ATOMS, createRng(fnv1a32('egnn big')));
add('max-atoms', `${M.MAX_ATOMS} atoms`, { points: Array.from({ length: M.MAX_ATOMS }, (_, i) => ({ x: big.pos[2 * i] / M.SCALE, y: big.pos[2 * i + 1] / M.SCALE, c: big.kind[i] })) }, true);
writeFileSync(join(out, 'cases.json'), JSON.stringify({ cases }));
writeFileSync(join(out, 'meta.json'), JSON.stringify({ pad: PAD, k: M.K, nmin: M.NMIN, nmax: M.NMAX, maxAtoms: M.MAX_ATOMS, sets }, null, 1));
console.log(`egnn data -> ${out}: ${Object.entries(sets).map(([k, v]) => `${k} ${v.n}`).join(', ')}; ${cases.length} cases`);
