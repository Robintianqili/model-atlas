// tinyseq2seq: write the training, validation and test pairs for Python from the package's TS generator
// (site/src/lib/explainers/tinyseq2seq/data.ts, the source of truth), so Python trains on exactly these strings.
//
//   node pipeline/tinyseq2seq/make_data.ts            (ATLAS_SITE picks the site folder, e.g. a lane; default: site/)
//
// Writes pipeline/tinyseq2seq/runs/data/{train,val,test}.json: { split, seed, n, src_vocab, tgt_vocab, pairs: [[src, tgt,
// format], ...], sha256 } where sha256 hashes the lines "src<TAB>tgt<LF>" of the pairs (claims.test.ts checks the same
// hash), and off.json: dates in formats the model never sees in training (the limits step's measured facts).
import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const site = process.env.ATLAS_SITE ? resolve(process.env.ATLAS_SITE) : resolve(here, '../../site');
const data = await import(pathToFileURL(join(site, 'src/lib/explainers/tinyseq2seq/data.ts')).href);
const { createRng } = await import(pathToFileURL(join(site, 'src/lib/atlas/core/rng.ts')).href);
const SPLITS = { train: { seed: 1, n: 60000 }, val: { seed: 2, n: 2000 }, test: { seed: 3, n: 5000 } };
const NL = String.fromCharCode(10);
const TAB = String.fromCharCode(9);
const out = join(here, 'runs', 'data');
mkdirSync(out, { recursive: true });
for (const [split, { seed, n }] of Object.entries(SPLITS)) {
	const pairs = data.dataset(seed, n).map((p: { src: string; tgt: string; fmt: string }) => [p.src, p.tgt, p.fmt]);
	const sha256 = createHash('sha256').update(pairs.map((p: string[]) => p[0] + TAB + p[1] + NL).join('')).digest('hex');
	writeFileSync(join(out, `${split}.json`), JSON.stringify({ split, seed, n, src_vocab: data.SRC, tgt_vocab: data.TGT, pairs, sha256 }) + NL);
	console.log(`${split}: ${n} pairs (seed ${seed}), sha256 ${sha256.slice(0, 16)}…`);
}

// Formats the model never sees in training, 1,000 dates each (seed 4). The target of 'impossible' is the text's own
// digits in ISO order ("March 35, 2021" -> 2021-03-35): it measures copying without a calendar, not a right answer.
const rng = createRng(4);
const two = (n: number) => String(n).padStart(2, '0');
const WORDS = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth'];
const OFF: Record<string, (y: number, m: number, d: number) => string> = {
	yearFirst: (y, m, d) => `${y}/${two(m)}/${two(d)}`,
	dashes: (y, m, d) => `${two(d)}-${two(m)}-${y}`,
	noComma: (y, m, d) => `${data.MONTHS[m - 1]} ${d} ${y}`,
	spelledDay: (y, m, d) => `${data.MONTHS[m - 1]} ${WORDS[d - 1]}, ${y}`
};
const off: string[][] = [];
for (const [fmt, write] of Object.entries(OFF))
	for (let i = 0; i < 1000; i++) {
		const y = 1900 + rng.int(200);
		const m = 1 + rng.int(12);
		const d = 1 + rng.int(fmt === 'spelledDay' ? 9 : data.daysIn(y, m));
		off.push([write(y, m, d), `${y}-${two(m)}-${two(d)}`, fmt]);
	}
for (let i = 0; i < 1000; i++) {
	const y = 1900 + rng.int(200);
	const m = 1 + rng.int(12);
	const d = data.daysIn(y, m) + 1 + rng.int(39 - data.daysIn(y, m));
	off.push([`${data.MONTHS[m - 1]} ${d}, ${y}`, `${y}-${two(m)}-${d}`, 'impossible']);
}
writeFileSync(join(out, 'off.json'), JSON.stringify({ split: 'off', seed: 4, n: off.length, pairs: off }) + NL);
console.log(`off: ${off.length} pairs in ${Object.keys(OFF).length + 1} formats the model never sees (seed 4)`);
console.log(`wrote ${out} (from ${site})`);
