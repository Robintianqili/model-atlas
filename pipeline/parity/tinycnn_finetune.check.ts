// Check tinycnn_finetune.ts against the Python head-only numbers.
// Run from the repo root: node pipeline/parity/tinycnn_finetune.check.ts
// Needs pipeline/tinycnn/out/ft_check.{bin,json} (python finetune.py).
// For each of the 5 seeds: train a head on the same 16 held-out drawings per class that Python used,
// then measure accuracy on the full held-out test split (3,000 drawings).
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { load } from '../../site/src/lib/models/tinycnn.ts';
import type { FetchLike } from '../../site/src/lib/models/tinycnn.ts';
import { featuresOf, trainHead, headAccuracy, projectPCA, createHeadTrainer } from '../../site/src/lib/models/tinycnn_finetune.ts';

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
const model = await load(fsFetch, dir);
const meta = JSON.parse(readFileSync(join(here, '..', 'tinycnn', 'out', 'ft_check.json'), 'utf8'));
const bin = readFileSync(join(here, '..', 'tinycnn', 'out', 'ft_check.bin'));
const img = (i: number) => Float32Array.from(bin.subarray(i * 784, (i + 1) * 784), (v) => v / 255);

const nSeeds: number = meta.seeds.length;
const perSeed = 2 * meta.n;
const trainLabels: number[] = meta.trainLabels;
const testLabels: number[] = meta.testLabels;
const t0 = performance.now();
const testFeats = testLabels.map((_, i) => featuresOf(model, img(nSeeds * perSeed + i)));
console.log(`features for ${testLabels.length} test drawings: ${(performance.now() - t0).toFixed(0)} ms`);

const accs: number[] = [];
for (let s = 0; s < nSeeds; s++) {
	const feats = Array.from({ length: perSeed }, (_, i) => featuresOf(model, img(s * perSeed + i)));
	const t1 = performance.now();
	const head = trainHead(feats, trainLabels, { epochs: 200, lr: 0.5, seed: s });
	const ms = performance.now() - t1;
	const acc = headAccuracy(head, testFeats, testLabels);
	accs.push(acc);
	console.log(`seed ${s}: JS head ${(acc * 100).toFixed(2)}%  python head ${(meta.pythonHeadAcc16[s] * 100).toFixed(2)}%  train loss ${head.lossHistory[0].toFixed(3)} -> ${head.lossHistory[head.lossHistory.length - 1].toFixed(4)}  (${ms.toFixed(1)} ms)`);
}
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
const jsMean = mean(accs);
const pyMean = mean(meta.pythonHeadAcc16);
console.log(`mean over seeds: JS ${(jsMean * 100).toFixed(2)}%  python ${(pyMean * 100).toFixed(2)}%  gap ${((jsMean - pyMean) * 100).toFixed(2)} points`);

// timing on 64 and 128 examples (features reused from the test set)
const timing: Record<string, number> = {};
for (const n of [64, 128]) {
	const f = testFeats.slice(0, n / 2).concat(testFeats.slice(testFeats.length - n / 2));
	const l = testLabels.slice(0, n / 2).concat(testLabels.slice(testLabels.length - n / 2));
	trainHead(f, l); // warm-up
	const t2 = performance.now();
	const reps = 5;
	for (let r = 0; r < reps; r++) trainHead(f, l, { epochs: 200 });
	timing[`${n} examples, 200 epochs (ms)`] = Number(((performance.now() - t2) / reps).toFixed(1));
}
console.log('timing', timing);

// determinism: two runs give identical weights; step-wise trainer equals trainHead
const fs0 = Array.from({ length: perSeed }, (_, i) => featuresOf(model, img(i)));
const h1 = trainHead(fs0, trainLabels, { seed: 3 });
const h2 = trainHead(fs0, trainLabels, { seed: 3 });
const tr = createHeadTrainer(fs0, trainLabels, { seed: 3 });
for (let i = 0; i < 20; i++) tr.step(10);
const h3 = tr.head();
let same = true;
for (let i = 0; i < h1.W.length; i++) if (h1.W[i] !== h2.W[i] || h1.W[i] !== h3.W[i]) same = false;
console.log(`deterministic: ${same}`);

// PCA projection sanity: the held-out drawings should land near their class centre in pca.json
const pca = JSON.parse(readFileSync(join(dir, 'pca.json'), 'utf8'));
for (const key of ['all', 'heldOut']) {
	const basis = pca.bases[key];
	const centre = (c: number) => {
		const p = basis.points.filter((q: { c: number }) => q.c === c);
		return [mean(p.map((q: { x: number }) => q.x)), mean(p.map((q: { y: number }) => q.y))];
	};
	let ok = 0;
	const nC = 200;
	for (let i = 0; i < nC; i++) {
		const j = i < nC / 2 ? i : testLabels.length - 1 - (i - nC / 2);
		const [x, y] = projectPCA(basis, testFeats[j]);
		const want = 10 + testLabels[j];
		const other = want === 10 ? 11 : 10;
		const [ax, ay] = centre(want);
		const [bx, by] = centre(other);
		if ((x - ax) ** 2 + (y - ay) ** 2 < (x - bx) ** 2 + (y - by) ** 2) ok++;
	}
	console.log(`PCA '${key}': ${ok}/${nC} held-out test drawings closer to their own class centre`);
}

const ftPath = join(dir, 'finetune.json');
const ft = JSON.parse(readFileSync(ftPath, 'utf8'));
ft.jsHeadCheck = {
	note: 'tinycnn_finetune.ts trainHead (full-batch gradient descent, 200 epochs, lr 0.5 on scale-normalised features, l2 1e-4) on the same N=16 per class subsets as the Python head-only runs; accuracy on the 3,000 held-out test drawings. Script: pipeline/parity/tinycnn_finetune.check.ts',
	n: meta.n,
	jsAcc: accs.map((a) => Number(a.toFixed(4))),
	jsMean: Number(jsMean.toFixed(4)),
	pythonMean: Number(pyMean.toFixed(4)),
	timingMs: timing,
	checked: new Date().toLocaleDateString('sv-SE')
};
writeFileSync(ftPath, JSON.stringify(ft, null, 1));
