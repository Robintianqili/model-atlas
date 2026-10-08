// Parity test: TinyCNN TypeScript forward pass vs the PyTorch fp32 reference.
// Run from the repo root:  node pipeline/parity/tinycnn.parity.ts
// - compares every traced layer on the reference.json inputs (bar: max |diff| <= 1e-4)
// - checks convDetail / poolDetail / denseDetail / softmaxDetail against the trace
// - writes the measured value into site/static/models/tinycnn/spec.json -> parity
// - --no-write: run every check but leave spec.json untouched; prints the record it would write as one
//   `PARITY_JSON {...}` line instead (the DevKit always runs legacy parity this way)
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { load } from '../../site/src/lib/models/tinycnn.ts';
import type { FetchLike } from '../../site/src/lib/models/tinycnn.ts';

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, '..', '..', 'site', 'static', 'models', 'tinycnn');
const BAR = 1e-4;
const NO_WRITE = process.argv.includes('--no-write');

const fsFetch: FetchLike = async (url: string) => {
	const buf = readFileSync(url);
	return {
		ok: true,
		status: 200,
		json: async () => JSON.parse(buf.toString('utf8')),
		arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer
	};
};

const model = await load(fsFetch, dir);
const ref = JSON.parse(readFileSync(join(dir, 'reference.json'), 'utf8')) as {
	inputs: { label: string; input: number[]; layers: Record<string, { shape: number[]; data: number[] }> }[];
};

let maxDiff = 0;
let worst = '';
let detailDiff = 0;
let detailWorst = '';
let checkedCells = 0;
const ids = model.spec.layers.map((l) => l.id);
for (const [n, r] of ref.inputs.entries()) {
	const tr = model.forward(r.input);
	if (tr.order.join() !== ids.join()) throw new Error('trace order differs from spec.layers');
	for (const id of ids) {
		const want = r.layers[id];
		const got = tr.t[id];
		if (!want) throw new Error(`reference lacks layer ${id}`);
		if (want.shape.join() !== got.shape.join()) throw new Error(`${id}: shape ${got.shape} vs ${want.shape}`);
		let d = 0;
		for (let i = 0; i < want.data.length; i++) d = Math.max(d, Math.abs(want.data[i] - got.data[i]));
		if (d > maxDiff) {
			maxDiff = d;
			worst = `${id} (input ${n}, ${r.label})`;
		}
	}
	// every cell of every conv / pool layer, every class of the dense layer
	for (const L of model.spec.layers) {
		const t = tr.t[L.id];
		if (L.op === 'conv2d' || L.op === 'maxpool2d') {
			const [C, H, W] = t.shape;
			for (let c = 0; c < C; c++) {
				for (let y = 0; y < H; y++) {
					for (let x = 0; x < W; x++) {
						const traced = t.data[(c * H + y) * W + x];
						let v: number;
						if (L.op === 'conv2d') {
							const det = model.convDetail(tr, L.id, c, y, x);
							v = det.total;
							// internal consistency: products and channel sums
							let s = det.bias;
							for (let k = 0; k < det.inChannels; k++) s += det.channelSums[k];
							const dc = Math.abs(Math.fround(s) - det.total);
							if (dc > detailDiff) {
								detailDiff = dc;
								detailWorst = `convDetail sum ${L.id}[${c},${y},${x}]`;
							}
						} else {
							v = model.poolDetail(tr, L.id, c, y, x).value;
						}
						const d = Math.abs(v - traced);
						if (d > detailDiff) {
							detailDiff = d;
							detailWorst = `${L.op} ${L.id}[${c},${y},${x}]`;
						}
						checkedCells++;
					}
				}
			}
		}
	}
	for (let k = 0; k < tr.t.dense.data.length; k++) {
		const det = model.denseDetail(tr, k);
		const d = Math.abs(det.logit - det.traced);
		if (d > detailDiff) {
			detailDiff = d;
			detailWorst = `dense[${k}]`;
		}
		checkedCells++;
	}
	const sm = model.softmaxDetail(tr);
	for (let k = 0; k < sm.probs.length; k++) {
		const d = Math.abs(sm.exps[k] / sm.sumExp - sm.probs[k]);
		if (d > detailDiff) {
			detailDiff = d;
			detailWorst = `softmax[${k}]`;
		}
	}
	console.log(`input ${n} (${r.label}): top = ${model.spec.classes[sm.top].en} p=${sm.probs[sm.top].toFixed(4)}`);
}

// receptive-field sanity: pool2 cell (3,3) must see input rows/cols 12..27
const rf = model.receptiveField('pool2', 3, 3);
if (rf.y0 !== 12 || rf.y1 !== 27 || rf.x0 !== 12 || rf.x1 !== 27) throw new Error(`receptiveField pool2(3,3) = ${JSON.stringify(rf)}`);

const pass = maxDiff <= BAR && detailDiff <= BAR;
console.log(`layers: max |diff| = ${maxDiff.toExponential(3)} at ${worst}`);
console.log(`details (${checkedCells} cells): max |diff| = ${detailDiff.toExponential(3)} at ${detailWorst || '-'}`);
console.log(pass ? 'PASS' : 'FAIL');

const specPath = join(dir, 'spec.json');
const spec = JSON.parse(readFileSync(specPath, 'utf8'));
const shippedMax = spec.parity?.max_abs_diff;
const parity = {
	against: 'pytorch-fp32',
	max_abs_diff: Number(maxDiff.toPrecision(3)),
	checked: new Date().toLocaleDateString('sv-SE'),
	bar: BAR,
	pass,
	inputs: ref.inputs.length,
	layers: ids.length,
	details_max_abs_diff: Number(detailDiff.toPrecision(3)),
	script: 'pipeline/parity/tinycnn.parity.ts'
};
if (NO_WRITE) {
	console.log(`--no-write: spec.json left unchanged (shipped max_abs_diff ${shippedMax}, measured ${parity.max_abs_diff})`);
	console.log('PARITY_JSON ' + JSON.stringify(parity));
} else {
	spec.parity = parity;
	writeFileSync(specPath, JSON.stringify(spec, null, 1));
}
if (!pass) process.exit(1);
