// The home page's data, at build time (redesign spec §6.3, §10.5): the catalog index (lessons, groups, counts, input
// counts), the learning paths, and the hero preview: the shipped TinyCNN weights run on the lesson's HOUSE drawing (the
// same preprocessing as the lesson), reduced to 8-bit maps. The page ships no catalog modules and no model code.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fromParts, preprocessStrokes, type TinyCNNSpec } from '$lib/models/tinycnn';
import { HOUSE } from '$lib/pages/tinycnn/sketches';
import { simplifyStroke } from '$lib/pages/tinycnn/data';
import { catalogIndex, learningPaths } from '$lib/kit/catalog/lessons';
import type { HeroData } from '$lib/kit/catalog/types';
import type { PageServerLoad } from './$types';

const b64 = (bytes: Uint8Array) => Buffer.from(bytes).toString('base64');
const maxAbs = (a: ArrayLike<number>) => {
	let m = 0;
	for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
	return m || 1;
};
/** signed values over [-max, max] as bytes: (v / max + 1) / 2 × 255 */
const signed = (a: ArrayLike<number>, max: number) => b64(Uint8Array.from(a, (v) => Math.round(((v / max + 1) / 2) * 255)));

function hero(): HeroData {
	const dir = resolve('static', 'models', 'tinycnn');
	const spec = JSON.parse(readFileSync(resolve(dir, 'spec.json'), 'utf8')) as TinyCNNSpec;
	const w = readFileSync(resolve(dir, 'weights.bin'));
	const model = fromParts(spec, w.buffer.slice(w.byteOffset, w.byteOffset + w.byteLength));
	const pre = preprocessStrokes(HOUSE.map((s) => simplifyStroke(s.points)).filter((p) => p.length > 0));
	const tr = model.forward(pre.input);
	const conv = tr.t.conv1_1.data;
	const pool1 = tr.t.pool1.data;
	const pool2 = tr.t.pool2.data;
	const probs = tr.t.softmax.data;
	const max = { conv: maxAbs(conv), pool1: maxAbs(pool1), pool2: maxAbs(pool2) };
	const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 5);
	return {
		input: b64(Uint8Array.from(pre.input, (v) => Math.round(Math.min(1, Math.max(0, v)) * 255))),
		conv: signed(conv, max.conv),
		pool1: signed(pool1, max.pool1),
		pool2: signed(pool2, max.pool2),
		max,
		top: order.map((k) => ({ en: spec.classes[k].en, zh: spec.classes[k].zh, p: probs[k] })),
		params: spec.params
	};
}

/** "Start here" (spec §6.3): four lessons, four kinds of input */
const START = ['tinycnn', 'tinygpt', 'loss-gradient-descent', 'mlp-playground'];

export const load: PageServerLoad = () => {
	const { lessons, groups, counts, inputs, sprites } = catalogIndex();
	return {
		start: START.map((s) => lessons.find((l) => l.slug === s)).filter((l) => l !== undefined),
		groups,
		counts,
		inputs,
		sprites,
		paths: learningPaths(lessons),
		hero: hero()
	};
};
