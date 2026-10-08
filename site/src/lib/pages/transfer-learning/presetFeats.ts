/**
 * Frozen features of the shipped ladder / envelope drawings (heldout.json: 32 train + 100 test per
 * class), computed once per page load and shared by the guided lesson and Part 1, so the worker
 * does the 264 forward passes only once. Same order as heldout.json.
 */
import { decodeB64, type HeldoutJson } from './data';
import type { Engine } from './engine';

export interface PresetFeats {
	/** model inputs (28×28, 0..1) by class: [ladder[], envelope[]] */
	inputs: { train: Float32Array[][]; test: Float32Array[][] };
	/** the 256 flatten features of each input, same layout */
	feats: { train: Float32Array[][]; test: Float32Array[][] };
	/** time for the features, ms, and where they were computed */
	ms: number;
	where: 'worker' | 'main';
}

let cache: { engine: Engine; heldout: HeldoutJson; p: Promise<PresetFeats> } | null = null;

export function presetFeatures(engine: Engine, heldout: HeldoutJson): Promise<PresetFeats> {
	if (cache && cache.engine === engine && cache.heldout === heldout) return cache.p;
	const p = compute(engine, heldout);
	cache = { engine, heldout, p };
	// a failed run may be retried by the next caller
	p.catch(() => {
		if (cache?.p === p) cache = null;
	});
	return p;
}

async function compute(engine: Engine, heldout: HeldoutJson): Promise<PresetFeats> {
	const inputs = {
		train: heldout.classes.slice(0, 2).map((c) => c.train.map(decodeB64)),
		test: heldout.classes.slice(0, 2).map((c) => c.test.map(decodeB64))
	};
	const t0 = performance.now();
	const flat = [...inputs.train[0], ...inputs.train[1], ...inputs.test[0], ...inputs.test[1]];
	const out = await engine.featuresMany(flat);
	const ms = performance.now() - t0;
	const n0 = inputs.train[0].length;
	const n1 = inputs.train[1].length;
	const m0 = inputs.test[0].length;
	return {
		inputs,
		feats: {
			train: [out.slice(0, n0), out.slice(n0, n0 + n1)],
			test: [out.slice(n0 + n1, n0 + n1 + m0), out.slice(n0 + n1 + m0)]
		},
		ms,
		where: engine.where()
	};
}
