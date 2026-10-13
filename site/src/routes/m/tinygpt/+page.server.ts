import { readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { PageServerLoad } from './$types';

// Build time only: the bytes of the three files this page fetches (spec.json, weights.bin, tokenizer.json), and the
// few spec facts the top bar, the cover and the About sheet show before the model loads (the full spec arrives with
// the engine). The checkpoints and replay files in the same folder belong to the next-token-prediction page and are
// never loaded here.
export const load: PageServerLoad = () => {
	const dir = resolve('static/models/tinygpt');
	let bytes = 0;
	for (const f of ['spec.json', 'weights.bin', 'tokenizer.json']) bytes += statSync(join(dir, f)).size;
	const spec = JSON.parse(readFileSync(join(dir, 'spec.json'), 'utf8'));
	return {
		// the lesson app (redesign spec §6.1): no site header or footer around this page
		chrome: 'lesson' as const,
		tinygptBytes: bytes,
		facts: { params: spec.params as number, precision: spec.precision as string, input: spec.input as { shape: (number | string)[] }, parity: spec.parity as { against?: string; max_abs_diff?: number } }
	};
};
