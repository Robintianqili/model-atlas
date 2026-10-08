import { statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { PageServerLoad } from './$types';

// Build time only: the bytes of the three files this page fetches (spec.json, weights.bin,
// tokenizer.json). The checkpoints and replay files in the same folder belong to the
// next-token-prediction page and are never loaded here.
export const load: PageServerLoad = () => {
	const dir = resolve('static/models/tinygpt');
	let bytes = 0;
	for (const f of ['spec.json', 'weights.bin', 'tokenizer.json']) bytes += statSync(join(dir, f)).size;
	return { tinygptBytes: bytes };
};
