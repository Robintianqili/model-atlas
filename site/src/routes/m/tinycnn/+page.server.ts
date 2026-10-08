import { readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { PageServerLoad } from './$types';
import type { TinyCNNSpec } from '$lib/models/tinycnn';

// Build time (prerender): embed spec.json in the page so the overview, the training facts and the
// article render before any fetch, and measure the files this page actually loads.
const DIR = resolve('static', 'models', 'tinycnn');
const size = (f: string) => statSync(join(DIR, f)).size;

export const load: PageServerLoad = () => {
	const spec = JSON.parse(readFileSync(join(DIR, 'spec.json'), 'utf8')) as TinyCNNSpec;
	return {
		spec,
		bytes: {
			// loaded on every visit
			auto: size('spec.json') + size('weights.bin') + size('presets.json') + size('channels.json'),
			// loaded only when the learner opens the out-of-distribution check
			heldout: size('heldout.json')
		}
	};
};
