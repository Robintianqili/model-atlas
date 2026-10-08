import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { FinetuneJson } from '$lib/pages/transfer-learning/data';
import type { PageServerLoad } from './$types';

// Build time only: the precomputed fine-tuning curves (8 KB) are read from the shipped
// static/models/tinycnn/finetune.json, so Part 2 is in the prerendered HTML (no fetch needed).
export const load: PageServerLoad = () => {
	const finetune = JSON.parse(readFileSync(resolve('static/models/tinycnn/finetune.json'), 'utf8')) as FinetuneJson;
	return { finetune };
};
