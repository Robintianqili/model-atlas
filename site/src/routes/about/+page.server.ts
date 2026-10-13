// /about/ at build time (redesign spec §6.7): static/licenses.json grouped by kind for the credits, the scope counts from
// the catalog (the page never goes stale when a lesson goes live), and TinyCNN's measured parity for the example label.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readLicenses, type LicenseEntry } from '$lib/server/assets';
import { MODELS } from '$lib/catalog/models';
import { CONCEPTS } from '$lib/catalog/concepts';
import { AREAS } from '$lib/catalog/areas';
import type { PageServerLoad } from './$types';

type Kind = 'data' | 'libs' | 'code' | 'papers';
/** licenses.json has no kind field: sort each row by its name, licence and use */
function kindOf(r: LicenseEntry): Kind {
	const s = `${r.name} ${r.license}`;
	if (/\bfont\b|Fontsource|KaTeX|\bSvelte(Kit)?\b/i.test(r.name)) return 'libs';
	if (/^(Papers?|Book|Article|Journal|arXiv|Web page)\b/i.test(r.license) || /\bpaper\b/i.test(r.license)) return 'papers';
	if (/dataset|vocabulary|tokenizer \(|encoding|karate club|TinyStories|\(numpy bitmaps\)/i.test(s) && !/ideas?\b/i.test(r.name)) return 'data';
	return 'code';
}

export const load: PageServerLoad = () => {
	const rows = readLicenses();
	const by = (k: Kind) => rows.filter((r) => kindOf(r) === k);
	const liveModels = MODELS.filter((m) => m.status === 'live');
	const spec = JSON.parse(readFileSync(resolve('static', 'models', 'tinycnn', 'spec.json'), 'utf8')) as { precision?: string; parity?: { max_abs_diff?: number; inputs?: number } };
	return {
		licences: [
			{ id: 'data', title: { en: 'Datasets and vocabularies', zh: '数据集与词表' }, rows: by('data') },
			{ id: 'libs', title: { en: 'Fonts and libraries', zh: '字体与程序库' }, rows: by('libs') },
			{ id: 'code', title: { en: 'Code and ideas from other projects', zh: '借鉴的代码与想法' }, rows: by('code') },
			{ id: 'papers', title: { en: 'Papers and books', zh: '论文与书籍' }, rows: by('papers') }
		],
		licenceCount: rows.length,
		parity: { precision: spec.precision ?? 'fp32', diff: spec.parity?.max_abs_diff, inputs: spec.parity?.inputs },
		scope: {
			liveModels: liveModels.length,
			liveConcepts: CONCEPTS.filter((c) => c.status === 'live').length,
			liveAreas: new Set(liveModels.map((m) => m.area)).size,
			models: MODELS.length,
			concepts: CONCEPTS.length,
			areas: AREAS.length
		}
	};
};
