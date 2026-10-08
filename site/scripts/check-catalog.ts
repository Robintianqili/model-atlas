/**
 * Catalog consistency check: `npm run check:catalog` (node runs this .ts file directly).
 * Fails (exit 1) on duplicate slugs, dangling references or missing Chinese UI labels
 * (including the plain-string runtime/params/size/engine texts, translated in catalog/tech-zh.ts).
 */
import { AREAS, AREA_BY_SLUG } from '../src/lib/catalog/areas.ts';
import { MODELS, MODEL_BY_SLUG } from '../src/lib/catalog/models.ts';
import { CONCEPTS, CONCEPT_BY_SLUG, SECTIONS } from '../src/lib/catalog/concepts.ts';
import { GLOSSARY, GLOSSARY_BY_ID } from '../src/lib/catalog/glossary.ts';
import { PATHS } from '../src/lib/catalog/paths.ts';
import { UI } from '../src/lib/i18n/strings.ts';
import { TECH_ZH, isLanguageNeutral } from '../src/lib/catalog/tech-zh.ts';

const errors: string[] = [];
const err = (m: string) => errors.push(m);
const bi = (where: string, v: { en?: string; zh?: string } | undefined, needZh = true) => {
	if (!v || !v.en) err(`${where}: missing English`);
	if (needZh && (!v || !v.zh)) err(`${where}: missing Chinese`);
};
const slugOk = (s: string) => /^[a-z0-9]+(-[a-z0-9]+)*$/.test(s);

const dup = (list: string[], what: string) => {
	const seen = new Set<string>();
	for (const s of list) {
		if (seen.has(s)) err(`duplicate ${what}: ${s}`);
		seen.add(s);
	}
};

dup(AREAS.map((a) => a.slug), 'area');
dup(MODELS.map((m) => m.slug), 'model slug');
dup(CONCEPTS.map((c) => c.slug), 'concept slug');
dup(GLOSSARY.map((g) => g.id), 'glossary id');

if (AREAS.length !== 11) err(`expected 11 areas, found ${AREAS.length}`);
for (const a of AREAS) {
	bi(`area ${a.slug} name`, a.name);
	bi(`area ${a.slug} blurb`, a.blurb);
}

for (const m of MODELS) {
	if (!slugOk(m.slug)) err(`model slug not lowercase-hyphenated: ${m.slug}`);
	if (!AREA_BY_SLUG[m.area]) err(`model ${m.slug}: unknown area ${m.area}`);
	bi(`model ${m.slug} name`, m.name);
	bi(`model ${m.slug} family`, m.family);
	bi(`model ${m.slug} hook`, m.hook);
	bi(`model ${m.slug} shows`, m.shows);
	if (m.notes) bi(`model ${m.slug} notes`, m.notes);
	if (m.tierNote) bi(`model ${m.slug} tierNote`, m.tierNote);
	for (const k of ['runtime', 'params', 'size'] as const) {
		const s = m[k];
		if (s && !isLanguageNeutral(s) && !TECH_ZH[s]) err(`model ${m.slug} ${k}: no Chinese in catalog/tech-zh.ts for "${s}"`);
	}
	if (m.inputs.length === 0) err(`model ${m.slug}: no inputs`);
	for (const p of m.prereqs) if (!MODEL_BY_SLUG[p]) err(`model ${m.slug}: unknown prereq ${p}`);
	if (m.prereqs.includes(m.slug)) err(`model ${m.slug}: lists itself as a prereq`);
}

const sectionSlugs = new Set(SECTIONS.map((s) => s.slug));
for (const c of CONCEPTS) {
	if (!slugOk(c.slug)) err(`concept slug not lowercase-hyphenated: ${c.slug}`);
	if (!sectionSlugs.has(c.section)) err(`concept ${c.slug}: unknown section ${c.section}`);
	bi(`concept ${c.slug} title`, c.title);
	bi(`concept ${c.slug} interaction`, c.interaction);
	if (!isLanguageNeutral(c.engine) && !TECH_ZH[c.engine]) err(`concept ${c.slug} engine: no Chinese in catalog/tech-zh.ts for "${c.engine}"`);
	for (const l of c.links) if (!MODEL_BY_SLUG[l]) err(`concept ${c.slug}: unknown model link ${l}`);
}

for (const g of GLOSSARY) {
	bi(`glossary ${g.id} term`, g.term);
	bi(`glossary ${g.id} def`, g.def);
	for (const r of g.related ?? []) if (!GLOSSARY_BY_ID[r]) err(`glossary ${g.id}: unknown related id ${r}`);
}

for (const p of PATHS) {
	for (const s of p.steps) {
		const ok = s.kind === 'model' ? MODEL_BY_SLUG[s.slug] : CONCEPT_BY_SLUG[s.slug];
		if (!ok) err(`path ${p.id}: unknown ${s.kind} ${s.slug}`);
	}
}

for (const [k, v] of Object.entries(UI)) bi(`UI string ${k}`, v as { en: string; zh: string });

const liveModels = MODELS.filter((m) => m.status === 'live').map((m) => m.slug);
const liveConcepts = CONCEPTS.filter((c) => c.status === 'live').map((c) => c.slug);
const byTier = (t: string) => MODELS.filter((m) => m.tier === t).length;
const perArea = AREAS.map((a) => `${a.slug} ${MODELS.filter((m) => m.area === a.slug).length}`).join(', ');

console.log(`areas ${AREAS.length}; models ${MODELS.length} (P0 ${byTier('P0')}, V1 ${byTier('V1')}, P1 ${byTier('P1')}, P2 ${byTier('P2')})`);
console.log(`models per area: ${perArea}`);
console.log(`concepts ${CONCEPTS.length} in ${SECTIONS.length} sections (v1: ${CONCEPTS.filter((c) => c.v1).length})`);
console.log(`glossary ${GLOSSARY.length} terms; paths ${PATHS.length}; UI strings ${Object.keys(UI).length}`);
console.log(`live models: ${liveModels.join(', ')}; live concepts: ${liveConcepts.join(', ')}`);

if (errors.length) {
	console.error(`\n${errors.length} problem(s):`);
	for (const e of errors) console.error(' - ' + e);
	process.exit(1);
}
console.log('catalog OK');
