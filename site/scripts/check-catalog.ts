/**
 * Catalog consistency check: `npm run check:catalog` (node runs this .ts file directly).
 * Fails (exit 1) on duplicate slugs, dangling references or missing Chinese UI labels
 * (including the plain-string runtime/params/size/engine texts, translated in catalog/tech-zh.ts), and when
 * src/lib/catalog/glossary-names.ts is stale (regenerate: node scripts/gen-glossary-names.ts).
 * Warns (redesign content rules; gates after the content wave): hooks over 100 English or 50 Chinese characters,
 * concepts without a hook, short names over 24 characters.
 */
import { AREAS, AREA_BY_SLUG } from '../src/lib/catalog/areas.ts';
import { MODELS, MODEL_BY_SLUG } from '../src/lib/catalog/models.ts';
import { CONCEPTS, CONCEPT_BY_SLUG, SECTIONS } from '../src/lib/catalog/concepts.ts';
import { GLOSSARY, GLOSSARY_BY_ID } from '../src/lib/catalog/glossary.ts';
import { PATHS } from '../src/lib/catalog/paths.ts';
import { CATALOG_UI, LESSON_UI, UI } from '../src/lib/i18n/strings.ts';
import { TECH_ZH, isLanguageNeutral } from '../src/lib/catalog/tech-zh.ts';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const errors: string[] = [];
const err = (m: string) => errors.push(m);
const warnings: string[] = [];
const warn = (m: string) => warnings.push(m);
/** characters as a reader counts them (a CJK character or an emoji is one) */
const chars = (s: string | undefined) => [...(s ?? '')].length;
const HOOK_EN = 100;
const HOOK_ZH = 50;
const SHORT = 24;
const hookRule = (where: string, v: { en?: string; zh?: string } | undefined) => {
	if (!v) return;
	if (chars(v.en) > HOOK_EN) warn(`${where} hook has ${chars(v.en)} English characters (at most ${HOOK_EN})`);
	if (chars(v.zh) > HOOK_ZH) warn(`${where} hook has ${chars(v.zh)} Chinese characters (at most ${HOOK_ZH})`);
};
const shortRule = (where: string, v: { en?: string; zh?: string } | undefined) => {
	if (!v) return;
	bi(`${where} short`, v);
	for (const l of ['en', 'zh'] as const) if (chars(v[l]) > SHORT) warn(`${where} short name '${v[l]}' has ${chars(v[l])} characters (at most ${SHORT})`);
};
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
	if (m.status === 'live') hookRule(`model ${m.slug}`, m.hook);
	shortRule(`model ${m.slug}`, m.short);
	if (m.lesson?.minutes !== undefined && !(Number.isInteger(m.lesson.minutes) && m.lesson.minutes > 0)) err(`model ${m.slug}: lesson.minutes must be a positive whole number`);
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
	if (c.hook) {
		bi(`concept ${c.slug} hook`, c.hook);
		hookRule(`concept ${c.slug}`, c.hook);
	}
	shortRule(`concept ${c.slug}`, c.short);
	if (c.lesson?.minutes !== undefined && !(Number.isInteger(c.lesson.minutes) && c.lesson.minutes > 0)) err(`concept ${c.slug}: lesson.minutes must be a positive whole number`);
}
const hookless = CONCEPTS.filter((c) => c.status === 'live' && !c.hook).map((c) => c.slug);
if (hookless.length) warn(`${hookless.length} live concepts have no hook yet (cards use the lesson summary): ${hookless.join(', ')}`);

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
for (const [k, v] of Object.entries(LESSON_UI)) bi(`LESSON_UI string ${k}`, v as { en: string; zh: string });
for (const [k, v] of Object.entries(CATALOG_UI)) bi(`CATALOG_UI string ${k}`, v as { en: string; zh: string });

// the names <Term> shows without loading the definitions (lazy glossary); a copy of this script without its generator
// next to it (promote's tests) skips the check
const gen = join(dirname(fileURLToPath(import.meta.url)), 'gen-glossary-names.ts');
if (existsSync(gen)) {
	const { namesStale } = (await import(pathToFileURL(gen).href)) as { namesStale: (site: string) => Promise<boolean> };
	if (await namesStale(join(dirname(fileURLToPath(import.meta.url)), '..'))) err('src/lib/catalog/glossary-names.ts is stale: run node scripts/gen-glossary-names.ts (atlas promote --apply does it)');
}

const liveModels = MODELS.filter((m) => m.status === 'live').map((m) => m.slug);
const liveConcepts = CONCEPTS.filter((c) => c.status === 'live').map((c) => c.slug);
const byTier = (t: string) => MODELS.filter((m) => m.tier === t).length;
const perArea = AREAS.map((a) => `${a.slug} ${MODELS.filter((m) => m.area === a.slug).length}`).join(', ');

console.log(`areas ${AREAS.length}; models ${MODELS.length} (P0 ${byTier('P0')}, V1 ${byTier('V1')}, P1 ${byTier('P1')}, P2 ${byTier('P2')})`);
console.log(`models per area: ${perArea}`);
console.log(`concepts ${CONCEPTS.length} in ${SECTIONS.length} sections (v1: ${CONCEPTS.filter((c) => c.v1).length})`);
console.log(`glossary ${GLOSSARY.length} terms; paths ${PATHS.length}; UI strings ${Object.keys(UI).length} + lesson ${Object.keys(LESSON_UI).length} + catalog ${Object.keys(CATALOG_UI).length}`);
console.log(`live models: ${liveModels.join(', ')}; live concepts: ${liveConcepts.join(', ')}`);

if (warnings.length) {
	console.log(`\n${warnings.length} warning(s) (content rules; gates after the content wave):`);
	for (const w of warnings) console.log(' - ' + w);
}

if (errors.length) {
	console.error(`\n${errors.length} problem(s):`);
	for (const e of errors) console.error(' - ' + e);
	process.exit(1);
}
console.log('catalog OK');
