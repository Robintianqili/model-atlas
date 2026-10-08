// atlas promote <slug> [--dry-run]   (design §4.12, §6.4 merge step 2)
// The only writer of shared files, and only of one: static/licenses.json, to which it appends the manifest's credits
// (re-read right before writing, append only; a row whose url is already listed, under any name, is skipped: K0, Quick,
// Draw! had two rows). Everything else is
// printed as a patch for the main session to apply by hand (rewriting TS catalog files by script is fragile):
//   - src/lib/catalog/models.ts: status 'live', lesson { steps, summary }, and the manifest's catalog fields
//   - src/lib/catalog/tech-zh.ts: Chinese for runtime / params / size strings that are not language-neutral
//   - src/lib/catalog/glossary.ts: manifest.glossary terms that the glossary does not have yet
// Then runs scripts/check-catalog.ts (npm run check:catalog) on the catalog as it is now: re-run it after applying the patch.
// K0: the patch also prints the entry's current hook and shows (stale promises: TinyViT's logit lens, gridworld's SARSA)
// and what the new catalog text costs (K1: since the prep for wave 2b the catalog tables reach the list pages and the
// /m/[slug] route through page data, not the flagship shells; glossary terms still reach every page that uses Term,
// the flagships included). A tech-zh row whose English shares most words with an existing row is printed with that row.
// The patch is also written to .atlas/<slug>/promote.patch.txt. devOnly packages are never promoted (AK701).
// K2: a concept package (type 'concept', /train/<slug>/) gets the concepts.ts patch instead (status 'live', lesson, and
// interaction / engine / mode / links only when the manifest's catalog block gives them); models.ts is never touched.
// concepts.ts text reaches every /train/<slug>/ page (ConceptPageShell imports CONCEPT_BY_SLUG) and the list pages.
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';
import type { Credit, I18n } from '../../../src/lib/atlas/types/index.ts';
import { AtlasError } from '../errors.ts';
import { runNode } from '../lib/child.ts';
import { posix, writeText } from '../lib/io.ts';
import { openPackage } from '../lib/pkg.ts';
import { roots } from '../root.ts';

const q = (s: string) => `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
const f = (t: I18n) => `f(${q(t.en)}, ${q(t.zh ?? t.en)})`;

/** Append credits to licenses.json (re-read, append only). Returns the rows added. */
export function appendCredits(file: string, credits: Credit[], dry: boolean): { added: Credit[]; skipped: Credit[]; sameUrl: { credit: Credit; listedAs: string }[] } {
	const text = existsSync(file) ? readFileSync(file, 'utf8').replace(/^﻿/, '') : '[]\n';
	const rows = JSON.parse(text) as Credit[];
	if (!Array.isArray(rows)) throw new AtlasError('AK704', `${posix(file)} is not a JSON array`);
	const added: Credit[] = [];
	const skipped: Credit[] = [];
	const sameUrl: { credit: Credit; listedAs: string }[] = [];
	for (const c of credits) {
		const listed = rows.find((r) => r.url === c.url) ?? added.find((r) => r.url === c.url);
		if (listed) {
			skipped.push(c);
			if (listed.name !== c.name) sameUrl.push({ credit: c, listedAs: listed.name });
		} else added.push({ name: c.name, url: c.url, license: c.license, usedFor: c.usedFor, attribution: c.attribution });
	}
	if (added.length && !dry) {
		const again = JSON.parse(readFileSync(file, 'utf8').replace(/^﻿/, '')) as Credit[]; // re-read right before writing
		writeText(file, JSON.stringify([...again, ...added], null, 2) + '\n');
	}
	return { added, skipped, sameUrl };
}

/** An existing tech-zh row whose English shares at least 70% of its words with `en` (K1). */
export function similarRow(en: string, rows: Record<string, string>): string | null {
	const words = (s: string) => new Set(s.toLowerCase().match(/[a-z0-9]+/g) ?? []);
	const a = words(en);
	let best: string | null = null;
	let bestScore = 0;
	for (const k of Object.keys(rows)) {
		const b = words(k);
		let both = 0;
		for (const w of a) if (b.has(w)) both++;
		const score = both / Math.max(1, Math.min(a.size, b.size));
		if (score >= 0.7 && score > bestScore) {
			best = k;
			bestScore = score;
		}
	}
	return best;
}

export async function main(args: string[]): Promise<number> {
	const dry = args.includes('--dry-run');
	const pos = args.filter((a) => !a.startsWith('--'));
	for (const a of args) if (a.startsWith('--') && a !== '--dry-run') throw new AtlasError('AK301', `unknown flag ${a}`);
	if (pos.length !== 1) throw new AtlasError('AK301', 'usage: atlas promote <slug> [--dry-run]');
	const slug = pos[0];
	const r = roots();
	const o = await openPackage(slug);
	const m = o.l.pkg.manifest;
	if (m.devOnly) throw new AtlasError('AK701', `'${slug}' is devOnly (lab only): it is never promoted to the catalog`);
	if ((m.type as string) === 'concept') return promoteConcept(o, dry);
	const catFile = join(r.site, 'src', 'lib', 'catalog', 'models.ts');
	const cat = ((await import(pathToFileURL(catFile).href)) as { MODELS: { slug: string; status: string; runtime: string; hook?: I18n; shows?: I18n }[] }).MODELS;
	const entry = cat.find((e) => e.slug === slug);
	if (!entry) throw new AtlasError('AK701', `'${slug}' is not in the catalog`, { fix: 'a maintainer adds the catalog entry first' });
	if (!Array.isArray(m.credits)) throw new AtlasError('AK704', 'manifest.credits is missing');
	for (const [i, c] of m.credits.entries())
		for (const k of ['name', 'url', 'license', 'usedFor', 'attribution'] as const)
			if (typeof c[k] !== 'string' || !c[k].trim()) throw new AtlasError('AK704', `credits[${i}] has no ${k}`);

	const lic = join(r.site, 'static', 'licenses.json');
	const { added, skipped, sameUrl } = appendCredits(lic, m.credits, dry);

	const { compiled } = await import('./lesson.ts');
	const { c, spec } = await compiled(o);
	const lessonSummary = o.l.pkg.lesson?.summary ?? c.summary;
	const tech = (await import(pathToFileURL(join(r.site, 'src', 'lib', 'catalog', 'tech-zh.ts')).href)) as { TECH_ZH: Record<string, string>; isLanguageNeutral(s: string): boolean };
	const glossary = (await import(pathToFileURL(join(r.site, 'src', 'lib', 'catalog', 'glossary.ts')).href)) as { GLOSSARY_BY_ID: Record<string, unknown> };
	const cf = m.catalog ?? {};
	const lines: string[] = [];
	lines.push(`// ---- src/lib/catalog/models.ts, the entry { slug: '${slug}', ... }: set or replace these fields`);
	lines.push(`status: 'live',`);
	lines.push(`lesson: { steps: ${c.steps.length}, summary: ${f(lessonSummary)} },`);
	if (cf.hook) lines.push(`hook: ${f(cf.hook)},`);
	if (cf.shows) lines.push(`shows: ${f(cf.shows)},`);
	const techRows: string[] = [];
	for (const k of ['runtime', 'params', 'size'] as const) {
		const v = cf[k];
		if (!v) continue;
		lines.push(`${k}: ${q(v.en)},`);
		if (!tech.isLanguageNeutral(v.en) && !tech.TECH_ZH[v.en]) {
			techRows.push(`${q(v.en)}: ${q(v.zh ?? '')},${v.zh ? '' : ' // TODO: zh missing (check:catalog fails without it)'}`);
			const near = similarRow(v.en, tech.TECH_ZH);
			if (near) techRows.push(`//   similar row already there: ${q(near)}: ${q(tech.TECH_ZH[near])} (reuse its string, or its wording)`);
		}
	}
	if (!cf.params && spec) lines.push(`// measured: params ${spec.params.toLocaleString('en-US')} (spec.json); set params: '≈${spec.params >= 1000 ? `${(spec.params / 1000).toFixed(1)}k` : spec.params}' if the catalog text is a design estimate`);
	if (cf.prereqs) lines.push(`prereqs: [${cf.prereqs.map(q).join(', ')}],`);
	lines.push('');
	lines.push('// ---- src/lib/catalog/tech-zh.ts, TECH_ZH: add');
	lines.push(...(techRows.length ? techRows : ['// (nothing: every string is language-neutral or already has an entry)']));
	lines.push('');
	lines.push("// ---- src/lib/catalog/glossary.ts, GLOSSARY: add only the terms other pages (or the /glossary/ page) need: every term");
	lines.push("//      reaches every page that uses Term (the flagships too), and the package's steps already show its own manifest.glossary terms (K0)");
	const newTerms = (m.glossary ?? []).filter((g) => !glossary.GLOSSARY_BY_ID[g.id]);
	lines.push(...(newTerms.length ? newTerms.map((g) => `{ id: ${q(g.id)}, term: ${f(g.term)}, def: ${f(g.def)}${g.related?.length ? `, related: [${g.related.map(q).join(', ')}]` : ''} },`) : ['// (nothing new)']));
	// the entry's promises as the catalog has them now: the page must show what they say
	lines.push('');
	lines.push('// ---- check by hand: the entry\'s hook and shows must promise only what the page has (replace the text if not)');
	for (const k of ['hook', 'shows'] as const) {
		const now = entry[k];
		if (now && !cf[k]) lines.push(`// ${k} now: ${now.en}`);
	}
	const patch = lines.join('\n') + '\n';
	// what the new catalog text costs (K1): glossary terms reach every page that uses Term (the flagship shells too);
	// models.ts and tech-zh.ts text reach the list pages and the /m/[slug] route (page data), not the flagships
	const gloText = newTerms.map((g) => `${g.id}${g.term.en}${g.term.zh ?? ''}${g.def.en}${g.def.zh ?? ''}`).join('\n');
	const gloCost = gloText ? gzipSync(Buffer.from(gloText)).length : 0;
	const addedText = lines.filter((l) => l && !l.startsWith('//')).join('\n');
	const shellCost = addedText ? gzipSync(Buffer.from(addedText)).length : 0;
	writeText(join(r.site, '.atlas', slug, 'promote.patch.txt'), patch);

	console.log(`atlas promote ${slug}${dry ? ' --dry-run' : ''}:`);
	console.log(`  licenses.json  ${dry ? 'would append' : 'appended'} ${added.length} credit(s)${skipped.length ? `; ${skipped.length} already listed` : ''}${added.length ? `: ${added.map((x) => x.name).join(', ')}` : ''}`);
	for (const x of sameUrl) console.log(`  licenses.json  '${x.credit.name}' not added: its url is already listed as '${x.listedAs}' (one row per source)`);
	console.log(`  text cost      ≈ ${shellCost} B gzip of catalog text in all; ≈ ${gloCost} B of it is glossary terms, which every page that uses Term carries (the flagships too: 300 KB budget in check:build); models and tech-zh text reach the list pages and the /m/[slug] route only`);
	console.log(`  catalog patch  (apply by hand; also in .atlas/${slug}/promote.patch.txt)\n`);
	console.log(patch.replace(/^/gm, '    '));
	if (entry.status === 'live') console.log(`  note           '${slug}' is already live in the catalog`);
	const script = join(r.site, 'scripts', 'check-catalog.ts');
	if (!existsSync(script)) {
		console.log('  check:catalog  skipped (no scripts/check-catalog.ts in this site)');
		return 0;
	}
	const cc = await runNode([script], { cwd: r.site, timeoutMs: 300_000 });
	console.log(`  check:catalog  ${cc.code === 0 ? 'ok' : 'FAILED'} on the current catalog (re-run it after applying the patch)`);
	if (cc.code !== 0) console.error((cc.stdout + cc.stderr).split('\n').filter((l) => /FAIL|error|✖/i.test(l)).slice(0, 8).join('\n'));
	return cc.code === 0 ? 0 : 1;
}

/** Lines of the tech-zh patch for one catalog string (K1 rules): nothing when language-neutral or already listed. */
function techRowsFor(en: string, zh: string | undefined, tech: { TECH_ZH: Record<string, string>; isLanguageNeutral(s: string): boolean }): string[] {
	if (tech.isLanguageNeutral(en) || tech.TECH_ZH[en]) return [];
	const out = [`${q(en)}: ${q(zh ?? '')},${zh ? '' : ' // TODO: zh missing (check:catalog fails without it)'}`];
	const near = similarRow(en, tech.TECH_ZH);
	if (near) out.push(`//   similar row already there: ${q(near)}: ${q(tech.TECH_ZH[near])} (reuse its string, or its wording)`);
	return out;
}

/** K2: promote a concept package: credits, then the concepts.ts / tech-zh.ts / glossary.ts patch, then check:catalog. */
async function promoteConcept(o: Awaited<ReturnType<typeof openPackage>>, dry: boolean): Promise<number> {
	const r = roots();
	const slug = o.slug;
	const m = o.l.pkg.manifest;
	const cat = ((await import(pathToFileURL(join(r.site, 'src', 'lib', 'catalog', 'concepts.ts')).href)) as { CONCEPTS: { slug: string; status: string; interaction?: I18n }[] }).CONCEPTS;
	const entry = cat.find((e) => e.slug === slug);
	if (!entry) throw new AtlasError('AK701', `'${slug}' is not in the concepts catalog (src/lib/catalog/concepts.ts)`, { fix: 'a maintainer adds the catalog entry first' });
	if (!Array.isArray(m.credits)) throw new AtlasError('AK704', 'manifest.credits is missing');
	for (const [i, c] of m.credits.entries())
		for (const k of ['name', 'url', 'license', 'usedFor', 'attribution'] as const)
			if (typeof c[k] !== 'string' || !c[k].trim()) throw new AtlasError('AK704', `credits[${i}] has no ${k}`);
	const lic = join(r.site, 'static', 'licenses.json');
	const { added, skipped, sameUrl } = appendCredits(lic, m.credits, dry);
	const { compiled } = await import('./lesson.ts');
	const { c } = await compiled(o);
	const lessonSummary = o.l.pkg.lesson?.summary ?? c.summary;
	const tech = (await import(pathToFileURL(join(r.site, 'src', 'lib', 'catalog', 'tech-zh.ts')).href)) as { TECH_ZH: Record<string, string>; isLanguageNeutral(s: string): boolean };
	const glossary = (await import(pathToFileURL(join(r.site, 'src', 'lib', 'catalog', 'glossary.ts')).href)) as { GLOSSARY_BY_ID: Record<string, unknown> };
	// the concept catalog fields are not in the frozen CatalogFields type: read them loosely
	const cf = (m.catalog ?? {}) as { interaction?: I18n; engine?: I18n | string; mode?: string; links?: string[] };
	const lines: string[] = [];
	lines.push(`// ---- src/lib/catalog/concepts.ts, the entry { slug: '${slug}', ... }: set or replace these fields`);
	lines.push(`status: 'live',`);
	lines.push(`lesson: { steps: ${c.steps.length}, summary: ${f(lessonSummary)} },`);
	if (cf.interaction) lines.push(`interaction: ${f(cf.interaction)},`);
	const techRows: string[] = [];
	if (cf.engine) {
		const en = typeof cf.engine === 'string' ? cf.engine : cf.engine.en;
		lines.push(`engine: ${q(en)},`);
		techRows.push(...techRowsFor(en, typeof cf.engine === 'string' ? undefined : cf.engine.zh, tech));
	}
	if (cf.mode) lines.push(`mode: ${q(cf.mode)},`);
	if (cf.links) lines.push(`links: [${cf.links.map(q).join(', ')}],`);
	lines.push('');
	lines.push('// ---- src/lib/catalog/tech-zh.ts, TECH_ZH: add');
	lines.push(...(techRows.length ? techRows : ['// (nothing: every string is language-neutral or already has an entry)']));
	lines.push('');
	lines.push('// ---- src/lib/catalog/glossary.ts, GLOSSARY: add only the terms other pages (or the /glossary/ page) need: every term');
	lines.push("//      reaches every page that uses Term (the flagships too), and the package's steps already show its own manifest.glossary terms (K0)");
	const newTerms = (m.glossary ?? []).filter((g) => !glossary.GLOSSARY_BY_ID[g.id]);
	lines.push(...(newTerms.length ? newTerms.map((g) => `{ id: ${q(g.id)}, term: ${f(g.term)}, def: ${f(g.def)}${g.related?.length ? `, related: [${g.related.map(q).join(', ')}]` : ''} },`) : ['// (nothing new)']));
	lines.push('');
	lines.push("// ---- check by hand: the entry's interaction must promise only what the page has (replace the text if not)");
	if (entry.interaction && !cf.interaction) lines.push(`// interaction now: ${entry.interaction.en}`);
	const patch = lines.join('\n') + '\n';
	const gloText = newTerms.map((g) => `${g.id}${g.term.en}${g.term.zh ?? ''}${g.def.en}${g.def.zh ?? ''}`).join('\n');
	const gloCost = gloText ? gzipSync(Buffer.from(gloText)).length : 0;
	const addedText = lines.filter((l) => l && !l.startsWith('//')).join('\n');
	const shellCost = addedText ? gzipSync(Buffer.from(addedText)).length : 0;
	writeText(join(r.site, '.atlas', slug, 'promote.patch.txt'), patch);
	console.log(`atlas promote ${slug}${dry ? ' --dry-run' : ''} (concept):`);
	console.log(`  licenses.json  ${dry ? 'would append' : 'appended'} ${added.length} credit(s)${skipped.length ? `; ${skipped.length} already listed` : ''}${added.length ? `: ${added.map((x) => x.name).join(', ')}` : ''}`);
	for (const x of sameUrl) console.log(`  licenses.json  '${x.credit.name}' not added: its url is already listed as '${x.listedAs}' (one row per source)`);
	console.log(`  text cost      ≈ ${shellCost} B gzip of catalog text in all; ≈ ${gloCost} B of it is glossary terms, which every page that uses Term carries (the flagships too: 300 KB budget in check:build); concepts.ts text reaches every /train/<slug>/ page (ConceptPageShell imports CONCEPT_BY_SLUG) and the list pages`);
	console.log(`  catalog patch  (apply by hand; also in .atlas/${slug}/promote.patch.txt; models.ts is not touched)\n`);
	console.log(patch.replace(/^/gm, '    '));
	if (entry.status === 'live') console.log(`  note           '${slug}' is already live in the concepts catalog`);
	console.log(`  page           /train/${slug}/ (check:build expects train/${slug}/index.html once the concept is live)`);
	const script = join(r.site, 'scripts', 'check-catalog.ts');
	if (!existsSync(script)) {
		console.log('  check:catalog  skipped (no scripts/check-catalog.ts in this site)');
		return 0;
	}
	const cc = await runNode([script], { cwd: r.site, timeoutMs: 300_000 });
	console.log(`  check:catalog  ${cc.code === 0 ? 'ok' : 'FAILED'} on the current catalog (re-run it after applying the patch)`);
	if (cc.code !== 0) console.error((cc.stdout + cc.stderr).split('\n').filter((l) => /FAIL|error|✖/i.test(l)).slice(0, 8).join('\n'));
	return cc.code === 0 ? 0 : 1;
}
