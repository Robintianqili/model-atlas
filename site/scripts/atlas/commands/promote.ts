// atlas promote <slug> [--dry-run | --verify | --apply] [--all-terms]   (design §4.12, §6.4 merge step 2; K4)
//
// What a package needs in the shared files to go live, as one patch:
//   - src/lib/catalog/models.ts (concepts.ts for a concept package): the entry { slug: '<slug>', ... } gets status 'live',
//     lesson { steps, summary }, and every field the manifest's `catalog` block gives. Models: name, family, inputs, mode
//     (K4), hook, shows, runtime, params, size, prereqs. Concepts: title (K4), interaction, engine, mode, links.
//   - src/lib/catalog/tech-zh.ts: Chinese for runtime / params / size / engine strings that are not language-neutral,
//     and (K4) the rows this patch leaves unused (the entry's old strings that no other entry uses): removed.
//   - src/lib/catalog/glossary.ts: manifest.glossary terms the site glossary does not have yet. K4: only the terms that
//     another package or page also uses (a shared term) reach the site glossary by default: every site term is in every
//     page that uses Term, the flagships too (300 KB budget), while the package's own steps already show its manifest
//     terms (K0). --all-terms adds the package-only terms too.
//   - static/licenses.json: the manifest's credits, one row per url (a row whose url is already listed is skipped).
// Modes:
//   (none)     append the credits to licenses.json, print the patch, run check:catalog on the catalog as it is
//   --dry-run  print the patch only (nothing is written)
//   --verify   K4: apply the whole patch to a copy of the catalog in .atlas/<slug>/verify/ (nothing in the tree changes),
//              then run check:catalog on the copy and check that the entry reads back as patched. Works in a lane.
//   --apply    K4, the main tree only: write the patch into the catalog files (each keeps its own line endings: models.ts
//              is CRLF, the others LF; the last row of a list gets the comma it lacked), append the credits, run
//              check:catalog. A lane never edits catalog files (AK702 at merge): use --verify there.
// The patch is also written to .atlas/<slug>/promote.patch.txt. devOnly packages are never promoted (AK701).
// K4 also prints, as checks for the integrator: the measured size of the weights file in the catalog's unit (KB = 1,024
// bytes) next to the manifest's size, the measured parameter count, the page's input kinds next to the entry's inputs,
// the entry's mode, the closest existing tech-zh rows (word overlap, so a short row such as 'JS' no longer wins), a
// manifest Chinese that an existing row overrides, and the gzip cost of the new glossary text.
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { gzipSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';
import type { Credit, I18n } from '../../../src/lib/atlas/types/index.ts';
import { AtlasError } from '../errors.ts';
import { addRows, f, PatchError, removeRows, sameStyle, setFields, tsValue, type FieldValue } from '../lib/catalogpatch.ts';
import { runNode } from '../lib/child.ts';
import { posix, writeBytes, writeText } from '../lib/io.ts';
import { openPackage, walk } from '../lib/pkg.ts';
import { roots } from '../root.ts';
import { namesSource } from '../../gen-glossary-names.ts';

const q = (s: string) => `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;

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

const words = (s: string) => new Set(s.toLowerCase().match(/[a-z0-9]{2,}|[一-鿿]/g) ?? []);

/** K4: existing tech-zh rows whose English shares the most words with `en` (Jaccard: shared words over all words of
 *  both, at least 0.45), best first. K1 measured the share of the SHORTER row, so 'JS' or 'Live' matched any long row. */
export function similarRows(en: string, rows: Record<string, string>, k = 2): string[] {
	const a = words(en);
	const scored: { row: string; s: number }[] = [];
	for (const row of Object.keys(rows)) {
		if (row === en) continue;
		const b = words(row);
		let both = 0;
		for (const w of a) if (b.has(w)) both++;
		const s = both / Math.max(1, a.size + b.size - both);
		if (s >= 0.45) scored.push({ row, s });
	}
	return scored.sort((x, y) => y.s - x.s || y.row.length - x.row.length).slice(0, k).map((x) => x.row);
}

/** The closest existing tech-zh row (similarRows), or null (K1 name). */
export function similarRow(en: string, rows: Record<string, string>): string | null {
	return similarRows(en, rows, 1)[0] ?? null;
}

/** K4: bytes in the catalog's unit: KB and MB of 1,024 (the catalog's "62 KB of weights" is 63,972 bytes). */
export function catalogBytes(b: number): string {
	if (b < 1024 * 1000) {
		const kb = b / 1024;
		return `${kb < 10 ? Number(kb.toFixed(1)) : Math.round(kb)} KB`;
	}
	const mb = b / (1024 * 1024);
	return `${mb < 10 ? Number(mb.toFixed(2)) : Number(mb.toFixed(1))} MB`;
}

/** K4: the bytes a size text names ("62 KB of weights", "1.46 MB"), with KB = 1,024 bytes; null when it names none or
 *  is a bound ("<1 MB (design)"). */
export function sizeBytes(text: string): number | null {
	const m = /^(?:about |≈)?([\d.]+)\s*(KB|MB)\b/i.exec(text.trim());
	if (!m) return null;
	return Number(m[1]) * (m[2].toUpperCase() === 'KB' ? 1024 : 1024 * 1024);
}

// ------------------------------------------------------------------ the plan
type Catalog = {
	MODELS: { slug: string; status: string; name: I18n; family?: I18n; inputs: string[]; mode: string; runtime: string; params?: string; size?: string; hook?: I18n; shows?: I18n }[];
};
type Concepts = { CONCEPTS: { slug: string; status: string; title: I18n; interaction?: I18n; hook?: I18n; engine: string; mode: string; links: string[] }[] };
type Tech = { TECH_ZH: Record<string, string>; isLanguageNeutral(s: string): boolean };
type Glossary = { GLOSSARY: { id: string; term: I18n; def: I18n; related?: string[] }[]; GLOSSARY_BY_ID: Record<string, unknown> };
type Term = { id: string; term: I18n; def: I18n; related?: string[] };

export interface PromotePlan {
	slug: string;
	concept: boolean;
	/** 'models.ts' or 'concepts.ts' */
	file: string;
	array: 'MODELS' | 'CONCEPTS';
	fields: [string, FieldValue][];
	techAdd: { en: string; zh: string }[];
	techRemove: string[];
	/** new glossary terms another package or page uses (with who), and the package-only ones */
	shared: { term: Term; users: string[] }[];
	own: Term[];
	/** redesign wave F: the regenerated src/lib/catalog/glossary-names.ts when shared terms are added (else null) */
	names: string | null;
	credits: Credit[];
	/** the patch as printed (and written to .atlas/<slug>/promote.patch.txt) */
	patch: string;
	/** checks for the integrator, one line each */
	notes: string[];
	cost: { all: number; shared: number; own: number; glossary: number };
	status: string;
	name: string;
}

async function load<T>(site: string, ...p: string[]): Promise<T> {
	return (await import(pathToFileURL(join(site, ...p)).href)) as T;
}

/** Glossary ids used outside the package: other packages' lesson/concept/explainer files, the hand-built pages' <Term>,
 *  the kit's op terms, and related ids of site glossary rows. */
export function termUsers(site: string, ids: string[], self: string, glossary: Term[]): Record<string, string[]> {
	const out: Record<string, string[]> = Object.fromEntries(ids.map((id) => [id, []]));
	if (!ids.length) return out;
	const look = (text: string, who: string) => {
		for (const id of ids) if (!out[id].includes(who) && (text.includes(`'${id}'`) || text.includes(`"${id}"`))) out[id].push(who);
	};
	const ex = join(site, 'src', 'lib', 'explainers');
	if (existsSync(ex))
		for (const slug of readdirSync(ex)) {
			if (slug === self || !statSync(join(ex, slug)).isDirectory()) continue;
			for (const fn of ['lesson.ts', 'concept.ts', 'explainer.ts']) if (existsSync(join(ex, slug, fn))) look(readFileSync(join(ex, slug, fn), 'utf8'), slug);
		}
	const pages = join(site, 'src', 'lib', 'pages');
	if (existsSync(pages)) for (const fl of walk(pages)) if (fl.endsWith('.svelte')) look(readFileSync(join(pages, fl), 'utf8'), `pages/${fl.split('/')[0]}`);
	const ops = join(site, 'src', 'lib', 'atlas', 'ops');
	if (existsSync(ops)) for (const fl of readdirSync(ops)) if (fl.endsWith('.ts') && !fl.endsWith('.test.ts')) look(readFileSync(join(ops, fl), 'utf8'), 'kit ops');
	for (const g of glossary) for (const r of g.related ?? []) if (out[r] && !out[r].includes(`glossary ${g.id}`)) out[r].push(`glossary ${g.id}`);
	return out;
}

const gloRow = (g: Term) => `{ id: ${q(g.id)}, term: ${f(g.term)}, def: ${f(g.def)}${g.related?.length ? `, related: [${g.related.map(q).join(', ')}]` : ''} }`;
const gz = (s: string) => (s ? gzipSync(Buffer.from(s)).length : 0);
const gloText = (ts: Term[]) => ts.map((g) => `${g.id}${g.term.en}${g.term.zh ?? ''}${g.def.en}${g.def.zh ?? ''}`).join('\n');

/** Everything promote would change, without changing anything. */
export async function planPromote(slug: string, o: { allTerms?: boolean } = {}): Promise<PromotePlan> {
	const r = roots();
	const op = await openPackage(slug);
	const m = op.l.pkg.manifest;
	if (m.devOnly) throw new AtlasError('AK701', `'${slug}' is devOnly (lab only): it is never promoted to the catalog`);
	if (!Array.isArray(m.credits)) throw new AtlasError('AK704', 'manifest.credits is missing');
	for (const [i, c] of m.credits.entries())
		for (const k of ['name', 'url', 'license', 'usedFor', 'attribution'] as const)
			if (typeof c[k] !== 'string' || !c[k].trim()) throw new AtlasError('AK704', `credits[${i}] has no ${k}`);
	const concept = (m.type as string) === 'concept';
	const tech = await load<Tech>(r.site, 'src', 'lib', 'catalog', 'tech-zh.ts');
	const glossary = await load<Glossary>(r.site, 'src', 'lib', 'catalog', 'glossary.ts');
	const models = (await load<Catalog>(r.site, 'src', 'lib', 'catalog', 'models.ts')).MODELS;
	const concepts = (await load<Concepts>(r.site, 'src', 'lib', 'catalog', 'concepts.ts')).CONCEPTS;
	const { compiled } = await import('./lesson.ts');
	const { c, spec } = await compiled(op);
	const lessonSummary = op.l.pkg.lesson?.summary ?? c.summary;
	const notes: string[] = [];
	const fields: [string, FieldValue][] = [
		['status', { kind: 'string', v: 'live' }],
		['lesson', { kind: 'lesson', steps: c.steps.length, summary: lessonSummary }]
	];
	const techStrings: { key: string; en: string; zh?: string }[] = [];
	// loosely typed: the K4 fields (name, family, inputs, mode, title) are additive
	const cf = (m.catalog ?? {}) as Record<string, unknown> & { [k: string]: unknown };
	const i18n = (v: unknown): I18n | null => (v && typeof v === 'object' && typeof (v as I18n).en === 'string' ? (v as I18n) : null);
	const textOf = (v: unknown): { en: string; zh?: string } | null => (typeof v === 'string' ? { en: v } : i18n(v));
	let entryName = slug;
	let old: [string, string | undefined][] = [];
	let status = 'planned';
	if (!concept) {
		const entry = models.find((e) => e.slug === slug);
		if (!entry) throw new AtlasError('AK701', `'${slug}' is not in the catalog`, { fix: 'a maintainer adds the catalog entry first' });
		status = entry.status;
		entryName = entry.name.en;
		old = [['runtime', entry.runtime], ['params', entry.params], ['size', entry.size]];
		const name = i18n(cf.name);
		if (name) fields.push(['name', { kind: 'i18n', v: name, n: true }]);
		const family = i18n(cf.family);
		if (family) fields.push(['family', { kind: 'i18n', v: family }]);
		const inputs = Array.isArray(cf.inputs) ? (cf.inputs as string[]) : null;
		if (inputs) fields.push(['inputs', { kind: 'list', v: inputs }]);
		if (typeof cf.mode === 'string') fields.push(['mode', { kind: 'string', v: cf.mode }]);
		// redesign wave F (types R1): a short name for path stations (at most 24 characters)
		const short = i18n(cf.short);
		if (short) fields.push(['short', { kind: 'i18n', v: short }]);
		for (const k of ['hook', 'shows'] as const) {
			const v = i18n(cf[k]);
			if (v) fields.push([k, { kind: 'i18n', v }]);
		}
		// the page's input kinds next to the entry's inputs (crnn-ocr promised drawing and image; its adapter takes a drawing)
		const ids = [m.input?.adapter, ...(m.inputs ?? []).map((x) => x.adapter)].filter((x): x is string => !!x);
		const kinds = [...new Set(ids.map((id) => op.l.pkg.adapters?.[id]?.kind).filter((k): k is string => !!k))];
		const promised = inputs ?? entry.inputs;
		if (kinds.length && promised.some((k) => !kinds.includes(k)))
			notes.push(`inputs: the entry ${inputs ? 'would promise' : 'promises'} [${promised.join(', ')}]; the page's adapter${kinds.length > 1 ? 's take' : ' takes'} [${kinds.join(', ')}]: set manifest.catalog.inputs`);
		else if (!kinds.length && !inputs) notes.push(`inputs: the entry promises [${entry.inputs.join(', ')}]; check it against the page (manifest.catalog.inputs sets it)`);
		if (typeof cf.mode !== 'string' && entry.mode !== 'live') notes.push(`mode: the entry says '${entry.mode}'; set manifest.catalog.mode ('live' when the page computes its numbers in the browser, 'both' when some are precomputed)`);
		for (const k of ['runtime', 'params', 'size'] as const) {
			const v = textOf(cf[k]);
			if (v) techStrings.push({ key: k, en: v.en, zh: v.zh });
		}
		// measured: parameters (spec.json) and the weights file in the catalog's unit (1 KB = 1,024 bytes)
		if (!cf.params && spec) notes.push(`params: measured ${spec.params.toLocaleString('en-US')} (spec.json); set manifest.catalog.params (e.g. '${spec.params.toLocaleString('en-US')}') unless the entry's text says it better`);
		const wfile = m.assets?.dir ? join(r.site, 'static', m.assets.dir, (spec as { files?: { weights?: string } } | null)?.files?.weights ?? 'weights.bin') : null;
		const wbytes = wfile && existsSync(wfile) ? statSync(wfile).size : null;
		const sizeText = textOf(cf.size)?.en;
		if (wbytes !== null) {
			const measured = `${catalogBytes(wbytes)} of weights`;
			const named = sizeText ? sizeBytes(sizeText) : null;
			// the measured size replaces a planned entry's design estimate, never a live entry's own wording
			const estimate = entry.status !== 'live' || !entry.size || /design|^[<≤≈~]/.test(entry.size);
			if (!sizeText && estimate) {
				techStrings.push({ key: 'size', en: measured, zh: `权重 ${catalogBytes(wbytes)}` });
				notes.push(`size: measured ${wbytes.toLocaleString('en-US')} bytes of weights = '${measured}' (1 KB = 1,024 bytes); used as the size text (manifest.catalog.size overrides it)`);
			} else if (!sizeText) {
				notes.push(`size: the entry says '${entry.size}'; the weights file has ${wbytes.toLocaleString('en-US')} bytes = '${measured}' (1 KB = 1,024 bytes)`);
			} else if (named !== null && Math.abs(named - wbytes) / wbytes > 0.06)
				notes.push(`size: the manifest says '${sizeText}', but the weights file has ${wbytes.toLocaleString('en-US')} bytes = '${measured}' (the catalog's KB is 1,024 bytes): fix manifest.catalog.size`);
		} else if (!sizeText && !(entry.size ?? '').match(/^no (files|weights)/i)) {
			const used = [...new Set(models.filter((e) => e.status === 'live' && /^no (files|weights)/i.test(e.size ?? '')).map((e) => e.size!))];
			notes.push(`size: no weights file; the entry says '${entry.size ?? '(none)'}'. Set manifest.catalog.size; texts other live pages use: ${used.map((s) => `'${s}'`).join(', ') || '(none)'}`);
		}
		if (Array.isArray(cf.prereqs)) fields.push(['prereqs', { kind: 'list', v: cf.prereqs as string[] }]);
		for (const k of ['hook', 'shows'] as const) if (entry[k] && !cf[k]) notes.push(`${k} now: ${entry[k]!.en} (must promise only what the page has; manifest.catalog.${k} replaces it)`);
		if (!cf.name) notes.push(`name now: ${entry.name.en} (the page's h1 and title; manifest.catalog.name renames it)`);
	} else {
		const entry = concepts.find((e) => e.slug === slug);
		if (!entry) throw new AtlasError('AK701', `'${slug}' is not in the concepts catalog (src/lib/catalog/concepts.ts)`, { fix: 'a maintainer adds the catalog entry first' });
		status = entry.status;
		entryName = entry.title.en;
		old = [['engine', entry.engine]];
		const title = i18n(cf.title);
		if (title) fields.push(['title', { kind: 'i18n', v: title }]);
		const inter = i18n(cf.interaction);
		if (inter) fields.push(['interaction', { kind: 'i18n', v: inter }]);
		const engine = textOf(cf.engine);
		if (engine) techStrings.push({ key: 'engine', en: engine.en, zh: engine.zh });
		if (typeof cf.mode === 'string') fields.push(['mode', { kind: 'string', v: cf.mode }]);
		if (Array.isArray(cf.links)) fields.push(['links', { kind: 'list', v: cf.links as string[] }]);
		// redesign wave F (types R1): a concept's card hook and short title
		for (const k of ['hook', 'short'] as const) {
			const v = i18n(cf[k]);
			if (v) fields.push([k, { kind: 'i18n', v }]);
		}
		if (!i18n(cf.hook) && !entry.hook) notes.push('hook: the concept has no card hook yet (cards use the lesson summary); manifest.catalog.hook sets one (at most 100 English and 50 Chinese characters)');
		if (entry.interaction && !inter) notes.push(`interaction now: ${entry.interaction.en} (must promise only what the page has; manifest.catalog.interaction replaces it)`);
		if (typeof cf.mode !== 'string') notes.push(`mode now: '${entry.mode}' (manifest.catalog.mode sets it)`);
	}
	// tech-zh: the strings and their Chinese
	const techAdd: { en: string; zh: string }[] = [];
	for (const s of techStrings) {
		fields.push([s.key, { kind: 'string', v: s.en }]);
		if (tech.isLanguageNeutral(s.en)) continue;
		if (tech.TECH_ZH[s.en] !== undefined) {
			if (s.zh && s.zh !== tech.TECH_ZH[s.en]) notes.push(`${s.key}: tech-zh.ts already has '${tech.TECH_ZH[s.en]}' for this text; the manifest's '${s.zh}' is not used (one Chinese text per string)`);
			continue;
		}
		if (techAdd.some((x) => x.en === s.en)) continue;
		techAdd.push({ en: s.en, zh: s.zh ?? '' });
		const near = similarRows(s.en, tech.TECH_ZH);
		if (near.length) notes.push(`${s.key}: similar rows already in tech-zh.ts: ${near.map((x) => `'${x}'`).join(', ')} (reuse one when it says the same, so the pages read alike)`);
	}
	// rows the patch leaves unused: the entry's old strings that no other entry uses
	const newStrings = new Set(techStrings.map((s) => s.en));
	const usedElsewhere = new Set([...models.filter((e) => e.slug !== slug).flatMap((e) => [e.runtime, e.params, e.size]), ...concepts.filter((e) => e.slug !== slug).map((e) => e.engine)].filter((s): s is string => !!s));
	const replaced = new Set(techStrings.map((s) => s.key));
	const oldStrings = old.map(([, v]) => v).filter((v): v is string => !!v);
	const techRemove = [...new Set(old.filter(([k, v]) => v && replaced.has(k) && !newStrings.has(v) && !usedElsewhere.has(v) && tech.TECH_ZH[v] !== undefined).map(([, v]) => v!))];
	const allUsed = new Set([...usedElsewhere, ...oldStrings, ...newStrings]);
	const stale = Object.keys(tech.TECH_ZH).filter((k) => !allUsed.has(k));
	if (stale.length) notes.push(`tech-zh.ts rows no entry uses (before this patch): ${stale.map((s) => `'${s}'`).join(', ')}`);
	// glossary: new terms, shared or package-only
	const newTerms = ((m.glossary ?? []) as Term[]).filter((g) => !glossary.GLOSSARY_BY_ID[g.id]);
	const users = termUsers(r.site, newTerms.map((g) => g.id), slug, glossary.GLOSSARY);
	const shared = newTerms.filter((g) => users[g.id].length || o.allTerms).map((term) => ({ term, users: users[term.id] }));
	const own = newTerms.filter((g) => !users[g.id].length && !o.allTerms);
	// the names <Term> shows without the definitions follow the glossary (lazy glossary, redesign wave F)
	const names = shared.length ? namesSource([...glossary.GLOSSARY, ...shared.map((x) => x.term)]) : null;
	// the printed patch: one line per value (the applied files wrap long texts as the catalog does)
	const lines: string[] = [];
	lines.push(`// ---- src/lib/catalog/${concept ? 'concepts' : 'models'}.ts, the entry { slug: '${slug}', ... }: set or replace these fields`);
	for (const [k, v] of fields) lines.push(`${k}: ${tsValue(k, v, '', 100000)},`);
	lines.push('');
	lines.push('// ---- src/lib/catalog/tech-zh.ts, TECH_ZH: add');
	lines.push(...(techAdd.length ? techAdd.map((t) => `${q(t.en)}: ${q(t.zh)},${t.zh ? '' : ' // TODO: zh missing (check:catalog fails without it)'}`) : ['// (nothing: every string is language-neutral or already has an entry)']));
	if (techRemove.length) lines.push('// ---- src/lib/catalog/tech-zh.ts, TECH_ZH: remove (no entry uses them after this patch)', ...techRemove.map((s) => `// ${q(s)}`));
	lines.push('');
	lines.push("// ---- src/lib/catalog/glossary.ts, GLOSSARY: add at the end (the last row there may lack its comma: --apply adds it)");
	lines.push(...(shared.length ? shared.map((x) => `${gloRow(x.term)},${x.users.length ? ` // also used by ${x.users.slice(0, 4).join(', ')}` : ''}`) : ['// (nothing new that another page uses)']));
	if (own.length) lines.push(`// package-only terms, NOT added (the package's steps show them; --all-terms adds them): ${own.map((g) => g.id).join(', ')}`);
	const patch = lines.join('\n') + '\n';
	const cost = { all: gz(lines.filter((l) => l && !l.startsWith('//')).join('\n')), shared: gz(gloText(shared.map((x) => x.term))), own: gz(gloText(own)), glossary: gz(gloText(glossary.GLOSSARY)) };
	return { slug, concept, file: concept ? 'concepts.ts' : 'models.ts', array: concept ? 'CONCEPTS' : 'MODELS', fields, techAdd, techRemove, shared, own, names, credits: m.credits, patch, notes, cost, status, name: entryName };
}

/** The patched texts of the catalog files (relative to the site) for a plan, read from `site`. */
export function patchedFiles(plan: PromotePlan, site: string): Record<string, string> {
	const cat = (f2: string) => join(site, 'src', 'lib', 'catalog', f2);
	const out: Record<string, string> = {};
	const read = (p: string) => readFileSync(p, 'utf8');
	try {
		const entryFile = cat(plan.file);
		const src = read(entryFile);
		out[`src/lib/catalog/${plan.file}`] = sameStyle(src, setFields(src, plan.array, plan.slug, plan.fields).text);
		const techFile = cat('tech-zh.ts');
		let tsrc = read(techFile);
		const add = plan.techAdd.filter((t) => t.zh).map((t) => `${q(t.en)}: ${q(t.zh)}`);
		tsrc = removeRows(addRows(tsrc, 'TECH_ZH', add), 'TECH_ZH', plan.techRemove).text;
		out['src/lib/catalog/tech-zh.ts'] = sameStyle(read(techFile), tsrc);
		if (plan.shared.length) {
			const gfile = cat('glossary.ts');
			const gsrc = read(gfile);
			out['src/lib/catalog/glossary.ts'] = sameStyle(gsrc, addRows(gsrc, 'GLOSSARY', plan.shared.map((x) => gloRow(x.term)), `// ---- ${plan.name} (added by the ${plan.slug} page)`));
		}
		if (plan.names) {
			const nfile = cat('glossary-names.ts');
			out['src/lib/catalog/glossary-names.ts'] = existsSync(nfile) ? sameStyle(read(nfile), plan.names) : plan.names;
		}
	} catch (e) {
		if (e instanceof PatchError) throw new AtlasError('AK701', `the catalog patch cannot be applied: ${e.message}`, { fix: 'apply it by hand from .atlas/<slug>/promote.patch.txt' });
		throw e;
	}
	// only the files the patch changes
	for (const rel of Object.keys(out)) if (existsSync(join(site, rel)) && out[rel] === readFileSync(join(site, rel), 'utf8')) delete out[rel];
	return out;
}

/** A copy of what check:catalog reads, in `dir`: scripts/check-catalog.ts and gen-glossary-names.ts, src/lib/catalog/,
 *  src/lib/i18n/, static/licenses.json. */
function catalogCopy(site: string, dir: string): void {
	rmSync(dir, { recursive: true, force: true });
	for (const p of [join('scripts', 'check-catalog.ts'), join('scripts', 'gen-glossary-names.ts'), join('src', 'lib', 'catalog'), join('src', 'lib', 'i18n'), join('static', 'licenses.json')]) {
		if (!existsSync(join(site, p))) continue;
		mkdirSync(join(dir, p, '..'), { recursive: true });
		cpSync(join(site, p), join(dir, p), { recursive: true });
	}
}

async function checkCatalog(site: string): Promise<{ ok: boolean; skipped?: boolean; out: string }> {
	const script = join(site, 'scripts', 'check-catalog.ts');
	if (!existsSync(script)) return { ok: true, skipped: true, out: '' };
	const cc = await runNode([script], { cwd: site, timeoutMs: 300_000 });
	return { ok: cc.code === 0, out: (cc.stdout + cc.stderr).split('\n').filter((l) => /FAIL|error|problem|✖| - /i.test(l)).slice(0, 10).join('\n') };
}

/** The entry as the patched catalog reads it back equals what the plan set (catches a patch that matched nothing). */
async function readBack(plan: PromotePlan, dir: string): Promise<string[]> {
	const mod = await import(`${pathToFileURL(join(dir, 'src', 'lib', 'catalog', plan.file)).href}?v=${Date.now()}`);
	const list = (plan.concept ? mod.CONCEPTS : mod.MODELS) as Record<string, unknown>[];
	const e = list.find((x) => x.slug === plan.slug);
	if (!e) return [`the patched ${plan.file} has no entry '${plan.slug}'`];
	const bad: string[] = [];
	for (const [k, v] of plan.fields) {
		const want = v.kind === 'string' ? v.v : v.kind === 'list' ? v.v : v.kind === 'i18n' ? { en: v.v.en, zh: v.v.zh ?? v.v.en } : v.kind === 'lesson' ? { steps: v.steps, summary: { en: v.summary.en, zh: v.summary.zh ?? v.summary.en } } : undefined;
		if (want !== undefined && JSON.stringify(e[k]) !== JSON.stringify(want)) bad.push(`${k} reads back as ${JSON.stringify(e[k])?.slice(0, 80)}`);
	}
	return bad;
}

export async function main(args: string[]): Promise<number> {
	const flags = new Set(args.filter((a) => a.startsWith('--')));
	for (const a of flags) if (!['--dry-run', '--verify', '--apply', '--all-terms'].includes(a)) throw new AtlasError('AK301', `unknown flag ${a}`, { fix: 'atlas promote <slug> [--dry-run | --verify | --apply] [--all-terms]' });
	const pos = args.filter((a) => !a.startsWith('--'));
	const modes = ['--dry-run', '--verify', '--apply'].filter((x) => flags.has(x));
	if (pos.length !== 1 || modes.length > 1) throw new AtlasError('AK301', 'usage: atlas promote <slug> [--dry-run | --verify | --apply] [--all-terms]');
	const mode = (modes[0] ?? '').slice(2) as '' | 'dry-run' | 'verify' | 'apply';
	const slug = pos[0];
	const r = roots();
	if (mode === 'apply' && r.lane) throw new AtlasError('AK702', `promote --apply edits the catalog, which a lane never does (merge would refuse the lane)`, { fix: `in the lane: npm run atlas -- promote ${slug} --verify; the main session runs --apply after the merge` });
	const plan = await planPromote(slug, { allTerms: flags.has('--all-terms') });
	const lic = join(r.site, 'static', 'licenses.json');
	const writes = mode === '' || mode === 'apply';
	const { added, skipped, sameUrl } = appendCredits(lic, plan.credits, !writes);
	writeText(join(r.site, '.atlas', slug, 'promote.patch.txt'), plan.patch);

	const verb = writes ? 'appended' : 'would append';
	console.log(`atlas promote ${slug}${mode ? ` --${mode}` : ''}${plan.concept ? ' (concept)' : ''}:`);
	console.log(`  licenses.json  ${verb} ${added.length} credit(s)${skipped.length ? `; ${skipped.length} already listed` : ''}${added.length ? `: ${added.map((x) => x.name).join(', ')}` : ''}`);
	for (const x of sameUrl) console.log(`  licenses.json  '${x.credit.name}' not added: its url is already listed as '${x.listedAs}' (one row per source; extend that row's usedFor by hand if it should name this page too)`);
	console.log(`  glossary       ${plan.shared.length} new term(s) for the site glossary (≈ ${plan.cost.shared} B gzip in every page that uses Term, the flagships too; the site glossary is ≈ ${(plan.cost.glossary / 1024).toFixed(1)} KB gzip now)${plan.own.length ? `; ${plan.own.length} package-only term(s) left out (≈ ${plan.cost.own} B): ${plan.own.map((g) => g.id).join(', ')}` : ''}`);
	console.log(`  catalog patch  (${mode === 'apply' ? 'applied below' : mode === 'verify' ? 'checked on a copy below' : 'apply with --apply in the main tree, or by hand'}; also in .atlas/${slug}/promote.patch.txt)\n`);
	console.log(plan.patch.replace(/^/gm, '    '));
	if (plan.notes.length) console.log(`  check by hand\n${plan.notes.map((n) => `    - ${n}`).join('\n')}`);
	if (plan.status === 'live') console.log(`  note           '${slug}' is already live in the catalog`);
	if (plan.concept) console.log(`  page           /train/${slug}/ (check:build expects train/${slug}/index.html once the concept is live)`);
	if (plan.techAdd.some((t) => !t.zh)) console.log('  tech-zh        a row has no Chinese yet: give the manifest string a zh (or add the row by hand); --verify and --apply leave it out, so check:catalog fails');

	if (mode === 'verify' || mode === 'apply') {
		const files = patchedFiles(plan, r.site);
		if (mode === 'verify') {
			const dir = join(r.site, '.atlas', slug, 'verify');
			catalogCopy(r.site, dir);
			for (const [rel, text] of Object.entries(files)) writeBytes(join(dir, rel), Buffer.from(text, 'utf8'));
			if (added.length) {
				const rows = JSON.parse(readFileSync(lic, 'utf8').replace(/^﻿/, '')) as Credit[];
				writeText(join(dir, 'static', 'licenses.json'), JSON.stringify([...rows, ...added], null, 2) + '\n');
			}
			const cc = await checkCatalog(dir);
			const back = await readBack(plan, dir);
			console.log(`  verify         on a copy in ${posix(relative(r.site, dir))}/: check:catalog ${cc.skipped ? 'skipped (no scripts/check-catalog.ts)' : cc.ok ? 'ok' : 'FAILED'}; the entry reads back ${back.length ? `WRONG (${back.join('; ')})` : 'as patched'}; files: ${Object.keys(files).join(', ')}`);
			if (!cc.ok) console.error(cc.out);
			return cc.ok && !back.length ? 0 : 1;
		}
		for (const [rel, text] of Object.entries(files)) writeBytes(join(r.site, rel), Buffer.from(text, 'utf8'));
		console.log(`  applied        ${Object.keys(files).join(', ')} (each with its own line endings)`);
	}
	const cc = await checkCatalog(r.site);
	console.log(`  check:catalog  ${cc.skipped ? 'skipped (no scripts/check-catalog.ts in this site)' : cc.ok ? 'ok' : 'FAILED'} on the ${mode === 'apply' ? 'patched' : 'current'} catalog${mode === 'apply' || cc.skipped ? '' : ' (promote --verify checks the patched one)'}`);
	if (!cc.ok) console.error(cc.out);
	return cc.ok ? 0 : 1;
}
