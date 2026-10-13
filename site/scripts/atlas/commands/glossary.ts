// atlas glossary [--json]   (K4: the flagship budget)
//
// Every term of the site glossary (src/lib/catalog/glossary.ts) is in every page that uses Term, the flagship pages
// /m/tinycnn/ and /m/tinygpt/ included (check:build: 300 KB gzip each). A package's own steps show the terms of its
// manifest.glossary without the site glossary (K0), so a site term that only one package uses, and that its manifest
// also defines, costs every page and serves one. This lists them, with the gzip bytes each costs, so the main session can
// take them out of glossary.ts (a catalog edit; nothing here writes). Read-only.
// Users of a term: the quoted id in another package's lesson.ts, concept.ts or explainer.ts, a <Term id> in the
// hand-built pages and routes, the kit (ops' terms, atlas UI), and the related lists of other site glossary rows.
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';
import type { I18n } from '../../../src/lib/atlas/types/index.ts';
import { AtlasError } from '../errors.ts';
import { walk } from '../lib/pkg.ts';
import { roots } from '../root.ts';

type Term = { id: string; term: I18n; def: I18n; related?: string[] };

export interface GlossaryAudit {
	terms: number;
	gzip: number;
	/** gzip bytes of the single-use terms together (about what taking them all out saves) */
	singleGzip: number;
	/** site terms only one package uses and also defines in its manifest: safe to take out of the site glossary */
	single: { id: string; pkg: string; bytes: number }[];
	/** site terms nothing uses (only the /glossary/ page lists them) */
	unused: string[];
}

/** Who uses each glossary id (package slugs, 'pages', 'kit', 'routes', 'glossary <id>'). */
export function glossaryUsers(site: string, ids: string[], glossary: Term[]): Record<string, Set<string>> {
	const out: Record<string, Set<string>> = Object.fromEntries(ids.map((id) => [id, new Set<string>()]));
	const quoted = new Map(ids.map((id) => [id, [`'${id}'`, `"${id}"`]]));
	const look = (text: string, who: string) => {
		for (const [id, qs] of quoted) if (qs.some((q) => text.includes(q))) out[id].add(who);
	};
	const ex = join(site, 'src', 'lib', 'explainers');
	if (existsSync(ex))
		for (const slug of readdirSync(ex)) {
			if (!statSync(join(ex, slug)).isDirectory()) continue;
			for (const fn of ['lesson.ts', 'concept.ts', 'explainer.ts']) if (existsSync(join(ex, slug, fn))) look(readFileSync(join(ex, slug, fn), 'utf8'), slug);
		}
	for (const [dir, who] of [
		[join(site, 'src', 'lib', 'pages'), 'pages'],
		[join(site, 'src', 'routes'), 'routes'],
		[join(site, 'src', 'lib', 'atlas'), 'kit']
	] as const)
		if (existsSync(dir)) for (const f of walk(dir)) if (/\.(svelte|ts)$/.test(f) && !f.endsWith('.test.ts')) look(readFileSync(join(dir, f), 'utf8'), who);
	for (const g of glossary) for (const rel of g.related ?? []) out[rel]?.add(`glossary ${g.id}`);
	return out;
}

export async function audit(site = roots().site): Promise<GlossaryAudit> {
	const { GLOSSARY } = (await import(pathToFileURL(join(site, 'src', 'lib', 'catalog', 'glossary.ts')).href)) as { GLOSSARY: Term[] };
	const users = glossaryUsers(site, GLOSSARY.map((g) => g.id), GLOSSARY);
	const own = (slug: string, id: string): boolean => {
		const f = join(site, 'src', 'lib', 'explainers', slug, 'explainer.ts');
		return existsSync(f) && new RegExp(`id:\\s*['"]${id.replace(/[-]/g, '\\-')}['"]`).test(readFileSync(f, 'utf8'));
	};
	const text = (g: Term) => `${g.id}${g.term.en}${g.term.zh ?? ''}${g.def.en}${g.def.zh ?? ''}`;
	const single: GlossaryAudit['single'] = [];
	const unused: string[] = [];
	for (const g of GLOSSARY) {
		const u = [...users[g.id]];
		if (!u.length) unused.push(g.id);
		else if (u.length === 1 && own(u[0], g.id)) single.push({ id: g.id, pkg: u[0], bytes: gzipSync(Buffer.from(text(g))).length });
	}
	const byId = new Map(GLOSSARY.map((g) => [g.id, g]));
	const singleGzip = single.length ? gzipSync(Buffer.from(single.map((x) => text(byId.get(x.id)!)).join('\n'))).length : 0;
	return { terms: GLOSSARY.length, gzip: gzipSync(Buffer.from(GLOSSARY.map(text).join('\n'))).length, singleGzip, single, unused };
}

export async function main(args: string[]): Promise<number> {
	for (const a of args) if (a !== '--json') throw new AtlasError('AK301', `unknown argument ${a}`, { fix: 'atlas glossary [--json]' });
	const a = await audit();
	if (args.includes('--json')) {
		console.log(JSON.stringify(a, null, 1));
		return 0;
	}
	console.log(`atlas glossary: ${a.terms} site terms, ≈ ${(a.gzip / 1024).toFixed(1)} KB gzip in every page that uses Term (the flagships too)`);
	console.log(`  ${a.single.length} used by one package that also defines it in its manifest (its steps show it without the site glossary; ≈ ${(a.singleGzip / 1024).toFixed(1)} KB gzip together; the bytes below are each alone):`);
	for (const x of a.single) console.log(`    ${x.id.padEnd(28)} ${x.pkg.padEnd(24)} ≈ ${x.bytes} B`);
	if (!a.single.length) console.log('    (none)');
	if (a.unused.length) console.log(`  used by nothing but the /glossary/ page: ${a.unused.join(', ')}`);
	console.log('  taking a term out is a catalog edit (src/lib/catalog/glossary.ts, the main session); keep the ones the /glossary/ page should list');
	return 0;
}
