/**
 * Post-build checks: `npm run build && npm run check:build`.
 *  1. Every expected page was prerendered (home, atlas + 11 areas, planned model and concept pages, …).
 *  2. No runtime requests to third-party hosts: scans build/ (HTML, JS, CSS) for absolute URLs and
 *     flags any that would be FETCHED (script/link/img src, CSS url()/@import, fetch/import calls).
 *     Plain <a href> links to other sites are allowed and listed separately.
 *  3. JS sizes: largest chunks (raw and gzip) and the JS each sample page preloads ("page shell"), with a 300 KB
 *     budget per sample page (including /m/tinycnn/ and /m/tinygpt/) and a tighter budget for planned model pages:
 *     their wave-0 shell baseline + 5 KB (DevKit design §5.1), so the auto-page code never leaks into them.
 *  4. Catalog modules in the shells (prep for wave 2b): each sample page prints which catalog modules (models, concepts,
 *     tech-zh, areas, glossary) its preloaded JS carries. /m/tinycnn/ and /m/tinygpt/ must carry none: a model page gets
 *     its slice of the catalog from the root layout's server load (src/lib/catalog/slices.ts), so entries going live or
 *     new catalog text never grow the flagship shells. Redesign wave F: no page but /glossary/ may carry the glossary's
 *     definitions (glossary.ts; <Term> shows names from glossary-names.ts and loads them on first use), and a flagship
 *     shell over 250 KB warns (the gate stays 300 KB; the redesign's target is 230 KB).
 * ATLAS_BUILD_DIR selects the build folder (default build/), as in svelte.config.js.
 * ATLAS_ONLY=<slug>[,<slug>...] (a package-only build, DevKit K2; svelte.config.js): check 1 expects the home page and at
 * least one page per package (/m/<slug>/, /train/<slug>/ or /dev/atlas/lab/<slug>/); check 3 measures the shells of the
 * pages that were built (the home page and the packages' pages, 300 KB each; a planned page's budget as usual) and gzips
 * only the chunks they preload plus the largest ones; checks 2 and 4 are unchanged.
 * Exit code 1 if a check fails.
 */
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative, extname } from 'node:path';
import { gzipSync } from 'node:zlib';
import { MODELS } from '../src/lib/catalog/models.ts';
import { CONCEPTS } from '../src/lib/catalog/concepts.ts';
import { AREAS } from '../src/lib/catalog/areas.ts';
import { TECH_ZH } from '../src/lib/catalog/tech-zh.ts';
import { GLOSSARY } from '../src/lib/catalog/glossary.ts';
import { candidatePages, onlyEntries, onlySlugs } from './atlas/lib/only.ts';

const BUILD = process.env.ATLAS_BUILD_DIR || 'build';
const ONLY = onlyEntries();
/** Shell JS (gzip bytes) of every planned /m/<slug>/ page at DevKit wave 0 (commit e4718c9): all 101 were 110,912. */
const PLANNED_SHELL_BASELINE = 110912;
const PLANNED_SHELL_BUDGET = PLANNED_SHELL_BASELINE + 5 * 1024;
let failed = false;
const fail = (m: string) => {
	failed = true;
	console.error('FAIL ' + m);
};

function walk(dir: string, out: string[] = []): string[] {
	for (const name of readdirSync(dir)) {
		const p = join(dir, name);
		if (statSync(p).isDirectory()) walk(p, out);
		else out.push(p);
	}
	return out;
}

if (!existsSync(BUILD)) {
	console.error(`${BUILD}/ not found: run npm run build first`);
	process.exit(1);
}
const files = walk(BUILD);
const rel = (p: string) => relative(BUILD, p).replaceAll('\\', '/');

// ------------------------------------------------------------------ 1. pages
const have = new Set(files.map(rel));
const pages = files.filter((f) => f.endsWith('index.html'));
/** a package-only build: each package's pages that were built */
const onlyPages = new Map<string, string[]>();
if (ONLY) {
	const page = (p: string) => `${p.slice(1)}index.html`;
	for (const slug of onlySlugs()) onlyPages.set(slug, candidatePages(slug).map(page).filter((p) => have.has(p)));
	console.log(`package-only build (ATLAS_ONLY=${onlySlugs().join(',')}): ${pages.length} pages prerendered`);
	if (!have.has('index.html')) fail('missing pages: index.html');
	for (const [slug, ps] of onlyPages) {
		if (ps.length) console.log(`  ${slug}: ${ps.map((p) => `/${p.replace('index.html', '')}`).join(', ')}`);
		else fail(`no page was built for package '${slug}' (none of ${candidatePages(slug).join(', ')}): is it in the catalog, or devOnly in a drafts build (ATLAS_INCLUDE_DRAFTS=1)?`);
	}
} else {
	const expected = [
		'index.html',
		'atlas/index.html',
		...AREAS.map((a) => `atlas/${a.slug}/index.html`),
		...MODELS.filter((m) => m.status !== 'live').map((m) => `m/${m.slug}/index.html`),
		'train/index.html',
		...CONCEPTS.filter((c) => c.status !== 'live').map((c) => `train/${c.slug}/index.html`),
		'glossary/index.html',
		'about/index.html',
		'dev/kit/index.html'
	];
	const missing = expected.filter((p) => !have.has(p));
	console.log(`pages prerendered: ${pages.length} (expected at least ${expected.length})`);
	if (missing.length) fail(`missing pages: ${missing.join(', ')}`);
	const pendingLive = [
		...MODELS.filter((m) => m.status === 'live').map((m) => `m/${m.slug}/index.html`),
		...CONCEPTS.filter((c) => c.status === 'live').map((c) => `train/${c.slug}/index.html`)
	];
	for (const p of pendingLive) console.log(`  live route ${p}: ${have.has(p) ? 'built' : 'not built yet (owned by a page stream)'}`);
}

// ------------------------------------------------------------------ 2. third-party URLs
const textFiles = files.filter((f) => ['.html', '.js', '.css', '.json', '.svg'].includes(extname(f)) && !rel(f).startsWith('models/'));
const urlRe = /https?:\/\/[^\s"'`)<>\\]+/g;
const fetchedPatterns: [string, RegExp][] = [
	['src attribute', /\bsrc\s*=\s*["']https?:\/\//i],
	['<link href> (stylesheet/preload/icon)', /<link[^>]+href\s*=\s*["']https?:\/\//i],
	['CSS url()', /url\(\s*["']?https?:\/\//i],
	['CSS @import', /@import\s+(url\()?["']?https?:\/\//i],
	['fetch()/import()', /\b(fetch|import)\s*\(\s*["'`]https?:\/\//i],
	['XMLHttpRequest/WebSocket/EventSource', /new\s+(WebSocket|EventSource)\s*\(\s*["'`](wss?|https?):\/\//i]
];
const linkHosts = new Map<string, number>();
const otherUrls = new Map<string, Set<string>>();
for (const f of textFiles) {
	const s = readFileSync(f, 'utf8');
	for (const [what, re] of fetchedPatterns) {
		const m = re.exec(s);
		if (m) fail(`${rel(f)}: runtime third-party request (${what}) near: ${s.slice(Math.max(0, m.index - 60), m.index + 120).replace(/\s+/g, ' ')}`);
	}
	for (const m of s.matchAll(urlRe)) {
		const url = m[0];
		const before = s.slice(Math.max(0, m.index! - 12), m.index);
		const host = url.replace(/^https?:\/\//, '').split('/')[0];
		if (/href=["']?$/.test(before)) linkHosts.set(host, (linkHosts.get(host) ?? 0) + 1);
		else {
			if (!otherUrls.has(host)) otherUrls.set(host, new Set());
			otherUrls.get(host)!.add(rel(f));
		}
	}
}
console.log(`\nthird-party hosts in <a href> links (navigation only, never fetched): ${[...linkHosts.keys()].join(', ') || 'none'}`);
console.log('other absolute URLs (strings in code: namespaces, license comments, catalog text); none is fetched:');
for (const [host, fs] of otherUrls) console.log(`  ${host}  (${[...fs].slice(0, 3).join(', ')}${fs.size > 3 ? ', …' : ''})`);

// ------------------------------------------------------------------ 3. JS sizes
const js = files.filter((f) => f.endsWith('.js') && rel(f).startsWith('_app/'));
// gzip (level 9) on demand: a full build gzips every chunk; a package-only build only the ones its pages preload and
// the largest by raw size (the "largest chunks" list), which takes seconds instead of a quarter of a minute
const gzCache = new Map<string, number>();
const rawOf = new Map(js.map((f) => [rel(f), statSync(f).size]));
const gzOfFile = (key: string): number => {
	let v = gzCache.get(key);
	if (v === undefined) {
		v = gzipSync(readFileSync(join(BUILD, key)), { level: 9 }).length;
		gzCache.set(key, v);
	}
	return v;
};
const byRaw = [...rawOf.keys()].sort((a, b) => rawOf.get(b)! - rawOf.get(a)!);
const sizes = (ONLY ? byRaw.slice(0, 24) : byRaw).map((file) => ({ file, raw: rawOf.get(file)!, gz: gzOfFile(file) }));
sizes.sort((a, b) => b.gz - a.gz);
const kb = (n: number) => `${(n / 1024).toFixed(1)} KB`;
const rawTotal = [...rawOf.values()].reduce((a, b) => a + b, 0);
console.log(`\nJS files: ${rawOf.size}; total ${kb(rawTotal)} raw${ONLY ? '' : `, ${kb(sizes.reduce((a, b) => a + b.gz, 0))} gzip`}`);
console.log(`largest JS chunks (gzip / raw)${ONLY ? '; of the 24 largest by raw size' : ''}:`);
for (const s of sizes.slice(0, 8)) console.log(`  ${kb(s.gz).padStart(9)} / ${kb(s.raw).padStart(9)}  ${s.file}`);

const jsKeys = [...rawOf.keys()];
function pageJs(page: string): { files: number; gz: number; keys: string[] } {
	const html = readFileSync(join(BUILD, page), 'utf8');
	const refs = new Set<string>();
	for (const m of html.matchAll(/(?:href|import\()\s*=?\s*["']([^"']+\.js)["']/g)) refs.add(m[1]);
	let gz = 0;
	const keys: string[] = [];
	for (const r of refs) {
		const clean = r.replace(/^(\.\.\/|\.\/)+/, '').replace(/^\//, '');
		const key = jsKeys.find((k) => k.endsWith(clean.replace(/^.*?_app\//, '_app/')));
		if (key) {
			gz += gzOfFile(key);
			keys.push(key);
		}
	}
	return { files: refs.size, gz, keys };
}

// Distinctive text of each catalog module: 3 entries spread over its table, as letters, spaces and commas only (the
// minifier's quoting cannot change them). A page whose preloaded JS contains 2 of the 3 carries that module.
const spread = <T>(xs: T[]): T[] => [xs[Math.floor(xs.length / 4)], xs[Math.floor(xs.length / 2)], xs[xs.length - 1]];
const lettersRun = (s: string) => (s.match(/[A-Za-z][A-Za-z ,]{19,}/g) ?? []).sort((a, b) => b.length - a.length)[0] ?? s;
const CATALOG_TEXT: Record<string, string[]> = {
	models: spread(MODELS.filter((m) => m.status !== 'live')).map((m) => lettersRun(m.shows.en)),
	concepts: spread(CONCEPTS.filter((c) => c.status !== 'live')).map((c) => lettersRun(c.interaction.en)),
	'tech-zh': spread(Object.values(TECH_ZH).filter((z) => z.length >= 8 && !/["'`]/.test(z))),
	areas: spread(AREAS).map((a) => lettersRun(a.blurb.en)),
	glossary: spread(GLOSSARY).map((g) => lettersRun(g.def.en))
};
function catalogIn(keys: string[]): string[] {
	const code = keys.map((k) => readFileSync(join(BUILD, k), 'utf8')).join('\n');
	return Object.entries(CATALOG_TEXT)
		.filter(([, texts]) => texts.filter((x) => code.includes(x)).length >= 2)
		.map(([name]) => name);
}

console.log('\npage shell JS preloaded per page (gzip; budget 300 KB) and the catalog modules in it:');
const FLAGSHIPS = ['m/tinycnn/index.html', 'm/tinygpt/index.html'];
/** redesign wave F: a flagship shell over this warns (target 230 KB; the gate stays 300 KB) */
const FLAGSHIP_WARN = 250 * 1024;
const SAMPLES = ['index.html', 'atlas/index.html', 'atlas/vision/index.html', 'm/resnet-18/index.html', 'm/tinycnn/index.html', 'm/tinygpt/index.html', 'train/lora/index.html', 'glossary/index.html', 'about/index.html', 'dev/kit/index.html'];
for (const p of [...new Set([...SAMPLES, ...[...onlyPages.values()].flat()])]) {
	if (!have.has(p)) continue;
	const r = pageJs(p);
	const cat = catalogIn(r.keys);
	const room = FLAGSHIPS.includes(p) ? `; ${kb(300 * 1024 - r.gz)} left` : '';
	console.log(`  ${kb(r.gz).padStart(9)}  (${r.files} files${room})  /${p.replace('index.html', '')}  catalog: ${cat.join(', ') || 'none'}`);
	if (r.gz > 300 * 1024) fail(`/${p} preloads ${kb(r.gz)} of JS (budget 300 KB gzip)`);
	else if (FLAGSHIPS.includes(p) && r.gz > FLAGSHIP_WARN) console.warn(`WARN /${p} preloads ${kb(r.gz)} of JS: over the 250 KB warning line (the redesign's target is 230 KB; move panels and views out of the shell)`);
	if (cat.includes('glossary') && p !== 'glossary/index.html')
		fail(`/${p} preloads the glossary definitions (glossary.ts): only /glossary/ imports it; <Term> shows names from glossary-names.ts and loads the definitions on first use`);
	// a flagship page reads its catalog slice from page.data, so catalog changes never grow it (prep for wave 2b)
	if (FLAGSHIPS.includes(p) && cat.filter((c) => c !== 'glossary').length)
		fail(`/${p} preloads catalog modules (${cat.join(', ')}): a model page reads its catalog slice from page.data (src/lib/catalog/slices.ts) and imports no catalog table`);
}

// planned model pages share /m/[slug]: each must stay within the wave-0 baseline + 5 KB. The sample m/resnet-18 always
// gates; the others gate too unless ATLAS_INCLUDE_DRAFTS is set (then a draft package may render its auto page there).
const drafts = !!process.env.ATLAS_INCLUDE_DRAFTS && process.env.ATLAS_INCLUDE_DRAFTS !== '0';
const plannedPages = MODELS.filter((m) => m.status !== 'live')
	.map((m) => `m/${m.slug}/index.html`)
	.filter((p) => have.has(p));
const plannedShells = plannedPages.map((p) => ({ p, gz: pageJs(p).gz }));
const overBudget = plannedShells.filter((s) => s.gz > PLANNED_SHELL_BUDGET);
const maxPlanned = Math.max(0, ...plannedShells.map((s) => s.gz));
console.log(
	`planned model pages: ${plannedShells.length}, largest shell ${kb(maxPlanned)} (${maxPlanned} B); budget ${kb(PLANNED_SHELL_BUDGET)} = wave-0 baseline ${PLANNED_SHELL_BASELINE} B + 5 KB`
);
for (const s of overBudget) {
	const msg = `/${s.p} preloads ${kb(s.gz)} of JS (planned-page budget ${kb(PLANNED_SHELL_BUDGET)} = wave-0 baseline + 5 KB)`;
	if (s.p === 'm/resnet-18/index.html' || !drafts) fail(msg);
	else console.warn(`WARN ${msg} (ATLAS_INCLUDE_DRAFTS build: expected for a draft package)`);
}

// total size without model assets
const appBytes = files.filter((f) => !rel(f).startsWith('models/')).reduce((a, f) => a + statSync(f).size, 0);
const modelBytes = files.filter((f) => rel(f).startsWith('models/')).reduce((a, f) => a + statSync(f).size, 0);
console.log(`\nbuild size: ${(appBytes / 1e6).toFixed(2)} MB site + ${(modelBytes / 1e6).toFixed(2)} MB model assets (static/models, owned by the model streams)`);

if (failed) process.exit(1);
console.log('\nbuild checks OK');
