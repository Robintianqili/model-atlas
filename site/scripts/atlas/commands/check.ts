// atlas check <slug> [<slug>...] [--quick | --fast] [--passes <list>] [--json] [--only gate,gate] [--port n]
//   [--keep-server | --no-keep-server]
// atlas check --all [<slug>...] [--e2e <slugs|all>] [--passes <list>] [--fast] [--json]     (design §6.3; K2 speed pack)
//
// Modes (K2):
//   --quick  while iterating: every gate below except site and e2e, plus `types` (svelte-check scoped to the package);
//            parity and op tests are reported from .atlas/<slug>/gate-cache.json when none of their inputs changed since
//            they last passed (lib/gatecache.ts; the lesson text, claims and Svelte views are not inputs unless a package
//            test imports them). Seconds to a minute, no build, no browser.
//   (none)   the full check, once at the end: every gate; the site gate's build is package-only (ATLAS_ONLY=<slug>: the
//            home page and the package's pages, svelte.config.js) and e2e runs the default 2 passes (desktop-en-light,
//            phone-zh-dark; --passes all|full|<names> for more) on a reused dev server (lib/devserver.ts).
//   --fast   the old fast mode: no types, site or e2e gates (check --all uses it for every package).
//   several slugs: each package's gates, then ONE package-only build of all of them and ONE e2e sequence (one dev
//            server, one browser).
//   --all    every package's fast gates, kit honesty, docs, the atlas tests, legacy parity, ONE full build (npm run check
//            over the whole site, every page) and, for the packages given (positional or --e2e a,b|all), ONE e2e sequence
//            with --passes all by default.
//
// Gates for one package, each blocking unless marked:
//   schema       manifest (slug, catalog AK701 unless devOnly, TODO markers AK501, credits AK704), spec 1.1 + weights,
//                reference (atlas-ref-1), facts (atlas-facts/1), tapes (atlas-tape/1); inferShape at the dim extremes (AK302)
//   parity       `atlas parity` (A, B, C; legacy scripts with --no-write)
//   op tests     node --test on the package's own tests (local ops, views), except claims.test.ts
//   lesson       compileLesson + lintLesson over every fixture in en and zh (the Node dry run), then claims.test.ts (AK408)
//   determinism  every process and train source: same seed twice = same frames; keyEvery 1 = 64; state(i) = replay
//   honesty      Node imports every pure file; the dependency rule and the Math.random ban (AK503)
//   budgets      each file <= 5 MB, the model folder <= 15 MB, tapes <= 2 MB, auto-loaded bytes <= 5 MB (AK303)
//   types        svelte-check over the package's files only, 0 errors and 0 warnings (K2; skipped by --fast)
//   site         check:catalog, a package-only build into .atlas/build-<slug>/ (ATLAS_INCLUDE_DRAFTS=1, ATLAS_ONLY=<slug>),
//                check:build on it (skipped by --quick and --fast)
//   e2e          atlas e2e <slug> (default passes; --passes) on a reused or started dev server (--port n, else the
//                package's port, else the first free port from 5210; the summary names it); axe scans (AK604) and
//                contact sheets (skipped by --quick and --fast)
//   paperwork    notes/<slug>.md has What / Decisions / Final numbers / How to re-run / Kit issues without TODO (a missing
//                file is a warning: the public repository has no notes/)
//                (a warning for devOnly packages); credits complete
//   ownership    inside a lane: nothing outside the owned paths changed (AK702; merge --dry-run rules, parallel I/O)
// The report also counts page-specific lines of code (files under explainers/<slug>/ except tests; comments and CSS not
// counted; target <= 1,000).
// --all: every package's non-build gates, docs (node scripts/atlas/docs.ts --check: DEVKIT.md's generated tables match
// the code), all atlas Node tests (npm run check:atlas, which also runs the docs check), both legacy parity scripts
// (--no-write), then one full build into .atlas/build-all/ with npm run check and check:build (unless --fast), then one
// e2e sequence over the given packages.
// Writes .atlas/<slug>/check.json (or .atlas/check-all.json).
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { ProcessDef } from '../../../src/lib/atlas/types/index.ts';
import { pythonChoice } from '../config.ts';
import { AtlasError, formatIssue, type AkCode } from '../errors.ts';
import { siteSteps } from '../lib/build.ts';
import { runNode, runNpm } from '../lib/child.ts';
import { readRecord, stopKept } from '../lib/devserver.ts';
import { inputsKey, readCached, writeCached } from '../lib/gatecache.ts';
import { posix, writeJson } from '../lib/io.ts';
import { KNOWN_LANES, modelOwns, scanLane } from '../lib/lanes.ts';
import { resolvePasses } from '../lib/passes.ts';
import { Issues, LEGACY, openPackage, sig, walk, type Opened } from '../lib/pkg.ts';
import { checkPackages, firstProblems } from '../lib/typecheck.ts';
import { listPackages } from '../load.ts';
import { roots } from '../root.ts';
import type { E2eResult } from './e2e.ts';

export type GateStatus = 'pass' | 'fail' | 'warn' | 'skip';
export interface Gate {
	name: string;
	status: GateStatus;
	ms: number;
	summary: string;
	issues: { code: string; level: string; message: string; where?: string; fix?: string }[];
}

export interface CheckOpts {
	fast?: boolean;
	/** K2: iterate: the package gates + types, parity from the cache when unchanged; no build, no e2e */
	quick?: boolean;
	only?: string[];
	echo?: boolean;
	/** the e2e gate's dev-server port (else e2e picks one: the package's port when free, else a free one from 5210) */
	port?: number;
	/** e2e passes (lib/passes.ts); default 'default' for packages, 'all' for --all */
	passes?: string;
	/** keep the dev server e2e starts (default true) */
	keepServer?: boolean;
	/** --all: the packages of the one e2e sequence ('all' = every package) */
	e2e?: string[] | 'all';
}

const GATES = ['schema', 'parity', 'op tests', 'lesson', 'determinism', 'honesty', 'budgets', 'types', 'site', 'e2e', 'paperwork', 'ownership'] as const;

async function gate(name: string, o: CheckOpts, fn: (is: Issues) => Promise<string>): Promise<Gate> {
	if (o.only && !o.only.includes(name)) return { name, status: 'skip', ms: 0, summary: 'not selected (--only)', issues: [] };
	const t0 = performance.now();
	const is = new Issues();
	let summary = '';
	try {
		summary = await fn(is);
	} catch (e) {
		if (e instanceof AtlasError) is.error(e.code, e.message, e.info);
		else is.error('AK301', `the gate crashed: ${(e as Error).stack?.split('\n').slice(0, 3).join(' | ') ?? String(e)}`);
	}
	const status: GateStatus = is.errors.length ? 'fail' : is.warnings.length ? 'warn' : 'pass';
	return { name, status, ms: performance.now() - t0, summary, issues: is.list.map((i) => ({ code: i.code, level: i.level, message: i.message, where: i.info?.where, fix: i.info?.fix })) };
}

const skipGate = (name: string, why: string): Gate => ({ name, status: 'skip', ms: 0, summary: why, issues: [] });

// ------------------------------------------------------------------ schema
async function schemaGate(o: Opened, is: Issues): Promise<string> {
	const m = o.l.pkg.manifest;
	const r = roots();
	const exFile = join(o.l.dir, 'explainer.ts');
	const text = readFileSync(exFile, 'utf8');
	const todos = text.split('\n').filter((l) => /\bTODO\b/.test(l)).length;
	if (todos) is.error('AK501', `explainer.ts still has ${todos} line(s) with TODO markers`, { where: posix(relative(r.site, exFile)), fix: 'fill the goals, prequestion, lessonNote and credits' });
	// K2: a concept package (type 'concept', routed at /train/<slug>/) is checked against the concepts catalog
	const concept = (m.type as string) === 'concept';
	if (m.type !== 'model' && !concept) is.error('AK301', `manifest type must be 'model' or 'concept' (got '${m.type}')`);
	if (!['draft', 'reviewed', 'flagship'].includes(m.quality)) is.error('AK301', `quality '${m.quality}' is not draft, reviewed or flagship`);
	const catName = concept ? 'concepts.ts' : 'models.ts';
	const catPath = join(r.site, 'src', 'lib', 'catalog', catName);
	const catalog = existsSync(catPath)
		? (((await import(pathToFileURL(catPath).href)) as { MODELS?: { slug: string }[]; CONCEPTS?: { slug: string }[] })[concept ? 'CONCEPTS' : 'MODELS'] ?? [])
		: [];
	const inCatalog = catalog.some((e) => e.slug === o.slug);
	if (!m.devOnly && !inCatalog) is.error('AK701', `'${o.slug}' is not in the ${concept ? 'concepts ' : ''}catalog (src/lib/catalog/${catName})`, { fix: 'use a catalog slug, or mark the package devOnly (lab only)' });
	if (!Array.isArray(m.credits)) is.error('AK704', 'manifest.credits must be an array of licenses.json rows');
	else
		for (const [i, c] of m.credits.entries())
			for (const k of ['name', 'url', 'license', 'usedFor', 'attribution'] as const)
				if (typeof c[k] !== 'string' || !c[k].trim()) is.error('AK704', `credits[${i}] has no ${k}`, { fix: 'every credit is a licenses.json row: name, url, license, usedFor, attribution' });
	for (const g of m.page?.goals ?? []) if (!g.zh?.trim()) is.error('AK406', `a page goal has no zh: "${g.en.slice(0, 40)}"`);
	const notes: string[] = [];
	const k = o.kit;
	const specFile = join(o.dir, 'spec.json');
	if (existsSync(specFile)) {
		const raw = JSON.parse(readFileSync(specFile, 'utf8').replace(/^﻿/, ''));
		const wfile = join(o.dir, (raw.files?.weights as string) ?? 'weights.bin');
		const weightsBytes = existsSync(wfile) ? statSync(wfile).size : undefined;
		const spec = k.legacy.normalizeSpec(raw);
		const issues = k.schemaSpec.validateSpec(raw, { ops: o.l.pkg.ops, weightsBytes, folder: o.slug } as never);
		for (const i of issues) (i.level === 'error' ? is.error.bind(is) : is.warn.bind(is))(i.code as AkCode, `spec.json ${i.path}: ${i.message}`, { fix: i.fix });
		if (spec.slug !== o.slug) is.error('AK301', `spec.json slug '${spec.slug}' differs from the package '${o.slug}'`);
		notes.push(`spec ${spec.layers.length} layers, ${spec.params} params`);
		const refFile = join(o.dir, (raw.files?.reference as string) ?? 'reference.json');
		if (existsSync(refFile)) {
			const ri = k.schemaRef.validateRef(JSON.parse(readFileSync(refFile, 'utf8')), { spec });
			for (const i of ri) (i.level === 'error' ? is.error.bind(is) : is.warn.bind(is))(i.code as AkCode, `reference.json ${i.path}: ${i.message}`);
			notes.push('reference ok');
		} else if (m.kinds.includes('trace')) is.error('AK301', 'no reference.json next to spec.json', { fix: 'export_model writes it; re-run npm run atlas -- export <slug>' });
	} else if (m.kinds.includes('trace') && Object.values(o.sources).some((s) => s.kind === 'trace' && !s.assets))
		is.error('AK301', `no ${posix(relative(r.site, specFile))}`, { fix: `export the model: npm run atlas -- export ${o.slug}` });
	const factsFile = join(o.dir, 'facts.json');
	if (existsSync(factsFile)) {
		const fi = k.schemaFacts.validateFacts(JSON.parse(readFileSync(factsFile, 'utf8')));
		for (const i of fi) (i.level === 'error' ? is.error.bind(is) : is.warn.bind(is))(i.code as AkCode, `facts.json ${i.path}: ${i.message}`);
	}
	for (const f of walk(join(o.dir, 'tapes')).filter((x) => x.endsWith('.tape.json'))) {
		const h = JSON.parse(readFileSync(join(o.dir, 'tapes', f), 'utf8')) as { bin?: { file?: string } };
		const bin = join(o.dir, 'tapes', h.bin?.file ?? f.replace(/\.json$/, '.bin'));
		for (const p of k.schemaTape.validateTapeHeader(h, existsSync(bin) ? statSync(bin).size : undefined)) is.error('AK301', `tapes/${f}: ${p}`);
	}
	for (const n of o.l.notes) is.warn('AK301', `loader: ${n}`);
	return [`${concept ? 'concept, ' : ''}kinds ${m.kinds.join('+')}`, m.devOnly ? 'devOnly' : inCatalog ? 'in catalog' : 'NOT in catalog', ...notes].join('; ');
}

// ------------------------------------------------------------------ op tests and claims (node --test)
async function nodeTests(files: string[], echo: boolean): Promise<{ code: number; pass: number; fail: number; out: string }> {
	if (!files.length) return { code: 0, pass: 0, fail: 0, out: '' };
	const r = await runNode(['--test', ...files], { echo: echo ? '    ' : false, timeoutMs: 540_000 });
	const num = (k: string) => Number(new RegExp(`^ℹ ${k} (\\d+)`, 'm').exec(r.stdout)?.[1] ?? 0);
	return { code: r.code, pass: num('pass'), fail: num('fail'), out: r.stdout + r.stderr };
}

// ------------------------------------------------------------------ determinism
async function determinismGate(o: Opened, is: Issues): Promise<string> {
	const out: string[] = [];
	const rt = await o.runtime();
	try {
		for (const [name, d] of Object.entries(o.sources)) {
			if (d.kind !== 'process' && d.kind !== 'train') continue;
			const key = d.kind === 'train' ? `train:${d.trainer}` : d.process;
			const p = o.l.pkg.processes[key] as ProcessDef<any, any, any> | undefined;
			if (!p) {
				if (!(d.kind === 'process' && d.precomputed)) is.error('AK301', `source '${name}' runs '${key}', which is not registered`);
				continue;
			}
			const opt: Record<string, unknown> = { ...((d.opt ?? {}) as Record<string, unknown>) };
			if (d.kind === 'train') opt.steps = Math.min(Number(opt.steps ?? (p.defaults as { steps?: number }).steps ?? 100), 120);
			const seed = d.kind === 'process' ? (d.seed ?? 1) : typeof opt.seed === 'number' ? (opt.seed as number) : 1;
			// K1: a train source does not run on an empty input; a process with after: reads that source in input(rt)
			if (d.kind === 'train' && rt.pre?.empty && !d.runOnEmpty) continue;
			if (d.kind === 'process' && d.after) await rt.ensure(d.after, 'last');
			const input = d.kind === 'process' && d.input ? d.input(rt) : rt.input;
			const recordDraws = opt.recordDraws === true || (p as { recordDraws?: boolean }).recordDraws === true;
			// K2: a process source's own detail too (word2vec's fine 'one' was checked as 3 coarse frames)
			const detail = d.detail ?? 'coarse';
			const res = o.kit.determinism.checkDeterminism(p, input, opt, { seed, detail, recordDraws, samples: 200 });
			out.push(`${name} ${res.frames} frames ${res.ok ? 'ok' : 'FAIL'}`);
			if (!res.ok)
				is.error('AK301', `process determinism: source '${name}' (${key}): ${res.problems.join('; ')}`, {
					likely: res.firstStateMismatch >= 0 ? `a 'changes' field was mutated without a change entry (first at frame ${res.firstStateMismatch})` : 'randomness outside ctx.rng, or state shared between runs',
					fix: 'draw only from ctx.rng; report every edit of a track: \'changes\' field in frame.changes; copy tensors you keep mutating'
				});
		}
	} finally {
		rt.close?.();
	}
	return out.join(', ') || 'no process or train sources';
}

// ------------------------------------------------------------------ honesty: imports, dependency rule, Math.random
const IMPORT_RE = /(?:^|[\s;])(?:import|export)\s+(?:type\s+)?(?:[^'";]*?\s+from\s+)?['"]([^'"]+)['"]|import\(\s*['"]([^'"]+)['"]\s*\)/g;

export function stripComments(src: string): string {
	return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:'"`\\])\/\/.*$/gm, '$1');
}

const isTestPath = (rel: string) => /\.test\.ts$/.test(rel) || rel.startsWith('tests/atlas/') || /(^|\/)fixtures\//.test(rel);

/** Node tools that live next to runtime code but are never bundled (they generate files): exempt from the rule. */
export const NODE_TOOLS = ['src/lib/atlas/ops/manifest.ts'];

/** AK503: a pure file imports only src/lib/atlas/** (outside ui/), src/lib/i18n/types.ts and its own package; relative
 *  .ts specifiers; no $app/$lib aliases, no .svelte, no packages, no node:*. Kit adapters may import their model's own
 *  preprocessing (design §3.1): drawing28 -> models/tinycnn.ts, text-bpe -> models/bpe.ts. */
export function dependencyProblems(site: string, file: string, src: string): string[] {
	const rel = posix(relative(site, file));
	if (isTestPath(rel.replace(/^src\/lib\/atlas\//, '').replace(/^src\/lib\/explainers\/[^/]+\//, '')) || isTestPath(rel) || NODE_TOOLS.includes(rel)) return [];
	const out: string[] = [];
	const pkg = /^src\/lib\/explainers\/([^/]+)\//.exec(rel)?.[1];
	// `typeof import('node:fs')` is a type query (erased): a lazily imported Node module in a Node-only code path
	const code = stripComments(src).replace(/typeof\s+import\(\s*['"]node:[^'"]+['"]\s*\)/g, 'unknown');
	for (const m of code.matchAll(IMPORT_RE)) {
		const spec = m[1] ?? m[2];
		if (!spec) continue;
		if (!spec.startsWith('.')) {
			out.push(`imports '${spec}' (pure files import only relative .ts paths: no $lib/$app aliases, packages or node:*)`);
			continue;
		}
		if (!spec.endsWith('.ts')) {
			out.push(`imports '${spec}' (needs the .ts extension; .svelte files are not allowed)`);
			continue;
		}
		const target = posix(relative(site, join(file, '..', spec)));
		const ok =
			(target.startsWith('src/lib/atlas/') && !target.startsWith('src/lib/atlas/ui/')) ||
			target === 'src/lib/i18n/types.ts' ||
			(pkg !== undefined && target.startsWith(`src/lib/explainers/${pkg}/`)) ||
			(rel.startsWith('src/lib/atlas/adapters/drawing28/') && target === 'src/lib/models/tinycnn.ts') ||
			(rel.startsWith('src/lib/atlas/adapters/text-bpe/') && target === 'src/lib/models/bpe.ts');
		if (!ok) out.push(`imports '${spec}' (${target}): outside src/lib/atlas/** (minus ui/), i18n/types.ts and its own package`);
	}
	return out;
}

export function mathRandomUses(src: string): number {
	return (stripComments(src).match(/\bMath\.random\b/g) ?? []).length;
}

/** The pure files of a package (design §3.1 dependency rules): every .ts file of the package except tests and
 *  fixtures (explainer.ts, lesson.ts, trainer.ts, process.ts, worker.ts, ops.ts, ops/*.ts, adapters/<id>/adapter.ts,
 *  views/<key>/meta.ts and any helper module they import). Svelte files (views, sections, content, Capture) may import
 *  the kit and are not checked. */
export function pureFiles(dir: string): string[] {
	return walk(dir, ['node_modules'])
		.filter((f) => f.endsWith('.ts') && !isTestPath(f))
		.map((f) => join(dir, f));
}

/** The pure kit files (for --all): atlas/{core,...}/** outside ui/, and the kit adapters. */
export function pureKitFiles(site: string): string[] {
	const atlas = join(site, 'src', 'lib', 'atlas');
	const dirs = ['core', 'ops', 'graph', 'schema', 'autograd', 'train', 'data', 'engine', 'process', 'lesson', 'runtime', 'types'];
	const out: string[] = [];
	for (const d of dirs) for (const f of walk(join(atlas, d))) if (f.endsWith('.ts') && !isTestPath(f)) out.push(join(atlas, d, f));
	for (const f of walk(join(atlas, 'adapters'))) if (/^[^/]+\/adapter\.ts$/.test(f)) out.push(join(atlas, 'adapters', f));
	return out;
}

async function honestyFiles(site: string, files: string[], is: Issues, importThem: boolean, mathScope: string[]): Promise<string> {
	let imported = 0;
	for (const f of files) {
		const src = readFileSync(f, 'utf8');
		for (const p of dependencyProblems(site, f, src)) is.error('AK503', `dependency rule: ${posix(relative(site, f))} ${p}`, { fix: 'pure files import only src/lib/atlas/** (outside ui/) and src/lib/i18n/types.ts, with relative .ts paths' });
		if (/^\s*(export\s+)?(const\s+)?enum\s|^\s*namespace\s|\bimport\s+\w+\s*=\s*require/m.test(stripComments(src))) is.error('AK503', `${posix(relative(site, f))}: non-erasable TypeScript (enum, namespace or import =)`);
		if (importThem && !/\/worker\.ts$/.test(posix(f))) {
			try {
				await import(pathToFileURL(f).href);
				imported++;
			} catch (e) {
				is.error('AK503', `${posix(relative(site, f))} does not load in Node: ${(e as Error).message.split('\n')[0]}`, { likely: 'a Vite-only import ($lib, .svelte) or non-erasable TypeScript' });
			}
		}
	}
	let mathFiles = 0;
	for (const dir of mathScope)
		for (const f of walk(dir, ['node_modules'])) {
			if (!/\.(ts|svelte|js)$/.test(f) || isTestPath(f)) continue;
			const full = join(dir, f);
			if (posix(full).endsWith('src/lib/atlas/core/rng.ts')) continue;
			const n = mathRandomUses(readFileSync(full, 'utf8'));
			if (n) {
				mathFiles++;
				is.error('AK503', `Math.random in ${posix(relative(site, full))} (${n}x)`, { fix: 'draw from ctx.rng (processes, trainers) or core/rng.ts createRng(seed): every number must be reproducible' });
			}
		}
	return `${files.length} pure files${importThem ? `, ${imported} imported in Node` : ''}; Math.random in ${mathFiles} file(s)`;
}

// ------------------------------------------------------------------ budgets
const MB = 1e6;
function budgetsGate(o: Opened, is: Issues): string {
	if (!existsSync(o.dir)) return 'no asset folder';
	const files = walk(o.dir);
	let total = 0;
	for (const f of files) {
		const n = statSync(join(o.dir, f)).size;
		total += n;
		if (n > 5 * MB) is.error('AK303', `${f} is ${(n / MB).toFixed(2)} MB (> 5 MB per file)`);
		if (/^tapes\//.test(f) && /\.tape\.(json|bin)$/.test(f)) {
			const stem = f.replace(/\.tape\.(json|bin)$/, '');
			const both = ['json', 'bin'].map((x) => join(o.dir, `${stem}.tape.${x}`)).filter(existsSync).reduce((s, p) => s + statSync(p).size, 0);
			if (f.endsWith('.json') && both > 2 * MB) is.error('AK303', `tape ${stem} is ${(both / MB).toFixed(2)} MB (> 2 MB for phones)`);
		}
	}
	if (total > 15 * MB) is.error('AK303', `the model folder is ${(total / MB).toFixed(2)} MB (> 15 MB)`);
	const m = o.l.pkg.manifest;
	const auto = [...(m.assets?.auto ?? []), ...(m.input?.presets && !(m.assets?.auto ?? []).includes(m.input.presets) ? [m.input.presets] : [])];
	let autoBytes = 0;
	for (const a of auto) if (existsSync(join(o.dir, a))) autoBytes += statSync(join(o.dir, a)).size;
	// a trace source on another model's folder (TinyViT's cnn: { assets: 'models/tinycnn' }) loads its spec and weights on
	// every visit too (K0: they were not counted)
	const staticDir = join(roots().site, 'static');
	for (const d of Object.values(m.sources ?? {}))
		if (d.kind === 'trace' && d.assets)
			for (const f of ['spec.json', 'weights.bin']) if (existsSync(join(staticDir, d.assets, f))) autoBytes += statSync(join(staticDir, d.assets, f)).size;
	const cap = m.budget?.autoLoadBytes ?? 5 * MB;
	if (autoBytes > cap) is.error('AK303', `assets.auto load ${(autoBytes / MB).toFixed(2)} MB on every visit (> ${(cap / MB).toFixed(2)} MB)`);
	return `${files.length} files, ${(total / 1024).toFixed(1)} KB; auto-loaded ${(autoBytes / 1024).toFixed(1)} KB`;
}

// ------------------------------------------------------------------ paperwork
export const NOTE_SECTIONS = ['What', 'Decisions', 'Final numbers', 'How to re-run', 'Kit issues'];
export function notesProblems(text: string): string[] {
	const out: string[] = [];
	for (const s of NOTE_SECTIONS) {
		const m = new RegExp(`^##\\s+${s.replace(/ /g, '\\s+')}\\s*$([\\s\\S]*?)(?=^##\\s|(?![\\s\\S]))`, 'mi').exec(text);
		if (!m) out.push(`section "## ${s}" is missing`);
		else if (!m[1].trim()) out.push(`section "## ${s}" is empty`);
		else if (/^\s*TODO\b/m.test(m[1])) out.push(`section "## ${s}" still starts with TODO`);
	}
	return out;
}

function paperworkGate(o: Opened, is: Issues): string {
	const f = join(roots().root, 'notes', `${o.slug}.md`);
	const dev = !!o.l.pkg.manifest.devOnly;
	const add = dev ? is.warn.bind(is) : is.error.bind(is);
	// a missing notes file is a warning for every package: the public repository has no notes/ (a fresh clone checks its
	// packages without them); a notes file that exists must still be complete
	if (!existsSync(f)) {
		is.warn('AK501', `notes/${o.slug}.md is missing`, { fix: 'atlas new model writes it; it needs What, Decisions, Final numbers, How to re-run and Kit issues' });
		return dev ? 'devOnly: notes optional' : 'notes missing (a warning)';
	}
	const probs = notesProblems(readFileSync(f, 'utf8'));
	for (const p of probs) add('AK501', `notes/${o.slug}.md: ${p}`);
	return probs.length ? `notes: ${probs.length} problem(s)` : 'notes complete';
}

// ------------------------------------------------------------------ lines of page code
/** Lines of code in one file: not blank, not only a comment (//, /* … *\/, <!-- … -->), not in a <style> block. K0:
 *  the count rewarded packing CSS rules onto long lines and shortening comments. */
export function codeLines(text: string, svelte = false): number {
	let n = 0;
	let inBlock = false;
	let inHtml = false;
	let inStyle = false;
	for (const raw of text.split('\n')) {
		let l = raw.trim();
		if (svelte && /^<style[\s>]/.test(l)) inStyle = true;
		if (inStyle) {
			if (/<\/style>/.test(l)) inStyle = false;
			continue;
		}
		if (inHtml) {
			if (l.includes('-->')) {
				inHtml = false;
				l = l.slice(l.indexOf('-->') + 3).trim();
			} else continue;
		}
		if (inBlock) {
			if (l.includes('*/')) {
				inBlock = false;
				l = l.slice(l.indexOf('*/') + 2).trim();
			} else continue;
		}
		if (l.startsWith('<!--')) {
			if (!l.includes('-->')) inHtml = true;
			continue;
		}
		if (l.startsWith('/*')) {
			if (!l.includes('*/')) inBlock = true;
			else if (!l.slice(l.indexOf('*/') + 2).trim()) continue;
			else l = l.slice(l.indexOf('*/') + 2).trim();
			if (inBlock) continue;
		}
		if (!l || l.startsWith('//')) continue;
		n++;
	}
	return n;
}

export function pageLines(dir: string): { files: number; lines: number; all: number } {
	let lines = 0;
	let all = 0;
	let files = 0;
	for (const f of walk(dir)) {
		if (/\.test\.ts$/.test(f) || !/\.(ts|svelte|js)$/.test(f)) continue;
		files++;
		const text = readFileSync(join(dir, f), 'utf8');
		all += text.split('\n').filter((l) => l.trim()).length;
		lines += codeLines(text, f.endsWith('.svelte'));
	}
	return { files, lines, all };
}

// ------------------------------------------------------------------ one package
type Result = { slug: string; gates: Gate[]; pass: boolean; lines: { files: number; lines: number; all: number } };

/** Package-relative paths of the package's code files that `files` import (relative imports inside the package,
 *  transitively). */
export function importClosure(dir: string, files: string[]): Set<string> {
	const seen = new Set<string>();
	const todo = files.map((f) => posix(relative(dir, f)));
	while (todo.length) {
		const rel = todo.pop()!;
		if (seen.has(rel) || rel.startsWith('..')) continue;
		const full = join(dir, rel);
		if (!existsSync(full)) continue;
		seen.add(rel);
		for (const m of stripComments(readFileSync(full, 'utf8')).matchAll(IMPORT_RE)) {
			const spec = m[1] ?? m[2];
			if (spec?.startsWith('.')) todo.push(posix(relative(dir, join(full, '..', spec))));
		}
	}
	return seen;
}

/** What the --quick cache of a gate keys on: the package's files except the ones that cannot change the gate's result
 *  (the lesson text, the claims, Svelte views, notes) unless `keep` (an import closure) needs them, the package's assets,
 *  its pipeline folder, atlas_kit, the kit, the CLI (scripts/atlas) and the configured Python. A test that loads the package at
 *  run time (openPackage) does not import lesson.ts: lesson fixtures added meanwhile are first tested by the full check. */
function gateInputs(op: Opened, keep: Set<string> = new Set()): string[] {
	const r = roots();
	const free = (f: string) => f === 'lesson.ts' || f === 'claims.test.ts' || /\.(svelte|md)$/.test(f);
	const pkg = walk(op.l.dir).filter((f) => keep.has(f) || !free(f)).map((f) => join(op.l.dir, f));
	return [...pkg, op.dir, join(r.root, 'pipeline', op.slug), join(r.root, 'pipeline', 'atlas_kit'), join(r.site, 'src', 'lib', 'atlas'), join(r.site, 'scripts', 'atlas')];
}

/** A gate that --quick may answer from the cache (lib/gatecache.ts) when its inputs did not change since it passed. */
async function cachedGate(op: Opened, o: CheckOpts, name: string, paths: string[], run: (is: Issues) => Promise<string>): Promise<Gate> {
	const r = roots();
	const key = inputsKey(paths, `python=${JSON.stringify(pythonChoice())}`);
	if (o.quick && !(o.only && !o.only.includes(name))) {
		const c = readCached(r.site, op.slug, name, key);
		if (c) return { name, status: c.status, ms: 0, summary: `cached (inputs unchanged since ${c.at}): ${c.summary}`, issues: c.issues };
	}
	const g = await gate(name, o, run);
	if (g.status === 'pass' || g.status === 'warn') writeCached(r.site, op.slug, name, { key, at: new Date().toLocaleString('sv-SE').slice(0, 16), summary: g.summary, status: g.status, issues: g.issues });
	else if (g.status === 'fail') writeCached(r.site, op.slug, name, null);
	return g;
}

async function parityGate(op: Opened, o: CheckOpts): Promise<Gate> {
	const slug = op.slug;
	return cachedGate(op, o, 'parity', gateInputs(op), async (is) => {
		const { parity } = await import('./parity.ts');
		const res = await parity(slug, { echo: false });
		for (const i of res.issues.list) is.list.push(i);
		const parts: string[] = [];
		if (res.report.forward) parts.push(`A ${sig(res.report.forward.max)} (bar ${sig(res.report.forward.bar)})`);
		for (const t of res.report.train) parts.push(t.runs.length ? `B ${t.source}: ${t.runs.map((x) => `${x.optimizer} ${sig(Math.max(x.losses, x.final))}`).join(', ')}` : `B ${t.source}: no twin`);
		// K2: a reference that failed to run compared nothing (it printed "identical" next to its AK204)
		for (const p of res.report.process) parts.push(p.reference ? `C ${p.source}: ${p.firstDivergence ? 'diverges' : p.pass ? `identical (${sig(p.maxFloat)})` : 'reference failed'}` : `C ${p.source}: no reference`);
		return parts.join('; ') || 'nothing to compare';
	});
}

/** svelte-check over packages only (K2). */
function typesGate(slugs: string[], name: string, o: CheckOpts): Promise<Gate> {
	return gate('types', o, async (is) => {
		const tc = await checkPackages(roots().site, slugs, name, { echo: o.echo ? '    [svelte-check] ' : false });
		if (!tc.ok) is.error('AK301', `svelte-check ${tc.scope}: ${tc.errors ?? '?'} errors, ${tc.warnings ?? '?'} warnings`, { likely: firstProblems(tc.out.replace(/\x1b\[[0-9;]*m/g, '')) || undefined, fix: `fix every error and warning (the bar is 0/0); the whole site: npm run check` });
		return `svelte-check of ${slugs.join(', ')}: ${tc.errors ?? '?'} errors, ${tc.warnings ?? '?'} warnings`;
	});
}

/** The gates before the site gate (schema ... budgets, and types unless --fast). */
async function packageGates(op: Opened, o: CheckOpts): Promise<Gate[]> {
	const r = roots();
	const slug = op.slug;
	const gates: Gate[] = [];
	const echo = !!o.echo;
	gates.push(await gate('schema', o, (is) => schemaGate(op, is)));
	gates.push(await parityGate(op, o));
	const tests = walk(op.l.dir).filter((f) => f.endsWith('.test.ts') && f !== 'claims.test.ts').map((f) => join(op.l.dir, f));
	gates.push(
		await cachedGate(op, o, 'op tests', gateInputs(op, importClosure(op.l.dir, tests)), async (is) => {
			const t = await nodeTests(tests, echo);
			if (t.code !== 0) is.error('AK301', `package tests failed (${t.fail} of ${t.pass + t.fail})`, { where: tests.map((f) => posix(relative(r.site, f))).join(', ') });
			return tests.length ? `${t.pass} passed, ${t.fail} failed (${tests.length} file(s))` : 'no package tests (local ops and views bring their own)';
		})
	);
	gates.push(
		await gate('lesson', o, async (is) => {
			const { lintPackage } = await import('./lesson.ts');
			const res = await lintPackage(op);
			for (const i of res.issues) (i.level === 'error' ? is.error.bind(is) : is.warn.bind(is))(i.code as AkCode, i.message, { where: [i.step && `step ${i.step}`, i.fixture && `fixture ${i.fixture}`, i.lang].filter(Boolean).join(', ') || undefined, fix: i.fix });
			const claims = join(op.l.dir, 'claims.test.ts');
			let claimText = 'no claims.test.ts';
			if (existsSync(claims)) {
				const t = await nodeTests([claims], echo);
				claimText = `claims ${t.pass} passed, ${t.fail} failed`;
				if (t.code !== 0) is.error('AK408', `claim tests failed (${t.fail})`, { where: posix(relative(r.site, claims)), likely: t.out.split('\n').filter((l) => /✖|AssertionError|Error:/.test(l)).slice(0, 4).join(' | ') });
			} else if (op.l.pkg.manifest.quality !== 'draft') is.error('AK408', 'a reviewed lesson needs claims.test.ts for its factual sentences');
			const errs = res.issues.filter((i) => i.level === 'error').length;
			return `${res.steps} steps x ${res.fixtures.length} fixtures x en/zh: ${errs} error(s), ${res.issues.length - errs} warning(s)${res.viewsChecked ? '' : ' (view keys not checked: no view metas)'}; ${claimText}`;
		})
	);
	gates.push(await gate('determinism', o, (is) => determinismGate(op, is)));
	gates.push(await gate('honesty', o, (is) => honestyFiles(r.site, pureFiles(op.l.dir), is, true, [op.l.dir])));
	gates.push(await gate('budgets', o, async (is) => budgetsGate(op, is)));
	gates.push(o.fast ? skipGate('types', '--fast') : await typesGate([slug], slug, o));
	return gates;
}

/** The ownership gate (inside a lane): planMerge's rules with parallel I/O (lib/lanes.ts scanLane, K2). */
async function ownershipGate(o: CheckOpts): Promise<Gate> {
	const r = roots();
	if (!r.lane) return skipGate('ownership', 'main tree (no lane)');
	return gate('ownership', o, async (is) => {
		const name = r.lane!;
		const laneDir = join(r.site, '..');
		const main = join(laneDir, '..', '..');
		const plan = await scanLane(main, r.site, join(laneDir, 'lane.json'), KNOWN_LANES[name] ?? modelOwns(name));
		if (plan.outside.length) is.error('AK702', `the lane changed ${plan.outside.length} file(s) outside its owned paths`, { where: plan.outside.slice(0, 8).join(', '), fix: `npm run atlas -- merge ${name} --dry-run lists them` });
		return `lane ${name}: ${plan.copy.length} owned file(s) to merge, ${plan.outside.length} outside`;
	});
}

function writeReport(res: Result): void {
	writeJson(join(roots().site, '.atlas', res.slug, 'check.json'), { ...res, checked: new Date().toISOString() }, 1);
}

const e2eGateFrom = (x: Pick<E2eResult, 'issues' | 'summary' | 'ms'>): Gate => {
	const issues = x.issues.map((i) => ({ code: String(i.code), level: i.level, message: i.message, where: i.info?.where, fix: i.info?.fix }));
	const status: GateStatus = issues.some((i) => i.level === 'error') ? 'fail' : issues.length ? 'warn' : 'pass';
	return { name: 'e2e', status, ms: x.ms, summary: x.summary, issues };
};

/** Several packages: each one's gates, ONE package-only build and ONE e2e sequence (K2). One package: the same. */
export async function checkPackages_(slugs: string[], o: CheckOpts = {}): Promise<{ results: Result[] }> {
	const r = roots();
	const opened: Opened[] = [];
	for (const s of slugs) opened.push(await openPackage(s, r.site));
	const pre = new Map<string, Gate[]>();
	for (const op of opened) pre.set(op.slug, await packageGates(op, o));
	const e2eGates = new Map<string, Gate>();
	const late = o.fast || o.quick;
	const why = o.quick ? '--quick' : '--fast';
	const name = slugs.length === 1 ? `build-${slugs[0]}` : `build-only-${slugs.length}`;
	const site = late ? skipGate('site', why) : await gate('site', o, async (is) => (await siteSteps(is, { site: r.site, name, slugs, echo: o.echo, noTypes: true })).summary);
	if (late) for (const s of slugs) e2eGates.set(s, skipGate('e2e', why));
	else if (o.only && !o.only.includes('e2e')) for (const s of slugs) e2eGates.set(s, skipGate('e2e', 'not selected (--only)'));
	else {
		const t0 = performance.now();
		try {
			const { e2eMany } = await import('./e2e.ts');
			const res = await e2eMany(slugs, { port: o.port, passes: o.passes, keepServer: o.keepServer, echo: o.echo });
			for (const x of res) e2eGates.set(x.slug, e2eGateFrom(x));
		} catch (e) {
			const msg = e instanceof AtlasError ? e : null;
			for (const s of slugs) e2eGates.set(s, { name: 'e2e', status: 'fail', ms: performance.now() - t0, summary: 'e2e could not run', issues: [{ code: msg?.code ?? 'AK301', level: 'error', message: msg?.message ?? (e as Error).message, where: msg?.info.where, fix: msg?.info.fix }] });
		}
	}
	const own = await ownershipGate(o);
	const results: Result[] = [];
	for (const op of opened) {
		const gates = [...pre.get(op.slug)!];
		gates.push(slugs.length === 1 ? site : { ...site, summary: `${site.summary}${site.status === 'skip' ? '' : ' (shared by ' + slugs.join(', ') + ')'}` });
		gates.push(e2eGates.get(op.slug)!);
		gates.push(await gate('paperwork', o, async (is) => paperworkGate(op, is)));
		gates.push(own);
		const res: Result = { slug: op.slug, gates, pass: gates.every((g) => g.status !== 'fail'), lines: pageLines(op.l.dir) };
		writeReport(res);
		results.push(res);
	}
	return { results };
}

export async function checkPackage(slug: string, o: CheckOpts = {}): Promise<Result> {
	return (await checkPackages_([slug], o)).results[0];
}

// ------------------------------------------------------------------ --all
async function checkAll(o: CheckOpts): Promise<{ results: { slug: string; pass: boolean; gates: Gate[] }[]; extra: Gate[]; pass: boolean }> {
	const r = roots();
	const results: { slug: string; pass: boolean; gates: Gate[] }[] = [];
	const all = listPackages(r.site);
	for (const slug of all) {
		const res = await checkPackage(slug, { ...o, fast: true, quick: false, only: o.only });
		results.push(res);
	}
	const extra: Gate[] = [];
	extra.push(
		await gate('kit honesty', o, (is) => honestyFiles(r.site, pureKitFiles(r.site), is, false, [join(r.site, 'src', 'lib', 'atlas'), join(r.site, 'src', 'lib', 'explainers')]))
	);
	extra.push(
		await gate('docs', o, async (is) => {
			// DEVKIT.md's generated tables (commands, codes, ports, ops, views) must match the code (R0: the op table drifted)
			const res = await runNode([join(r.site, 'scripts', 'atlas', 'docs.ts'), '--check'], { timeoutMs: 120_000 });
			if (res.code !== 0) is.error('AK301', 'DEVKIT.md is out of date', { fix: 'node scripts/atlas/docs.ts --write, then review the diff' });
			return res.code === 0 ? 'DEVKIT.md generated tables up to date' : 'DEVKIT.md out of date';
		})
	);
	extra.push(
		await gate('atlas tests', o, async (is) => {
			const res = await runNpm('check:atlas', [], { echo: o.echo ? '    ' : false, timeoutMs: 540_000 });
			const pass = Number(/^ℹ pass (\d+)/m.exec(res.stdout)?.[1] ?? 0);
			const fail = Number(/^ℹ fail (\d+)/m.exec(res.stdout)?.[1] ?? 0);
			if (res.code !== 0) is.error('AK301', `npm run check:atlas: ${fail} failed`);
			return `${pass} passed, ${fail} failed`;
		})
	);
	for (const slug of LEGACY)
		extra.push(
			await gate(`legacy parity ${slug}`, o, async (is) => {
				const { parity } = await import('./parity.ts');
				const res = await parity(slug, { echo: false });
				for (const i of res.issues.list) is.list.push(i);
				const L = res.report.legacy!;
				return `measured ${sig(L.measured)} (shipped ${sig(L.shipped)}); spec.json ${L.specSha.unchanged ? 'unchanged' : 'CHANGED'}`;
			})
		);
	if (!o.fast) extra.push(await gate('site (one build)', o, async (is) => (await siteSteps(is, { site: r.site, name: 'build-all', slugs: null, echo: o.echo })).summary));
	else extra.push(skipGate('site (one build)', '--fast'));
	// K2: one e2e sequence over the given packages (one dev server, one browser), --passes all by default
	const e2eSlugs = o.e2e === 'all' ? all : (o.e2e ?? []);
	if (e2eSlugs.length && !o.fast) {
		const { e2eMany } = await import('./e2e.ts');
		const res = await e2eMany(e2eSlugs, { port: o.port, passes: o.passes ?? 'all', keepServer: o.keepServer, echo: o.echo });
		for (const x of res) extra.push({ ...e2eGateFrom(x), name: `e2e ${x.slug}` });
	} else extra.push(skipGate('e2e (one sequence)', o.fast ? '--fast' : 'no packages given (atlas check --all <slug> ... or --e2e all)'));
	const pass = results.every((x) => x.pass) && extra.every((g) => g.status !== 'fail');
	writeJson(join(r.site, '.atlas', 'check-all.json'), { results, extra, pass, checked: new Date().toISOString() }, 1);
	return { results, extra, pass };
}

// ------------------------------------------------------------------ printing
const MARK: Record<GateStatus, string> = { pass: 'ok  ', fail: 'FAIL', warn: 'warn', skip: 'skip' };
function printGates(title: string, gates: Gate[]): void {
	console.log(title);
	for (const g of gates) console.log(`  ${MARK[g.status]}  ${g.name.padEnd(18)} ${(g.ms / 1000).toFixed(1).padStart(5)} s  ${g.summary}`);
}
function printIssues(gates: Gate[]): void {
	const lines: string[] = [];
	for (const g of gates)
		for (const i of g.issues) lines.push(formatIssue(i.code, `${i.level === 'warn' ? '(warn) ' : ''}[${g.name}] ${i.message}`, { where: i.where, fix: i.fix }));
	if (lines.length) console.error(lines.join('\n'));
}

export const USAGE = 'atlas check <slug> [<slug>...] [--quick|--fast] [--passes <list>] [--json] [--only gate,gate] [--port n] [--keep-server|--no-keep-server] [--stop-server] | check --all [<slug>...] [--e2e <slugs|all>] [--passes <list>] [--fast] [--json]';

export async function main(args: string[]): Promise<number> {
	const t0 = performance.now();
	const o: CheckOpts & { json?: boolean; all?: boolean; stop?: boolean } = { echo: false };
	const pos: string[] = [];
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		if (a === '--fast') o.fast = true;
		else if (a === '--quick') o.quick = true;
		else if (a === '--json') o.json = true;
		else if (a === '--all') o.all = true;
		else if (a === '--verbose') o.echo = true;
		else if (a === '--only') o.only = (args[++i] ?? '').split(',').map((s) => s.trim());
		else if (a === '--passes') o.passes = args[++i] ?? '';
		else if (a === '--keep-server') o.keepServer = true;
		else if (a === '--no-keep-server') o.keepServer = false;
		else if (a === '--stop-server') o.stop = true;
		else if (a === '--e2e') {
			const v = (args[++i] ?? '').trim();
			o.e2e = v === 'all' ? 'all' : v.split(',').map((s) => s.trim()).filter(Boolean);
		} else if (a === '--port') {
			o.port = Number(args[++i]);
			if (!Number.isInteger(o.port) || o.port < 1024 || o.port > 65535) throw new AtlasError('AK301', '--port needs a port number (1024-65535)');
		} else if (a.startsWith('--')) throw new AtlasError('AK301', `unknown flag ${a}`, { fix: USAGE });
		else pos.push(...a.split(',').map((s) => s.trim()).filter(Boolean));
	}
	if (o.only) for (const g of o.only) if (!(GATES as readonly string[]).includes(g)) throw new AtlasError('AK301', `unknown gate '${g}'`, { fix: `gates: ${GATES.join(', ')}` });
	if (o.quick && o.fast) throw new AtlasError('AK301', '--quick and --fast: pick one (--quick adds the package type check and the parity cache)');
	if (o.passes !== undefined) resolvePasses(o.passes);
	const done = async () => {
		// --json: stdout is the report alone; the notes go to stderr
		const say = o.json ? console.error : console.log;
		if (o.stop) say(`  ${(await stopKept()).note}`);
		else {
			const rec = readRecord();
			if (rec) say(`  dev server kept on http://localhost:${rec.port}/ (pid ${rec.pid}; it stops itself after ${rec.idleMin} idle minutes; stop it now: npm run atlas -- e2e --stop-server)`);
		}
		say(`  took ${((performance.now() - t0) / 1000).toFixed(1)} s`);
	};
	if (o.all) {
		if (o.quick) throw new AtlasError('AK301', 'check --all has no --quick (it runs every package\'s fast gates; --fast skips the build and e2e)');
		if (pos.length) o.e2e = o.e2e === 'all' ? 'all' : [...new Set([...(o.e2e ?? []), ...pos])];
		const res = await checkAll(o);
		if (o.json) console.log(JSON.stringify(res));
		else {
			for (const x of res.results) printGates(`atlas check ${x.slug} (--fast): ${x.pass ? 'PASS' : 'FAIL'}`, x.gates);
			printGates(`atlas check --all: ${res.pass ? 'PASS' : 'FAIL'}`, res.extra);
			printIssues([...res.results.flatMap((x) => x.gates), ...res.extra]);
		}
		await done();
		return res.pass ? 0 : 1;
	}
	if (!pos.length) {
		if (o.stop) {
			console.log(`atlas check --stop-server: ${(await stopKept()).note}`);
			return 0;
		}
		throw new AtlasError('AK301', `usage: ${USAGE}`);
	}
	const { results } = await checkPackages_([...new Set(pos)], o);
	if (o.json) console.log(JSON.stringify(results.length === 1 ? results[0] : results));
	else {
		const mode = o.quick ? ' --quick' : o.fast ? ' --fast' : '';
		for (const res of results) {
			printGates(`atlas check ${res.slug}${mode}: ${res.pass ? 'PASS' : 'FAIL'}`, res.gates);
			console.log(`  page code: ${res.lines.lines} lines of code in ${res.lines.files} files under src/lib/explainers/${res.slug}/ (target <= 1,000; comments, CSS and tests not counted; ${res.lines.all} non-blank lines in all)`);
			console.log(`  report: .atlas/${res.slug}/check.json`);
			printIssues(res.gates);
		}
		if (o.quick) console.log('  --quick: no build and no browser; run the full check (atlas check <slug>) once at the end');
	}
	await done();
	return results.every((x) => x.pass) ? 0 : 1;
}
