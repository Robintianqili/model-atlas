// atlas lesson <slug> draft|lint|print [--lang en|zh] [--fixture id] [--json] [--cap n]   (design §4.12, §5.2, §6.2)
//   draft  the auto plan of the package (planLesson on its spec, ops, adapters and processes): step ids to use with
//          auto('<id>') / autoAll({ until }) in lesson.ts, their views, sources and template narration.
//   lint   compileLesson + lintLesson: every step x every fixture (lesson fixtures, adapter edge fixtures, presets) x
//          en/zh on a real Node runtime. Errors exit 1 (AK401 AK403 AK404 AK405 AK406 AK407 AK409 AK411 AK501 AK502); AK402,
//          AK410, AK412 and lesson-shape advice are warnings. View keys are checked (AK405) when the tree has view metas, step
//          terms (AK411) against the site glossary and manifest.glossary.
//   print  every narration resolved on each fixture (or --fixture), as Markdown tables for translators and reviewers;
//          also written to .atlas/<slug>/lesson.<lang>.md.
// K1: lint and print run their sources through the kit's sharedRunner(): a run with the same process, input, options and
// seed on two fixtures (a source that ignores the learner's input, a preset equal to the start input) is recorded once.
// --cap n (lint and print, while iterating; never a gate) caps live training at n steps, as e2e's ?atlas-e2e=1 does: the
// narration then quotes other numbers, and a cursor past the cap is reported (AK404).
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { LintIssue, Narration, StepDef } from '../../../src/lib/atlas/types/index.ts';
import { AtlasError } from '../errors.ts';
import { writeJson, writeText } from '../lib/io.ts';
import { openPackage, type Opened } from '../lib/pkg.ts';
import { roots } from '../root.ts';

type Lang = 'en' | 'zh';

/** Registered view keys of a site: kit views (atlas/ui/views/<key>/meta.ts) and package views ('<slug>/<key>'). */
export function viewKeys(site: string): Set<string> {
	const keys = new Set<string>();
	const kitDir = join(site, 'src', 'lib', 'atlas', 'ui', 'views');
	if (existsSync(kitDir)) for (const k of readdirSync(kitDir)) if (existsSync(join(kitDir, k, 'meta.ts'))) keys.add(k);
	const ex = join(site, 'src', 'lib', 'explainers');
	if (existsSync(ex))
		for (const slug of readdirSync(ex)) {
			const vd = join(ex, slug, 'views');
			if (existsSync(vd)) for (const k of readdirSync(vd)) if (existsSync(join(vd, k, 'meta.ts'))) keys.add(`${slug}/${k}`);
		}
	return keys;
}

/** The site glossary's ids (src/lib/catalog/glossary.ts of the site), or null when the site has none. */
export async function glossaryIds(site: string): Promise<Set<string> | null> {
	const f = join(site, 'src', 'lib', 'catalog', 'glossary.ts');
	if (!existsSync(f)) return null;
	try {
		const m = (await import(pathToFileURL(f).href)) as { GLOSSARY?: { id: string }[] };
		return m.GLOSSARY ? new Set(m.GLOSSARY.map((g) => g.id)) : null;
	} catch {
		return null;
	}
}

export async function compiled(o: Opened) {
	const rt = await o.runtime();
	const spec = rt.spec;
	rt.close?.();
	const views = viewKeys(o.site);
	const terms = await glossaryIds(o.site);
	const ctx = {
		spec,
		ops: o.l.pkg.ops,
		manifest: o.l.pkg.manifest,
		adapters: o.l.pkg.adapters,
		processes: o.l.pkg.processes,
		...(views.size ? { views: (k: string) => views.has(k) } : {}),
		...(terms ? { terms: (id: string) => terms.has(id) } : {})
	};
	return { c: o.kit.compile.compileLesson(o.l.pkg.lesson, ctx), viewsChecked: views.size > 0, spec };
}

/** compile + lint over the fixtures; the issues (static and runtime) and the fixture ids used. */
export async function lintPackage(o: Opened, f: { fixture?: string; langs?: Lang[]; cap?: number } = {}): Promise<{ issues: LintIssue[]; fixtures: string[]; steps: number; viewsChecked: boolean; runs?: { shared: number; recorded: number } }> {
	const { c, viewsChecked } = await compiled(o);
	let fixtures = await o.fixtures();
	if (f.fixture) {
		fixtures = fixtures.filter((x) => x.id === f.fixture);
		if (!fixtures.length) throw new AtlasError('AK404', `fixture '${f.fixture}' not found`, { fix: `fixtures: ${(await o.fixtures()).map((x) => x.id).join(', ')}` });
	}
	if (!fixtures.length) fixtures = [{ id: 'default', label: { en: 'the default input', zh: '默认输入' }, raw: undefined, kind: 'default' }];
	const runner = o.kit.local.sharedRunner?.({ cap: f.cap });
	try {
		const issues = await o.kit.lint.lintLesson(c, (fx) => o.runtime(fx.raw, { runner }), fixtures, { langs: f.langs });
		return { issues, fixtures: fixtures.map((x) => x.id), steps: c.steps.length, viewsChecked, ...(runner ? { runs: { ...runner.stats } } : {}) };
	} finally {
		runner?.clear();
	}
}

const cell = (s: string | undefined) => (s ?? '').replace(/\|/g, '\\|').replace(/\n/g, ' ');
const short = (s: string, n = 90) => (s.length > n ? s.slice(0, n - 1) + '…' : s);

async function draft(o: Opened, json: boolean): Promise<number> {
	const rt = await o.runtime();
	const plan: StepDef[] = o.kit.plan.planLesson(rt.spec, o.l.pkg.ops, o.l.pkg.manifest, { adapters: o.l.pkg.adapters, processes: o.l.pkg.processes });
	rt.close?.();
	const { c } = await compiled(o);
	const used = new Set(c.steps.filter((s) => s.autoId).map((s) => s.autoId!));
	const rows = plan.map((s, i) => {
		const n = s.narration as Narration<never>;
		const view = typeof s.view === 'string' ? s.view : s.view.key;
		return { n: i + 1, id: s.id, used: used.has(s.id), view, source: s.source ?? '', title: s.title, narration: { en: n?.en ?? '', zh: n?.zh ?? '' } };
	});
	writeJson(join(roots().site, '.atlas', o.slug, 'lesson-draft.json'), { slug: o.slug, steps: rows }, 1);
	if (json) {
		console.log(JSON.stringify({ slug: o.slug, steps: rows }));
		return 0;
	}
	console.log(`atlas lesson ${o.slug} draft: ${plan.length} auto steps (the lesson uses ${used.size} of them; ${c.steps.length} steps in total)`);
	console.log('');
	console.log('| # | auto id | in lesson | view | source | title (en / zh) | narration (en template) |');
	console.log('|---|---|---|---|---|---|---|');
	for (const r of rows) console.log(`| ${r.n} | \`${r.id}\` | ${r.used ? 'yes' : ''} | ${r.view} | ${r.source} | ${cell(r.title.en)} / ${cell(r.title.zh)} | ${cell(short(r.narration.en))} |`);
	console.log('');
	console.log("Use them in lesson.ts: { autoAll: {} } (every step), { autoAll: { until: '<id>' } }, or { auto: '<id>', override: { title, narration } }.");
	return 0;
}

async function lint(o: Opened, fixture: string | undefined, langs: Lang[] | undefined, json: boolean, cap?: number): Promise<number> {
	const r = await lintPackage(o, { fixture, langs, cap });
	const errors = r.issues.filter((i) => i.level === 'error');
	const warns = r.issues.filter((i) => i.level === 'warn');
	if (json) {
		console.log(JSON.stringify({ slug: o.slug, steps: r.steps, fixtures: r.fixtures, errors: errors.length, warnings: warns.length, issues: r.issues }));
		return errors.length ? 1 : 0;
	}
	console.log(`atlas lesson ${o.slug} lint: ${r.steps} steps x ${r.fixtures.length} fixtures (${r.fixtures.join(', ')}) x ${(langs ?? ['en', 'zh']).join('/')}: ${errors.length} error(s), ${warns.length} warning(s)${r.viewsChecked ? '' : '; view keys not checked (no view metas in this tree)'}${r.runs ? `; ${r.runs.recorded} run(s) recorded, ${r.runs.shared} shared` : ''}${cap ? `; training capped at ${cap} steps (not a gate)` : ''}`);
	if (r.issues.length) console.error(o.kit.lint.formatIssues([...errors, ...warns]));
	return errors.length ? 1 : 0;
}

async function print(o: Opened, fixture: string | undefined, langs: Lang[], cap?: number): Promise<number> {
	const { c } = await compiled(o);
	const runner = o.kit.local.sharedRunner?.({ cap });
	let fixtures = await o.fixtures();
	if (fixture) fixtures = fixtures.filter((x) => x.id === fixture);
	if (!fixtures.length) fixtures = [{ id: 'default', label: { en: 'the default input', zh: '默认输入' }, raw: undefined, kind: 'default' }];
	const out: Record<Lang, string[]> = { en: [], zh: [] };
	for (const lang of langs) out[lang].push(`# ${o.slug}: lesson narration (${lang})`, '', `${c.steps.length} steps. Generated by \`npm run atlas -- lesson ${o.slug} print --lang ${lang}\`.`, '');
	for (const fx of fixtures) {
		const rt = await o.runtime(fx.raw, { runner });
		for (const lang of langs) out[lang].push(`## ${fx.id}: ${fx.label[lang] ?? fx.label.en}`, '', '| step | title | narration | look |', '|---|---|---|---|');
		for (const s of c.steps) {
			const pos = await o.kit.sources.stepAt(rt, s);
			for (const lang of langs) {
				if (pos.skipped) {
					out[lang].push(`| \`${s.id}\` | ${cell(s.title[lang] ?? s.title.en)} | (empty input: the stage shows the adapter's note) | |`);
					continue;
				}
				const n = o.kit.narrate.render(s.src.narration as Narration<never>, rt as never, pos.at, lang);
				const lk = s.src.look ? o.kit.narrate.render(s.src.look as Narration<never>, rt as never, pos.at, lang).text : '';
				out[lang].push(`| \`${s.id}\` | ${cell(s.title[lang] ?? s.title.en)} | ${cell(n.text)}${pos.error ? ` (AK404: ${cell(pos.error)})` : ''} | ${cell(lk)} |`);
			}
		}
		for (const lang of langs) out[lang].push('');
		rt.close?.();
	}
	runner?.clear();
	for (const lang of langs) {
		const text = out[lang].join('\n') + '\n';
		writeText(join(roots().site, '.atlas', o.slug, `lesson.${lang}.md`), text);
		process.stdout.write(text);
	}
	return 0;
}

export async function main(args: string[]): Promise<number> {
	const pos: string[] = [];
	let lang: Lang | undefined;
	let fixture: string | undefined;
	let json = false;
	let cap: number | undefined;
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		if (a === '--cap') {
			cap = Number(args[++i]);
			if (!Number.isInteger(cap) || cap < 1) throw new AtlasError('AK301', '--cap needs a step count (e.g. --cap 200)');
		} else if (a === '--lang') {
			const v = args[++i];
			if (v !== 'en' && v !== 'zh') throw new AtlasError('AK301', '--lang is en or zh');
			lang = v;
		} else if (a === '--fixture') fixture = args[++i];
		else if (a === '--json') json = true;
		else if (a.startsWith('--')) throw new AtlasError('AK301', `unknown flag ${a}`);
		else pos.push(a);
	}
	const [slug, what] = pos;
	if (!slug || !['draft', 'lint', 'print'].includes(what ?? '') || pos.length !== 2) throw new AtlasError('AK301', 'usage: atlas lesson <slug> draft|lint|print [--lang en|zh] [--fixture id] [--json] [--cap n]');
	const o = await openPackage(slug);
	if (what === 'draft') return draft(o, json);
	if (what === 'lint') return lint(o, fixture, lang ? [lang] : undefined, json, cap);
	return print(o, fixture, lang ? [lang] : ['en', 'zh'], cap);
}
