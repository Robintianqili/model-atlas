// DevKit K2 speed pack: e2e passes, package-only builds (ATLAS_ONLY), the scoped type check, contact sheets and the
// lesson print, axe results, the gate cache, the parallel ownership scan and the dev-server identity probe.
//   node --test scripts/atlas/tests/k2-speed.test.ts   (from site/)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { legacyPasses, parsePass, resolvePasses } from '../lib/passes.ts';
import { candidatePages, isCandidate, onlyAllows, onlyEntries } from '../lib/only.ts';
import { lessonPrint, numbersIn, sheetHtml, sheetLayout, type StepText } from '../lib/contact.ts';
import { AK_A11Y, axeIssues, axeText, type StepScan } from '../lib/axe.ts';
import { parseSvelteCheck, scopedTsconfig } from '../lib/typecheck.ts';
import { inputsKey, readCached, writeCached } from '../lib/gatecache.ts';
import { scanLane, sha256, siteFiles } from '../lib/lanes.ts';
import { probeSite, readRecord } from '../lib/devserver.ts';
import { buildEnv } from '../lib/build.ts';
import { Issues } from '../lib/pkg.ts';
import { parseE2eArgs } from '../commands/e2e.ts';
import { importClosure } from '../commands/check.ts';
import { SITE } from './helpers.ts';

const tmp = () => mkdtempSync(join(tmpdir(), 'atlas-k2-'));

test('passes: default is desktop-en-light + phone-zh-dark; the phone pass probes 360 px; presets and names', () => {
	const d = resolvePasses();
	assert.deepEqual(d.map((p) => p.name), ['desktop-en-light', 'phone-zh-dark']);
	assert.equal(d[0].width, 1280);
	assert.ok(d[0].badge, 'the English desktop pass checks the badges');
	assert.equal(d[1].width, 390);
	assert.equal(d[1].lang, 'zh');
	assert.equal(d[1].scheme, 'dark');
	assert.ok(d[1].scroll && d[1].probe360, 'a 390 px pass checks scroll and probes the 360 px gate width');
	const all = resolvePasses('all');
	assert.equal(all.length, 5);
	assert.ok(all.some((p) => p.name === 'narrow-en-dark' && p.width === 360 && p.scroll && !p.probe360));
	assert.ok(all.some((p) => p.reduced && !p.shots));
	assert.equal(resolvePasses('full').length, 8);
	assert.deepEqual(resolvePasses('default,narrow-en-dark,phone-zh-dark').map((p) => p.name), ['desktop-en-light', 'phone-zh-dark', 'narrow-en-dark']);
	assert.equal(parsePass('414-en-light').width, 414);
	assert.throws(() => parsePass('tablet-en-light'), /unknown e2e pass/);
	assert.throws(() => parsePass('100-en-light'), /outside/);
	assert.equal(legacyPasses({ full: true }), 'full');
	assert.equal(legacyPasses({ phone: 414 }), 'default,414-en-light');
	assert.equal(legacyPasses({ phone: 360 }), 'default');
	assert.equal(resolvePasses('phone-zh-dark').find((p) => p.badge), undefined, 'no English pass: no badge check');
});

test('e2e arguments: several packages, --passes checked before anything starts', () => {
	const a = parseE2eArgs(['tinyvit,gcn', 'pinn', '--passes', 'all', '--axe', 'warn', '--no-keep-server']);
	assert.deepEqual(a.slugs, ['tinyvit', 'gcn', 'pinn']);
	assert.equal(a.o.passes, 'all');
	assert.equal(a.o.axe, 'warn');
	assert.equal(a.o.keepServer, false);
	assert.throws(() => parseE2eArgs(['x', '--passes', 'huge-en-light']), /unknown e2e pass/);
	assert.throws(() => parseE2eArgs(['x', '--axe', 'maybe']), /fail, warn or off/);
	assert.equal(parseE2eArgs(['--stop-server']).stop, true);
});

test('ATLAS_ONLY: the pages of a package-only build and what its server hook lets through', () => {
	assert.equal(onlyEntries({}), null);
	assert.equal(onlyEntries({ ATLAS_ONLY: ' ' }), null);
	const e = onlyEntries({ ATLAS_ONLY: 'gcn, hello-xor', ATLAS_ONLY_PAGES: 'atlas,/glossary/' })!;
	assert.deepEqual(e, ['/', ...candidatePages('gcn'), ...candidatePages('hello-xor'), '/atlas/', '/glossary/']);
	assert.ok(onlyAllows('/', e));
	assert.ok(onlyAllows('/__data.json', e));
	assert.ok(onlyAllows('/m/gcn/', e));
	assert.ok(onlyAllows('/m/gcn/__data.json', e));
	assert.ok(onlyAllows('/dev/atlas/lab/hello-xor', e));
	assert.ok(onlyAllows('/[fallback]', e));
	assert.ok(onlyAllows('/%5Bfallback%5D', e));
	assert.ok(!onlyAllows('/m/tinyvit/', e));
	assert.ok(!onlyAllows('/devkit/guide/', e));
	assert.ok(isCandidate('/model-atlas/train/gcn/', e), 'a candidate page (with a base path) may 404');
	assert.ok(!isCandidate('/m/tinyvit/', e));
	assert.ok(!isCandidate('/', e), 'the home page must exist');
	assert.equal(buildEnv('build-x', ['a', 'b']).ATLAS_ONLY, 'a,b');
	assert.equal(buildEnv('build-all', null).ATLAS_ONLY, '', 'a full build clears an inherited ATLAS_ONLY');
});

test('svelte.config.js: ATLAS_ONLY switches the prerender to the package pages; unset, the config is the full build', () => {
	const read = (env: Record<string, string>) => {
		const r = spawnSync(process.execPath, ['-e', "import('./svelte.config.js').then((m) => { const k = m.default.kit; console.log(JSON.stringify({ e: k.prerender.entries, c: k.prerender.crawl, u: k.prerender.handleUnseenRoutes ?? null, h: k.files?.hooks?.server ?? null })); })"], { cwd: SITE, env: { ...process.env, ATLAS_ONLY: '', ...env }, encoding: 'utf8' });
		assert.equal(r.status, 0, r.stderr);
		return JSON.parse(r.stdout) as { e: string[]; c: boolean; u: string | null; h: string | null };
	};
	assert.deepEqual(read({}), { e: ['*'], c: true, u: null, h: null });
	const o = read({ ATLAS_ONLY: 'gcn' });
	assert.deepEqual(o.e, ['/', '/m/gcn/', '/train/gcn/', '/dev/atlas/lab/gcn/']);
	assert.equal(o.c, false);
	assert.equal(o.u, 'ignore');
	assert.equal(o.h, 'scripts/atlas/lib/only-hooks');
});

test('check-build: a package-only build needs the home page and a page per package', () => {
	const dir = tmp();
	try {
		const page = (p: string, js: string) => {
			mkdirSync(join(dir, p), { recursive: true });
			writeFileSync(join(dir, p, 'index.html'), `<html><link rel="modulepreload" href="../_app/immutable/${js}"></html>`);
		};
		mkdirSync(join(dir, '_app', 'immutable'), { recursive: true });
		writeFileSync(join(dir, '_app', 'immutable', 'a.js'), 'console.log("a");'.repeat(50));
		page('', 'a.js');
		page('m/gcn', 'a.js');
		const run = (only: string) => spawnSync(process.execPath, ['scripts/check-build.ts'], { cwd: SITE, env: { ...process.env, ATLAS_BUILD_DIR: dir, ATLAS_ONLY: only, ATLAS_INCLUDE_DRAFTS: '1' }, encoding: 'utf8' });
		const ok = run('gcn');
		assert.equal(ok.status, 0, ok.stdout + ok.stderr);
		assert.match(ok.stdout, /package-only build \(ATLAS_ONLY=gcn\)/);
		assert.match(ok.stdout, /\/m\/gcn\//);
		const bad = run('gcn,tinyvit');
		assert.equal(bad.status, 1);
		assert.match(bad.stderr, /no page was built for package 'tinyvit'/);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test('types: the scoped tsconfig includes only the packages (and the ambient SvelteKit types); summary parsing', () => {
	const { file, config } = scopedTsconfig(SITE, ['gcn', 'pinn'], 'k2-test');
	assert.match(file.replace(/\\/g, '/'), /\.atlas\/k2-test\/tsconfig\.check\.json$/);
	assert.equal(config.extends, '../../tsconfig.json');
	const inc = config.include as string[];
	assert.ok(inc.includes('../../src/lib/explainers/gcn/**/*.ts'));
	assert.ok(inc.includes('../../src/lib/explainers/pinn/**/*.svelte'));
	assert.ok(!inc.some((p) => p.includes('/atlas/')), 'the kit is not diagnosed');
	assert.deepEqual(parseSvelteCheck('\x1b[31msvelte-check found 1 error and 0 warnings in 1 file'), { errors: 1, warnings: 0, files: 1 });
	assert.deepEqual(parseSvelteCheck('svelte-check found 0 errors and 0 warnings'), { errors: 0, warnings: 0, files: null });
	assert.deepEqual(parseSvelteCheck('1791409317467 COMPLETED 1193 FILES 0 ERRORS 2 WARNINGS 1 FILES_WITH_PROBLEMS'), { errors: 0, warnings: 2, files: 1193 });
	assert.equal(parseSvelteCheck('crashed'), null);
});

const step = (n: number, say: string, stage: string, title = `Step ${n}`): StepText => ({ step: n, steps: 2, title, say, look: '', badge: 'Live', stage });

test('contact: numbers as printed, the sheet layout and the lesson print per language', () => {
	assert.deepEqual(numbersIn('0.731 and −1.2e-3, 83.1% of 29,626 (step 7)'), ['0.731', '−1.2e-3', '83.1%', '29,626', '7']);
	assert.deepEqual(numbersIn('x1 and x2 are 2 numbers; W_k1, σ2 = 0.5; 第1步'), ['2', '0.5', '1'], 'not the digits of a name');
	const phone = sheetLayout({ width: 390 }, 12);
	assert.equal(phone.cols, 6);
	assert.equal(phone.tileW, 390);
	const desk = sheetLayout({ width: 1280 }, 2);
	assert.equal(desk.cols, 2);
	const many = sheetLayout({ width: 390 }, 120);
	assert.ok(many.maxTileH * Math.ceil(120 / many.cols) < 16_000, 'a tall lesson still fits one image');
	const html = sheetHtml('t <x>', [{ label: '1/2  A & B', src: 'data:image/png;base64,AA' }], desk);
	assert.match(html, /1\/2  A &amp; B/);
	assert.match(html, /t &lt;x&gt;/);
	const passes = resolvePasses('desktop-en-light,phone-zh-dark,narrow-en-dark');
	const text = lessonPrint({
		slug: 's',
		path: '/m/s/',
		date: '2026-10-07 12:00',
		passes: [
			{ pass: passes[0], steps: [step(1, 'Sum is 1.88.', 'x 0.199 4.42'), step(2, 'Done.', '')] },
			{ pass: passes[1], steps: [step(1, '和是 1.88。', 'x 0.199 4.42', '第 1 步'), step(2, '完成。', '')] },
			{ pass: passes[2], steps: [step(1, 'Sum is 1.89.', 'x 0.199 4.42'), step(2, 'Done.', '')] }
		]
	});
	assert.match(text, /Step 1\/2/);
	assert.match(text, /en {2}title {6}Step 1/);
	assert.match(text, /zh {2}title {6}第 1 步/);
	assert.match(text, /numbers {4}narration \[1\.88\]; stage \[0\.199 4\.42\]/);
	assert.match(text, /narrow-en-dark narration differs: Sum is 1\.89\./);
});

const viol = (id: string, impact: 'critical' | 'serious' | 'moderate' | 'minor', nodes = 1) => ({ id, impact, help: `${id} help`, helpUrl: `https://x/${id}`, nodes: Array.from({ length: nodes }, (_, i) => ({ target: [`.n${i}`], html: '<p>', summary: 'fix' })) });

test('axe: serious and critical fail with one issue per rule (steps per pass), moderate and minor warn', () => {
	const scans: StepScan[] = [
		{ pass: 'desktop-en-light', step: 1, steps: 5, title: 'a', scope: 'page', ms: 10, violations: [viol('color-contrast', 'serious', 2), viol('region', 'moderate')] },
		{ pass: 'desktop-en-light', step: 2, steps: 5, title: 'b', scope: 'lesson', ms: 10, violations: [viol('color-contrast', 'serious')] },
		{ pass: 'desktop-en-light', step: 3, steps: 5, title: 'c', scope: 'lesson', ms: 10, violations: [viol('color-contrast', 'serious')] },
		{ pass: 'desktop-en-light', step: 5, steps: 5, title: 'e', scope: 'lesson', ms: 10, violations: [viol('color-contrast', 'serious')] },
		{ pass: 'phone-zh-dark', step: 4, steps: 5, title: 'd', scope: 'lesson', ms: 10, violations: [viol('button-name', 'critical')] },
		{ pass: 'phone-zh-dark', step: 5, steps: 5, title: 'e', scope: 'lesson', ms: 10, violations: [] }
	];
	const is = new Issues();
	const r = axeIssues(scans, is);
	assert.equal(r.blocking, 2);
	assert.equal(r.warnings, 1);
	assert.equal(is.errors.length, 2);
	assert.ok(is.list.every((i) => i.code === AK_A11Y));
	assert.match(is.errors[0].message, /button-name \(critical\)/, 'critical first');
	assert.match(is.errors[1].message, /color-contrast \(serious\).*5 node\(s\) on desktop-en-light steps 1-3, 5/);
	assert.match(is.warnings[0].message, /region \(moderate\)/);
	const txt = axeText('s', scans);
	assert.match(txt, /step  5\/5 \(lesson\) e: clean/);
	assert.match(txt, /FAIL color-contrast \(serious\) x2/);
	assert.match(txt, /warn region \(moderate\)/);
});

test('gate cache: the key follows files, sizes and mtimes; a passing entry is read back, a failed one removed', () => {
	const dir = tmp();
	try {
		const a = join(dir, 'pkg');
		mkdirSync(join(a, 'runs'), { recursive: true });
		writeFileSync(join(a, 'x.ts'), 'one');
		const k1 = inputsKey([a, join(dir, 'missing')], 'py=1');
		writeFileSync(join(a, 'runs', 'ckpt.bin'), 'state');
		assert.equal(inputsKey([a, join(dir, 'missing')], 'py=1'), k1, 'training state (runs/) is not an input');
		assert.notEqual(inputsKey([a, join(dir, 'missing')], 'py=2'), k1);
		writeFileSync(join(a, 'x.ts'), 'two');
		utimesSync(join(a, 'x.ts'), new Date(), new Date(Date.now() + 5000));
		const k2 = inputsKey([a, join(dir, 'missing')], 'py=1');
		assert.notEqual(k2, k1);
		const site = join(dir, 'site');
		writeCached(site, 's', 'parity', { key: k2, at: 'now', summary: 'A 1e-7', status: 'pass', issues: [] });
		assert.equal(readCached(site, 's', 'parity', k2)?.summary, 'A 1e-7');
		assert.equal(readCached(site, 's', 'parity', k1), null);
		writeCached(site, 's', 'parity', null);
		assert.equal(readCached(site, 's', 'parity', k2), null);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test('quick cache inputs: the package code a test imports (transitively), not the rest', () => {
	const dir = tmp();
	try {
		writeFileSync(join(dir, 'a.test.ts'), "import { b } from './b.ts';\n// import { x } from './lesson.ts';\nexport const a = b;\n");
		writeFileSync(join(dir, 'b.ts'), "export { c as b } from './c.ts';\nconst lazy = () => import('./d.ts');\nimport '../outside.ts';\n");
		writeFileSync(join(dir, 'c.ts'), 'export const c = 1;\n');
		writeFileSync(join(dir, 'd.ts'), 'export const d = 1;\n');
		writeFileSync(join(dir, 'lesson.ts'), "import { c } from './c.ts';\n");
		assert.deepEqual([...importClosure(dir, [join(dir, 'a.test.ts')])].sort(), ['a.test.ts', 'b.ts', 'c.ts', 'd.ts']);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test('ownership scan: the same answer as merge --dry-run (planMerge), with parallel I/O', async () => {
	const base = tmp();
	const repo = join(base, 'repo');
	const main = join(repo, 'site');
	const put = (root: string, rel: string, text: string) => {
		const f = join(root, ...rel.split('/'));
		mkdirSync(join(f, '..'), { recursive: true });
		writeFileSync(f, text);
	};
	try {
		put(repo, 'CONTRACTS.md', '# temp\n');
		put(repo, 'pipeline/README.md', 'p\n');
		for (const [rel, text] of [
			['package.json', '{ "name": "s", "type": "module" }\n'],
			['svelte.config.js', 'export default {};\n'],
			['vite.config.ts', 'export default {};\n'],
			['src/lib/kit/a.ts', 'a\n'],
			['src/lib/kit/b.ts', 'b\n'],
			['src/lib/explainers/demo/explainer.ts', 'e\n'],
			['static/models/demo/spec.json', '{}\n'],
			['package-lock.json', '{}\n']
		])
			put(main, rel, text);
		const laneDir = join(main, '.work', 'demo');
		const lane = join(laneDir, 'site');
		const snapshot: Record<string, string> = {};
		for (const rel of siteFiles(main)) {
			put(lane, rel, readFileSync(join(main, ...rel.split('/')), 'utf8'));
			snapshot[rel] = sha256(join(main, ...rel.split('/')));
		}
		const owns = ['site/src/lib/explainers/demo/**', 'site/static/models/demo/**'];
		writeFileSync(join(laneDir, 'lane.json'), JSON.stringify({ format: 'atlas-lane/1', name: 'demo', created: 'now', owns, snapshot }));
		put(lane, 'src/lib/explainers/demo/explainer.ts', 'e2\n'); // owned change
		put(lane, 'src/lib/explainers/demo/lesson.ts', 'l\n'); // owned add
		rmSync(join(lane, 'static', 'models', 'demo', 'spec.json')); // owned remove
		put(lane, 'src/lib/kit/a.ts', 'a2\n'); // outside
		put(main, 'src/lib/kit/b.ts', 'b2\n'); // main moved on: not a lane change
		put(lane, 'vite.config.ts', 'lane setup\n'); // never counted
		const { planMerge } = await import('../commands/worktree.ts');
		const { plan } = planMerge('demo', { root: repo, main });
		const scan = await scanLane(main, lane, join(laneDir, 'lane.json'), []);
		assert.deepEqual(scan.copy, plan.copy);
		assert.deepEqual(scan.remove, plan.remove);
		assert.deepEqual(scan.outside, plan.outside);
		assert.deepEqual(scan.both, plan.both);
		assert.deepEqual(scan.outside, ['src/lib/kit/a.ts']);
		assert.deepEqual(scan.copy, ['src/lib/explainers/demo/explainer.ts', 'src/lib/explainers/demo/lesson.ts']);
		assert.deepEqual(scan.remove, ['static/models/demo/spec.json']);
	} finally {
		rmSync(base, { recursive: true, force: true });
	}
});

test('dev server identity: only a server that serves this site folder answers the probe', async () => {
	const site = tmp();
	const other = tmp();
	mkdirSync(join(site, '.svelte-kit'), { recursive: true });
	// a stand-in for vite: serves /.svelte-kit/<file> from its own root
	const serve = (root: string) =>
		createServer((req, res) => {
			try {
				res.end(readFileSync(join(root, decodeURIComponent((req.url ?? '/').split('?')[0]))));
			} catch {
				res.statusCode = 404;
				res.end('nope');
			}
		});
	const mine = serve(site);
	const theirs = serve(other);
	await new Promise<void>((r) => mine.listen(0, '127.0.0.1', r));
	await new Promise<void>((r) => theirs.listen(0, '127.0.0.1', r));
	try {
		const port = (s: ReturnType<typeof serve>) => (s.address() as { port: number }).port;
		assert.equal(await probeSite(port(mine), site), true);
		assert.equal(await probeSite(port(theirs), site), false, 'another site folder never matches');
		assert.equal(readRecord(site), null);
	} finally {
		mine.close();
		theirs.close();
		rmSync(site, { recursive: true, force: true });
		rmSync(other, { recursive: true, force: true });
	}
});
