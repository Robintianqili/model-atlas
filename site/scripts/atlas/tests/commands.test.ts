// W1-E2 commands: parity (legacy, A, B, C), lesson, check, the honesty rules, export, review and promote.
// The legacy parity run is read-only (--no-write; spec.json is hashed before and after). Everything that writes runs in
// a throwaway repo (ATLAS_SITE / ATLAS_ROOT point there), except .atlas/ reports in this site (git-ignored).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { dependencyProblems, mathRandomUses, notesProblems, stripComments } from '../commands/check.ts';
import { CODES } from '../errors.ts';
import { PYTHON, ROOT, SITE, cli, tempRepo } from './helpers.ts';

const sha = (f: string) => createHash('sha256').update(readFileSync(f)).digest('hex');

test('acceptance 3: parity tinycnn runs the legacy script with --no-write, reports 2.86e-6 and leaves spec.json unchanged', () => {
	const main = join(ROOT, 'site', 'static', 'models', 'tinycnn', 'spec.json');
	const before = sha(main);
	const r = cli(['parity', 'tinycnn', '--json']);
	assert.equal(r.status, 0, r.stderr);
	const rep = JSON.parse(r.stdout);
	assert.equal(rep.legacy.exit, 0);
	assert.ok(Math.abs(rep.legacy.measured - 2.86e-6) <= 1e-7, `measured ${rep.legacy.measured}`);
	assert.equal(rep.legacy.specSha.unchanged, true);
	assert.equal(sha(main), before);
	assert.ok(rep.legacy.kit.max <= 1e-4, 'the kit executor on the shipped assets');
});

test('honesty rules (AK503): dependency rule with the test, fixture and node:* exemptions; Math.random outside comments', () => {
	const site = 'C:/s';
	const f = (rel: string) => `${site}/${rel}`;
	assert.deepEqual(dependencyProblems(site, f('src/lib/explainers/x/lesson.ts'), "import type { A } from '../../atlas/types/index.ts';\nimport { h } from './helpers.ts';"), []);
	assert.equal(dependencyProblems(site, f('src/lib/explainers/x/lesson.ts'), "import { base } from '$app/paths';").length, 1);
	assert.equal(dependencyProblems(site, f('src/lib/explainers/x/lesson.ts'), "import V from './views/v/View.svelte';").length, 1);
	assert.equal(dependencyProblems(site, f('src/lib/explainers/x/lesson.ts'), "import type { T } from '../../kit/types.ts';").length, 1, 'not even import type from the kit');
	assert.equal(dependencyProblems(site, f('src/lib/explainers/x/process.ts'), "import { s } from '../../atlas/ui/strings.ts';").length, 1, 'atlas/ui is off limits');
	assert.equal(dependencyProblems(site, f('src/lib/explainers/x/process.ts'), "import { readFileSync } from 'node:fs';").length, 1);
	assert.equal(dependencyProblems(site, f('src/lib/explainers/x/process.ts'), "import { y } from '../other/process.ts';").length, 1, 'another package');
	assert.deepEqual(dependencyProblems(site, f('src/lib/explainers/x/claims.test.ts'), "import { readFileSync } from 'node:fs';\nimport { loadPackage } from '../../../../scripts/atlas/load.ts';"), []);
	assert.deepEqual(dependencyProblems(site, f('src/lib/atlas/process/fixtures/paths.ts'), "import { readFileSync } from 'node:fs';"), []);
	assert.deepEqual(dependencyProblems(site, f('tests/atlas/autograd/support.ts'), "import { x } from 'node:crypto';"), []);
	assert.deepEqual(dependencyProblems(site, f('src/lib/atlas/adapters/drawing28/adapter.ts'), "import { preprocessStrokes } from '../../../models/tinycnn.ts';"), []);
	assert.equal(dependencyProblems(site, f('src/lib/atlas/adapters/text-bpe/adapter.ts'), "import { x } from '../../../models/tinycnn.ts';").length, 1);
	assert.equal(mathRandomUses('// never Math.random here\nconst u = ctx.rng.next(); /* Math.random */'), 0);
	assert.equal(mathRandomUses('const u = Math.random();'), 1);
	assert.equal(stripComments("const url = 'http://x'; // c"), "const url = 'http://x'; ");
	assert.deepEqual(notesProblems('## What\nx\n## Decisions\ny\n## Final numbers\n1\n## How to re-run\nz\n## Kit issues\nnone\n'), []);
	assert.equal(notesProblems('## What\nTODO: x\n## Decisions\n\n').length, 5);
	for (const code of ['AK501', 'AK503', 'AK702', 'AK703']) assert.ok(code in CODES);
});

test('hello-xor: parity A (reference), B (twin, SGD and Adam) and C (Python reference); lesson lint; check --fast', { skip: !PYTHON && 'no configured Python' }, () => {
	const p = cli(['parity', 'hello-xor', '--json']);
	assert.equal(p.status, 0, p.stderr);
	const rep = JSON.parse(p.stdout);
	assert.ok(rep.forward.max <= 1e-4 && rep.forward.explain.max <= 1e-5, 'kind A');
	assert.equal(rep.forward.cases, 6);
	const runs = rep.train[0].runs;
	assert.deepEqual(runs.map((x: { optimizer: string }) => x.optimizer), ['sgd', 'adam']);
	assert.ok(runs[0].losses <= 1e-10 && runs[0].grads0 <= 1e-10 && runs[0].final <= 1e-10, 'SGD 1e-10 relative');
	assert.ok(runs[1].losses <= 1e-7 && runs[1].final <= 1e-7, 'Adam 1e-7 relative');
	assert.equal(rep.process[0].firstDivergence, null);
	assert.equal(rep.process[0].frames, rep.process[0].refFrames);
	const l = cli(['lesson', 'hello-xor', 'lint', '--json']);
	assert.equal(l.status, 0, l.stdout + l.stderr);
	const lint = JSON.parse(l.stdout);
	assert.equal(lint.errors, 0);
	assert.equal(lint.steps, 9);
	const d = cli(['lesson', 'hello-xor', 'draft', '--json']);
	assert.equal(d.status, 0, d.stderr);
	assert.ok(JSON.parse(d.stdout).steps.some((s: { id: string }) => s.id === 'proc:event.update@step1'));
	const c = cli(['check', 'hello-xor', '--fast', '--json', '--only', 'schema,op tests,lesson,determinism,honesty,budgets,paperwork']);
	const res = JSON.parse(c.stdout);
	for (const g of res.gates) assert.notEqual(g.status, 'fail', `${g.name}: ${JSON.stringify(g.issues)}`);
	assert.ok(res.lines.lines <= 1000, `page code ${res.lines.lines} lines`);
});

const TEMPLATES = ['process-blank', 'points2d-train', 'series-train', 'graph-node', 'drawing-classifier', 'char-seq'];

test('the six templates scaffold packages whose sources run, pass determinism, honesty and parity (with Python)', { timeout: 540_000 }, () => {
	const t = tempRepo({ kit: true });
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo, PYTHONPATH: join(ROOT, 'pipeline') };
	try {
		for (const tpl of TEMPLATES) {
			const slug = `t-${tpl}`;
			const n = cli(['new', 'model', slug, '--dev', '--template', tpl], env);
			assert.equal(n.status, 0, `${tpl}: ${n.stderr}${n.stdout}`);
			const traceKind = tpl === 'drawing-classifier' || tpl === 'char-seq';
			if (traceKind && PYTHON) {
				const ex = cli(['export', slug, '--no-draft'], env);
				assert.equal(ex.status, 0, `${tpl} export: ${ex.stdout}\n${ex.stderr}`);
				assert.ok(existsSync(join(t.site, 'static', 'models', slug, 'reference.json')));
				const spec = JSON.parse(readFileSync(join(t.site, 'static', 'models', slug, 'spec.json'), 'utf8'));
				assert.ok(spec.parity.max_abs_diff <= 1e-4 && spec.parity.pass === true, `${tpl}: spec.parity written by parity --write`);
			}
			const only = ['determinism', 'honesty', 'budgets', ...(PYTHON && (!traceKind || PYTHON) ? ['parity'] : [])];
			const c = cli(['check', slug, '--fast', '--json', '--only', only.join(',')], env);
			const res = JSON.parse(c.stdout || '{"gates":[]}');
			for (const g of res.gates.filter((x: { status: string }) => x.status !== 'skip'))
				assert.notEqual(g.status, 'fail', `${tpl} ${g.name}: ${JSON.stringify(g.issues).slice(0, 600)}`);
			const lint = cli(['lesson', slug, 'lint', '--json'], env);
			const lj = JSON.parse(lint.stdout);
			// a kind-A scaffold cannot run before its kit adapter exists in the tree (drawing28, text-char: W1-D1) or before export
			const adapter = tpl === 'drawing-classifier' ? 'drawing28' : 'text-char';
			const noRun = traceKind && (!PYTHON || !existsSync(join(t.site, 'src', 'lib', 'atlas', 'adapters', adapter, 'adapter.ts')));
			const unexpected = lj.issues.filter((i: { level: string; code: string }) => i.level === 'error' && i.code !== 'AK501' && !(noRun && i.code === 'AK404'));
			assert.deepEqual(unexpected, [], `${tpl}: only TODO markers (AK501) may fail the scaffold's lint`);
			const ex = readFileSync(join(t.site, 'src', 'lib', 'explainers', slug, 'explainer.ts'), 'utf8');
			assert.match(ex, /TODO/, 'the scaffold asks for its goals');
		}
	} finally {
		t.cleanup();
	}
});

test('review writes notes/review-<slug>.md (checklist kept on re-run); promote appends credits once and prints the catalog patch', () => {
	const t = tempRepo({ kit: true });
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		assert.equal(cli(['new', 'model', 'gridworld', '--template', 'process-blank'], env).status, 0);
		const exFile = join(t.site, 'src', 'lib', 'explainers', 'gridworld', 'explainer.ts');
		writeFileSync(
			exFile,
			readFileSync(exFile, 'utf8').replace('credits: []', "credits: [{ name: 'Gridworld idea', url: 'https://example.org/gw', license: 'MIT', usedFor: 'the maze editor idea', attribution: 'Example' }]")
		);
		mkdirSync(join(t.site, 'static'), { recursive: true });
		writeFileSync(join(t.site, 'static', 'licenses.json'), '[]\n');
		mkdirSync(join(t.site, 'scripts'), { recursive: true });
		cpSync(join(SITE, 'src', 'lib', 'i18n', 'strings.ts'), join(t.site, 'src', 'lib', 'i18n', 'strings.ts'));
		cpSync(join(SITE, 'scripts', 'check-catalog.ts'), join(t.site, 'scripts', 'check-catalog.ts'));
		const rv = cli(['review', 'gridworld'], env);
		assert.equal(rv.status, 0, rv.stderr + rv.stdout);
		const file = join(t.repo, 'notes', 'review-gridworld.md');
		let text = readFileSync(file, 'utf8');
		assert.match(text, /^## Checklist/m);
		assert.match(text, /- \[ \] 10\. Credits are correct/);
		assert.match(text, /## Narration on/);
		writeFileSync(file, text.replace('- [ ] 1.', '- [x] 1.'));
		assert.equal(cli(['review', 'gridworld'], env).status, 0);
		text = readFileSync(file, 'utf8');
		assert.match(text, /- \[x\] 1\./, 'ticks survive a re-run');
		const p1 = cli(['promote', 'gridworld'], env);
		assert.match(p1.stdout, /appended 1 credit/);
		assert.match(p1.stdout, /status: 'live',/);
		assert.match(p1.stdout, /lesson: \{ steps: \d+, summary: f\(/);
		const lic = JSON.parse(readFileSync(join(t.site, 'static', 'licenses.json'), 'utf8'));
		assert.equal(lic.length, 1);
		const p2 = cli(['promote', 'gridworld'], env);
		assert.match(p2.stdout, /appended 0 credit\(s\); 1 already listed/);
		assert.equal(JSON.parse(readFileSync(join(t.site, 'static', 'licenses.json'), 'utf8')).length, 1);
		const dev = cli(['new', 'model', 'x-dev', '--dev', '--kind', 'process'], env);
		assert.equal(dev.status, 0);
		const pd = cli(['promote', 'x-dev'], env);
		assert.equal(pd.status, 1);
		assert.match(pd.stderr, /^AK701 /m);
	} finally {
		t.cleanup();
	}
});

test('worker.ts follows the engine contract (design §4.10): generated when atlas/engine/host.ts exists, and by `new worker` later', () => {
	const t = tempRepo({ kit: true });
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		// simulate a kit from before W1-B2 (no engine), then add a stub of its entry point
		rmSync(join(t.site, 'src', 'lib', 'atlas', 'engine'), { recursive: true, force: true });
		assert.equal(cli(['new', 'model', 'early', '--dev', '--kind', 'train,process'], env).status, 0);
		const pkg = join(t.site, 'src', 'lib', 'explainers', 'early');
		assert.ok(!existsSync(join(pkg, 'worker.ts')), 'no worker.ts while the engine is missing');
		assert.equal(cli(['new', 'worker', 'early'], env).status, 1, 'new worker refuses without the engine');
		// the engine's entry point (W1-B2): serve({ trainers, processes })
		mkdirSync(join(t.site, 'src', 'lib', 'atlas', 'engine'), { recursive: true });
		writeFileSync(join(t.site, 'src', 'lib', 'atlas', 'engine', 'host.ts'), 'export function serve(_m: { trainers?: Record<string, unknown>; processes?: Record<string, unknown> }): null {\n\treturn null;\n}\n');
		const w = cli(['new', 'worker', 'early'], env);
		assert.equal(w.status, 0, w.stderr);
		const text = readFileSync(join(pkg, 'worker.ts'), 'utf8');
		assert.match(text, /import \{ serve \} from '\.\.\/\.\.\/atlas\/engine\/host\.ts';/);
		assert.match(text, /import \* as trainers from '\.\/trainer\.ts';/);
		assert.match(text, /import \* as processes from '\.\/process\.ts';/);
		assert.match(text, /serve\(\{ trainers, processes \}\);/);
		assert.equal(cli(['new', 'worker', 'early'], env).status, 1, 'never overwritten');
		assert.equal(cli(['new', 'model', 'later', '--dev', '--kind', 'process'], env).status, 0);
		const later = readFileSync(join(t.site, 'src', 'lib', 'explainers', 'later', 'worker.ts'), 'utf8');
		assert.match(later, /serve\(\{ processes \}\);/);
		assert.doesNotMatch(later, /trainer/);
	} finally {
		t.cleanup();
	}
});
