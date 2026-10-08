// Acceptance 5 (W1-E1): `atlas new model demo-x --dev --kind process` scaffolds files that Node imports.
// Everything runs in a throwaway repo (ATLAS_SITE / ATLAS_ROOT point there), so the real tree is never touched.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import type { ExplainerManifest, LessonDef, ProcessDef, ViewMeta } from '../../../src/lib/atlas/types/index.ts';
import { loadPackage } from '../load.ts';
import { PYTHON, ROOT, cli, tempRepo } from './helpers.ts';

const imp = async <T>(f: string) => (await import(pathToFileURL(f).href)) as T;

test('new model demo-x --dev --kind process: files, Node import, run, Python reference twin', async () => {
	const t = tempRepo();
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		const r = cli(['new', 'model', 'demo-x', '--dev', '--kind', 'process'], env);
		assert.equal(r.status, 0, r.stderr + r.stdout);
		const pkg = join(t.site, 'src', 'lib', 'explainers', 'demo-x');
		for (const f of ['explainer.ts', 'lesson.ts', 'process.ts']) assert.ok(existsSync(join(pkg, f)), f);
		for (const f of ['pipeline/demo-x/README.md', 'pipeline/demo-x/reference.py', 'notes/demo-x.md']) assert.ok(existsSync(join(t.repo, f)), f);
		assert.ok(!existsSync(join(pkg, 'worker.ts')), 'no worker.ts while atlas/engine/host.ts is missing');
		assert.match(r.stdout, /worker\.ts not generated/);

		// Node imports every generated file (type stripping) ...
		const m = (await imp<{ default: ExplainerManifest }>(join(pkg, 'explainer.ts'))).default;
		assert.equal(m.slug, 'demo-x');
		assert.equal(m.devOnly, true);
		assert.deepEqual(m.kinds, ['process']);
		assert.deepEqual(m.sources, { run: { kind: 'process', process: 'demoX', run: 'main' } });
		const l = (await imp<{ default: LessonDef }>(join(pkg, 'lesson.ts'))).default;
		assert.ok(l.steps.length >= 2);
		const p = (await imp<{ demoX: ProcessDef }>(join(pkg, 'process.ts'))).demoX;
		assert.equal(p.id, 'demoX');
		// ... and the package loader turns them into PackageModules
		const loaded = await loadPackage('demo-x', t.site);
		assert.deepEqual(Object.keys(loaded.pkg.processes), ['demoX']);
		assert.equal(loaded.pkg.lesson?.title.en.startsWith('TODO'), true);
		// the lesson's author step resolves its placeholders from a real frame in en and zh
		const step = l.steps[1] as { id: string } & { narration: { en: string; zh: string; vars: (rt: unknown, at: unknown) => Record<string, number> } };
		assert.equal(step.id, 'estimate-wobbles');
		const run = cli(['run', 'demo-x', 'run', '--json'], env);
		assert.equal(run.status, 0, run.stderr);
		const j = JSON.parse(run.stdout);
		assert.equal(j.frames, 40);
		assert.equal(j.process, 'demoX');
		// the author step 'estimate-wobbles' reads the last update frame and the source's opt
		const lastUpdate = { metrics: { heads: 13, n: 20, estimate: 0.65 } };
		const vars = step.narration.vars(null, { index: 39, frame: lastUpdate, tape: { opt: { n: 20, p: 0.7 } } });
		for (const name of ['n', 'estimate', 'p']) {
			assert.ok(Number.isFinite(vars[name]), name);
			assert.ok(step.narration.en.includes(`{${name}`) && step.narration.zh.includes(`{${name}`), `${name} in en and zh`);
		}

		// --save writes the tape and the draws; the scaffolded Python reference writes the same tape
		const saved = cli(['run', 'demo-x', 'run', '--save'], env);
		assert.equal(saved.status, 0, saved.stderr);
		assert.ok(existsSync(join(t.site, '.atlas', 'demo-x', 'run.tape.json')));
		assert.ok(existsSync(join(t.site, '.atlas', 'demo-x', 'demoX.draws.json')));
		if (PYTHON) {
			const py = spawnSync(
				PYTHON,
				['-c', `import importlib.util, atlas_kit as ak\nspec = importlib.util.spec_from_file_location("ref", r"${join(t.repo, 'pipeline', 'demo-x', 'reference.py')}")\nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\nm.run(record_draws=True).write(ak.site_path(".atlas", "demo-x", "demoX.ref"))`],
				{ env: { ...process.env, ...env, PYTHONPATH: join(ROOT, 'pipeline'), PYTHONUTF8: '1' }, encoding: 'utf8' }
			);
			assert.equal(py.status, 0, py.stderr);
			const a = JSON.parse(readFileSync(join(t.site, '.atlas', 'demo-x', 'run.tape.json'), 'utf8'));
			const b = JSON.parse(readFileSync(join(t.site, '.atlas', 'demo-x', 'demoX.ref.tape.json'), 'utf8'));
			delete a.bin;
			delete b.bin;
			assert.ok(isDeepStrictEqual(a, b), 'TS tape and Python reference tape differ');
			assert.ok(readFileSync(join(t.site, '.atlas', 'demo-x', 'run.tape.bin')).equals(readFileSync(join(t.site, '.atlas', 'demo-x', 'demoX.ref.tape.bin'))));
		}

		// nothing is overwritten
		const again = cli(['new', 'model', 'demo-x', '--dev', '--kind', 'process'], env);
		assert.equal(again.status, 1);
		assert.match(again.stderr, /^AK301 {2}'demo-x' already exists/m);
	} finally {
		t.cleanup();
	}
});

test('catalog rule: a slug outside the catalog needs --dev (AK701); a catalog slug is seeded from its entry', () => {
	const t = tempRepo();
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		const r = cli(['new', 'model', 'not-a-model', '--kind', 'process'], env);
		assert.equal(r.status, 1);
		assert.match(r.stderr, /^AK701 /m);
		const g = cli(['new', 'model', 'gridworld', '--kind', 'process', '--input', 'gridworld'], env);
		assert.equal(g.status, 0, g.stderr);
		const notes = readFileSync(join(t.repo, 'notes', 'gridworld.md'), 'utf8');
		assert.match(notes, /Pull slot-machine arms yourself/); // the catalog hook
		assert.match(notes, /--port 5193/); // the port table
		const ex = readFileSync(join(t.site, 'src', 'lib', 'explainers', 'gridworld', 'explainer.ts'), 'utf8');
		assert.doesNotMatch(ex, /devOnly/);
		assert.match(ex, /input: \{ adapter: 'gridworld' \}/);
	} finally {
		t.cleanup();
	}
});

test('kind trace: the scaffolded export.py exports a valid model folder', { skip: !PYTHON && 'no configured Python' }, () => {
	const t = tempRepo();
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		const r = cli(['new', 'model', 'demo-t', '--dev', '--kind', 'trace', '--input', 'points2d'], env);
		assert.equal(r.status, 0, r.stderr);
		const ex = spawnSync(PYTHON!, [join(t.repo, 'pipeline', 'demo-t', 'export.py')], { env: { ...process.env, ...env, PYTHONPATH: join(ROOT, 'pipeline'), PYTHONUTF8: '1' }, encoding: 'utf8' });
		assert.equal(ex.status, 0, ex.stderr + ex.stdout);
		const out = join(t.site, 'static', 'models', 'demo-t');
		const v = spawnSync(PYTHON!, ['-m', 'atlas_kit.validate', out], { env: { ...process.env, PYTHONPATH: join(ROOT, 'pipeline'), PYTHONUTF8: '1' }, encoding: 'utf8' });
		assert.equal(v.status, 0, v.stdout + v.stderr);
		const spec = JSON.parse(readFileSync(join(out, 'spec.json'), 'utf8'));
		assert.deepEqual(spec.layers.map((l: { id: string }) => l.id), ['input', 'fc1', 'act1', 'fc2']);
		assert.equal(spec.inputs[0].adapter, 'points2d');
		assert.deepEqual(spec.atlas.kinds, ['trace']);
	} finally {
		t.cleanup();
	}
});

test('kind train + process: trainer.ts, twin.py and the worker note; new op and new view (local)', async () => {
	const t = tempRepo();
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		const r = cli(['new', 'model', 'demo-y', '--dev', '--kind', 'train,process'], env);
		assert.equal(r.status, 0, r.stderr);
		const pkg = join(t.site, 'src', 'lib', 'explainers', 'demo-y');
		assert.ok(existsSync(join(pkg, 'trainer.ts')) && existsSync(join(pkg, 'process.ts')));
		assert.ok(existsSync(join(t.repo, 'pipeline', 'demo-y', 'twin.py')));
		assert.match(r.stdout, /W1-B1/);

		const op = cli(['new', 'op', 'scale', '--local', 'demo-y'], env);
		assert.equal(op.status, 0, op.stderr);
		const test = spawnSync(process.execPath, ['--test', join(pkg, 'ops', 'scale.test.ts')], { encoding: 'utf8' });
		assert.equal(test.status, 0, test.stdout + test.stderr);
		assert.ok(existsSync(join(t.repo, 'pipeline', 'demo-y', 'ops_scale.py')));
		if (PYTHON) {
			const py = spawnSync(PYTHON, ['-c', 'import sys, torch; sys.path.insert(0, sys.argv[1]); import ops_scale as m; print(float(m.reference([torch.tensor(3.0)], {"alpha": 2.0}, {})))', join(t.repo, 'pipeline', 'demo-y')], { encoding: 'utf8' });
			assert.equal(py.stdout.trim(), '6.0', py.stderr);
		}

		const view = cli(['new', 'view', 'bars', '--local', 'demo-y'], env);
		assert.equal(view.status, 0, view.stderr);
		const meta = (await imp<{ default: ViewMeta }>(join(pkg, 'views', 'bars', 'meta.ts'))).default;
		assert.equal(meta.key, 'demo-y/bars');
		const demo = await meta.demo();
		assert.equal(demo.provenance, 'illustrative');
		const d = meta.describe(demo as Parameters<ViewMeta['describe']>[0]);
		assert.ok(d.en.trim() && d.zh?.trim());
		assert.ok(existsSync(join(pkg, 'views', 'bars', 'View.svelte')));
	} finally {
		t.cleanup();
	}
});

// Integration 1a: the same scaffold on the real kit (W1-C record/saveTape instead of the CLI fallback). Covers the parts
// of `atlas check demo-x --fast` that exist in wave 1a: Node imports, the process determinism gate, the reference
// tape (TS tape = Python tape, header and bytes), and the lesson compile + lint over the adapter-free fixture.
test('integration 1a: demo-x on the real kit: determinism, run --save equals the Python reference, lesson compiles and lints', async () => {
	const t = tempRepo({ kit: true });
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		assert.equal(cli(['new', 'model', 'demo-x', '--dev', '--kind', 'process'], env).status, 0);
		const atlas = join(t.site, 'src', 'lib', 'atlas');
		const loaded = await loadPackage('demo-x', t.site);
		const p = loaded.pkg.processes.demoX;
		const { checkDeterminism } = await imp<typeof import('../../../src/lib/atlas/process/determinism.ts')>(join(atlas, 'process', 'determinism.ts'));
		const det = checkDeterminism(p, null, {}, { seed: 1, recordDraws: true });
		assert.ok(det.ok, det.problems.join(' | '));
		const run = cli(['run', 'demo-x', 'run', '--save'], env);
		assert.equal(run.status, 0, run.stderr);
		assert.match(run.stdout, /using {3}atlas\/process\/record\.ts/);
		const dir = join(t.site, '.atlas', 'demo-x');
		const a = JSON.parse(readFileSync(join(dir, 'run.tape.json'), 'utf8'));
		assert.equal(a.bin.file, 'run.tape.bin', 'the header names the .bin written next to it');
		if (PYTHON) {
			const py = spawnSync(
				PYTHON,
				['-c', `import importlib.util, atlas_kit as ak\nspec = importlib.util.spec_from_file_location("ref", r"${join(t.repo, 'pipeline', 'demo-x', 'reference.py')}")\nm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\nm.run(record_draws=True).write(ak.site_path(".atlas", "demo-x", "demoX.ref"))`],
				{ env: { ...process.env, ...env, PYTHONPATH: join(ROOT, 'pipeline'), PYTHONUTF8: '1' }, encoding: 'utf8' }
			);
			assert.equal(py.status, 0, py.stderr);
			const b = JSON.parse(readFileSync(join(dir, 'demoX.ref.tape.json'), 'utf8'));
			delete a.bin;
			delete b.bin;
			assert.ok(isDeepStrictEqual(a, b), 'TS tape (W1-C saveTape) and Python reference tape differ');
			assert.ok(readFileSync(join(dir, 'run.tape.bin')).equals(readFileSync(join(dir, 'demoX.ref.tape.bin'))));
		}
		const { compileLesson } = await imp<typeof import('../../../src/lib/atlas/lesson/compile.ts')>(join(atlas, 'lesson', 'compile.ts'));
		const { lintLesson } = await imp<typeof import('../../../src/lib/atlas/lesson/lint.ts')>(join(atlas, 'lesson', 'lint.ts'));
		const { createRuntime } = await imp<typeof import('../../../src/lib/atlas/runtime/runtime.ts')>(join(atlas, 'runtime', 'runtime.ts'));
		const { fsFetch } = await imp<typeof import('../../../src/lib/atlas/core/io.ts')>(join(atlas, 'core', 'io.ts'));
		const ctx = { spec: null, ops: loaded.pkg.ops, manifest: loaded.pkg.manifest, adapters: loaded.pkg.adapters, processes: loaded.pkg.processes };
		const c = compileLesson(loaded.pkg.lesson, ctx);
		assert.deepEqual(c.issues.filter((i) => i.level === 'error'), []);
		assert.ok(c.steps.some((s) => s.id === 'estimate-wobbles'));
		assert.ok(c.steps.some((s) => s.autoId === 'proc:event.update@run'), 'the phase narration gives the auto lesson a real update step');
		const issues = await lintLesson(c, (fx) => createRuntime(loaded.pkg, { fetch: fsFetch, base: join(t.site, 'static'), input: fx.raw }), [{ id: 'none', label: { en: 'no input', zh: '无输入' }, raw: null }]);
		assert.deepEqual(issues.filter((i) => i.level === 'error').map((i) => `${i.code} ${i.step ?? ''} ${i.lang ?? ''} ${i.message}`), []);
	} finally {
		t.cleanup();
	}
});
