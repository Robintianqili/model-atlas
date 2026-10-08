// K2 concept packages: `atlas new concept` (scaffold, --like a concept, --like a model, refusals) in a throwaway repo
// (ATLAS_SITE / ATLAS_ROOT point there), so the real tree is never touched.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { cpSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { LessonDef, PackageModules } from '../../../src/lib/atlas/types/index.ts';
import { loadPackage } from '../load.ts';
import { CONCEPT_TEMPLATE, asConceptManifest, templateNames } from '../commands/new.ts';
import { SITE, cli, tempRepo } from './helpers.ts';

const imp = async <T>(f: string) => (await import(pathToFileURL(f).href)) as T;

test('new concept <slug>: the base scaffold + the concept overlay; type concept; Node loads it; the panels compile; notes name /train/<slug>/', async () => {
	const t = tempRepo({ kit: true });
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		const r = cli(['new', 'concept', 'backprop', '--port', '5299'], env);
		assert.equal(r.status, 0, r.stderr + r.stdout);
		assert.match(r.stdout, /\/train\/backprop\//);
		const pkg = join(t.site, 'src', 'lib', 'explainers', 'backprop');
		for (const f of ['explainer.ts', 'lesson.ts', 'concept.ts', 'concept.test.ts', 'process.ts', 'worker.ts']) assert.ok(existsSync(join(pkg, f)), f);
		for (const f of ['pipeline/backprop/reference.py', 'notes/backprop.md']) assert.ok(existsSync(join(t.repo, f)), f);
		const notes = readFileSync(join(t.repo, 'notes', 'backprop.md'), 'utf8');
		assert.match(notes, /concept package/);
		assert.match(notes, /`\/train\/backprop\/`/);
		assert.match(notes, /port 5299/);
		const ex = readFileSync(join(pkg, 'explainer.ts'), 'utf8');
		assert.match(ex, /type ConceptManifest = Omit<ExplainerManifest, 'type'> & \{ type: 'concept' \}/);
		const loaded = await loadPackage('backprop', t.site);
		const m = loaded.pkg.manifest as unknown as { type: string; kinds: string[]; sources: Record<string, { kind: string }> };
		assert.equal(m.type, 'concept');
		assert.deepEqual(m.kinds, ['process']);
		assert.equal(m.sources.run.kind, 'process');
		assert.ok(loaded.pkg.processes.backprop, 'the scaffold process is registered');
		const l = (await imp<{ default: LessonDef }>(join(pkg, 'lesson.ts'))).default;
		assert.match(l.title.en, /Backpropagation and autograd/);
		// concept.ts: the panels compile like lesson steps (their TODO narration is the AK501 the author must clear)
		const { conceptOf, panelLesson } = await imp<typeof import('../../../src/lib/atlas/ui/concept/panels.ts')>(join(t.site, 'src', 'lib', 'atlas', 'ui', 'concept', 'panels.ts'));
		const { compileLesson } = await imp<typeof import('../../../src/lib/atlas/lesson/compile.ts')>(join(t.site, 'src', 'lib', 'atlas', 'lesson', 'compile.ts'));
		const def = conceptOf(await imp<Record<string, unknown>>(join(pkg, 'concept.ts')));
		const pl = panelLesson(def, loaded.pkg.manifest);
		assert.deepEqual(pl.keys, ['toy']);
		const c = compileLesson(pl, { spec: null, ops: loaded.pkg.ops, manifest: loaded.pkg.manifest, processes: loaded.pkg.processes, plan: [] });
		assert.equal(c.steps[0].source, 'run');
		assert.ok(c.issues.some((i) => i.code === 'AK501'), 'TODO markers are reported');
		assert.ok(!c.issues.some((i) => i.code === 'AK404'), 'the toy names a declared source');
		// nothing is overwritten
		const again = cli(['new', 'concept', 'backprop'], env);
		assert.notEqual(again.status, 0);
		assert.match(again.stderr, /already exists/);
	} finally {
		t.cleanup();
	}
});

test('new concept: refusals (not a concept AK701, a model slug, a model --template concept) and --dev; concept is not a model template', () => {
	assert.ok(!templateNames().includes(CONCEPT_TEMPLATE));
	const t = tempRepo();
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		const notConcept = cli(['new', 'concept', 'tinycnn-concept'], env);
		assert.notEqual(notConcept.status, 0);
		assert.match(notConcept.stderr, /AK701/);
		assert.match(notConcept.stderr, /concepts\.ts/);
		const model = cli(['new', 'concept', 'resnet-18'], env);
		assert.notEqual(model.status, 0, 'resnet-18 is a model, not a concept');
		const shared = cli(['new', 'concept', 'logit-lens'], env);
		assert.notEqual(shared.status, 0);
		assert.match(shared.stderr, /both a model and a concept/);
		const tpl = cli(['new', 'model', 'resnet-18', '--template', 'concept'], env);
		assert.notEqual(tpl.status, 0);
		assert.match(tpl.stderr, /new concept/);
		const dev = cli(['new', 'concept', 'my-idea', '--dev', '--no-worker'], env);
		assert.equal(dev.status, 0, dev.stderr);
		const ex = readFileSync(join(t.site, 'src', 'lib', 'explainers', 'my-idea', 'explainer.ts'), 'utf8');
		assert.match(ex, /devOnly: true/);
		assert.match(ex, /type: 'concept'/);
	} finally {
		t.cleanup();
	}
});

test('new concept --like: a concept package is copied and renamed; a model package becomes a concept (type, alias, concept.ts added)', async () => {
	const t = tempRepo({ kit: true });
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		for (const p of ['loss-gradient-descent', 'hello-xor']) cpSync(join(SITE, 'src', 'lib', 'explainers', p), join(t.site, 'src', 'lib', 'explainers', p), { recursive: true });
		const r = cli(['new', 'concept', 'optimizers', '--like', 'loss-gradient-descent'], env);
		assert.equal(r.status, 0, r.stderr + r.stdout);
		const pkg = join(t.site, 'src', 'lib', 'explainers', 'optimizers');
		const ex = readFileSync(join(pkg, 'explainer.ts'), 'utf8');
		assert.match(ex, /slug: 'optimizers'/);
		assert.match(ex, /type: 'concept'/);
		assert.ok(!/loss-gradient-descent/.test(readFileSync(join(pkg, 'concept.test.ts'), 'utf8')), 'the concept test names the new slug');
		assert.match(readFileSync(join(t.repo, 'notes', 'optimizers.md'), 'utf8'), /new concept optimizers --like loss-gradient-descent/);
		const loaded = await loadPackage('optimizers', t.site);
		assert.equal((loaded.pkg.manifest as unknown as { type: string }).type, 'concept');

		const m = cli(['new', 'concept', 'xor-concept', '--like', 'hello-xor', '--dev'], env);
		assert.equal(m.status, 0, m.stderr + m.stdout);
		const pkg2 = join(t.site, 'src', 'lib', 'explainers', 'xor-concept');
		const ex2 = readFileSync(join(pkg2, 'explainer.ts'), 'utf8');
		assert.match(ex2, /type: 'concept'/);
		assert.match(ex2, /const manifest: ConceptManifest =/);
		assert.ok(existsSync(join(pkg2, 'concept.ts')) && existsSync(join(pkg2, 'concept.test.ts')));
		assert.match(m.stdout, /added concept\.ts/);
		const p2: PackageModules = (await loadPackage('xor-concept', t.site)).pkg;
		assert.equal((p2.manifest as unknown as { type: string }).type, 'concept');
	} finally {
		t.cleanup();
	}
});

test('asConceptManifest: rewrites a model manifest once and leaves a concept manifest alone', () => {
	const model = "import type { ExplainerManifest } from '../../atlas/types/index.ts';\n\nconst manifest: ExplainerManifest = {\n\tslug: 'x',\n\ttype: 'model',\n};\n";
	const once = asConceptManifest(model);
	assert.match(once, /type: 'concept'/);
	assert.match(once, /type ConceptManifest = Omit<ExplainerManifest, 'type'>/);
	assert.match(once, /const manifest: ConceptManifest =/);
	assert.equal(asConceptManifest(once), once);
});
