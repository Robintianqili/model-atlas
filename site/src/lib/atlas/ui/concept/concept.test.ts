// Concept packages (K2): the pure helpers (panels.ts), the concept lint (lint.ts) and the route's rules, in Node.
// Run from site/:  node --test src/lib/atlas/ui/concept/concept.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import type { ExplainerManifest, PackageModules, StepDef } from '../../types/index.ts';
import { createRuntime } from '../../runtime/runtime.ts';
import { compileLesson } from '../../lesson/compile.ts';
import { fsFetch } from '../../core/io.ts';
import { CORE_OPS } from '../../ops/index.ts';
import { drawing28 } from '../../adapters/drawing28/adapter.ts';
import { conceptIncluded, conceptOf, furtherHref, hookText, isConceptManifest, panelLesson, panelPosition, pitfallsOf, SHAPE_ONLY } from './panels.ts';
import { lintConcept } from './lint.ts';
import { STATIC_CONCEPT_ROUTES } from './routes.ts';

const SITE = fileURLToPath(new URL('../../../../../', import.meta.url));
const STATIC = join(SITE, 'static');
const page: ExplainerManifest['page'] = {
	goals: [{ en: 'g', zh: '目标' }],
	prequestion: { id: 'demo-concept-pre', question: { en: 'q', zh: '问' }, options: [{ en: 'a', zh: '甲' }], answer: 0, explain: { en: 'e', zh: '解' } }
};

/** A concept package built in memory: TinyCNN's exported weights (a live model's assets) traced on a drawing. */
function demo(over: Partial<ExplainerManifest> = {}): PackageModules {
	const manifest = {
		slug: 'demo-concept',
		type: 'concept',
		kinds: ['trace'],
		quality: 'draft',
		input: { adapter: 'drawing28' },
		sources: { cnn: { kind: 'trace', assets: 'models/tinycnn' } },
		page,
		credits: [],
		...over
	} as unknown as ExplainerManifest;
	return { slug: 'demo-concept', manifest, lesson: { title: { en: 't', zh: '课' }, summary: { en: 's', zh: '概' }, steps: [{ id: 'look', title: { en: 'Look', zh: '看' }, source: 'cnn', at: 'last', view: 'layer.tensor', provenance: 'live', narration: { en: 'The output has {n} numbers.', zh: '输出有 {n} 个数。', vars: (_rt, at) => ({ n: (at.state as { out?: { data: ArrayLike<number> } } | undefined)?.out?.data.length ?? 10 }) } }] }, ops: CORE_OPS, processes: {}, adapters: { drawing28 } };
}

const toy: StepDef = {
	id: 'whatever',
	title: { en: 'The last layer', zh: '最后一层' },
	source: 'cnn',
	at: 'last',
	view: 'layer.tensor',
	provenance: 'live',
	narration: { en: 'TinyCNN traced your drawing through {n} layers.', zh: 'TinyCNN 把你的画依次算过 {n} 层。', vars: (rt) => ({ n: rt.spec?.layers.length ?? (rt.sources.cnn?.tape?.length ?? NaN) }) }
};

test('conceptOf: default export or named exports; hook as text or stage; non-steps are ignored', () => {
	const named = conceptOf({ hook: { en: 'h', zh: '钩' }, toy, real: { not: 'a step' }, pitfalls: [{ claim: { en: 'c' }, truth: { en: 't' } }] });
	assert.deepEqual(Object.keys(named).sort(), ['hook', 'pitfalls', 'toy']);
	assert.deepEqual(hookText(named), { en: 'h', zh: '钩' });
	const viaDefault = conceptOf({ default: { hook: toy, compare: toy } });
	assert.equal(hookText(viaDefault), null);
	assert.deepEqual(panelLesson(viaDefault, { slug: 'x' }).keys, ['hook', 'compare']);
	assert.deepEqual(conceptOf(null), {});
});

test('panelLesson: page order, ids forced to the panel keys, its own deep-link prefix; pitfalls: the manifest misconception first', () => {
	const pl = panelLesson({ compare: toy, toy, real: toy }, { slug: 'x' });
	assert.equal(pl.id, 'concept');
	assert.deepEqual(pl.keys, ['toy', 'real', 'compare']);
	assert.deepEqual(pl.steps.map((s) => (s as StepDef).id), ['toy', 'real', 'compare']);
	const m = { page: { ...page, misconception: { claim: { en: 'm' }, truth: { en: 't' } } } };
	assert.deepEqual(pitfallsOf({ pitfalls: [{ claim: { en: 'p' }, truth: { en: 'q' } }] }, m).map((p) => p.claim.en), ['m', 'p']);
	assert.ok(SHAPE_ONLY.has('AK407') && SHAPE_ONLY.has('AK409'));
	assert.equal(furtherHref('/m/tinycnn/', '/model-atlas'), '/model-atlas/m/tinycnn/');
	assert.equal(furtherHref('https://d2l.ai/', '/model-atlas'), 'https://d2l.ai/');
});

test('conceptIncluded: a concept manifest, a catalog status, drafts only when included, never devOnly', () => {
	const m = { type: 'concept' };
	assert.equal(conceptIncluded(m, 'live', false), true);
	assert.equal(conceptIncluded(m, 'planned', false), false);
	assert.equal(conceptIncluded(m, 'planned', true), true);
	assert.equal(conceptIncluded({ ...m, devOnly: true }, 'live', true), false);
	assert.equal(conceptIncluded({ type: 'model' }, 'live', true), false, 'a model package is never a concept page');
	assert.equal(conceptIncluded(m, undefined, true), false, 'not in the concept catalog');
	assert.equal(isConceptManifest({ type: 'concept' }), true);
});

test('a concept package reuses a live model\'s assets (trace on models/tinycnn); panels resolve without touching the lesson selection', async () => {
	const pkg = demo();
	const rt = await createRuntime(pkg, { fetch: fsFetch, base: STATIC });
	try {
		rt.sel = { layer: 'conv1_1', unit: [0, 3, 3] };
		const sel = rt.sel;
		const c = compileLesson(panelLesson({ toy }, pkg.manifest), { spec: rt.spec, ops: pkg.ops, manifest: pkg.manifest, adapters: pkg.adapters, plan: [] });
		const pos = await panelPosition(rt, c.steps[0]);
		assert.equal(pos.error, undefined);
		assert.ok(pos.index > 0 && pos.found, `the toy's frame: ${pos.index}`);
		assert.equal(rt.sel, sel, 'panelPosition never replaces the lesson selection');
		const n = c.steps[0].narration(rt, pos.at);
		assert.match(n.en ?? '', /through \d+ layers/);
	} finally {
		rt.close();
	}
});

test('lintConcept: clean on a good page; catches the type, a view, zh, a pitfall step, a further link and placeholders', async () => {
	const mk = (pkg: PackageModules) => (raw: unknown) => createRuntime(pkg, { fetch: fsFetch, base: STATIC, ...(raw !== undefined ? { input: raw } : {}) });
	const good = demo();
	const clean = await lintConcept({ pkg: good, concept: { hook: { en: 'h', zh: '钩' }, toy, pitfalls: [{ claim: { en: 'c', zh: '误' }, truth: { en: 't', zh: '真' }, step: 'look' }] }, makeRuntime: mk(good), fixtures: [], views: (k) => k === 'layer.tensor' });
	assert.deepEqual(clean.filter((i) => i.level === 'error'), []);
	const bad = demo({ type: 'model' } as Partial<ExplainerManifest>);
	const issues = await lintConcept({
		pkg: bad,
		concept: {
			hook: { en: 'no chinese' },
			toy: { ...toy, view: 'no-such-view' },
			real: { ...toy, narration: { en: 'A {missing} value.', zh: '一个 {missing} 值。' } },
			pitfalls: [{ claim: { en: 'c', zh: '误' }, truth: { en: 't', zh: '真' }, step: 'nope' }],
			further: [{ label: { en: 'x', zh: '外' }, href: 'ftp://example.com' }]
		},
		makeRuntime: mk(bad),
		fixtures: [],
		views: (k) => k === 'layer.tensor'
	});
	const codes = (code: string) => issues.filter((i) => i.code === code && i.level === 'error');
	assert.ok(codes('AK301').some((i) => /type 'model'/.test(i.message)), 'the manifest type');
	assert.ok(codes('AK301').some((i) => /ftp:/.test(i.message)), 'a further link');
	assert.ok(codes('AK406').some((i) => i.step === 'hook'), 'the hook has no zh');
	assert.ok(codes('AK404').some((i) => /'nope'/.test(i.message)), 'a pitfall links a missing lesson step');
	assert.ok(codes('AK405').some((i) => i.step === 'toy'), 'an unknown view');
	assert.ok(codes('AK401').some((i) => i.step === 'real'), 'an unresolved placeholder');
	assert.ok(!issues.some((i) => SHAPE_ONLY.has(i.code)), 'no lesson-shape rules on panels');
});

test('route rules: the static concept routes exist; catalog slugs of models and concepts are disjoint; every live concept has a page', async () => {
	const { CONCEPTS } = (await import(pathToFileURL(join(SITE, 'src', 'lib', 'catalog', 'concepts.ts')).href)) as { CONCEPTS: { slug: string; status: string }[] };
	const { MODELS } = (await import(pathToFileURL(join(SITE, 'src', 'lib', 'catalog', 'models.ts')).href)) as { MODELS: { slug: string }[] };
	for (const s of STATIC_CONCEPT_ROUTES) assert.ok(existsSync(join(SITE, 'src', 'routes', 'train', s, '+page.svelte')), `static route ${s}`);
	// One explainers/<slug>/ folder per slug: a concept package cannot share its slug with a model. The catalog has one such
	// slug today (logit-lens: a precomputed model page and a TinyGPT concept page); it must be renamed on one side before
	// either gets a package (notes/k2-concepts.md). Any other overlap fails here.
	const KNOWN_SHARED = ['logit-lens'];
	const models = new Set(MODELS.map((m) => m.slug));
	assert.deepEqual(CONCEPTS.filter((c) => models.has(c.slug) && !KNOWN_SHARED.includes(c.slug)).map((c) => c.slug), [], 'a slug is both a model and a concept');
	for (const s of KNOWN_SHARED) assert.ok(!existsSync(join(SITE, 'src', 'lib', 'explainers', s, 'explainer.ts')), `${s} has a package although its slug is both a model and a concept`);
	for (const c of CONCEPTS.filter((x) => x.status === 'live')) {
		if (STATIC_CONCEPT_ROUTES.includes(c.slug)) continue;
		const ex = join(SITE, 'src', 'lib', 'explainers', c.slug, 'explainer.ts');
		assert.ok(existsSync(ex) && /type:\s*'concept'/.test(readFileSync(ex, 'utf8')), `live concept ${c.slug} has no concept package`);
	}
	// the route prerenders every concept but the static ones, and keeps them in sync with routes.ts
	const pageTs = readFileSync(join(SITE, 'src', 'routes', 'train', '[slug]', '+page.server.ts'), 'utf8');
	assert.match(pageTs, /STATIC_CONCEPT_ROUTES\.includes/);
});
