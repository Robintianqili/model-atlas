// W1-C acceptance 3 (the four reviewer findings are caught with their AK codes) and 4 (compileLesson output passes a
// contract test with fakeRuntime), plus the narration language, built-in facts and the planner.
// Run from site/:  node --test src/lib/atlas/lesson/lesson.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import type { At, LessonDef, StepDef } from '../types/lesson.ts';
import type { OpDef } from '../types/op.ts';
import type { ExplainerManifest } from '../types/explainer.ts';
import type { AtlasSpec } from '../types/spec.ts';
import type { Tensor } from '../types/core.ts';
import { compileLesson } from './compile.ts';
import { errorsOf, fixtureCases, formatIssues, lintLesson, samePrint } from './lint.ts';
import { formatValue, literalNumbers, placeholderNames, render, sentenceCount } from './narrate.ts';
import { builtinFact, layerStat, parseLayerPath } from './facts.ts';
import { inferGroups, planLesson } from './plan.ts';
import { auto, autoAll } from './define.ts';
import { FINDINGS } from './fixtures/findings.ts';
import { lesson6 } from './fixtures/lesson6.ts';
import { SEQ_SPEC, SEQ_WEIGHTS, seqAdapter, seqFetch, seqManifest, seqPackage, unroll } from './fixtures/seq.ts';
import { STUB_OPS } from './fixtures/stub-ops.ts';
import { createRuntime } from '../runtime/runtime.ts';
import { fakeRuntime } from '../runtime/fake.ts';
import { fromParts } from '../runtime/seams.ts';
import { record } from '../process/record.ts';
import { ATLAS_SITE, readJson } from '../process/fixtures/paths.ts';

const ctxFor = (lesson: LessonDef | null) => {
	const pkg = seqPackage(lesson);
	return { pkg, ctx: { spec: SEQ_SPEC, ops: pkg.ops, manifest: pkg.manifest, adapters: pkg.adapters, processes: pkg.processes } };
};
const makeRt = (lesson: LessonDef | null) => (fx: { raw: unknown }) => createRuntime(seqPackage(lesson), { fetch: seqFetch, base: '', input: fx.raw });

// ------------------------------------------------------------------ narration language
test('narrate: placeholders, formats, fallback, unresolved and non-finite values', () => {
	const rt = fakeRuntime({ facts: { 'top.label': { en: 'cat', zh: '猫' }, 'fact:acc': 0.9681, 'layer.conv1.shape': '[8×26×26]' } });
	const at: At = { index: 0 };
	const t = {
		en: 'Top is {top.label} ({fact:acc|pct}); {n|int} items, {s|signed}, {x}, {tok|tok}, {shape}, {layer.conv1.shape}, {bad}, {nan}.',
		zh: '最高是 {top.label}（{fact:acc|pct}）；{n|int} 项，{s|signed}，{x}，{tok|tok}，{shape}，{layer.conv1.shape}，{bad}，{nan}。',
		vars: () => ({ n: 12345.4, s: 0.5, x: -0.0001234, tok: ' the', shape: { v: 3, fmt: 'int' }, nan: NaN })
	};
	const en = render(t, rt, at, 'en');
	assert.equal(en.text, 'Top is cat (96.8%); 12,345 items, +0.500, −1.23e−4, “␣the”, 3, [8×26×26], …, ….');
	assert.deepEqual(en.missing, ['bad', 'nan']);
	// without a format, exact integers print as integers; |n3 forces 3 significant digits
	assert.equal(render({ en: '{k} {k|n3} {big} {f}', vars: () => ({ k: 9, big: 6722, f: 2.5 }) }, rt, at, 'en').text, '9 9.00 6,722 2.50');
	// |n4 (and n2, n5, n6): more significant digits, for a small change that n3 would hide (afternoon test)
	const small = { en: 'The loss falls from {a} to {b}, from {a|n4} to {b|n4}.', vars: () => ({ a: 0.69913, b: 0.69874 }) };
	const r4 = render(small, rt, at, 'en');
	assert.equal(r4.text, 'The loss falls from 0.699 to 0.699, from 0.6991 to 0.6987.');
	// AK410: two different numbers that print the same are found (the |n4 pair is not)
	assert.deepEqual(samePrint(r4), { a: 'a', b: 'b', va: 0.69913, vb: 0.69874, text: '0.699' });
	assert.equal(samePrint(render({ en: '{a|n4} to {b|n4}; {k} and {k}; {m|int} and {n|int}', vars: () => ({ a: 0.69913, b: 0.69874, k: 0.5, m: 4.2, n: 4.4 }) }, rt, at, 'en')), null);
	// narrating the change itself is the other remedy: no warning when the difference is in the narration too
	assert.equal(samePrint(render({ en: 'From {a} it falls by {drop} to {b}.', vars: () => ({ a: 0.69913, b: 0.69874, drop: 0.69913 - 0.69874 }) }, rt, at, 'en')), null);
	const zh = render(t, rt, at, 'zh');
	assert.ok(zh.text.startsWith('最高是 猫（96.8%）'));
	assert.equal(zh.fallback, false);
	assert.equal(render({ en: 'only {x}' }, rt, at, 'zh').fallback, true);
	const thrown = render({ en: 'a {y}', vars: () => { throw new Error('boom'); } }, rt, at, 'en');
	assert.equal(thrown.error, 'boom');
	assert.deepEqual(thrown.missing, ['y']);
	assert.equal(formatValue(3, 'weird', 'en'), null, 'unknown format');
	assert.deepEqual(placeholderNames('{a} {b|int} {a} {layer.L0.attn.scores#h2.max}'), ['a', 'b', 'layer.L0.attn.scores#h2.max']);
	// AK402 ignores single digits, identifiers and placeholders
	assert.deepEqual(literalNumbers('A 3×3 kernel in conv1_1 and L0, block 2, {n} items'), []);
	assert.deepEqual(literalNumbers('It is right 97% of the time, 0.5 on average, 28×28 pixels'), ['97%', '0.5', '28', '28']);
	assert.equal(sentenceCount('One. Two {x}. Three!', 'en'), 3);
	assert.equal(sentenceCount('一。二！三', 'zh'), 3);
});

test('facts: layer paths parse right to left; stats; top/runner-up; sel', () => {
	assert.deepEqual(parseLayerPath('layer.L0.attn.scores#h2.max'), { id: 'L0.attn.scores#h2', stat: 'max' });
	assert.deepEqual(parseLayerPath('layer.conv1_1.strongestChannel'), { id: 'conv1_1', stat: 'strongestChannel' });
	assert.equal(parseLayerPath('top.p'), null);
	const t: Tensor = { shape: [2, 2, 2], data: Float32Array.from([0, -1, 2, 0, 0.5, 7, -3, 1]) };
	assert.equal(layerStat(t, 'max'), 7);
	assert.equal(layerStat(t, 'argmax'), 5);
	assert.equal(layerStat(t, 'negShare'), 0.25);
	assert.equal(layerStat(t, 'zeroShare'), 0.25);
	assert.equal(layerStat(t, 'shape'), '[2×2×2]');
	assert.equal(layerStat(t, 'strongestChannel'), 1);
	const model = fromParts(SEQ_SPEC, SEQ_WEIGHTS.buffer as ArrayBuffer, STUB_OPS);
	const pre = seqAdapter.preprocess('hello', null, {});
	const trace = model.forward(pre.tensors);
	const rt = fakeRuntime({ spec: SEQ_SPEC, model, trace, sel: { layer: 'B0.lin', unit: [1, 2] } });
	const p = trace.t.prob.data;
	const order = [0, 1, 2].sort((a, b) => p[b] - p[a]);
	assert.equal(rt.num('top.p'), p[order[0]]);
	assert.ok(Math.abs(rt.num('margin') - (p[order[0]] - p[order[1]])) < 1e-7);
	assert.equal((rt.fact('top.label') as { en: string }).en, ['calm', 'busy', 'odd'][order[0]]);
	assert.equal(rt.fact('sel.cell'), '(1, 2)');
	assert.equal(rt.num('sel.value'), trace.t['B0.lin'].data[1 * 4 + 2]);
	assert.ok(Math.abs(rt.num('unit.result') - rt.num('sel.value')) < 1e-6, 'explain() equals the trace');
	assert.equal(builtinFact({ spec: SEQ_SPEC, model, sources: {}, sel: {} }, 'top.p'), undefined);
	// a sequence output [T, V] is read at its last row (next-token prediction), not flattened
	const seqOut: Tensor = { shape: [2, 3], data: Float32Array.from([0.9, 0.05, 0.05, 0.1, 0.2, 0.7]) };
	const rt2 = fakeRuntime({ trace: { order: ['probs'], t: { probs: seqOut } } });
	assert.ok(Math.abs(rt2.num('top.p') - 0.7) < 1e-7);
	assert.equal(rt2.fact('top.label'), '#2');
	assert.ok(Math.abs(rt2.num('margin') - 0.5) < 1e-7);
});

// ------------------------------------------------------------------ planner
test('planLesson: input, first occurrences, the group summary, shape-only rule, output, teach hints, process motifs', () => {
	const plan = planLesson(SEQ_SPEC, STUB_OPS, seqManifest, { adapters: { 'stub-seq': seqAdapter }, processes: { unroll } });
	assert.deepEqual(plan.map((s) => s.id), ['input', 'op:emb', 'op:B0.lin', 'op:B0.act', 'group:B1', 'op:pool', 'output', 'proc:event.read@unroll', 'proc:run@unroll']);
	assert.deepEqual(inferGroups(SEQ_SPEC).map((g) => [g.id, g.sig]), [['B0', 'stub_linear|stub_tanh'], ['B1', 'stub_linear|stub_tanh'], ['B2', 'stub_linear|stub_tanh']]);
	const g = plan.find((s) => s.id === 'group:B1')!;
	assert.deepEqual(g.title, { en: '2 more blocks like this one', zh: '还有 2 个同样的块' });
	const scan = plan.find((s) => s.id === 'op:B0.lin')!;
	assert.equal(scan.source, 'scan:B0.lin');
	assert.equal(scan.waitForDone, true);
	// teach hints: skip the embedding, force a step for the head
	const hinted: AtlasSpec = { ...SEQ_SPEC, layers: SEQ_SPEC.layers.map((l) => (l.id === 'emb' ? { ...l, teach: { lesson: 'skip' as const } } : l.id === 'head' ? { ...l, teach: { lesson: 'step' as const, title: { en: 'Scores', zh: '分数' } } } : l)) };
	const p2 = planLesson(hinted, STUB_OPS, seqManifest).map((s) => s.id);
	assert.ok(!p2.includes('op:emb') && p2.includes('op:head'));
});

test('planLesson on the shipped TinyCNN and TinyGPT specs (stand-in op metadata; reported metric, not a gate)', (t) => {
	const op = (name: string, category: OpDef['category'], extra: Partial<OpDef> = {}): OpDef => ({
		op: name,
		version: 1,
		attrs: {},
		category,
		label: { en: name, zh: name },
		inferShape: (ins) => ins[0],
		forward: (xs) => xs[0] as Tensor,
		narration: { what: { en: `${name}.`, zh: `${name}。` } },
		zoom: 'none',
		parity: { bar: 1e-4 },
		...extra
	});
	const ops: OpDef[] = [
		op('input', 'io'),
		op('conv2d', 'conv', { unit: { axes: (a) => a, scan: 'raster' }, distinct: (p, prev) => (prev.inChannels as number) === 1 && (p.inChannels as number) > 1 }),
		op('relu', 'activation'),
		op('maxpool2d', 'pool'),
		op('flatten', 'shape'),
		op('linear', 'linear', { aliases: { dense: {}, linear_tied: {} } }),
		op('softmax', 'activation'),
		op('embedding', 'embedding'),
		op('position_embedding', 'embedding'),
		op('add', 'merge'),
		op('layernorm', 'norm'),
		op('attention_scores', 'attention'),
		op('causal_mask', 'shape'),
		op('attention_mix', 'attention'),
		op('gelu', 'activation', { aliases: { gelu_tanh: { approx: 'tanh' } } })
	];
	const cnn = readJson<AtlasSpec>(join(ATLAS_SITE, 'static', 'models', 'tinycnn', 'spec.json'));
	const man = (slug: string): ExplainerManifest => ({ slug, type: 'model', kinds: ['trace'], quality: 'draft', page: seqManifest.page, credits: [] });
	const pc = planLesson(cnn, ops, man('tinycnn')).map((s) => s.id);
	const hand: Record<string, string | null> = { pixels: 'input', slide: 'op:conv1_1', kernels: null, relu: 'op:relu1_1', combine: 'op:conv1_2', pool: 'op:pool1', grow: null, flatten: 'op:flatten', dense: 'op:dense', softmax: 'op:softmax' };
	const covered = Object.entries(hand).filter(([, id]) => id && pc.includes(id)).map(([k]) => k);
	t.diagnostic(`TinyCNN auto plan (${pc.length} steps): ${pc.join(', ')}`);
	t.diagnostic(`TinyCNN hand-made ids covered: ${covered.length}/10 (${covered.join(', ')}); author-only: kernels, grow`);
	assert.ok(covered.length >= 7, `covered ${covered.length}`);
	const gpt = readJson<AtlasSpec>(join(ATLAS_SITE, 'static', 'models', 'tinygpt', 'spec.json'));
	const pg = planLesson(gpt, ops, man('tinygpt')).map((s) => s.id);
	t.diagnostic(`TinyGPT auto plan (${pg.length} steps): ${pg.join(', ')}`);
	assert.ok(pg.includes('group:L1'));
});

// ------------------------------------------------------------------ acceptance 4
const CURSOR_KEYS = new Set(['phase', 'step', 'sub', 'layer', 'last', 'where']);
const isI18n = (v: unknown) => !!v && typeof v === 'object' && typeof (v as { en: unknown }).en === 'string' && ((v as { zh?: unknown }).zh === undefined || typeof (v as { zh?: unknown }).zh === 'string');

test('acceptance 4: compileLesson output passes the CompiledStep contract with fakeRuntime', async () => {
	const { ctx } = ctxFor(lesson6);
	const compiled = compileLesson(lesson6, ctx);
	assert.deepEqual(compiled.issues, [], formatIssues(compiled.issues));
	assert.equal(compiled.id, 'lesson');
	assert.equal(compiled.steps.length, 6);
	// a fake runtime: a real trace of the stub model, a recorded 'unroll' tape, nothing fetched
	const model = fromParts(SEQ_SPEC, SEQ_WEIGHTS.buffer as ArrayBuffer, STUB_OPS);
	const pre = seqAdapter.preprocess('a quiet day', null, {});
	const trace = model.forward(pre.tensors);
	const rt = fakeRuntime({ slug: 'fixture-seq', spec: SEQ_SPEC, model, pre, input: 'a quiet day', trace, tapes: { unroll: record(unroll, 'a quiet day', {}, { seed: 1 }) } });
	// the implicit scan source is the runtime's job; the fake one gets it explicitly
	const { scanTape } = await import('../process/adapters.ts');
	rt.sources['scan:B0.lin'] = { kind: 'scan', status: 'ready', provenance: 'illustrative', tape: scanTape(model, trace, 'B0.lin', [0]) };
	const ids = new Set<string>();
	for (const s of compiled.steps) {
		assert.match(s.id, /^[A-Za-z0-9_-]+$/, 'deep-link safe id');
		assert.ok(!ids.has(s.id), 'unique id');
		ids.add(s.id);
		assert.ok(s.origin === 'auto' || s.origin === 'author');
		if (s.origin === 'auto') assert.ok(typeof s.autoId === 'string' && s.autoId.length > 0);
		assert.ok(isI18n(s.title) && s.title.zh, `${s.id}: title en + zh`);
		assert.ok(typeof s.view === 'string' && s.view.length > 0);
		assert.ok(['live', 'precomputed', 'illustrative'].includes(s.provenance.mode));
		assert.ok(Array.isArray(s.terms) && s.terms.every((x) => typeof x === 'string'));
		assert.equal(typeof s.waitForDone, 'boolean');
		for (const k of ['linger', 'duration'] as const) assert.ok(s[k] === undefined || typeof s[k] === 'number');
		assert.ok(rt.sources[s.source], `${s.id}: source ${s.source}`);
		rt.sel = s.select?.(rt) ?? {};
		const c = s.at(rt);
		assert.ok(c === 'first' || c === 'last' || (typeof c === 'number' && Number.isInteger(c) && c >= 0) || (typeof c === 'object' && Object.keys(c).every((k) => CURSOR_KEYS.has(k))), `${s.id}: cursor ${JSON.stringify(c)}`);
		const index = await rt.ensure(s.source, c);
		const tape = rt.sources[s.source].tape!;
		assert.ok(index >= 0 && index < tape.length, `${s.id}: index ${index} in [0, ${tape.length})`);
		const at: At = { index, frame: tape.frame(index), state: tape.state(index), tape };
		if (s.source === 'fwd') {
			at.layer = (at.state as { layer: string }).layer;
			if (rt.sel.unit && rt.sel.layer === at.layer) at.unit = model.explain(trace, at.layer, rt.sel.unit);
		}
		if (s.source.startsWith('scan:')) at.unit = (at.state as { explain: At['unit'] }).explain;
		const n = s.narration(rt, at);
		assert.ok(isI18n(n) && n.en.trim() && n.zh?.trim(), `${s.id}: narration en + zh`);
		assert.ok(!n.en.includes('…') && !n.zh!.includes('…'), `${s.id}: no unresolved placeholder: ${n.en}`);
		const look = s.look?.(rt, at);
		assert.ok(look === undefined || isI18n(look));
		const props = s.props(rt, at);
		assert.ok(props && typeof props === 'object' && !Array.isArray(props));
		const focus = s.focus(rt, at);
		assert.ok(Array.isArray(focus) && focus.every((f) => typeof f.field === 'string'));
		if (s.predict) assert.ok(s.predict.options.length > s.predict.answer && isI18n(s.predict.question) && isI18n(s.predict.explain));
		if (s.task) assert.equal(typeof s.task.done(rt, at), 'boolean');
		// it maps onto a kit WalkStep (types/kit re-check in CONTRACTS.md)
		const walk = { id: s.id, title: s.title, narration: () => s.narration(rt, at), look: () => s.look?.(rt, at) ?? { en: '' }, waitForDone: s.waitForDone, linger: s.linger, duration: s.duration };
		assert.equal(typeof walk.narration().en, 'string');
	}
	// predict / task / limits are where the author put them; the override kept the auto narration
	const byId = Object.fromEntries(compiled.steps.map((s) => [s.id, s]));
	assert.equal(byId['op-B0-lin'].autoId, 'op:B0.lin');
	assert.equal(byId['op-B0-lin'].title.en, 'A weighted sum, cell by cell');
	assert.equal(byId['op-B0-lin'].waitForDone, true);
	assert.equal(byId.count.waitForDone, true, 'predict waits');
	assert.equal(byId.nudge.waitForDone, true, 'task waits');
	assert.equal(byId.output.kindOf, 'limits');
	assert.equal(byId.squash.waitForDone, false);
});

test('compileLesson: autoAll ranges, explicit auto wins, static rules', () => {
	const { ctx } = ctxFor(null);
	const all = compileLesson(null, ctx);
	assert.equal(all.steps.length, all.plan.length);
	const ranged = compileLesson({ title: { en: 't', zh: 't' }, summary: { en: 's', zh: 's' }, steps: [autoAll({ from: 'op:emb', until: 'output', skip: ['op:pool'] }), auto('op:B0.act', { title: { en: 'Squash', zh: '压缩' } })] }, ctx);
	assert.deepEqual(ranged.steps.map((s) => s.autoId), ['op:emb', 'op:B0.lin', 'group:B1', 'op:B0.act']);
	const bad: StepDef = { id: 'x', title: { en: 'No zh' }, source: 'nope', view: 'v', provenance: { mode: 'illustrative', note: { en: 'made up', zh: '编的' } }, narration: { en: 'Uses {a} and {b|weird}. Twelve is 12.', zh: '只用 {a}。' } };
	const r = compileLesson({ title: { en: 't' }, summary: { en: 's' }, steps: [bad, bad, auto('op:nothing')] }, { ...ctx, views: (k) => k !== 'v' });
	const codes = new Set(r.issues.map((i) => i.code));
	for (const c of ['AK401', 'AK402', 'AK403', 'AK404', 'AK405', 'AK406', 'AK407']) assert.ok(codes.has(c), `${c} in ${[...codes].join(' ')}`);
	const reviewed = compileLesson(null, { ...ctx, manifest: { ...ctx.manifest, quality: 'reviewed' } });
	assert.ok(reviewed.issues.some((i) => i.code === 'AK409' && i.level === 'error'));
});

// ------------------------------------------------------------------ acceptance 3
test('acceptance 3: the four past reviewer findings are caught with their AK codes (and the clean lesson is clean)', async (t) => {
	const { pkg, ctx } = ctxFor(lesson6);
	const rt0 = await createRuntime(pkg, { fetch: seqFetch, base: '' });
	const fixtures = fixtureCases(pkg, rt0.presets);
	const clean = await lintLesson(compileLesson(lesson6, ctx), makeRt(lesson6), fixtures);
	assert.deepEqual(clean, [], formatIssues(clean));
	for (const f of FINDINGS) {
		const issues = await lintLesson(compileLesson(f.lesson, ctx), makeRt(f.lesson), fixtures);
		const hit = issues.filter((i) => i.code === f.expect.code && i.step === f.expect.step);
		assert.ok(hit.length > 0, `${f.id}: expected ${f.expect.code}, got:\n${formatIssues(issues)}`);
		for (const h of hit) {
			assert.equal(h.level, f.expect.level);
			if (f.expect.fixture) assert.equal(h.fixture, f.expect.fixture);
			if (f.expect.lang) assert.ok(hit.some((x) => x.lang === f.expect.lang));
		}
		// nothing else is reported: the lint is specific
		const other = issues.filter((i) => !(i.code === f.expect.code && i.step === f.expect.step));
		assert.deepEqual(other, [], `${f.id}: unexpected\n${formatIssues(other)}`);
		t.diagnostic(`${f.id}: ${hit.map((h) => `${h.code} ${h.level} step=${h.step}${h.fixture ? ` fixture=${h.fixture}` : ''}${h.lang ? ` lang=${h.lang}` : ''}`).join('; ')}`);
	}
	assert.equal(errorsOf(clean).length, 0);
});
