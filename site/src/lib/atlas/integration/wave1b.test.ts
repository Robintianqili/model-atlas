// Integration 1b (design §8.2): checks that need two wave-1b packages at once, plus the additive types revision 1b.
//   1. Every view key the TinyCNN / TinyGPT auto lessons, the kit adapters' layer-0 steps and hello-xor's lesson name is a
//      registered view (W1-C planner + W1-D1 adapters + W1-D2 / W1-B2 view folders): no AK405 after the merge.
//   2. Controls queued between two frames ('set lr' then 'input'; 'set' twice) reach the generator in order on the local
//      recorder (W1-C) exactly as on the engine host (W1-B2): equal frame and state hashes.
//   3. Adapter contracts across lanes: text-char reads the char-seq template's `alphabet` (+ `unk`) key; text-bpe fills
//      Preprocessed.tokens / vocab when it has the tokenizer (the token views of W1-D2 read them).
//   4. Types revision 1b: the executor's OpCtx / UnitQuery have has(slot) and optW() uses it; defineProcess() infers a
//      ProcessDef whose frames differ in shape.
// Run from site/: node --test src/lib/atlas/integration/wave1b.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CORE_OPS } from '../ops/index.ts';
import { optW } from '../ops/common.ts';
import { fromParts } from '../graph/executor.ts';
import { normalizeSpec } from '../graph/legacy.ts';
import { compileLesson } from '../lesson/compile.ts';
import { record, recorder } from '../process/record.ts';
import { formulaOf } from '../ui/views/zoom.formula/formula.ts';
import { scrubMarks } from '../ui/scrub.ts';
import { fakeRuntime } from '../runtime/fake.ts';
import { defineProcess, type ProcessRun } from '../process/define.ts';
import { makeLocalRunner } from '../runtime/local.ts';
import { makeEngineRunner } from '../engine/runner.ts';
import { registerModules } from '../engine/host.ts';
import { channelWorker } from '../engine/fixtures/channel.ts';
import { clsTrainer, DATA } from '../engine/fixtures/trainers.ts';
import drawing28 from '../adapters/drawing28/adapter.ts';
import { textBpe } from '../adapters/text-bpe/adapter.ts';
import { textChar } from '../adapters/text-char/adapter.ts';
import points2d from '../adapters/points2d/adapter.ts';
import series from '../adapters/series/adapter.ts';
import type { AtlasSpec, ExplainerManifest, InputAdapter, LiveTape, OpDef, PackageModules, ProcessDef, Tape, Tensor } from '../types/index.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const STATIC = join(SITE, 'static');
const VIEWS = join(SITE, 'src', 'lib', 'atlas', 'ui', 'views');
const isView = (key: string) => existsSync(join(VIEWS, key, 'meta.ts'));
const loadSpec = (slug: string): AtlasSpec => normalizeSpec(JSON.parse(readFileSync(join(STATIC, 'models', slug, 'spec.json'), 'utf8')));
const MANIFEST = (slug: string, adapter: string): ExplainerManifest => ({
	slug,
	type: 'model',
	kinds: ['trace'],
	quality: 'draft',
	devOnly: true,
	assets: { dir: `models/${slug}`, auto: ['spec.json', 'weights.bin'] },
	input: { adapter },
	output: { view: 'kit.probbars' },
	page: { goals: [], prequestion: { id: 'q', question: { en: 'q', zh: 'q' }, options: [{ en: 'a', zh: 'a' }], answer: 0, explain: { en: 'e', zh: 'e' } } },
	credits: []
});

// ------------------------------------------------------------------ 1. view keys
test('every view the auto lessons, the kit adapters and hello-xor name is registered (no AK405)', async () => {
	const adapters: Record<string, InputAdapter<any>> = { drawing28, 'text-bpe': textBpe, 'text-char': textChar, points2d, series };
	for (const [id, a] of Object.entries(adapters)) assert.ok(isView(a.layer0.view), `adapter ${id}: layer0 view '${a.layer0.view}'`);
	for (const [slug, adapter] of [['tinycnn', 'drawing28'], ['tinygpt', 'text-bpe']]) {
		const m = MANIFEST(slug, adapter);
		const c = compileLesson(null, { spec: loadSpec(slug), ops: CORE_OPS, manifest: m, adapters, views: isView });
		const views = [...new Set(c.steps.map((s) => s.view))];
		console.log(`ℹ ${slug} auto lesson views: ${views.join(', ')}`);
		assert.deepEqual(c.issues.filter((i) => i.code === 'AK405'), [], `${slug}: unregistered views`);
	}
	const hx = join(SITE, 'src', 'lib', 'explainers', 'hello-xor', 'lesson.ts');
	if (existsSync(hx)) {
		const src = readFileSync(hx, 'utf8');
		const keys = [...src.matchAll(/view:\s*(?:\{\s*key:\s*)?'([^']+)'/g)].map((m) => m[1]);
		assert.ok(keys.length > 0, 'hello-xor names views');
		for (const k of keys) assert.ok(isView(k) || k.includes('/'), `hello-xor view '${k}'`);
		console.log(`ℹ hello-xor views: ${[...new Set(keys)].join(', ')}`);
	}
});

// ------------------------------------------------------------------ 2. queued controls: recorder = engine host
test('controls queued between two frames reach the generator in order: local recorder = engine host', async () => {
	const mods = { trainers: { clsTrainer }, processes: {} };
	const pkg: PackageModules = {
		slug: 'wave1b',
		manifest: { ...MANIFEST('wave1b', 'none'), kinds: ['train'] },
		lesson: null,
		ops: [],
		processes: registerModules(mods) as Record<string, ProcessDef<any, any, any>>,
		adapters: {},
		worker: (() => channelWorker(mods)) as unknown as () => Worker
	};
	const half = { ...DATA, xy: DATA.xy.slice(0, 64), label: DATA.label.slice(0, 32), n: 32 };
	const o = { slug: pkg.slug, source: 's', process: 'train:cls', input: DATA, opt: { steps: 40, snapshotEvery: 10 }, seed: 3, detail: 'coarse' as const };
	const drive = async (t: LiveTape) => {
		t.control({ type: 'step', n: 6 });
		await t.ensure(5);
		// back to back, before the next frame: a set, then an input (the old recorder kept only the input)
		t.control({ type: 'set', opt: { lr: 0.02 } });
		t.control({ type: 'input', input: half });
		t.control({ type: 'step', n: 8 });
		await t.ensure(13);
		// two sets back to back are merged
		t.control({ type: 'set', opt: { lr: 0.005 } });
		t.control({ type: 'set', opt: { snapshotEvery: 5 } });
		t.control({ type: 'step', n: 10 });
		await t.ensure(23);
	};
	const eng = makeEngineRunner();
	const loc = makeLocalRunner({ autorun: false });
	const a = eng.open(o, pkg);
	const b = loc.open(o, pkg);
	try {
		await drive(a);
		await drive(b);
		const sa = (a as unknown as { store: { frameHash(): string; stateHash(): string; length: number; frame(i: number): { metrics?: Record<string, number> } } }).store;
		const sb = (b as unknown as { store: typeof sa }).store;
		assert.equal(a.where, 'worker');
		assert.equal(sa.length, sb.length);
		assert.equal(sa.frameHash(), sb.frameHash(), 'frame hash');
		assert.equal(sa.stateHash(), sb.stateHash(), 'state hash');
		const lrs = Array.from({ length: sb.length }, (_, i) => sb.frame(i).metrics?.lr).filter((v) => v !== undefined);
		assert.ok(lrs.includes(0.02), 'the set lr before the input took effect');
		assert.equal(lrs[lrs.length - 1], 0.005, 'merged sets');
	} finally {
		a.close();
		b.close();
		eng.dispose();
	}
});

// ------------------------------------------------------------------ 3. adapter contracts
test("text-char reads the char-seq template's alphabet and unk keys", () => {
	const spec = { alphabet: ['?', ' ', 'a', 'b'], unk: '?', dims: { T: { min: 1, max: 4, label: { en: 'T', zh: 'T' } } } };
	const p = textChar.preprocess('ab zab', spec, {});
	assert.deepEqual(Array.from(p.tensors.input.data), [1, 0, 2, 3], 'the last 4 of a b ␣ z a b; z became unk');
	assert.equal(p.notes?.length, 2);
});

test('text-bpe fills Preprocessed.tokens and vocab when it has the tokenizer', () => {
	const tokenizer = JSON.parse(readFileSync(join(STATIC, 'models', 'tinygpt', 'tokenizer.json'), 'utf8'));
	const spec = loadSpec('tinygpt');
	const p = textBpe.preprocess('Once upon a time', spec, { tokenizer });
	const n = p.tensors.input.shape[0];
	assert.ok(n > 1);
	assert.equal(p.tokens?.length, n);
	assert.equal(p.tokens!.join(''), 'Once upon a time');
	assert.equal(typeof p.vocab, 'function');
	assert.equal((p.vocab as (id: number) => string)(97), 'a');
	const bare = textBpe.preprocess({ ids: [97, 98] }, spec, {});
	assert.equal(bare.tokens, undefined, 'no tokenizer: no token text (views show #id)');
});

// ------------------------------------------------------------------ 4. types revision 1b
test('OpCtx.has / UnitQuery.has: the executor provides them and optW() uses them', () => {
	const seen: boolean[] = [];
	type P = Record<string, never>;
	const probe: OpDef<P> = {
		op: 'probe',
		version: 1,
		attrs: {},
		category: 'linear',
		label: { en: 'probe', zh: 'probe' },
		weights: [{ slot: 'weight' }, { slot: 'bias', optional: true }],
		inferShape: (ins) => ins[0],
		forward(xs, _p, ctx) {
			seen.push(ctx.has!('weight'), ctx.has!('bias'));
			const w = ctx.w('weight');
			const b = optW(ctx, 'bias');
			const x = xs[0] as Tensor;
			return { shape: x.shape, data: Float32Array.from(x.data, (v, i) => v * w.data[i] + (b ? b.data[i] : 0)) };
		},
		explain(q) {
			seen.push(q.has!('weight'), q.has!('bias'));
			const i = q.unit[0];
			const x = (q.ins[0] as Tensor).data[i];
			const w = q.w('weight').data[i];
			return { kind: 'elementwise', layer: q.layer.id, unit: q.unit, reads: [], values: {}, result: x * w, traced: q.out.data[i] };
		},
		narration: { what: { en: 'probe', zh: 'probe' } },
		zoom: 'elementwise',
		parity: { bar: 1e-6 }
	};
	const spec = normalizeSpec({
		slug: 'probe', name: { en: 'probe', zh: 'probe' }, version: '1', params: 2, precision: 'fp32',
		atlas: { spec: '1.1', kinds: ['trace'], exporter: 'test' },
		inputs: [{ name: 'input', adapter: 'none', kind: 'vector', shape: [2] }],
		tensors: [{ name: 'p.weight', shape: [2], offset: 0, length: 2 }],
		layers: [{ id: 'p', op: 'probe', label: { en: 'p', zh: 'p' }, in: ['input'], outShape: [2] }],
		parity: { against: 'none', max_abs_diff: null, checked: null }
	});
	const m = fromParts(spec, { 'p.weight': { shape: [2], data: Float32Array.from([2, 3]) } }, [probe as unknown as OpDef]);
	const tr = m.forward({ input: { shape: [2], data: Float32Array.from([1, 1]) } });
	assert.deepEqual(Array.from(tr.t.p.data), [2, 3]);
	const e = m.explain(tr, 'p', [1]);
	assert.equal(e.result, e.traced);
	assert.deepEqual(seen, [true, false, true, false]);
	// a context without has() still works through MissingWeightError
	assert.equal(optW({ w: () => { const err = new Error('x'); err.name = 'MissingWeightError'; throw err; } }, 'bias'), null);
});

test('defineProcess infers a ProcessDef whose frames differ in shape, and it records', () => {
	type WalkState = { x: number; path: number[] };
	type WalkOpt = { steps: number };
	const walk = defineProcess({
		id: 'walk',
		kind: 'algorithm',
		title: { en: 'A random walk', zh: '随机游走' },
		phases: { move: { label: { en: 'move', zh: '移动' }, kind: 'act' }, end: { label: { en: 'end', zh: '结束' }, kind: 'output' } },
		schema: { x: { kind: 'scalar', label: { en: 'position', zh: '位置' } }, path: { kind: 'json', label: { en: 'path', zh: '路径' } } },
		defaults: { steps: 20 },
		recordDraws: true,
		*run(_input: null, ctx): ProcessRun<WalkState, WalkOpt> {
			let x = 0;
			const path: number[] = [];
			for (let k = 0; k < ctx.opt.steps; k++) {
				x += ctx.rng.next() < 0.5 ? -1 : 1;
				path.push(x);
				yield { phase: 'move', step: k, state: { x } };
			}
			yield { phase: 'end', state: { path } };
		}
	});
	const t = record(walk, null, { steps: 12 }, { seed: 5, recordDraws: walk.recordDraws });
	assert.equal(t.length, 13);
	assert.equal(t.frame(12).phase, 'end');
	assert.equal(t.frame(3).draws?.length, 1);
});

// ------------------------------------------------------------------ 5. phase formulas on live tapes
test('phase formulas reach zoom.formula on live tapes (recorder and engine), not only on a TapeStore', async () => {
	type St = { w: number };
	const sgd = defineProcess({
		id: 'sgd1',
		kind: 'training',
		title: { en: 'one step', zh: '一步' },
		phases: { update: { label: { en: 'Update', zh: '更新' }, kind: 'update', formula: { tex: 'w_{new} = {{w}} - {{lr}} g', values: (f) => ({ lr: 0.5, w: Number((f.state as St).w) }) } } },
		schema: { w: { kind: 'scalar', label: { en: 'w', zh: 'w' } } },
		defaults: { steps: 3 },
		*run(_i: null, ctx): ProcessRun<St, { steps: number }> {
			let w = 1;
			for (let k = 0; k < ctx.opt.steps; k++) yield { phase: 'update', step: k, state: { w: (w -= 0.5 * w) }, metrics: { g: w } };
		}
	});
	const rec = recorder(sgd, null, {}, { seed: 1, autorun: false });
	const i = await rec.ensure(1);
	assert.ok(rec.phases?.update, 'recorder exposes phases');
	const rt = fakeRuntime({});
	const stage = { rt, source: 's', at: { index: i, frame: rec.frame(i), tape: rec }, focus: [], sel: {}, provenance: 'live' as const, autoplay: false, onend: () => {}, props: {} };
	const f = formulaOf(stage);
	assert.equal(f?.from, 'phase');
	assert.equal(f?.values.lr, 0.5);
	rec.close();
	const mods = { trainers: {}, processes: { sgd } };
	const pkg: PackageModules = { slug: 'wave1b', manifest: { ...MANIFEST('wave1b', 'none'), kinds: ['process'] }, lesson: null, ops: [], processes: registerModules(mods) as Record<string, ProcessDef<any, any, any>>, adapters: {}, worker: (() => channelWorker(mods)) as unknown as () => Worker };
	const eng = makeEngineRunner();
	const t = eng.open({ slug: 'wave1b', source: 's', process: 'sgd1', input: null, opt: {}, seed: 1, detail: 'coarse' }, pkg);
	try {
		await t.ensure('last');
		assert.equal(t.where, 'worker');
		assert.ok(t.phases?.update && 'formula' in t.phases.update, 'engine tape exposes phases with formulas');
	} finally {
		t.close();
		eng.dispose();
	}
});

// ------------------------------------------------------------------ 6. scrubber marks on a one-row scan
test('scan jump marks: rows for a 2-D scan, features for a scan along one row (TinyGPT [T, D] layers; R0)', () => {
	const fake = (units: number[][]) => ({ frame: () => ({ phase: 'scan', state: {} }), state: (i: number) => ({ unit: units[i] }) }) as unknown as Tape;
	const grid = Array.from({ length: 36 }, (_, i) => [0, Math.floor(i / 6), i % 6]);
	const rows = scrubMarks(fake(grid), grid.map((_, i) => i), { unit: 'position' });
	assert.ok(new Set(rows.map((m) => m.label.en)).size > 1 && rows[0].label.en.startsWith('row'));
	const line = Array.from({ length: 76 }, (_, d) => [13, d]);
	const pos = scrubMarks(fake(line), line.map((_, i) => i), { unit: 'position' });
	assert.equal(new Set(pos.map((m) => m.label.en)).size, pos.length);
	assert.equal(pos[0].label.en, 'output 1', 'without axes: the place in the scan');
	// with the layer's axes [t, d], the marks name the feature the scan is on, counting from 0 (not "position")
	const feat = scrubMarks(fake(line), line.map((_, i) => i), { unit: 'position', axes: ['t', 'd'] });
	assert.equal(feat[0].label.en, 'feature 0');
	assert.equal(feat[feat.length - 1].label.en, 'feature 75');
	assert.equal(feat[0].label.zh, '特征 0');
});
