// DevKit K1: the kit fixes after wave 2b (LOG.md "K1"). One test per fix that has kit code behind it.
// Run from site/:  node --test src/lib/atlas/integration/k1.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { LiveTape, ProcessDef, ProcessRun } from '../types/process.ts';
import type { OpenSource, PackageModules, SourceRunner } from '../types/runtime.ts';
import type { AtlasSpec } from '../types/spec.ts';
import type { Trace } from '../types/core.ts';
import { createRuntime } from '../runtime/runtime.ts';
import { localRunner, runKey, sharedRunner } from '../runtime/local.ts';
import { stepAt } from '../runtime/sources.ts';
import { makeEngineRunner } from '../engine/runner.ts';
import { lesson6 } from '../lesson/fixtures/lesson6.ts';
import { memoryFetch, SEQ_FILES, seqManifest, seqPackage } from '../lesson/fixtures/seq.ts';
import { compileLesson } from '../lesson/compile.ts';
import { lintLesson, samePrint, saturated } from '../lesson/lint.ts';
import { render } from '../lesson/narrate.ts';
import { metricName, metricRows, metricText, trainingFrame } from '../ui/training.ts';
import { isAnswerLayer, outputItemLabel } from '../ui/fallback.ts';
import { alignTex, breakSum } from '../ui/views/kit.parts/tex.ts';
import { textChar } from '../adapters/text-char/adapter.ts';
import seriesAdapter, { sameSeries, seriesPresetRaw } from '../adapters/series/adapter.ts';
import { fmt3 } from '../core/format.ts';
import { sourceBadge } from '../ui/kitspec.ts';

type O = { steps: number; lr: number };
type S = { n: number; lr: number; k: number; w?: number };
/** A toy trainer: one frame per step, the data size and lr in every frame, a "weight" w that grows with k. */
const toy: ProcessDef<string, O, S> = {
	id: 'train:toy',
	kind: 'training',
	title: { en: 'toy', zh: '玩具' },
	phases: { init: { label: { en: 'Init', zh: '初始' }, kind: 'input' }, snapshot: { label: { en: 'Snapshot', zh: '快照' }, kind: 'update' } },
	schema: {
		n: { kind: 'int', label: { en: 'n', zh: 'n' } },
		lr: { kind: 'scalar', label: { en: 'lr', zh: 'lr' } },
		k: { kind: 'int', label: { en: 'k', zh: 'k' } },
		w: { kind: 'scalar', label: { en: 'w', zh: 'w' } }
	},
	defaults: { steps: 4, lr: 0.1 },
	*run(input, ctx): ProcessRun<S, O> {
		const data = String(input ?? '');
		yield { phase: 'init', step: 0, state: { n: data.length, lr: ctx.opt.lr, k: 0, w: 0 } };
		for (let k = 1; k <= ctx.opt.steps; k++) yield { phase: 'snapshot', step: k, state: { n: data.length, lr: ctx.opt.lr, k, w: k * data.length }, metrics: { loss: 1 / k, n: data.length } };
	}
};
type PO = { scale: number };
type PS = { v: number };
let opened = 0;
/** A process whose input(rt) reads the toy trainer's last frame (after: 'train'). */
const after: ProcessDef<{ w: number; tag?: string }, PO, PS> = {
	id: 'after',
	kind: 'algorithm',
	title: { en: 'after', zh: '之后' },
	phases: { out: { label: { en: 'Out', zh: '出' }, kind: 'output' } },
	schema: { v: { kind: 'scalar', label: { en: 'v', zh: 'v' } } },
	defaults: { scale: 2 },
	*run(input, ctx): ProcessRun<PS, PO> {
		opened++;
		yield { phase: 'out', step: 0, state: { v: input.w * ctx.opt.scale }, metrics: { v: input.w * ctx.opt.scale } };
	}
};

function pkgWith(sources: Record<string, unknown>, extra: Record<string, ProcessDef<any, any, any>> = {}): PackageModules {
	const base = seqPackage(lesson6);
	return {
		...base,
		manifest: { ...seqManifest, kinds: ['trace', 'train', 'process'], sources: { ...seqManifest.sources, ...sources } as never },
		processes: { ...base.processes, 'train:toy': toy as ProcessDef<any, any, any>, after: after as ProcessDef<any, any, any>, ...extra }
	};
}
const opts = (input: unknown, runner: SourceRunner = localRunner) => ({ fetch: memoryFetch(SEQ_FILES), base: '', input, runner });

test('K1 run main: the source passes run to the runner, and the engine runner keeps a run: main source off the worker', async () => {
	const seen: OpenSource[] = [];
	const spy: SourceRunner = { open: (o, p) => (seen.push(o), localRunner.open(o, p)) };
	const rt = await createRuntime(pkgWith({ m: { kind: 'process', process: 'after', run: 'main', input: () => ({ w: 1 }) } }), opts('abc', spy));
	await rt.ensure('m', 'last');
	assert.equal(seen.at(-1)?.run, 'main');
	let made = 0;
	const eng = makeEngineRunner({ readyTimeoutMs: 50 });
	const pkg = { ...pkgWith({}), worker: () => (made++, { postMessage() {}, terminate() {}, onmessage: null, onerror: null }) as never };
	const t = eng.open({ slug: 'x', source: 'm', process: 'after', input: { w: 3 }, opt: {}, seed: 1, detail: 'coarse', run: 'main' }, pkg);
	await t.ensure('last');
	assert.equal(made, 0, 'no worker was started for run: main');
	assert.equal(t.notice, null, 'no fallback notice: running on the main thread is what the source asked for');
	assert.equal((t.state(0) as PS).v, 6);
	t.close();
	eng.dispose();
});

test('K1 worker inputs: in Node, a worker source whose input a worker cannot receive fails like the page would', async () => {
	const rt = await createRuntime(pkgWith({ bad: { kind: 'process', process: 'after', input: () => ({ w: 1, f: () => 1 }) } }), opts('abc'));
	const pos = await stepAt(rt, { source: 'bad', at: () => 'last' });
	assert.match(String(pos.error ?? rt.sources.bad.error), /cannot be sent to a worker/);
	const ok = await createRuntime(pkgWith({ fine: { kind: 'process', process: 'after', run: 'main', input: () => ({ w: 1, f: () => 1 }) } }), opts('abc'));
	assert.equal(await ok.ensure('fine', 'last'), 0, "run: 'main' takes any input");
});

test('K1 after: a process opens once its source has run to the end, and reads its frames', async () => {
	const rt = await createRuntime(
		pkgWith({ train: { kind: 'train', trainer: 'toy' }, post: { kind: 'process', process: 'after', after: 'train', input: (r: { sources: Record<string, { tape?: LiveTape | null }> }) => { const t = r.sources.train.tape!; return { w: (t.state(t.length - 1) as S).w! }; } } }),
		opts('abcd')
	);
	await rt.ensure('post', 'last');
	assert.equal(rt.sources.train.tape!.length, 5, 'the train source ran to its end first');
	assert.equal((rt.sources.post.tape!.state(0) as PS).v, 2 * 4 * 4, 'w after 4 steps on 4 letters, times scale 2');
	rt.setInput('ab');
	await rt.ensure('post', 'last');
	assert.equal((rt.sources.post.tape!.state(0) as PS).v, 2 * 4 * 2, 'a new input reruns both');
});

test('K1 empty input: a train source does not run (its steps are skipped); runOnEmpty runs it', async () => {
	const rt = await createRuntime(pkgWith({ train: { kind: 'train', trainer: 'toy' }, lm: { kind: 'process', process: 'after', after: 'train', input: () => ({ w: 1 }) } }), opts(''));
	assert.ok(rt.pre?.empty);
	const pos = await stepAt(rt, { source: 'train', at: () => 'last' });
	assert.equal(pos.skipped, true);
	assert.equal(rt.sources.train.tape?.length ?? 0, 0, 'no frames (an empty, finished tape)');
	assert.equal((await stepAt(rt, { source: 'lm', at: () => 'last' })).skipped, true, 'a process after it is skipped too');
	rt.setInput('abc');
	await rt.ensure('train', 'last');
	assert.equal(rt.sources.train.tape!.length, 5, 'a later input starts a fresh run');
	rt.setInput('');
	assert.equal(rt.sources.train.tape?.length ?? 0, 0, 'an empty input stops it again');
	await rt.ensure('train', 'last');
	assert.equal(rt.sources.train.tape?.length ?? 0, 0, 'and it does not run on it');
	rt.setInput('abcd');
	await rt.ensure('train', 'last');
	assert.equal(rt.sources.train.tape!.length, 5, 'after an empty input, the next input runs it again');
	const on = await createRuntime(pkgWith({ train: { kind: 'train', trainer: 'toy', runOnEmpty: true } }), opts(''));
	await on.ensure('train', 'last');
	assert.equal(on.sources.train.tape!.length, 5);
});

test('K1 unchanged input: a process whose input(rt) gives the same input keeps its run on a new learner input', async () => {
	opened = 0;
	const rt = await createRuntime(pkgWith({ fixed: { kind: 'process', process: 'after', input: () => ({ w: 5 }) }, follows: { kind: 'process', process: 'after', input: (r: { input: unknown }) => ({ w: String(r.input).length }) } }), opts('abc'));
	await rt.ensure('fixed', 'last');
	await rt.ensure('follows', 'last');
	const t1 = rt.sources.fixed.tape;
	rt.setInput('abcdef');
	assert.equal(rt.sources.fixed.tape, t1, 'same input: the run is kept');
	assert.equal(rt.sources.follows.tape ?? null, null, 'changed input: closed, rerun on the next ensure');
	await rt.ensure('follows', 'last');
	assert.equal((rt.sources.follows.tape!.state(0) as PS).v, 12);
	assert.equal(opened, 3);
});

test('K1 live-model at first: the starting weights, without running the train source to its end', async () => {
	// liveSpec records which state it was asked to trace (and stops there: the trace itself is tested in K0 and 1b)
	let asked = -1;
	const toyLive = {
		...toy,
		liveSpec: (s: S) => {
			asked = s.w ?? -1;
			throw new Error('stop here');
		}
	} as unknown as ProcessDef<any, any, any>;
	const rt = await createRuntime(pkgWith({ train: { kind: 'train', trainer: 'toy' }, init: { kind: 'live-model', of: 'train', at: 'first' } }, { 'train:toy': toyLive }), opts('abc'));
	await rt.ensure('init', 'first');
	assert.equal(asked, 0, 'the init frame (w = 0) was traced');
	assert.equal(rt.sources.train.tape!.length < 5, true, 'the train source ran only as far as its first frame');
});

test('K1 sharedRunner: identical runs are recorded once; a control gives a view its own run', async () => {
	const shared = sharedRunner();
	const pkg = pkgWith({ train: { kind: 'train', trainer: 'toy' } });
	const a = await createRuntime(pkg, opts('abc', shared));
	const b = await createRuntime(pkg, opts('abc', shared));
	await a.ensure('train', 'last');
	await b.ensure('train', 'last');
	assert.deepEqual(shared.stats, { shared: 1, recorded: 1 });
	a.close();
	assert.equal(b.sources.train.tape!.length, 5, "closing one runtime leaves the other's view alone");
	const c = await createRuntime(pkg, opts('abcd', shared));
	await c.ensure('train', 'last');
	assert.equal(shared.stats.recorded, 2, 'another input is another run');
	(b.sources.train.tape as LiveTape).control({ type: 'set', opt: { lr: 9 } });
	await b.ensure('train', 'last');
	const d = await createRuntime(pkg, opts('abc', shared));
	await d.ensure('train', 'last');
	assert.equal((d.sources.train.tape!.state(4) as S).lr, 0.1, "b's control did not reach the shared run");
	assert.equal(runKey('x', { slug: 'x', source: 's', process: 'p', input: { f: () => 1 }, opt: {}, seed: 1, detail: 'coarse' }), null, 'a function in the input: never shared');
	const capped = sharedRunner({ cap: 2 });
	const e = await createRuntime(pkg, opts('abc', capped));
	await e.ensure('train', 'last');
	assert.equal(e.sources.train.tape!.length, 3, 'cap 2: init + 2 steps');
	shared.clear();
});

test('K1 opt.<name>: resolves from the step source and its process defaults before the source has run', async () => {
	const rt = await createRuntime(pkgWith({ train: { kind: 'train', trainer: 'toy', opt: { steps: 3 } } }), opts('abc'));
	const r = render({ en: 'lr {opt.lr}, steps {opt.steps}', zh: 'lr {opt.lr}，{opt.steps} 步' }, rt, { index: -1, source: 'train' }, 'en');
	assert.equal(r.text, 'lr 0.100, steps 3');
});

test('K1 lint: AK410 leaves saturated values alone; AK412 reads the look and the title; resolved narration over 90 words warns', async () => {
	assert.ok(saturated('1.00') && saturated('−1.00') && saturated('100.0%') && !saturated('0.699'));
	const r = { lang: 'en' as const, text: '', fallback: false, missing: [], placeholders: [
		{ name: 'i', value: 0.99991, ok: true, from: 'vars' as const },
		{ name: 'g', value: 0.99962, ok: true, from: 'vars' as const }
	] };
	assert.equal(samePrint(r), null, 'two saturated gates both print 1.00: not a hidden change');
	const pkg = pkgWith({ train: { kind: 'train', trainer: 'toy' } });
	const long = Array.from({ length: 30 }, () => '{w}').join(' ');
	const lesson = {
		title: { en: 'K1', zh: 'K1' },
		summary: { en: 's', zh: 's' },
		steps: [
			{ id: 'q', title: { en: 'Which one is right', zh: '哪个对' }, narration: { en: 'Guess. Then look.', zh: '猜。再看。' }, look: { en: 'the bigger loss falls fastest here', zh: '看' }, source: 'train', view: 'tape.table', provenance: 'live' as const,
			  predict: { id: 'k1-q', question: { en: 'Which?', zh: '哪个？' }, options: [{ en: 'the bigger loss falls fastest', zh: '大' }, { en: 'no', zh: '不' }], answer: 0, explain: { en: 'It does.', zh: '是的。' } } },
			{ id: 'w', title: { en: 'Words', zh: '词' }, narration: { en: `Here: ${long}. And more.`, zh: `这里：${long.split(' ').join('')}。还有。` , vars: () => ({ w: { en: 'many words here', zh: '很多词' } }) }, source: 'train', view: 'tape.table', provenance: 'live' as const }
		]
	};
	const c = compileLesson(lesson as never, { spec: null, ops: [], manifest: pkg.manifest });
	assert.ok(c.issues.some((i) => i.code === 'AK412' && /the look/.test(i.message)), 'AK412 on the look');
	const issues = await lintLesson(c, (fx) => createRuntime(pkg, opts(fx.raw)), [{ id: 'f', label: { en: 'f', zh: 'f' }, raw: 'abc' }], { langs: ['en'] });
	assert.ok(issues.some((i) => i.code === 'AK407' && /once its numbers are filled in/.test(i.message)), 'resolved narration over 90 words');
});

test('K1 Training section: readable metric names and formats; the curve at the last training snapshot', () => {
	assert.deepEqual(metricName('test_acc').label, { en: 'Right on the test set', zh: '测试集上的正确率' });
	assert.equal(metricName('val_bpc').label.en, 'Bits per character on the validation set');
	assert.equal(metricName('gradNorm').label.zh, '梯度大小');
	assert.equal(metricName('my_metric', { my_metric: { label: { en: 'Mine', zh: '我的' }, fmt: 'int' } }).label.en, 'Mine');
	assert.equal(metricText('test_acc', 0.9715), '97.15%');
	assert.equal(metricText('test_n', 5000), '5,000');
	assert.equal(metricRows({ step: 3, loss: 0.123456, junk: [1] as never }, undefined, ['step']).map((r) => `${r.key}=${r.text}`).join(), 'loss=0.1235');
	const frames = [{ phase: 'init' }, { phase: 'snapshot', metrics: { loss: 1 } }, { phase: 'step', metrics: { loss: 0.5 } }, { phase: 'move' }, { phase: 'move' }];
	assert.equal(trainingFrame({ length: frames.length, frame: (i: number) => frames[i] as never }), 1, 'the last snapshot, not the sampling frames after it');
	const noSnap = [{ phase: 'forward' }, { phase: 'loss', metrics: { loss: 1 } }, { phase: 'update' }];
	assert.equal(trainingFrame({ length: noSnap.length, frame: (i: number) => noSnap[i] as never }), 1);
});

test('K1 the answer line: one answer only; per-node outputs and image outputs have none; vocabulary words, not #id', () => {
	const spec = (axes: string[], outShape: number[], classes?: unknown[]) => ({ layers: [{ id: 'out', op: 'softmax', axes, outShape }], outputs: ['out'], ...(classes ? { classes } : {}) }) as unknown as AtlasSpec;
	const tr = (shape: number[]): Trace => ({ order: ['out'], t: { out: { shape, data: new Float32Array(shape.reduce((a, b) => a * b, 1)) } } });
	assert.equal(isAnswerLayer({ spec: spec(['class'], [2]) }, 'out', tr([2])), true);
	assert.equal(isAnswerLayer({ spec: spec(['n', 'class'], [34, 2]) }, 'out', tr([34, 2])), false, 'a GCN: one answer per node');
	assert.equal(isAnswerLayer({ spec: spec(['t', 'vocab'], [5, 74]) }, 'out', tr([5, 74])), true, 'a sequence: its last row');
	assert.equal(isAnswerLayer({ spec: spec(['c', 'y', 'x'], [1, 28, 28], ['a', 'b']) }, 'out', tr([1, 28, 28])), false, 'an image output, though the spec lists classes');
	const pre = { tensors: {}, steps: [], empty: false, vocab: (i: number) => 'abc'[i] };
	assert.equal(outputItemLabel({ spec: spec(['t', 'vocab'], [5, 3]), pre }, 1), '“b”');
});

test('K1 adapters: text-char plain quotes and presets; series presets from a file and the chip of the input', () => {
	const spec = { vocab: 'ab"\'-. ' };
	const p = textChar.preprocess('a“b”—a…', spec, {});
	assert.equal(p.notes, undefined, 'no unknown characters left');
	assert.deepEqual(Array.from(p.tensors.input.data), [0, 2, 1, 2, 4, 0, 5, 5, 5]);
	const off = textChar.preprocess('a“b', spec, { normalize: false });
	assert.equal(off.notes?.length, 1);
	assert.deepEqual(textChar.presets!({ 'presets.json': { texts: [{ id: 'x', en: 'X', zh: '叉', text: 'ab' }] } }).map((x) => [x.id, x.raw]), [['x', 'ab']]);
	const list = seriesAdapter.presets!({ 'presets.json': { presets: [{ id: 'eight', name: 'oscillator', en: 'Eight points', zh: '八个点', seed: 3, noise: 0.05 }, { id: 'sine' }] } });
	assert.deepEqual(list.map((x) => x.id), ['eight', 'sine']);
	assert.ok(sameSeries(list[0].raw, seriesPresetRaw('oscillator', { seed: 3, noise: 0.05 })));
	assert.ok(!sameSeries(list[0].raw, seriesPresetRaw('oscillator', { seed: 4, noise: 0.05 })), 'another seed of the same generator is another chip');
});

test('K1 formulas and numbers: long sums break into lines; fmt3 as a map callback; the parity check at a glance', () => {
	const tex = alignTex('c = f c + i g = {{a}} \\cdot {{b}} + {{c}} \\cdot {{d}} + {{e}} \\cdot {{f}} = {{y}}');
	assert.match(tex, /\\\\ &\\quad \+/);
	assert.equal(breakSum('a + b'), 'a + b');
	assert.deepEqual([0.5, 0.25].map(fmt3), ['0.500', '0.250']);
	assert.equal(sourceBadge({ scope: 'process', against: 'python-reference', pass: true, max: 0, checked: 'x' }).short?.en, '= Python run');
	assert.equal(sourceBadge({ scope: 'process', against: 'python-reference', pass: false, max: 1, checked: 'x' }).short, undefined);
});
