// Contract tests for the frozen DevKit types (research/devkit/DEVKIT_DESIGN.md §4).
// Run from site/:  node --test src/lib/atlas/types/contract.test.ts   (also part of `npm run check:atlas`)
//
//  1. types/index.ts loads under Node type stripping (type-only: no runtime exports).
//  2. The §2.2 StepDef examples (kinds A, B and C) type-check (svelte-check) and their narration placeholders
//     resolve from vars() with the same names in en and zh (AK401/AK403 in miniature).
//  3. Sample ops, processes, manifests and engine messages type-check against the contracts, and a ProcessDef
//     generator runs with the documented control protocol.
//  4. The mulberry32 test vectors documented on `Rng` (core.ts) are what mulberry32 returns.
//  5. View contract: every views/<key>/meta.ts (kit views under atlas/ui/views/, package views under
//     explainers/<slug>/views/) default-exports a ViewMeta whose key matches its folder, whose demo() resolves in Node,
//     whose describe(demo) is non-empty in en and zh, and whose demo provenance is 'illustrative'.
//     Zero views is a pass (wave 0); views arrive in wave 1b.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import type {
	At,
	AutoRef,
	CaptureProps,
	Control,
	EngineReply,
	EngineRequest,
	ExplainerManifest,
	ExplainerRuntime,
	Frame,
	LessonDef,
	OpCtx,
	OpDef,
	PackageModules,
	Preprocessed,
	ProcessDef,
	ProcessRun,
	Rng,
	SourceRunner,
	StepDef,
	TapeHeader,
	Tensor,
	Tmpl,
	TmplVal,
	TrainState,
	UnitExplanation,
	ViewMeta
} from './index.ts';

const here = dirname(fileURLToPath(import.meta.url));
const LIB = join(here, '..', '..'); // site/src/lib

// ------------------------------------------------------------------ fixtures shared by the tests
const facts: Record<string, number> = { 'layer.conv1_1.strongestChannel': 3, 'opt.gamma': 0.9, 'opt.lr': 0.1 };
const rt: ExplainerRuntime = {
	slug: 'contract',
	spec: null,
	model: null,
	input: null,
	pre: null,
	sources: {},
	sel: {},
	fact: (p) => facts[p],
	num: (p) => (typeof facts[p] === 'number' ? facts[p] : NaN),
	ensure: async () => 0,
	setInput: () => {}
};

/** A stand-in for train/helpers.ts updateOf (W1-B1): { old, g, lr, new } for one SGD weight. */
function updateOf(at: At, key: string, index: number[]): Record<string, TmplVal> {
	const s = at.state as TrainState;
	const w = s.params[key];
	const g = s.grads![key];
	const i = index.reduce((acc, k, d) => acc * w.shape[d] + k, 0);
	const lr = 0.1;
	return { old: w.data[i], g: g.data[i], lr, new: w.data[i] - lr * g.data[i] };
}

// The three examples of design §2.2, verbatim.
const kindA: StepDef = { id: 'slide', source: 'scan1', view: 'zoom.weighted-sum', provenance: 'live',
  title: { en: 'One kernel slides over your drawing', zh: '一个卷积核在你的画上滑动' },
  select: (rt) => ({ layer: 'conv1_1', ch: rt.num('layer.conv1_1.strongestChannel') }),
  play: { from: 'first', to: 'last', interval: 14 },
  narration: {
    en: 'A kernel is {n|int} learned numbers. At each stop the window is multiplied by it, the {n|int} products are added, and a bias is added: {s} + {b} = {y}.',
    zh: '卷积核就是 {n|int} 个学来的数。每到一处，窗口和它逐个相乘，{n|int} 个乘积相加，再加上偏置：{s} + {b} = {y}。',
    vars: (rt, at) => ({ n: at.unit!.terms!.length, s: at.unit!.values.s, b: at.unit!.bias!, y: at.unit!.result }) } };

const kindB: StepDef = { id: 'update', source: 'sgd1', at: { phase: 'update', step: 0 }, view: 'train-step', provenance: 'live',
  title: { en: 'Nudge every weight a little', zh: '把每个权重推动一点点' },
  focus: () => [{ field: 'params', key: 'net.0.weight', index: [0, 1], role: 'param' }],
  narration: {
    en: 'This weight was {old}. Its gradient is {g}, so with learning rate {lr} it becomes {old} − {lr} × {g} = {new}.',
    zh: '这个权重原来是 {old}。它的梯度是 {g}，学习率是 {lr}，所以它变成 {old} − {lr} × {g} = {new}。',
    vars: (rt, at) => updateOf(at, 'net.0.weight', [0, 1]) } };   // train/helpers.ts: { old, g, lr, new }

const kindC: StepDef = { id: 'target', source: 'q', at: { phase: 'update', where: (f) => f.metrics!.r !== 0 }, view: 'zoom.formula', provenance: 'live',
  title: { en: 'The target: reward now plus the best guess for later', zh: '目标：眼前的奖励加上对以后的最好估计' },
  narration: {
    en: 'The agent got {r}. The target is {r} + {gamma} × {next} = {target}.',
    zh: '智能体得到奖励 {r}。目标是 {r} + {gamma} × {next} = {target}。',
    vars: (rt, at) => ({ r: at.frame!.metrics!.r, gamma: rt.num('opt.gamma'), next: at.frame!.metrics!.next, target: at.frame!.metrics!.target }) } };

const f32 = (shape: number[], values: number[]): Tensor => ({ shape, data: Float32Array.from(values) });
const unit: UnitExplanation = {
	kind: 'weighted-sum',
	layer: 'conv1_1',
	unit: [3, 10, 12],
	reads: [{ tensor: 'input', cells: [0, 1, 2], box: { y0: 10, x0: 12, y1: 12, x1: 14 } }],
	terms: Array.from({ length: 9 }, (_, k) => ({ x: k / 8, w: 0.5, p: (k / 8) * 0.5 })),
	bias: -0.25,
	values: { s: 2.25 },
	result: 2,
	traced: 2
};
const atA: At = { index: 0, unit, layer: 'conv1_1' };
const atB: At = {
	index: 4,
	state: { step: 0, loss: 0.69, params: { 'net.0.weight': f32([2, 2], [0.5, -0.3, 0.2, 0.1]) }, grads: { 'net.0.weight': f32([2, 2], [0, 0.4, 0, 0]) } }
};
const atC: At = { index: 7, frame: { phase: 'update', state: {}, metrics: { r: 1, next: 0.5, target: 1.45 } } };

const PLACEHOLDER = /\{([A-Za-z_][\w.:#-]*)(?:\|[a-z0-9]+)?\}/g;
const names = (s: string) => new Set([...s.matchAll(PLACEHOLDER)].map((m) => m[1]));

// ------------------------------------------------------------------ 1
test('types/index.ts loads under type stripping and has no runtime exports', async () => {
	const m = await import('./index.ts');
	assert.deepEqual(Object.keys(m), []);
});

// ------------------------------------------------------------------ 2
test('design §2.2 examples: placeholders resolve from vars() and en/zh use the same names', () => {
	for (const [step, at] of [
		[kindA, atA],
		[kindB, atB],
		[kindC, atC]
	] as const) {
		const n = step.narration as Tmpl<ExplainerRuntime>;
		assert.ok(n.zh, `${step.id}: zh narration`);
		const en = names(n.en);
		const zh = names(n.zh!);
		assert.deepEqual([...en].sort(), [...zh].sort(), `${step.id}: en/zh placeholder sets differ (AK403)`);
		const vars = n.vars!(rt, at);
		for (const name of en) {
			const v = vars[name];
			const num = typeof v === 'object' ? ('v' in v ? v.v : v.en) : v; // an I18n value (types revision K0) prints as is
			assert.ok(num !== undefined && (typeof num === 'string' || Number.isFinite(num)), `${step.id}: {${name}} unresolved or non-finite (AK401)`);
		}
	}
	assert.deepEqual(kindA.select!(rt), { layer: 'conv1_1', ch: 3 });
	assert.deepEqual(kindB.focus!(rt, atB), [{ field: 'params', key: 'net.0.weight', index: [0, 1], role: 'param' }]);
	const whereC = (kindC.at as { where: (f: Frame, s: unknown) => boolean }).where;
	assert.equal(whereC(atC.frame!, {}), true);
});

// ------------------------------------------------------------------ 3
type ConvP = { kernel: number; stride: number };
const conv: OpDef<ConvP> = {
	op: 'contract_conv',
	version: 1,
	attrs: { kernel: { type: 'int', required: true }, stride: { type: 'int', default: 1 } },
	aliases: { contract_conv_alias: { stride: 1 } },
	category: 'conv',
	label: { en: 'Conv', zh: '卷积' },
	weights: [{ slot: 'weight' }, { slot: 'bias', optional: true }],
	inferShape: (ins, p) => [ins[0][0], ins[0][1], ins[0][2]].map((d, i) => (i === 0 || typeof d === 'string' ? d : Math.floor((d - p.kernel) / p.stride) + 1)),
	forward: (xs) => ({ shape: xs[0].shape, data: Float32Array.from(xs[0].data) }),
	receptive: (b, p) => ({ y0: b.y0 * p.stride, x0: b.x0 * p.stride, y1: b.y1 * p.stride + p.kernel - 1, x1: b.x1 * p.stride + p.kernel - 1 }),
	narration: { what: { en: 'A kernel of {kernel} × {kernel} numbers slides over the grid.', zh: '一个 {kernel} × {kernel} 的卷积核在网格上滑动。' } },
	zoom: 'weighted-sum',
	parity: { bar: 1e-4 }
};
type CountOpt = { n: number };
type CountState = { x: number };
const count: ProcessDef<number, CountOpt, CountState> = {
	id: 'count',
	kind: 'algorithm',
	title: { en: 'Count up', zh: '往上数' },
	phases: { add: { label: { en: 'Add one', zh: '加一' }, kind: 'update' } },
	schema: { x: { kind: 'scalar', label: { en: 'x', zh: 'x' }, tol: 'exact' } },
	defaults: { n: 3 },
	*run(input, ctx) {
		let x = input;
		let n = ctx.opt.n;
		for (let i = 0; i < n && !ctx.signal.cancelled; i++) {
			x += 1;
			const c: Control<CountOpt> | undefined = yield { phase: 'add', step: i, state: { x }, metrics: { x } };
			if (c?.type === 'set' && c.opt.n !== undefined) n = c.opt.n;
		}
	}
};
const manifest: ExplainerManifest = {
	slug: 'contract-demo',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'draft',
	devOnly: true,
	sources: { fwd: { kind: 'trace' }, cnt: { kind: 'process', process: 'count', run: 'main' } },
	page: {
		goals: [{ en: 'Count to three', zh: '数到三' }],
		prequestion: { id: 'contract-q', question: { en: 'Where does it stop?', zh: '它停在哪里？' }, options: [{ en: '3', zh: '3' }, { en: '4', zh: '4' }], answer: 0, explain: { en: 'n = 3.', zh: 'n = 3。' } }
	},
	credits: []
};
const lesson: LessonDef = {
	title: { en: 'Contract lesson', zh: '契约课程' },
	summary: { en: 'Type check only.', zh: '只做类型检查。' },
	steps: [{ autoAll: { until: 'op:conv1_1' } }, { auto: 'op:conv1_1', override: { title: { en: 'Slide', zh: '滑动' } } } satisfies AutoRef<ExplainerRuntime>, kindA, kindB, kindC]
};
const pkg: PackageModules = { slug: manifest.slug, manifest, lesson, ops: [conv], processes: { count }, adapters: {} };
const req: EngineRequest[] = [
	{ id: 1, channel: 'cnt', type: 'stream', method: 'open', args: { process: 'count', input: 0, opt: { n: 3 }, seed: 1, detail: 'coarse' }, maxFps: 30 },
	{ channel: 'cnt', type: 'control', control: { type: 'set', opt: { lr: 0.01 } } },
	{ channel: 'cnt', type: 'cancel' }
];
const rep: EngineReply[] = [
	{ id: 1, type: 'schema', schema: { x: { kind: 'scalar', label: { en: 'x' } } } },
	{ id: 1, type: 'frames', frames: [{ phase: 'add', state: { x: 1 } }], done: true },
	{ type: 'status', status: 'ready' }
];
const header: TapeHeader = {
	format: 'atlas-tape/1',
	process: 'count',
	version: '1',
	seed: 1,
	opt: { n: 3 },
	provenance: 'precomputed',
	schema: { x: { kind: 'scalar', label: { en: 'x' }, derived: false } },
	phases: { add: { label: { en: 'Add one' }, kind: 'update' } },
	frames: []
};

test('sample op, process, manifest, lesson, package and engine messages fit the contracts', () => {
	// OpDef<P> with a specific P is usable where OpDef[] is expected (CORE_OPS); a type alias P satisfies the
	// Record<string, unknown> constraint (an interface P would not).
	const list: OpDef[] = [conv];
	assert.equal(list[0].op, 'contract_conv');
	assert.deepEqual(conv.inferShape([[1, 28, 28]], { kernel: 3, stride: 1 }, {}), [1, 26, 26]);
	assert.deepEqual(conv.inferShape([[8, 'H', 26]], { kernel: 3, stride: 1 }, {}), [8, 'H', 24]);
	assert.deepEqual(conv.receptive!({ y0: 0, x0: 0, y1: 0, x1: 0 }, { kernel: 3, stride: 1 }), { y0: 0, x0: 0, y1: 2, x1: 2 });
	assert.equal(pkg.processes.count.id, 'count');
	assert.equal(lesson.steps.length, 5);
	assert.equal(req.length + rep.length, 6);
	assert.equal(header.format, 'atlas-tape/1');
});

test('a ProcessDef generator runs and receives controls through gen.next(control)', () => {
	const rng: Rng = { seed: 1, next: () => 0.5, normal: () => 0, int: (n) => Math.floor(0.5 * n), fork: () => rng, getState: () => 1, setState: () => {} };
	const gen = count.run(10, { rng, opt: { ...count.defaults }, detail: 'coarse', signal: { cancelled: false } });
	const frames: Frame<CountState>[] = [];
	let r = gen.next();
	while (!r.done) {
		frames.push(r.value);
		// after the first frame, ask for 5 steps instead of 3
		r = gen.next(frames.length === 1 ? { type: 'set', opt: { n: 5 } } : undefined);
	}
	assert.deepEqual(
		frames.map((f) => f.state.x),
		[11, 12, 13, 14, 15]
	);
});

test('types revision 1b (integration 1b): the optional additions fit and old shapes still fit', () => {
	// every addition is optional: an OpCtx, a Preprocessed, a SourceRunner and a ProcessDef written for wave 0 still type-check
	const ctx0: OpCtx = { layer: { id: 'l', op: 'x', label: { en: 'l' }, in: [], outShape: [1] }, dims: {}, w: () => ({ shape: [1], data: new Float32Array(1) }), emit: () => {} };
	const ctx1: OpCtx = { ...ctx0, has: (slot) => slot === 'weight' };
	const pre0: Preprocessed = { tensors: {}, steps: [], empty: true };
	const pre1: Preprocessed = { ...pre0, tokens: ['Once', ' upon'], vocab: (id) => `#${id}` };
	const runner0: SourceRunner = { open: () => { throw new Error('unused'); } };
	const runner1: SourceRunner = { ...runner0, dispose: () => {} };
	type S = { x: number; done: boolean };
	const proc: ProcessDef<null, { steps: number }, S> = {
		id: 'p', kind: 'algorithm', title: { en: 'p', zh: 'p' }, phases: {}, schema: {}, defaults: { steps: 2 }, recordDraws: true,
		*run(_i, c): ProcessRun<S, { steps: number }> {
			for (let k = 0; k < c.opt.steps; k++) yield { phase: 'step', state: { x: k } };
			yield { phase: 'end', state: { done: true } };
		},
		liveSpec: () => { throw new Error('unused'); }
	};
	const cap: CaptureProps<number[]> = { raw: [1, 2], presets: [{ id: 'a', label: { en: 'a', zh: 'a' }, raw: [0, 0] }], options: { classes: 2 }, onchange: (r, o) => void (r.length + Number(!!o?.live)), compact: true };
	const open: EngineRequest = { id: 1, channel: 'c', type: 'stream', method: 'open', args: { process: 'p', input: null, opt: {}, seed: 1, detail: 'coarse', keyEvery: 8, keep: { every: 2 }, autorun: true } };
	assert.equal(ctx1.has!('weight'), true);
	assert.equal(pre1.tokens!.length, 2);
	assert.equal(typeof runner1.dispose, 'function');
	assert.equal(proc.recordDraws, true);
	assert.equal(cap.raw!.length, 2);
	assert.equal(open.type, 'stream');
});

// ------------------------------------------------------------------ 4
test('Rng doc test vectors are mulberry32', () => {
	function mulberry32(seed: number): () => number {
		let a = seed >>> 0;
		return () => {
			a = (a + 0x6d2b79f5) | 0;
			let t = Math.imul(a ^ (a >>> 15), 1 | a);
			t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
			return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
		};
	}
	const r42 = mulberry32(42);
	assert.deepEqual([r42(), r42(), r42(), r42()].map((v) => v.toFixed(10)), ['0.6011037519', '0.4482905590', '0.8524657935', '0.6697340414']);
	const r1 = mulberry32(1);
	assert.deepEqual([r1(), r1(), r1()].map((v) => v.toFixed(10)), ['0.6270739406', '0.0027357212', '0.5274470400']);
});

// ------------------------------------------------------------------ 5
function viewMetaFiles(): { file: string; key: string }[] {
	const out: { file: string; key: string }[] = [];
	const dirs = (p: string) => (existsSync(p) ? readdirSync(p).filter((n) => statSync(join(p, n)).isDirectory()) : []);
	const kitViews = join(LIB, 'atlas', 'ui', 'views');
	for (const key of dirs(kitViews)) {
		const file = join(kitViews, key, 'meta.ts');
		if (existsSync(file)) out.push({ file, key });
	}
	const explainers = join(LIB, 'explainers');
	for (const slug of dirs(explainers)) {
		const views = join(explainers, slug, 'views');
		for (const key of dirs(views)) {
			const file = join(views, key, 'meta.ts');
			if (existsSync(file)) out.push({ file, key: `${slug}/${key}` });
		}
	}
	return out;
}

test('view contract: every views/<key>/meta.ts default-exports a ViewMeta with a working demo', async (t) => {
	const files = viewMetaFiles();
	t.diagnostic(`${files.length} view meta file(s) found`);
	for (const { file, key } of files) {
		const mod = (await import(pathToFileURL(file).href)) as { default?: ViewMeta; meta?: ViewMeta };
		const meta = mod.default ?? mod.meta;
		assert.ok(meta, `${key}: meta.ts must default-export its ViewMeta`);
		assert.equal(meta.key, key, `${key}: meta.key must be '${key}'`);
		assert.ok(meta.title?.en && meta.title?.zh, `${key}: title needs en and zh`);
		assert.ok(Array.isArray(meta.kinds) && meta.kinds.length > 0, `${key}: kinds`);
		const demo = await meta.demo();
		assert.equal(demo.provenance, 'illustrative', `${key}: demo provenance must be 'illustrative'`);
		const d = meta.describe(demo);
		assert.ok(d.en?.trim(), `${key}: describe(demo).en is empty (AK504)`);
		assert.ok(d.zh?.trim(), `${key}: describe(demo).zh is empty (AK504)`);
	}
});
