// DevKit K0: the kit fixes from wave 2a (LOG.md "K0"). One test per fix that has kit code behind it.
// Run from site/:  node --test src/lib/atlas/integration/k0.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Frame, LiveTape, ProcessDef, ProcessRun } from '../types/process.ts';
import type { PackageModules } from '../types/runtime.ts';
import type { PredictQuestion } from '../types/core.ts';
import { createRuntime } from '../runtime/runtime.ts';
import { localRunner } from '../runtime/local.ts';
import { alignedIndex, stepAt } from '../runtime/sources.ts';
import { recorder, record } from '../process/record.ts';
import { saveTape } from '../process/io.ts';
import { lesson6 } from '../lesson/fixtures/lesson6.ts';
import { memoryFetch, SEQ_FILES, seqManifest, seqPackage } from '../lesson/fixtures/seq.ts';
import { compileLesson, giveaway } from '../lesson/compile.ts';
import { plural, num } from '../lesson/say.ts';
import { renderI18n } from '../lesson/narrate.ts';
import { countLabel, frameLabel, scrubMarks } from '../ui/scrub.ts';

type O = { steps: number; lr: number };
type S = { n: number; lr: number; k: number };
/** A toy "trainer": one frame per step; every frame records the data size and the learning rate it ran with. */
const toy: ProcessDef<string, O, S> = {
	id: 'train:toy',
	kind: 'training',
	title: { en: 'toy', zh: '玩具' },
	phases: { step: { label: { en: 'Step', zh: '步' }, kind: 'update' } },
	schema: {
		n: { kind: 'int', label: { en: 'n', zh: 'n' } },
		lr: { kind: 'scalar', label: { en: 'lr', zh: 'lr' } },
		k: { kind: 'int', label: { en: 'k', zh: 'k' } }
	},
	defaults: { steps: 4, lr: 0.1 },
	*run(input, ctx): ProcessRun<S, O> {
		let data = String(input ?? '');
		let lr = ctx.opt.lr;
		for (let k = 0; k <= ctx.opt.steps; k++) {
			const c = yield { phase: 'step', step: k, state: { n: data.length, lr, k }, metrics: { n: data.length, lr } };
			if (c?.type === 'input') data = String(c.input ?? '');
			if (c?.type === 'set' && typeof c.opt.lr === 'number') lr = c.opt.lr;
		}
	}
};

function pkgWith(sources: Record<string, unknown>): PackageModules {
	const base = seqPackage(lesson6);
	return {
		...base,
		manifest: { ...seqManifest, kinds: ['trace', 'train'], sources: { ...seqManifest.sources, ...sources } as never },
		processes: { ...base.processes, 'train:toy': toy as ProcessDef<any, any, any> }
	};
}

test('K0 issue 1: a finished train source restarts on a new input and keeps the options the learner set', async () => {
	const rt = await createRuntime(pkgWith({ train: { kind: 'train', trainer: 'toy' } }), { fetch: memoryFetch(SEQ_FILES), base: '', input: 'abc', runner: localRunner });
	await rt.ensure('train', 'last');
	const tape = rt.sources.train.tape as LiveTape;
	assert.equal(tape.frame(tape.length - 1).metrics!.n, 3);
	tape.control({ type: 'set', opt: { lr: 9 } });
	rt.setInput('abcdefg');
	assert.equal(rt.sources.train.tape, tape, 'the same LiveTape object (a TrainSession bound to it keeps working)');
	await rt.ensure('train', 'last');
	const t2 = rt.sources.train.tape!;
	assert.equal(t2.frame(t2.length - 1).metrics!.n, 7, 'the new input');
	assert.equal(t2.frame(0).metrics!.lr, 9, "a fresh run with the learner's lr");
	assert.equal(t2.frame(0).step, 0, 'from the start');
	rt.close();
});

test('K0 issue 1: the recorder restarts a reset run on the latest input and options', async () => {
	const r = recorder(toy, 'ab', {}, { seed: 1, autorun: false });
	await r.ensure('last');
	r.control({ type: 'input', input: 'abcd' });
	r.control({ type: 'set', opt: { lr: 5 } });
	r.control({ type: 'reset' });
	await r.ensure('last');
	assert.equal(r.frame(0).metrics!.n, 4);
	assert.equal(r.frame(0).metrics!.lr, 5);
	r.close();
});

test('K0: a precomputed tape serves the start input only; a new input runs live', async () => {
	const t = record(toy, 'start', {}, { seed: 1 });
	const { header, blob } = saveTape(t);
	const files = { ...SEQ_FILES, 'models/fixture-seq/tapes/toy.tape.json': JSON.parse(JSON.stringify(header)), 'models/fixture-seq/tapes/toy.tape.bin': blob };
	const rt = await createRuntime(pkgWith({ train: { kind: 'train', trainer: 'toy', precomputed: 'tapes/toy' } }), {
		fetch: memoryFetch(files),
		base: '',
		input: 'start',
		runner: localRunner,
		preferPrecomputed: true
	});
	await rt.ensure('train', 'last');
	assert.equal(rt.sources.train.provenance, 'precomputed');
	rt.setInput('a much longer input');
	await rt.ensure('train', 'last');
	assert.equal(rt.sources.train.provenance, 'live');
	assert.equal(rt.sources.train.tape!.frame(0).metrics!.n, 19);
	rt.close();
});

test('K0 issue 8: process sources take detail; derive(s, opt) sees the run options', async () => {
	let seen = '';
	const p: ProcessDef<unknown, { act: string }, { x: number }> = {
		id: 'p',
		kind: 'algorithm',
		title: { en: 'p', zh: 'p' },
		phases: { a: { label: { en: 'a', zh: 'a' }, kind: 'other' } },
		schema: {
			x: { kind: 'scalar', label: { en: 'x', zh: 'x' } },
			twice: { kind: 'scalar', label: { en: '2x', zh: '2x' }, derive: (s: { x: number }, o?: { act?: string }) => (o?.act === 'double' ? 2 * s.x : s.x) }
		},
		defaults: { act: 'same' },
		*run(_i, ctx): ProcessRun<{ x: number }, { act: string }> {
			seen = ctx.detail;
			yield { phase: 'a', state: { x: 3 } };
		}
	};
	const base = seqPackage(lesson6);
	const pkg: PackageModules = {
		...base,
		manifest: { ...seqManifest, sources: { ...seqManifest.sources, fine: { kind: 'process', process: 'p', detail: 'fine', opt: { act: 'double' } } } },
		processes: { ...base.processes, p: p as ProcessDef<any, any, any> }
	};
	const rt = await createRuntime(pkg, { fetch: memoryFetch(SEQ_FILES), base: '', runner: localRunner });
	await rt.ensure('fine', 'last');
	assert.equal(seen, 'fine');
	assert.equal((rt.sources.fine.tape!.state(0) as unknown as { twice: number }).twice, 6);
	rt.close();
});

test('K0 issue 7: a step compare runs both sources to the step cursor; alignedIndex matches by cursor, not by index', async () => {
	const rt = await createRuntime(pkgWith({ short: { kind: 'train', trainer: 'toy', opt: { steps: 2 } }, long: { kind: 'train', trainer: 'toy', opt: { steps: 6 } } }), {
		fetch: memoryFetch(SEQ_FILES),
		base: '',
		input: 'ab',
		runner: localRunner
	});
	const pos = await stepAt(rt, { source: 'short', at: () => 'last', compare: { a: 'short', b: 'long', label: [{ en: 'a' }, { en: 'b' }] } });
	assert.equal(pos.index, 2);
	assert.equal(rt.sources.long.tape!.length, 7, 'the compared source ran to its own end');
	assert.equal(alignedIndex(rt, 'long', 'last'), 6, "'last' on each side");
	assert.equal(alignedIndex(rt, 'long', 'last', rt.sources.short.tape!.frame(1)), 1, 'a playing frame: the same phase and step');
	rt.close();
});

test('K0 issue 12 / R5: I18n vars, plural() and num()', () => {
	const pts = { one: '{n} point', many: '{n} points', zh: '{n} 个点' };
	assert.deepEqual(plural(1, pts), { en: '1 point', zh: '1 个点' });
	assert.deepEqual(plural(1200, pts), { en: '1,200 points', zh: '1,200 个点' });
	assert.ok(Number.isNaN(plural(NaN, { one: 'a', many: 'b' }) as number), 'non-finite stays visible to lint');
	assert.deepEqual(num(-0.000771, 'n2'), { en: '−7.7e−4', zh: '−7.7e−4' });
	const tm = { en: 'Your data has {pts}.', zh: '你的数据有 {pts}。', vars: () => ({ pts: plural(1, pts) }) };
	assert.equal(renderI18n(tm as never, {} as never, { index: 0 }).en, 'Your data has 1 point.');
});

test('K0: lint AK411 (a step term in no glossary) and AK412 (a predict narration gives the answer away)', () => {
	const q: PredictQuestion = {
		id: 'k0-q',
		question: { en: 'Will greedy find the best arm?', zh: '贪心会找到最好的臂吗？' },
		options: [
			{ en: 'Yes, eventually', zh: '会' },
			{ en: 'No, it gets stuck on a worse arm', zh: '不会' }
		],
		answer: 1,
		explain: { en: 'Greedy never tries the other arms again once one looks best, so it keeps pulling a worse arm.', zh: '……' }
	};
	assert.match(giveaway('Greedy pulls arm 2. No, it gets stuck on a worse arm here.', q) ?? '', /quotes the right option/);
	assert.match(giveaway('Once one looks best, so it keeps pulling a worse arm forever.', q) ?? '', /repeats the explanation/);
	assert.equal(giveaway('Greedy always pulls the arm whose average is highest so far. What happens next?', q), null);
	const pkg = seqPackage(lesson6);
	const manifest = { ...pkg.manifest, glossary: [{ id: 'own-term', term: { en: 'own', zh: '自己的' }, def: { en: 'd', zh: 'd' } }] };
	const steps = [...lesson6.steps];
	const i = steps.findIndex((s) => !('auto' in s) && !('autoAll' in s));
	steps[i] = { ...(steps[i] as object), terms: ['own-term', 'nope', 'softmax'] } as never;
	const c = compileLesson({ ...lesson6, steps }, { spec: null, ops: pkg.ops, manifest, terms: (id) => id === 'softmax' });
	const bad = c.issues.filter((x) => x.code === 'AK411');
	assert.equal(bad.length, 1, JSON.stringify(c.issues));
	assert.match(bad[0].message, /'nope'/);
});

test('K0: the scrubber counts in the step unit (play.unit / by)', () => {
	const f: Frame = { phase: 'update', step: 7, sub: 3, state: {} };
	assert.deepEqual(countLabel(f), { en: 'step 7', zh: '第 7 步' });
	assert.deepEqual(countLabel(f, { unit: { en: 'episode', zh: '回合' }, by: 'sub' }), { en: 'episode 3', zh: '回合 3' });
	const t = record(toy, 'abc', { steps: 60 }, { seed: 1 });
	const frames = Array.from({ length: t.length }, (_, i) => i);
	const marks = scrubMarks(t, frames, { count: { unit: { en: 'epoch', zh: '轮' } } });
	assert.ok(marks.length > 1 && marks.every((m) => /^epoch \d+$/.test(m.label.en)), marks.map((m) => m.label.en).join(', '));
	assert.equal(frameLabel(t, 3, { unit: { en: 'move', zh: '步' } }).en, 'Step · move 3');
});
