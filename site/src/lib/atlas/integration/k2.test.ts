// DevKit K2 (lane k2-fixes): the open kit issues after K1 and the 13-lane merge (LOG.md). One test per fix with kit code.
// Run from site/:  node --test src/lib/atlas/integration/k2.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { ProcessDef, ProcessRun } from '../types/process.ts';
import type { InputAdapter, PackageModules, StepDef, Tensor, TrainerDef, Trace } from '../types/index.ts';
import { createRuntime } from '../runtime/runtime.ts';
import { localRunner } from '../runtime/local.ts';
import { stepAt } from '../runtime/sources.ts';
import { lesson6 } from '../lesson/fixtures/lesson6.ts';
import { memoryFetch, SEQ_FILES, seqAdapter, seqManifest, seqPackage } from '../lesson/fixtures/seq.ts';
import { compileStep } from '../lesson/compile.ts';
import { fixtureCases, lintLesson } from '../lesson/lint.ts';
import { EMPTY_LINE, extraEllipses, render, sentences } from '../lesson/narrate.ts';
import { answerRow, builtinFact, outputProbs, outputRow } from '../lesson/facts.ts';
import { topAnswers } from '../ui/fallback.ts';
import { unitText, countLabel } from '../ui/scrub.ts';
import { fmtNum, fmtSetting, paren } from '../core/format.ts';
import { createRng } from '../core/rng.ts';
import { MLP } from '../autograd/nn.ts';
import { input as inputVar, type Var } from '../autograd/var.ts';
import { crossEntropy } from '../autograd/loss.ts';
import { trainingProcess } from '../train/trainer.ts';
import { sameFramesAsTrainer, wrappedTrainer } from '../train/wraps.ts';
import { moons, sampleBatch, type Points2D } from '../data/points2d.ts';

// ------------------------------------------------------------------ a toy train source (as in k1.test.ts)
type O = { steps: number; lr: number };
type S = { n: number; k: number };
const toy: ProcessDef<string, O, S> = {
	id: 'train:toy',
	kind: 'training',
	title: { en: 'toy', zh: '玩具' },
	phases: { init: { label: { en: 'Init', zh: '初始' }, kind: 'input' }, snapshot: { label: { en: 'Snapshot', zh: '快照' }, kind: 'update' } },
	schema: { n: { kind: 'int', label: { en: 'n', zh: 'n' } }, k: { kind: 'int', label: { en: 'k', zh: 'k' } } },
	defaults: { steps: 4, lr: 0.1 },
	*run(input): ProcessRun<S, O> {
		const data = String(input ?? '');
		yield { phase: 'init', step: 0, state: { n: data.length, k: 0 } };
		for (let k = 1; k <= 4; k++) yield { phase: 'snapshot', step: k, state: { n: data.length, k }, metrics: { loss: 1 / k } };
	}
};
function pkgWith(sources: Record<string, unknown>, more: Partial<PackageModules> = {}): PackageModules {
	const base = seqPackage(lesson6);
	return {
		...base,
		...more,
		manifest: { ...seqManifest, kinds: ['trace', 'train'], sources: { ...seqManifest.sources, ...sources } as never, ...(more.manifest ?? {}) },
		processes: { ...base.processes, 'train:toy': toy as ProcessDef<any, any, any> }
	};
}
const opts = (input: unknown) => ({ fetch: memoryFetch(SEQ_FILES), base: '', input, runner: localRunner });

const stepOf = (narration: StepDef['narration'], look?: StepDef['look']) =>
	compileStep({ id: 'loss', title: { en: 'Loss', zh: '损失' }, narration, ...(look ? { look } : {}), source: 'train', at: 'last', view: 'loss-curve', provenance: 'live' }, { origin: 'author', defaultSource: 'train' });

// ------------------------------------------------------------------ 1. empty input: no "…" next to the empty note
test('K2 empty input: a skipped step leaves out the sentences without numbers and says why; its look drops them silently', async () => {
	const rt = await createRuntime(pkgWith({ train: { kind: 'train', trainer: 'toy' } }), opts(''));
	const s = stepOf(
		{ en: 'Training looks at examples. After {frame.step} steps the loss is {frame.loss}. Watch it fall.', zh: '训练要看样例。{frame.step} 步之后损失是 {frame.loss}。看它下降。' },
		{ en: 'The dot at step {frame.step}.', zh: '第 {frame.step} 步的点。' }
	);
	const pos = await stepAt(rt, s);
	assert.equal(pos.skipped, true);
	assert.equal(pos.at.skipped, true, 'the At says so');
	const n = s.narration(rt, pos.at);
	assert.equal(n.en, `Training looks at examples. Watch it fall. ${EMPTY_LINE.en}`);
	assert.equal(n.zh, `训练要看样例。看它下降。${EMPTY_LINE.zh}`);
	assert.ok(!n.en.includes('…') && !n.zh!.includes('…'));
	assert.deepEqual(s.look!(rt, pos.at), { en: '', zh: '' }, 'a look made only of numbers is left out');
	// a narration that needs no numbers is unchanged; on a real input the numbers are there
	assert.equal(stepOf({ en: 'No numbers here.', zh: '这里没有数。' }).narration(rt, pos.at).en, 'No numbers here.');
	rt.setInput('abc');
	const p2 = await stepAt(rt, s);
	assert.equal(p2.skipped, false);
	assert.equal(s.narration(rt, p2.at).en, 'Training looks at examples. After 4 steps the loss is 0.250. Watch it fall.');
	// a vars() that throws on the empty input (it reads frames that do not exist) is not an error there
	const r = render({ en: 'One. Two {x}.', zh: '一。二 {x}。', vars: () => { throw new Error('no frames'); } }, rt, { index: -1, skipped: true }, 'en');
	assert.equal(r.text, `One. ${EMPTY_LINE.en}`);
	assert.equal(r.error, undefined);
	// sentences() keeps the text whole and does not cut at a decimal point
	assert.deepEqual(sentences('The loss is 0.693. Then it falls! Done', 'en'), ['The loss is 0.693. ', 'Then it falls! ', 'Done']);
	assert.equal(sentences('损失是 0.693。然后下降！结束', 'zh').join(''), '损失是 0.693。然后下降！结束');
});

test('K2 lint: "…" printed by a vars() value is AK401 on a step that ran; on a skipped step its sentence is left out', async () => {
	const pkg = pkgWith({ train: { kind: 'train', trainer: 'toy' } });
	// at frame 0 the fallback prints "…" (a step that ran); on the empty input the step is skipped and the sentence dropped
	const fallback = compileStep(
		{ id: 'fallback', title: { en: 'W', zh: 'W' }, narration: { en: 'Look. The winner is {who}.', zh: '看。赢家是 {who}。', vars: (_rt, at) => ({ who: at.index > 0 ? 'a' : '…' }) }, source: 'train', at: 'first', view: 'loss-curve', provenance: 'live' },
		{ origin: 'author', defaultSource: 'train' }
	);
	const lit = stepOf({ en: 'It goes 1, 2, 3, … and the loss is {frame.loss}.', zh: '它是 1、2、3、……，损失是 {frame.loss}。' });
	const clean = stepOf({ en: 'Watch the loss: {frame.loss}.', zh: '看损失：{frame.loss}。' });
	const fixtures = [
		{ id: 'empty', label: { en: 'empty', zh: '空' }, raw: '' },
		{ id: 'abc', label: { en: 'abc', zh: 'abc' }, raw: 'abc' }
	];
	const issues = await lintLesson([{ ...fallback, id: 'fallback' }, { ...lit, id: 'lit' }, { ...clean, id: 'clean' }], (fx) => createRuntime(pkg, opts(fx.raw)), fixtures);
	const ak401 = issues.filter((i) => i.code === 'AK401');
	assert.ok(ak401.some((i) => i.step === 'fallback' && i.fixture === 'abc' && /{who} prints "…"/.test(i.message)), JSON.stringify(ak401));
	assert.ok(!ak401.some((i) => i.step === 'fallback' && i.fixture === 'empty'), 'skipped: the sentence is left out, nothing prints "…"');
	const rtE = await createRuntime(pkg, opts(''));
	const pe = await stepAt(rtE, fallback);
	assert.equal(fallback.narration(rtE, pe.at).en, `Look. ${EMPTY_LINE.en}`);
	assert.ok(!ak401.some((i) => i.step === 'lit' || i.step === 'clean'), 'a literal ellipsis in the template, and dropped sentences, are fine');
	assert.equal(extraEllipses('a … b …', 'a … b {x}'), 1);
	assert.equal(extraEllipses('a …… b', 'a …… {x}'), 0);
});

test('K2 lint fixtures: the page start input is linted when no other fixture is the same input', () => {
	const pkg = pkgWith({}, { manifest: { ...seqManifest, input: { adapter: 'stub-seq', start: 'hello' } } });
	assert.ok(fixtureCases(pkg).some((f) => f.id === 'start' && f.kind === 'start' && f.raw === 'hello'));
	const same = fixtureCases(pkg, [{ id: 'hi', label: { en: 'hi', zh: 'hi' }, raw: 'hello' }]);
	assert.ok(!same.some((f) => f.id === 'start'), 'a preset with the same input already covers it');
});

test('K2 presets made by the adapter: manifest input.presetsFromAdapter fills rt.presets without a file', async () => {
	const made: InputAdapter<string> = { ...seqAdapter, presets: () => [{ id: 'p1', label: { en: 'P1', zh: 'P1' }, raw: 'xyz' }] };
	const pkg = pkgWith({}, { adapters: { 'stub-seq': made as InputAdapter }, manifest: { ...seqManifest, input: { adapter: 'stub-seq', presetsFromAdapter: true } } });
	const rt = (await createRuntime(pkg, opts(undefined))) as unknown as { presets: { id: string }[]; input: unknown };
	assert.deepEqual(rt.presets.map((p) => p.id), ['p1']);
	assert.equal(rt.input, 'xyz', 'the first preset is the start input when none is given');
});

// ------------------------------------------------------------------ 2. the answer row of a [T, V] output
test('K2 answer row: manifest output.row picks the row of a [T, V] output (default the last)', () => {
	assert.equal(answerRow(undefined, 5), 4);
	assert.equal(answerRow('last', 5), 4);
	assert.equal(answerRow('first', 5), 0);
	assert.equal(answerRow(2, 5), 2);
	assert.equal(answerRow(-2, 5), 3);
	assert.equal(answerRow(99, 5), 4, 'clamped');
	assert.equal(answerRow(() => 1, 5, {} as never), 1);
	assert.equal(answerRow(() => { throw new Error('x'); }, 5, {} as never), 4, 'a throwing function falls back to the last row');
	// rows 0..2 of a [3, 4] distribution: the answer is in row 1 (a [MASK] position)
	const t: Tensor = { shape: [3, 4], data: Float32Array.from([0.7, 0.1, 0.1, 0.1, 0.05, 0.05, 0.8, 0.1, 0.25, 0.25, 0.25, 0.25]) };
	assert.equal(outputProbs(t)[0], 0.25, 'default: the last row');
	assert.ok(Math.abs(outputProbs(t, 1)[2] - 0.8) < 1e-6);
	const rt = { pkg: { manifest: { output: { row: (r: { input: string }) => r.input.indexOf('_') } } }, input: 'a_c', spec: null, pre: null };
	assert.equal(outputRow(rt, t), 1);
	assert.equal(topAnswers(rt as never, t, 1)[0].i, 2, 'the side panel reads the [MASK] row');
	const trace = { order: ['prob'], t: { prob: t } } as unknown as Trace;
	const ctx = { spec: null, model: null, sel: {}, sources: { fwd: { kind: 'trace', status: 'ready', provenance: 'live', trace, tape: null } } as never };
	assert.equal(builtinFact({ ...ctx, outputRow: (x) => outputRow(rt, x) }, 'top.p'), outputProbs(t, 1)[2]);
	assert.equal(builtinFact(ctx, 'top.p'), 0.25, 'without a row: the last');
});

// ------------------------------------------------------------------ 3. number formats
test('K2 formats: counts as themselves, settings as set, negative factors in parentheses, unit templates', () => {
	assert.equal(fmtNum(50), '50', 'the loss-curve readout said "step 50.0"');
	assert.equal(fmtNum(1500), '1,500');
	assert.equal(fmtNum(0.5), '0.500');
	assert.equal(fmtSetting(0.05), '0.05', 'a learning rate as set, not "0.0500"');
	assert.equal(fmtSetting(1.5), '1.5');
	assert.equal(fmtSetting(3e-4), '0.0003');
	assert.equal(fmtSetting(-1e-7), '−1e−7');
	assert.equal(fmtSetting(1 / 3), '0.333');
	assert.equal(paren(-0.0566), '(−0.0566)');
	assert.equal(paren(0.5), '0.500');
	assert.deepEqual(unitText({ en: 'second', zh: '第 {n} 秒' }, 18), { en: 'second 18', zh: '第 18 秒' });
	assert.deepEqual(countLabel({ phase: 'x', step: 3, state: {} }, { unit: { en: 'episode', zh: '回合' } }), { en: 'episode 3', zh: '回合 3' });
});

// ------------------------------------------------------------------ 4. a process that wraps a trainer gets its twin
type TO = { seed: number; lr: number; batch: number; steps: number; snapshotEvery: number; optimizer: 'sgd' | 'momentum' | 'adam'; hidden: number };
const seen: { model: boolean }[] = [];
const PLAY: TrainerDef<Points2D, MLP, { x: Var; y: Int32Array }, TO> = {
	id: 'play',
	title: { en: 'Play', zh: '演练' },
	defaults: { seed: 1, lr: 0.03, batch: 16, steps: 30, snapshotEvery: 10, optimizer: 'adam', hidden: 4 },
	model: (rng, opt) => new MLP('net', [2, opt.hidden, 2], 'tanh', rng),
	batch(data, _k, rng, opt, model) {
		seen.push({ model: model instanceof MLP });
		const b = sampleBatch(data, opt.batch, rng);
		return { x: inputVar(b.x), y: b.label };
	},
	loss: (m, b, rec) => ({ loss: crossEntropy(m.forward(b.x, rec), b.y) })
};

test('K2 wraps: trainingProcess names its trainer; a wrapper that passes the frames on is the trainer, one that adds metrics is not', () => {
	const inner = trainingProcess(PLAY);
	assert.equal(wrappedTrainer(inner), 'play');
	const fit: ProcessDef<Points2D, TO, any> = {
		...inner,
		id: 'fit',
		*run(input, ctx) {
			if (!input.n) return;
			yield* inner.run(input, ctx);
		}
	};
	const pair: ProcessDef<Points2D, TO, any> = {
		...inner,
		id: 'pair',
		*run(input, ctx) {
			for (const f of inner.run(input, ctx)) yield { ...f, metrics: { ...f.metrics, hiLoss: 1 } };
		}
	};
	assert.equal(wrappedTrainer(fit), 'play', 'a spread wrapper inherits the trainer');
	const data = moons(64, createRng(5), { noise: 0.1 });
	seen.length = 0;
	const a = sameFramesAsTrainer(fit, inner, data, { lr: 0.05 }, 3);
	assert.equal(a.same, true, a.why);
	assert.ok(a.frames > 3);
	assert.ok(seen.length > 0 && seen.every((x) => x.model), 'batch() received the current model (K2)');
	const fine = sameFramesAsTrainer(fit, inner, data, { lr: 0.05 }, 3, { detail: 'fine' });
	assert.equal(fine.same, true, fine.why);
	assert.equal(fine.detail, 'fine');
	const b = sameFramesAsTrainer(pair, inner, data, {}, 3);
	assert.equal(b.same, false);
	assert.match(b.why!, /other metrics/);
	assert.equal(wrappedTrainer(toy), null, 'a hand-written process wraps nothing');
});
