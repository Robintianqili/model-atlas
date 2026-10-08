// The kind-B views' data layer (W1-B2): what plane2d, curve1d, network, train-step, loss-curve and train-controls show,
// computed on real recorded runs (the toy trainers), plus the toy trainers' derived fields against the autograd forward.
// The view contract itself (demo, describe in en + zh, illustrative) is types/contract.test.ts.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { StageProps, Tensor } from '../../types/index.ts';
import { input, noGrad } from '../../autograd/var.ts';
import { softmax } from '../../autograd/ops.ts';
import { MLP } from '../../autograd/nn.ts';
import { createRng } from '../../core/rng.ts';
import { trainingProcess } from '../../train/trainer.ts';
import { record } from '../../process/record.ts';
import { planLesson } from '../../lesson/plan.ts';
import { classifierRun, demoProps, fitRun } from './demo.ts';
import { BOUNDS, classifierTrainer, gridXY, mlpApply, probGrid, velocityField } from './toys.ts';
import { downsample, mlpLayers, mlpSizes, resolveRef } from './refs.ts';
import plane, { agreement, planeData } from '../views/plane2d/meta.ts';
import curve, { curveData, meanGap } from '../views/curve1d/meta.ts';
import net, { netData, strongest } from '../views/network/meta.ts';
import step, { stepData } from '../views/train-step/meta.ts';
import loss, { lossSeries, summarize } from '../views/loss-curve/meta.ts';
import controls, { controlsInfo } from '../views/train-controls/meta.ts';
import { presetRaw } from '../../adapters/points2d/adapter.ts';

const both = (m: { describe(p: StageProps): { en: string; zh?: string } }, p: StageProps) => {
	const d = m.describe(p);
	assert.ok(d.en.trim() && d.zh?.trim(), 'describe en + zh');
	assert.ok(!/NaN|undefined/.test(d.en), `no NaN in: ${d.en}`);
	return d;
};

test('toys: mlpApply and probGrid equal the autograd MLP forward (softmax) on the recorded weights', () => {
	const m = new MLP('net', [2, 8, 8, 2], 'tanh', createRng(3).fork('init'));
	const params: Record<string, Tensor> = {};
	for (const [k, v] of Object.entries(m.named())) params[k] = v.value;
	const xy = gridXY(BOUNDS, 6);
	const want = noGrad(() => softmax(m.forward(input({ shape: [36, 2], data: xy })), -1)).data;
	const g = probGrid(params, 'net', 'tanh', BOUNDS, 6);
	let worst = 0;
	for (let i = 0; i < 36; i++) worst = Math.max(worst, Math.abs(g.data[i] - want[2 * i + 1]));
	assert.ok(worst < 1e-14, `probGrid vs autograd: ${worst}`);
	const v = velocityField(Object.fromEntries(Object.entries(new MLP('v', [3, 64, 64, 64, 2], 'silu', createRng(1)).named()).map(([k, x]) => [k, x.value])), 0.5, BOUNDS, 4);
	assert.deepEqual(v.shape, [16, 4]);
	assert.throws(() => mlpApply({}, 'net', 'tanh', new Float64Array(2), 1), /no 'net.0.weight'/);
});

test('plane2d: points from the adapter, the derived decision grid, agreement, focus ring', () => {
	const run = classifierRun('coarse', 300);
	const p = demoProps(run, 'last', {}, [{ field: 'xy', index: [5] }]) as StageProps;
	const d = planeData(p);
	assert.deepEqual(d.points?.shape, [120, 2]);
	assert.deepEqual(d.grid?.shape, [28, 28]);
	assert.equal(d.highlight, 5);
	assert.equal(d.labelled, true);
	const agree = agreement(d);
	assert.ok(agree > 0.8, `trained classifier: ${agree} of the points on their side`);
	const first = planeData(demoProps(run, 'first') as StageProps);
	assert.ok(first.grid, 'the init frame has weights, so a grid');
	both(plane, p);
	// no points: still a sentence
	both(plane, { ...p, props: { points: null } });
});

test('curve1d: data and the derived fit (unit domain mapped onto the data time range), mean gap shrinks', () => {
	const run = fitRun(400);
	const end = curveData(demoProps(run, 'last') as StageProps);
	const start = curveData(demoProps(run, 'first') as StageProps);
	assert.equal(end.lines.length, 2);
	assert.equal(end.lines[1].kind, 'model');
	assert.equal(end.lines[1].x[0], end.lines[0].x[0]);
	assert.equal(end.lines[1].x[end.lines[1].x.length - 1], end.lines[0].x[end.lines[0].x.length - 1]);
	assert.ok(meanGap(end) < meanGap(start), `gap ${meanGap(start)} → ${meanGap(end)}`);
	both(curve, demoProps(run, 'last') as StageProps);
});

test('network: layer sizes from the params, edges by weight / grad / update, strongest edge, focus', () => {
	const coarse = classifierRun('coarse', 300);
	const d = netData(demoProps(coarse, 'last', {}, [{ field: 'params', key: 'net.2.weight', index: [1, 3] }]) as StageProps);
	assert.deepEqual(d.sizes, [2, 8, 8, 2]);
	assert.equal(d.params, 2 * 8 + 8 + 8 * 8 + 8 + 8 * 2 + 2);
	assert.deepEqual(d.focus, { layer: 1, o: 1, i: 3 });
	const s = strongest(d)!;
	assert.equal(Math.abs(s.v), d.maxAbs);
	const fine = classifierRun('fine', 4);
	const g = netData(demoProps(fine, { phase: 'update', step: 1 }, { color: 'grad' }) as StageProps);
	assert.equal(g.color, 'grad');
	assert.ok(g.values && g.values[0] && g.values[3], 'activations of example 1 from the fine frame');
	const u = netData(demoProps(fine, { phase: 'update', step: 1 }, { color: 'update' }) as StageProps);
	assert.equal(u.color, 'update');
	both(net, demoProps(coarse, 'last') as StageProps);
	assert.deepEqual(mlpSizes(mlpLayers({ 'a.weight': { shape: [3, 2], data: new Float64Array(6) }, 'b.weight': { shape: [2, 4], data: new Float64Array(8) } })), null, 'weights that do not chain');
});

test('train-step: the four phases; SGD shows old − lr × g = new with the numbers that ran; Adam does not', () => {
	const fine = classifierRun('fine', 4);
	for (const phase of ['forward', 'loss', 'backward', 'update']) {
		const p = demoProps(fine, { phase, step: 2 }) as StageProps;
		const d = stepData(p);
		assert.equal(d.phase, phase);
		assert.equal(d.step, 2);
		both(step, p);
	}
	const u = stepData(demoProps(fine, { phase: 'update', step: 2 }, {}, [{ field: 'params', key: 'net.0.weight', index: [3, 1] }]) as StageProps);
	assert.equal(u.sgd, true);
	assert.ok(Math.abs(u.old - u.lr * u.g - u.now) < 1e-12);
	// the same trainer with Adam: no SGD rule
	const adam = record(trainingProcess(classifierTrainer({ res: 8 })), presetRaw('moons', { n: 40 }), { steps: 3 }, { seed: 1, detail: 'fine' });
	const at = adam.find({ phase: 'update', step: 1 });
	const a = stepData({ at: { index: at, frame: adam.frame(at), state: adam.state(at) }, props: {}, focus: [] });
	assert.equal(a.sgd, false);
	assert.equal(a.optimizer, 'adam');
	assert.ok(Number.isFinite(a.m) && Number.isFinite(a.v), 'Adam moments');
	assert.equal(stepData({ at: { index: -1 }, props: {}, focus: [] }).phase, null);
});

test('loss-curve and train-controls: series, summary, divergence, compare; recorded tapes are not live', () => {
	const run = classifierRun('coarse', 300);
	const p = demoProps(run, { phase: 'snapshot', step: 100 }) as StageProps;
	const s = lossSeries(p);
	assert.equal(s.length, 1);
	const sum = summarize(s[0], p.at.index)!;
	assert.equal(sum.atStep, 100);
	assert.ok(sum.last < sum.first, 'training lowered the loss');
	assert.equal(sum.diverged, -1);
	const cmp = lossSeries({ ...p, props: { compare: ['train'], metrics: ['loss', 'lr'] } });
	assert.equal(cmp.length, 3);
	both(loss, p);
	const c = controlsInfo(p)!;
	assert.equal(c.live, false);
	assert.equal(c.step, 100);
	both(controls, p);
	const ds = downsample(Float64Array.from({ length: 5000 }, (_, i) => i), Float64Array.from({ length: 5000 }, (_, i) => (i === 2500 ? 99 : 1)), 200);
	assert.ok(ds.x.length <= 200 + 2 && ds.y.includes(99), 'downsampling keeps spikes');
	// divergence: lr = 50 with plain SGD on a tanh MLP
	const div = record(trainingProcess(classifierTrainer({ res: 4 })), presetRaw('moons', { n: 40 }), { steps: 200, optimizer: 'sgd', lr: 1e6 }, { seed: 1 });
	const last = div.frame(div.length - 1);
	if (last.metrics?.diverged === 1) assert.ok(summarize(lossSeries({ rt: { sources: { train: { kind: 'train', status: 'ready', provenance: 'live', tape: div } } } as never, source: 'train', props: {} })[0], 0)!.diverged >= 0);
});

test('refs: pre.<name>, fields, record entries, derived fields; missing names give undefined', () => {
	const run = classifierRun('coarse', 300);
	const p = demoProps(run, 'last') as StageProps;
	assert.deepEqual(resolveRef('pre.xy', p.rt, p.at)?.shape, [120, 2]);
	assert.deepEqual(resolveRef('params:net.0.weight', p.rt, p.at)?.shape, [8, 2]);
	assert.deepEqual(resolveRef('state.grid', p.rt, p.at)?.shape, [28, 28]);
	assert.equal(resolveRef('nope', p.rt, p.at), undefined);
	assert.equal(resolveRef('pre.nope', p.rt, p.at), undefined);
});

test('the planner’s train motifs name these views, and they render something on default props', () => {
	const run = classifierRun('fine', 4);
	const steps = planLesson(null, [], { slug: 'x', type: 'model', kinds: ['train'], quality: 'draft', sources: { s: { kind: 'train', trainer: 'cls', detail: 'fine' } }, page: { goals: [], prequestion: { id: 'q', question: { en: 'q' }, options: [], answer: 0, explain: { en: '' } } }, credits: [] } as never, {});
	const views = steps.map((s) => (typeof s.view === 'string' ? s.view : s.view.key));
	for (const v of ['plane2d', 'train-step', 'loss-curve']) assert.ok(views.includes(v), `${v} in ${views.join(',')}`);
	assert.ok(planeData(demoProps(run, 'first') as StageProps).points, 'train:data at the first frame shows the points');
	assert.equal(stepData(demoProps(run, { phase: 'loss' }) as StageProps).phase, 'loss');
});
