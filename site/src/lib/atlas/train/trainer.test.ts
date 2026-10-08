// W1-B1 acceptance 6: trainingProcess passes the process determinism checks (a local copy of W1-C's harness in
// tests/atlas/autograd/tape-check.ts): same seed → identical frame hash; keyEvery 1 = keyEvery 64; state(i) = replay
// for 200 random i. Also: frame structure, controls (set lr, input, reset), divergence, derived fields.
// Run from site/:  node --test src/lib/atlas/train/trainer.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Control, Frame, ProcessDef, TrainOpt, TrainState, TrainerDef } from '../types/index.ts';
import { Var, input, noGrad, param } from '../autograd/var.ts';
import { mul, sum } from '../autograd/ops.ts';
import { crossEntropy } from '../autograd/loss.ts';
import { MLP } from '../autograd/nn.ts';
import { moons, circles, sampleBatch, type Points2D } from '../data/points2d.ts';
import { accuracy, decisionGrid } from './metrics.ts';
import { trainingProcess } from './trainer.ts';
import { checkDeterminism, hashOf, runProcess } from '../../../../tests/atlas/autograd/tape-check.ts';
import { makeRng } from '../../../../tests/atlas/autograd/rng-stub.ts';

type Opt = TrainOpt & { hidden: number };
const BOUNDS = { x0: -1.5, x1: 1.5, y0: -1.5, y1: 1.5 };

const playground: TrainerDef<Points2D, MLP, { x: Var; y: Int32Array }, Opt> = {
	id: 'playground',
	title: { en: 'MLP playground', zh: 'MLP 游乐场' },
	defaults: { seed: 1, lr: 0.03, batch: 16, steps: 120, snapshotEvery: 10, optimizer: 'adam', hidden: 6 },
	model: (rng, opt) => new MLP('net', [2, opt.hidden, opt.hidden, 2], 'tanh', rng),
	batch(data, _step, rng, opt) {
		const b = sampleBatch(data, opt.batch, rng);
		return { x: input(b.x), y: b.label };
	},
	loss(m, b, rec) {
		return { loss: crossEntropy(m.forward(b.x, rec), b.y) };
	},
	derived: {
		grid: {
			kind: 'tensor',
			label: { en: 'Decision regions', zh: '决策区域' },
			derive: (s: TrainState) => {
				const m = new MLP('net', [2, 6, 6, 2], 'tanh', makeRng(0));
				for (const [k, v] of Object.entries(m.named())) v.data.set(s.params[k].data);
				return decisionGrid((x) => m.forward(x), BOUNDS, 8);
			}
		}
	},
	metrics: (m, data) => ({ acc: accuracy(noGrad(() => m.forward(input({ shape: [data.n, 2], data: data.xy }))), data.label) })
};

const data = () => moons(120, makeRng(99), { noise: 0.1 });

test('coarse frames: init, then one frame per step with snapshots every snapshotEvery and a final frame', () => {
	const frames = runProcess(trainingProcess(playground), data(), { seed: 4, opt: { steps: 25, snapshotEvery: 10 } });
	assert.equal(frames[0].phase, 'init');
	assert.equal(frames[0].step, undefined);
	const steps = frames.slice(1);
	assert.deepEqual(
		steps.map((f) => f.step),
		Array.from({ length: 26 }, (_, k) => k)
	);
	assert.deepEqual(
		steps.filter((f) => f.phase === 'snapshot').map((f) => f.step),
		[0, 10, 20, 25]
	);
	for (const f of steps) {
		assert.ok(Number.isFinite(f.metrics!.loss));
		assert.equal(f.metrics!.lr, 0.03);
		assert.equal(!!f.state.params, f.phase === 'snapshot');
		if (f.step! < 25) assert.ok(f.metrics!.gradNorm! > 0);
	}
	assert.ok(typeof steps[0].metrics!.acc === 'number' && steps[0].metrics!.acc >= 0 && steps[0].metrics!.acc <= 1);
	const l0 = steps[0].metrics!.loss;
	const lN = steps[25].metrics!.loss;
	assert.ok(lN < l0, `loss went ${l0} → ${lN}`);
});

for (const detail of ['coarse', 'fine'] as const) {
	test(`determinism (${detail}): same seed → same hash; keyEvery 1 = 64; state(i) = replay for 200 random i`, (t) => {
		const p = trainingProcess(playground);
		const r = checkDeterminism(p, data, { seed: 11, detail, opt: { steps: detail === 'fine' ? 60 : 300 } });
		t.diagnostic(JSON.stringify(r));
		assert.ok(r.sameSeed, 'same seed, same frames');
		assert.ok(r.keyEvery, 'keyEvery 1 equals keyEvery 64');
		assert.ok(r.replay, 'state(i) equals a replay');
		assert.ok(r.frames > 64, 'enough frames to cross a keyframe');
		const other = runProcess(p, data(), { seed: 12, detail, opt: { steps: 5 } });
		const same = runProcess(p, data(), { seed: 11, detail, opt: { steps: 5 } });
		assert.notEqual(hashOf(other), hashOf(same), 'a different seed gives different frames');
	});
}

test('determinism holds with controls in the stream (lr change, new data, reset)', (t) => {
	const p = trainingProcess(playground);
	const controlAt = (i: number): Control<Opt> | undefined => {
		if (i === 20) return { type: 'set', opt: { lr: 0.1 } };
		if (i === 40) return { type: 'input', input: circles(80, makeRng(5)) };
		if (i === 70) return { type: 'reset' };
		return undefined;
	};
	const r = checkDeterminism(p, data, { seed: 21, opt: { steps: 100 }, controlAt });
	t.diagnostic(JSON.stringify(r));
	assert.ok(r.sameSeed && r.keyEvery && r.replay);
});

test('controls: set lr applies at the next step; input starts a segment and keeps the weights; reset restarts', () => {
	const p = trainingProcess(playground);
	let sent = false;
	const frames = runProcess(p, data(), {
		seed: 7,
		opt: { steps: 30, snapshotEvery: 1 },
		controlAt: (i, f) => {
			if (f.step === 5 && !sent) {
				sent = true;
				return { type: 'set', opt: { lr: 0.2 } };
			}
			if (f.step === 10 && f.phase === 'snapshot' && i < 15) return { type: 'input', input: circles(60, makeRng(3)) };
			return undefined;
		}
	});
	const byStep = (k: number) => frames.filter((f) => f.step === k && f.phase !== 'init');
	assert.equal(byStep(5)[0].metrics!.lr, 0.03, 'the step that received the control still used the old lr');
	assert.equal(byStep(6)[0].metrics!.lr, 0.2, 'the next step uses the new lr');
	const segStart = frames.findIndex((f, i) => i > 0 && f.phase === 'init');
	assert.ok(segStart > 0, 'a new segment starts after the input control');
	assert.deepEqual(frames[segStart].caption, { en: 'New data', zh: '换了新数据' });
	// the step-10 frame showed θ10; its update ran before the swap, so the new segment starts from θ11 and so does step 11
	const after = frames[segStart].state.params!;
	assert.equal(frames[segStart + 1].step, 11, 'training continues with the next step');
	assert.equal(hashOf(frames[segStart + 1].state.params), hashOf(after), 'weights are kept on new data');
	assert.notEqual(hashOf(frames[segStart - 1].state.params), hashOf(after), 'the pending update of step 10 still ran');

	// reset without a seed reproduces the start exactly
	const fr = runProcess(p, data(), { seed: 7, opt: { steps: 12 }, controlAt: (i) => (i === 6 ? { type: 'reset' } : undefined) });
	const inits = fr.map((f, i) => (f.phase === 'init' ? i : -1)).filter((i) => i >= 0);
	assert.equal(inits.length, 2);
	assert.equal(hashOf(fr[inits[1]].state.params), hashOf(fr[0].state.params), 'reset → the same first weights');
	assert.equal(fr[inits[1] + 1].step, 0);
});

test('two trainers with the same seed share initial weights (fork("init"))', () => {
	const a = runProcess(trainingProcess(playground), data(), { seed: 8, opt: { steps: 0 } });
	const b = runProcess(trainingProcess({ ...playground, id: 'other', loss: (m, bt, rec) => ({ loss: crossEntropy(m.forward(bt.x, rec), bt.y) }) }), data(), {
		seed: 8,
		opt: { steps: 0, lr: 1 }
	});
	assert.equal(hashOf(a[0].state.params), hashOf(b[0].state.params));
});

test('divergence: a non-finite loss ends the run with a captioned frame instead of crashing', () => {
	// loss w², SGD with lr 1e10: w grows ×(1 − 2e10) per step and overflows to Infinity after about 30 steps
	const def: TrainerDef<null, { w: Var; params(): Var[]; named(): Record<string, Var> }, null, TrainOpt> = {
		id: 'blowup',
		title: { en: 'Blow-up', zh: '爆炸' },
		defaults: { seed: 1, lr: 1e10, batch: 1, steps: 400, snapshotEvery: 50, optimizer: 'sgd' },
		model: () => {
			const w = param('w', { shape: [1], data: [0.5] });
			return { w, params: () => [w], named: () => ({ w }) };
		},
		batch: () => null,
		loss: (m) => ({ loss: sum(mul(m.w, m.w)) })
	};
	const frames = runProcess(trainingProcess(def), null, { seed: 2 });
	const last = frames[frames.length - 1];
	assert.ok(frames.length < 100, `stopped early (${frames.length} frames)`);
	assert.equal(last.metrics!.diverged, 1);
	assert.equal(last.phase, 'snapshot');
	assert.ok(!Number.isFinite(last.metrics!.loss));
	assert.ok(last.caption && /diverged/.test(last.caption.en));
	const fine = runProcess(trainingProcess(def), null, { seed: 2, detail: 'fine' });
	assert.equal(fine[fine.length - 1].phase, 'loss', 'fine detail stops at the loss frame');
	assert.equal(fine[fine.length - 1].metrics!.diverged, 1);
});

test('frames hold copies: mutating the weights afterwards does not change recorded frames', () => {
	const p = trainingProcess(playground);
	const frames = runProcess(p, data(), { seed: 9, detail: 'fine', opt: { steps: 3 } });
	assert.equal(hashOf(runProcess(p, data(), { seed: 9, detail: 'fine', opt: { steps: 3 } })), hashOf(frames));
	const upd = frames.filter((f) => f.phase === 'update');
	assert.notEqual(hashOf(upd[0].state.params), hashOf(upd[1].state.params), 'each update frame has its own weights');
	assert.ok(upd[0].state.opt!['net.0.weight'].m && upd[0].state.opt!['net.0.weight'].v, 'Adam moments are recorded');
	const fwd = frames.find((f) => f.phase === 'forward')!;
	assert.deepEqual(Object.keys(fwd.state.acts!).sort(), ['net.0', 'net.1', 'net.2', 'net.3', 'net.4']);
	assert.deepEqual(Object.keys(fwd.state.batch!).sort(), ['x', 'y']);
});

test('the harness itself catches an unreported in-place mutation', () => {
	const leaky: ProcessDef<null, { n: number }, { w: { shape: number[]; data: Float64Array } }> = {
		id: 'leaky',
		kind: 'algorithm',
		title: { en: 'Leaky', zh: '泄漏' },
		phases: { add: { label: { en: 'Add', zh: '加' }, kind: 'update' } },
		schema: { w: { kind: 'tensor', label: { en: 'w', zh: 'w' } } },
		defaults: { n: 130 },
		*run(_i, ctx) {
			const w = { shape: [1], data: new Float64Array(1) };
			for (let k = 0; k < ctx.opt.n; k++) {
				w.data[0] += 1; // mutates the tensor that earlier frames already hold
				yield { phase: 'add', step: k, state: { w } } as Frame<{ w: typeof w }>;
			}
		}
	};
	const r = checkDeterminism(leaky, () => null, { seed: 1 });
	assert.equal(r.keyEvery, false, 'keyEvery 1 vs 64 differs when frames share mutable tensors');
});

test('derived fields are declared in the schema, not stored in frames', () => {
	const p = trainingProcess(playground);
	assert.ok(p.schema.grid?.derive, 'derived field in schema');
	const frames = runProcess(p, data(), { seed: 1, opt: { steps: 3 } });
	for (const f of frames) assert.equal('grid' in f.state, false);
	const g = p.schema.grid.derive!(frames[1].state as TrainState) as { shape: number[]; data: Float64Array };
	assert.deepEqual(g.shape, [8, 8]);
	assert.ok(g.data.every((v) => v >= 0 && v <= 1));
});
