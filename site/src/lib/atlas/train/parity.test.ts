// W1-B1 acceptance 3 and 4: training parity with PyTorch float64 twins (pipeline/atlas_fixtures/autograd/make.py).
//  3. MLP 2→8→8→2 through trainingProcess: 50 SGD steps ≤ 1e-10 relative, 50 Adam steps ≤ 1e-7 (losses and params);
//     also momentum and AdamW, a ReLU variant, step-1 gradients ≤ 1e-10 and the params after 10 steps.
//  4. PINN: tanh MLP u′, u″ (grad with createGraph, twice) and the physics-loss parameter gradients ≤ 1e-10 relative
//     against torch.autograd.grad(create_graph=True); 20 Adam steps on the PINN loss ≤ 1e-7.
// Initial weights and data come from TS (tests/atlas/autograd/{train,pinn}_inputs.json), as design §4.11 prescribes.
// Run from site/:  node --test src/lib/atlas/train/parity.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Rng, Tensor, TrainOpt, TrainerDef } from '../types/index.ts';
import { Var, grad, input } from '../autograd/var.ts';
import { add, div, mean, mul, scale, sub, sum } from '../autograd/ops.ts';
import { crossEntropy, mse } from '../autograd/loss.ts';
import { MLP, loadParams } from '../autograd/nn.ts';
import { AdamW } from '../autograd/optim.ts';
import { trainingProcess } from './trainer.ts';
import { updateOf } from './helpers.ts';
import { loadFixture, relErr } from '../../../../tests/atlas/autograd/support.ts';
import { runProcess } from '../../../../tests/atlas/autograd/tape-check.ts';
import { makeRng } from '../../../../tests/atlas/autograd/rng-stub.ts';
import type { PinnInputs, TrainInputs, TrainRun } from '../../../../tests/atlas/autograd/inputs.ts';

type J = { shape: number[]; data: number[] };
type Named = Record<string, J>;
const inputs = loadFixture<TrainInputs>('train_inputs.json');
const torchTrain = loadFixture<{ torch: string; runs: Record<string, { losses: number[]; grads0: Named; params10: Named; final: Named }> }>('train_torch.json');

function namedErr(a: Record<string, Tensor> | undefined, b: Named): number {
	assert.ok(a, 'missing tensors');
	assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), 'parameter names');
	return Math.max(...Object.keys(b).map((k) => relErr(a[k].data, b[k].data)));
}

type MlpBatch = { x: Var; y: number[] };

function mlpTrainer(run: TrainRun): TrainerDef<TrainInputs, MLP, MlpBatch, TrainOpt> {
	return {
		id: `mlp-${run.id}`,
		title: { en: `MLP ${run.id}`, zh: `MLP ${run.id}` },
		defaults: {
			seed: 1,
			lr: run.lr,
			batch: inputs.batchSize,
			steps: run.steps,
			snapshotEvery: 1,
			optimizer: run.optimizer === 'adamw' ? 'adam' : run.optimizer,
			momentum: run.momentum,
			betas: run.betas,
			eps: run.eps,
			weightDecay: run.weightDecay
		},
		model(rng) {
			const m = new MLP(inputs.name, inputs.sizes, run.act, rng);
			loadParams(m, inputs.init);
			return m;
		},
		batch(data, step) {
			const b = data.batches[step];
			return { x: input({ shape: [b.y.length, 2], data: b.x }), y: b.y };
		},
		loss(m, b, rec) {
			return { loss: crossEntropy(m.forward(b.x, rec), b.y) };
		},
		optimizer: run.optimizer === 'adamw' ? (m, opt) => new AdamW(m.named(), { lr: opt.lr, weightDecay: run.weightDecay }) : undefined
	};
}

for (const run of inputs.runs) {
	test(`MLP 2-8-8-2 ${run.id}: ${run.steps} steps vs torch fp64 (bar ${run.bar})`, (t) => {
		const ref = torchTrain.runs[run.id];
		const p = trainingProcess(mlpTrainer(run));
		const frames = runProcess(p, inputs, { seed: 3, detail: 'coarse' });
		const steps = frames.filter((f) => f.phase === 'snapshot' || f.phase === 'step');
		assert.equal(steps.length, run.steps + 1, 'one frame per step plus the final weights');
		const losses = steps.map((f) => f.metrics!.loss);
		const eLoss = relErr(losses, ref.losses);
		const eFinal = namedErr(steps[run.steps].state.params, ref.final);
		const e10 = namedErr(steps[10].state.params, ref.params10);
		// step-1 gradients from a fine-detail run (its 'backward' frame of step 0)
		const fine = runProcess(p, inputs, { seed: 3, detail: 'fine', opt: { steps: 2 } });
		const back0 = fine.find((f) => f.phase === 'backward' && f.step === 0)!;
		const eG0 = namedErr(back0.state.grads, ref.grads0);
		t.diagnostic(
			`losses ${eLoss.toExponential(2)}  final params ${eFinal.toExponential(2)}  params@10 ${e10.toExponential(2)}  step-1 grads ${eG0.toExponential(2)}  ` +
				`(loss ${losses[0].toFixed(4)} → ${losses[run.steps].toFixed(4)})`
		);
		assert.ok(eLoss <= run.bar, `losses rel err ${eLoss} > ${run.bar}`);
		assert.ok(eFinal <= run.bar, `final params rel err ${eFinal} > ${run.bar}`);
		assert.ok(e10 <= run.bar, `params after 10 steps rel err ${e10} > ${run.bar}`);
		assert.ok(eG0 <= 1e-10, `step-1 grads rel err ${eG0} > 1e-10`);
	});
}

test('fine detail gives the same weights as coarse, and updateOf narrates the SGD steps that ran', () => {
	const run = inputs.runs.find((r) => r.id === 'sgd-tanh')!;
	const p = trainingProcess(mlpTrainer(run));
	const coarse = runProcess(p, inputs, { seed: 3, detail: 'coarse', opt: { steps: 5 } });
	const fine = runProcess(p, inputs, { seed: 3, detail: 'fine', opt: { steps: 5 } });
	assert.deepEqual(
		fine.map((f) => f.phase),
		['init', ...Array.from({ length: 5 }, () => ['forward', 'loss', 'backward', 'update']).flat()]
	);
	const lastUpdate = fine[fine.length - 1];
	const finalCoarse = coarse[coarse.length - 1];
	assert.equal(finalCoarse.step, 5);
	for (const [k, v] of Object.entries(lastUpdate.state.params!)) assert.deepEqual(Array.from(v.data), Array.from(finalCoarse.state.params![k].data), k);
	// updateOf at the first update frame, with the state the tape would hold there
	const i = fine.findIndex((f) => f.phase === 'update' && f.step === 0);
	let state: Record<string, unknown> = {};
	for (let j = 0; j <= i; j++) state = { ...state, ...fine[j].state };
	const u = updateOf({ index: i, frame: fine[i], state }, 'net.0.weight', [0, 1]);
	assert.equal(u.lr, run.lr);
	assert.equal(u.old + -u.lr * u.g, u.new);
	assert.equal(u.old, inputs.init['net.0.weight'].data[1]);
	// Adam: updateOf refuses to narrate the SGD rule
	const adam = inputs.runs.find((r) => r.id === 'adam-tanh')!;
	const fa = runProcess(trainingProcess(mlpTrainer(adam)), inputs, { seed: 3, detail: 'fine', opt: { steps: 1 } });
	let sa: Record<string, unknown> = {};
	for (const f of fa) sa = { ...sa, ...f.state };
	assert.throws(() => updateOf({ index: fa.length - 1, frame: fa[fa.length - 1], state: sa }, 'net.0.weight', [0, 1]), /only describes plain SGD/);
});

// ------------------------------------------------------------------ PINN

const pin = loadFixture<PinnInputs>('pinn_inputs.json');
const torchPinn = loadFixture<{
	u: J;
	du: J;
	ddu: J;
	r: J;
	losses0: Record<'data' | 'phys' | 'ic' | 'total', number>;
	gradTotal: Named;
	gradPhys: Named;
	adam: { losses: number[]; final: Named };
}>('pinn_torch.json');

function pinnParts(m: MLP) {
	const w = pin.omega;
	const z = pin.zeta;
	const n = pin.tColl.length;
	const t = input({ shape: [n, 1], data: pin.tColl }, true);
	const u = m.forward(t);
	const du = grad(u, t, { createGraph: true });
	const ddu = grad(du, t, { createGraph: true });
	const r = div(add(add(ddu, scale(du, 2 * z * w)), scale(u, w * w)), w * w);
	const phys = mean(mul(r, r));
	const k = pin.tObs.length;
	const data = mse(m.forward(input({ shape: [k, 1], data: pin.tObs })), { shape: [k, 1], data: Float64Array.from(pin.uObs) });
	const t0 = input({ shape: [1, 1], data: [0] }, true);
	const u0 = m.forward(t0);
	const du0 = grad(u0, t0, { createGraph: true });
	const e0 = sub(u0, pin.x0);
	const e1 = div(sub(du0, pin.v0), w);
	const ic = add(sum(mul(e0, e0)), sum(mul(e1, e1)));
	const total = add(add(data, scale(phys, pin.lambdaPhys)), scale(ic, pin.lambdaIc));
	return { u, du, ddu, r, phys, data, ic, total };
}

function pinnModel(rng: Rng) {
	const m = new MLP(pin.name, pin.sizes, 'tanh', rng);
	loadParams(m, pin.init);
	return m;
}

test("PINN: u′, u″ by autograd (createGraph) and the physics-loss gradients match torch ≤ 1e-10", (t) => {
	const m = pinnModel(makeRng(0));
	const p = pinnParts(m);
	const eu = relErr(p.u.data, torchPinn.u.data);
	const edu = relErr(p.du.data, torchPinn.du.data);
	const eddu = relErr(p.ddu.data, torchPinn.ddu.data);
	const er = relErr(p.r.data, torchPinn.r.data);
	const eL = relErr([p.data.item(), p.phys.item(), p.ic.item(), p.total.item()], [torchPinn.losses0.data, torchPinn.losses0.phys, torchPinn.losses0.ic, torchPinn.losses0.total]);
	const names = Object.keys(m.named());
	const params = Object.values(m.named());
	const gTotal = grad(p.total, params);
	const gPhys = grad(p.phys, params);
	const eGT = Math.max(...names.map((k, i) => relErr(gTotal[i].data, torchPinn.gradTotal[k].data)));
	const eGP = Math.max(...names.map((k, i) => relErr(gPhys[i].data, torchPinn.gradPhys[k].data)));
	t.diagnostic(`u ${eu.toExponential(2)}  u′ ${edu.toExponential(2)}  u″ ${eddu.toExponential(2)}  residual ${er.toExponential(2)}  losses ${eL.toExponential(2)}  ∂total/∂θ ${eGT.toExponential(2)}  ∂physics/∂θ ${eGP.toExponential(2)}`);
	for (const [label, e] of [['u', eu], ["u'", edu], ["u''", eddu], ['residual', er], ['losses', eL], ['grad total', eGT], ['grad physics', eGP]] as const) {
		assert.ok(e <= 1e-10, `${label} rel err ${e}`);
	}
});

test('PINN: 20 Adam steps through trainingProcess match torch ≤ 1e-7', (t) => {
	const def: TrainerDef<PinnInputs, MLP, null, TrainOpt> = {
		id: 'pinn',
		title: { en: 'PINN', zh: 'PINN' },
		defaults: { seed: 1, lr: pin.lr, batch: pin.tColl.length, steps: pin.steps, snapshotEvery: 5, optimizer: 'adam' },
		model: (rng) => pinnModel(rng),
		batch: () => null,
		loss(m) {
			const p = pinnParts(m);
			return { loss: p.total, parts: { data: p.data, physics: p.phys, ic: p.ic } };
		}
	};
	const frames = runProcess(trainingProcess(def), pin, { seed: 5 });
	const steps = frames.filter((f) => f.phase === 'snapshot' || f.phase === 'step');
	const losses = steps.map((f) => f.metrics!.loss);
	const eL = relErr(losses, torchPinn.adam.losses);
	const eP = namedErr(steps[steps.length - 1].state.params, torchPinn.adam.final);
	t.diagnostic(`losses ${eL.toExponential(2)}  final params ${eP.toExponential(2)}  (loss ${losses[0].toFixed(5)} → ${losses[losses.length - 1].toFixed(5)}; parts at step 0: ${JSON.stringify(steps[0].state.parts)})`);
	assert.ok(eL <= 1e-7, `losses rel err ${eL}`);
	assert.ok(eP <= 1e-7, `final params rel err ${eP}`);
	assert.equal(steps[0].metrics!.physics, steps[0].state.parts!.physics, 'parts are metrics too');
});
