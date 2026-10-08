// Speed of the flow-2d training step in Node (W1-B1 acceptance 5: ≥ 50 steps/s at batch 256).
//   node tests/atlas/autograd/bench-flow.ts [steps=600] [batch=256]
// Prints steps/s through trainingProcess (coarse frames, snapshot every 50) and for a bare loop.
import { smiley } from '../../../src/lib/atlas/data/points2d.ts';
import { trainingProcess } from '../../../src/lib/atlas/train/trainer.ts';
import { backward, zeroGrad } from '../../../src/lib/atlas/autograd/var.ts';
import { Adam } from '../../../src/lib/atlas/autograd/optim.ts';
import { flowTrainer } from './flow.ts';
import { makeRng } from './rng-stub.ts';

export function benchFlow(steps = 600, batch = 256, warm = 60): { trainer: number; bare: number; lossFrom: number; lossTo: number; batch: number } {
	const data = smiley(1000, makeRng(1));
	// through the process (what a worker runs)
	const p = trainingProcess(flowTrainer);
	const gen = p.run(data, { rng: makeRng(2), opt: { ...flowTrainer.defaults, batch, steps: steps + warm }, detail: 'coarse', signal: { cancelled: false } });
	let r = gen.next();
	let n = 0;
	let t0 = 0;
	let lossFrom = NaN;
	let lossTo = NaN;
	while (!r.done) {
		const f = r.value;
		if (f.step === warm) {
			t0 = performance.now();
			lossFrom = f.metrics!.loss;
		}
		if (f.step !== undefined && f.step > warm) n++;
		lossTo = f.metrics?.loss ?? lossTo;
		r = gen.next();
	}
	const trainer = (n / (performance.now() - t0)) * 1000;
	// the bare loop: batch, forward, backward, Adam
	const rng = makeRng(3);
	const m = flowTrainer.model(rng.fork('init'), { ...flowTrainer.defaults, batch });
	const opt = new Adam(m.named(), { lr: 1e-3 });
	const br = rng.fork('batch');
	let t1 = 0;
	for (let k = 0; k < steps + warm; k++) {
		if (k === warm) t1 = performance.now();
		const b = flowTrainer.batch(data, k, br, { ...flowTrainer.defaults, batch });
		zeroGrad(m);
		const { loss } = flowTrainer.loss(m, b, () => {}, flowTrainer.defaults);
		backward(loss as Parameters<typeof backward>[0]);
		opt.step();
	}
	const bare = (steps / (performance.now() - t1)) * 1000;
	return { trainer, bare, lossFrom, lossTo, batch };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/').replace(/^\//, '')}` || process.argv[1]?.endsWith('bench-flow.ts')) {
	const steps = Number(process.argv[2] ?? 600);
	const batch = Number(process.argv[3] ?? 256);
	const res = benchFlow(steps, batch);
	console.log(
		`flow MLP 3-64-64-64-2 SiLU, Adam, batch ${res.batch}: ${res.trainer.toFixed(1)} steps/s through trainingProcess, ` +
			`${res.bare.toFixed(1)} steps/s bare loop (loss ${res.lossFrom.toFixed(4)} → ${res.lossTo.toFixed(4)}); node ${process.version}`
	);
}
