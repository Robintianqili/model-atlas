// TS-side inputs for the torch fp64 twins (W1-B1). TS is the source of truth for initial weights and synthetic data:
// these functions build them, make-inputs.ts writes them as JSON, and pipeline/atlas_fixtures/autograd/make.py reads
// them, runs PyTorch float64 on the CPU and writes the *_torch.json results the Node tests compare against.
import { MLP } from '../../../src/lib/atlas/autograd/nn.ts';
import { moons, sampleBatch } from '../../../src/lib/atlas/data/points2d.ts';
import { dampedOscillator, linspace, observations } from '../../../src/lib/atlas/data/series.ts';
import { makeRng } from './rng-stub.ts';

export type Named = Record<string, { shape: number[]; data: number[] }>;

function snapshot(m: MLP): Named {
	const out: Named = {};
	for (const [k, v] of Object.entries(m.named())) out[k] = { shape: v.shape.slice(), data: Array.from(v.data) };
	return out;
}

export interface TrainRun {
	id: string;
	act: 'tanh' | 'relu';
	optimizer: 'sgd' | 'momentum' | 'adam' | 'adamw';
	lr: number;
	steps: number;
	momentum?: number;
	betas?: [number, number];
	eps?: number;
	weightDecay?: number;
	/** the design §6.2 bar for this run (relative) */
	bar: number;
}

export interface TrainInputs {
	format: 'atlas-autograd-train/1';
	note: string;
	seed: number;
	name: string;
	sizes: number[];
	batchSize: number;
	init: Named;
	/** steps + 1 batches: batch k feeds step k; the last one gives the loss of the final weights */
	batches: { x: number[]; y: number[] }[];
	runs: TrainRun[];
}

export const TRAIN_SEED = 7;

export function makeTrainInputs(): TrainInputs {
	const rng = makeRng(TRAIN_SEED);
	const data = moons(200, rng.fork('data'), { noise: 0.1 });
	const sizes = [2, 8, 8, 2];
	const init = snapshot(new MLP('net', sizes, 'tanh', rng.fork('init')));
	const br = rng.fork('batch');
	const batches: { x: number[]; y: number[] }[] = [];
	for (let k = 0; k <= 50; k++) {
		const b = sampleBatch(data, 16, br);
		batches.push({ x: Array.from(b.x.data), y: Array.from(b.label) });
	}
	return {
		format: 'atlas-autograd-train/1',
		note: 'Written by site/tests/atlas/autograd/make-inputs.ts. MLP 2-8-8-2 on 200 moons points, batch 16.',
		seed: TRAIN_SEED,
		name: 'net',
		sizes,
		batchSize: 16,
		init,
		batches,
		runs: [
			{ id: 'sgd-tanh', act: 'tanh', optimizer: 'sgd', lr: 0.1, steps: 50, bar: 1e-10 },
			{ id: 'adam-tanh', act: 'tanh', optimizer: 'adam', lr: 0.01, betas: [0.9, 0.999], eps: 1e-8, steps: 50, bar: 1e-7 },
			{ id: 'momentum-tanh', act: 'tanh', optimizer: 'momentum', lr: 0.05, momentum: 0.9, steps: 50, bar: 1e-10 },
			{ id: 'sgd-relu', act: 'relu', optimizer: 'sgd', lr: 0.1, steps: 50, bar: 1e-10 },
			{ id: 'adamw-relu', act: 'relu', optimizer: 'adamw', lr: 0.01, weightDecay: 0.01, steps: 50, bar: 1e-7 }
		]
	};
}

export interface PinnInputs {
	format: 'atlas-autograd-pinn/1';
	note: string;
	seed: number;
	name: string;
	sizes: number[];
	init: Named;
	zeta: number;
	omega: number;
	x0: number;
	v0: number;
	tColl: number[];
	tObs: number[];
	uObs: number[];
	lambdaPhys: number;
	lambdaIc: number;
	steps: number;
	lr: number;
}

export const PINN_SEED = 11;

export function makePinnInputs(): PinnInputs {
	const rng = makeRng(PINN_SEED);
	const sizes = [1, 32, 32, 1];
	const init = snapshot(new MLP('net', sizes, 'tanh', rng.fork('init')));
	const osc = { zeta: 0.15, omega: 12, x0: 1, v0: 0 };
	const obs = observations((t) => dampedOscillator(t, osc), rng.fork('obs'), { n: 8, tMax: 1, frac: 0.4, noise: 0.02 });
	return {
		format: 'atlas-autograd-pinn/1',
		note:
			'Written by site/tests/atlas/autograd/make-inputs.ts. u_net(t) = MLP 1-32-32-1 tanh; residual r = (u″ + 2ζω u′ + ω² u) / ω²; ' +
			'loss = mse(u(tObs), uObs) + λp·mean(r²) + λi·((u(0) − x0)² + (u′(0)/ω − v0/ω)²).',
		seed: PINN_SEED,
		name: 'net',
		sizes,
		init,
		...osc,
		tColl: Array.from(linspace(0, 1, 24)),
		tObs: Array.from(obs.t),
		uObs: Array.from(obs.u),
		lambdaPhys: 0.1,
		lambdaIc: 1,
		steps: 20,
		lr: 0.01
	};
}
