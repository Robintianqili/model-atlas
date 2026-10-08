// flow-2d: the processes every source runs. Both are trainer 'fit' (trainingProcess, unchanged frames) with two
// additions, so they can stay train sources (the twin badge, live-model, restart on a new input, the e2e step cap):
//   - no frames at all on an empty input (a train source would record its init frame, and every step would have to
//     narrate an empty training run); the stage then shows the adapter's "no points yet" note;
//   - coarse runs with opt.sample (default) end with Euler sampling on the final weights: 'noise' (opt.particles noise
//     points from rng.fork('noise')), then 'move' × opt.K steps x ← x + h·v(x, t), h = 1/K; then 'noiseFew' (the same
//     noise) and 'moveFew' × opt.Kfew. Metric dist: the mean distance to the nearest data point (noise and last frames).
//   train:flow   (the default export) registered under the name the loaders give trainers, so { kind: 'train',
//                trainer: 'flow' } runs it
//   eulerCheck   the same run as a process source, for the Python reference (pipeline/flow-2d/reference.py: the twin
//                trains on the replayed draws, then samples with Euler): "Euler frames equal the twin"
import type { Control, Frame, PhaseDef, ProcessDef, ProcessRun, Tensor, TrainState } from '../../atlas/types/index.ts';
import { trainingProcess, type LiveSpecFn } from '../../atlas/train/trainer.ts';
import { fit, type FlowOpt } from './trainer.ts';
import { dataOf, meanNearest, velocity, type Params } from './flow.ts';
import type { Points2D } from '../../atlas/data/points2d.ts';

export type FlowState = TrainState & { x: Tensor };
const core = trainingProcess(fit);
const L = (en: string, zh: string) => ({ en, zh });

const PHASES: Record<string, PhaseDef> = {
	...core.phases,
	noise: { label: L('Start from noise', '从噪声出发'), kind: 'input' },
	move: {
		label: L('One Euler step', '欧拉法走一步'),
		kind: 'update',
		formula: { tex: 'x_{k+1} = x_k + h\\,v(x_k, t_k),\\quad h = {{h}}', values: (f) => ({ h: f.metrics?.h ?? NaN }) }
	},
	noiseFew: { label: L('The same noise, fewer steps', '同样的噪声，更少的步数'), kind: 'input' },
	moveFew: { label: L('One big Euler step', '欧拉法走一大步'), kind: 'update' }
};

function* euler(params: Params, data: Points2D, noise: Float64Array, K: number, names: [string, string], signal: { cancelled: boolean }): ProcessRun<FlowState, FlowOpt> {
	const n = noise.length / 2;
	const x = noise.slice();
	const h = 1 / K;
	const pos = (): Tensor => ({ shape: [n, 2], data: x.slice() });
	yield { phase: names[0], step: 0, state: { x: pos() }, metrics: { t: 0, h, K, dist: meanNearest(x, n, data) } };
	for (let k = 0; k < K && !signal.cancelled; k++) {
		const v = velocity(params, x, k * h, n);
		for (let i = 0; i < 2 * n; i++) x[i] += h * v[i];
		yield { phase: names[1], step: k, state: { x: pos() }, metrics: { t: (k + 1) * h, h, ...(k === K - 1 ? { dist: meanNearest(x, n, data) } : {}) } };
	}
}

function flowProcess(id: string): ProcessDef<unknown, FlowOpt, FlowState> & LiveSpecFn<FlowOpt> {
	return {
		...core,
		id,
		phases: PHASES,
		schema: { ...core.schema, x: { kind: 'tensor', label: L('Sampled points', '采样的点'), shape: ['N', 2], role: 'output', tol: 1e-9 } },
		*run(input, ctx): ProcessRun<FlowState, FlowOpt> {
			const opt = { ...fit.defaults, ...ctx.opt };
			const data = dataOf(input);
			if (data.n === 0) return;
			const g = core.run(input, ctx);
			let params: Params | undefined;
			let c: Control<FlowOpt> | undefined;
			for (let r = g.next(); !r.done; r = g.next(c)) {
				if (r.value.state.params) params = r.value.state.params as Params;
				c = yield r.value as Frame<FlowState>;
			}
			if (ctx.detail === 'fine' || !opt.sample || !params || ctx.signal.cancelled) return;
			const rng = ctx.rng.fork('noise');
			const noise = new Float64Array(2 * Math.max(1, Math.floor(opt.particles)));
			for (let i = 0; i < noise.length; i++) noise[i] = rng.normal();
			yield* euler(params, data, noise, Math.max(1, Math.floor(opt.K)), ['noise', 'move'], ctx.signal);
			yield* euler(params, data, noise, Math.max(1, Math.floor(opt.Kfew)), ['noiseFew', 'moveFew'], ctx.signal);
		}
	};
}

const flow = flowProcess('train:flow');
export default flow;
export const eulerCheck = flowProcess('eulerCheck');
