// fno: the processes (kind C).
//   data   the training pairs: a random starting curve (heat.ts initial(), one of four families in turn) and the
//          solver's answer after STEPS steps, one frame per chunk; pair i of a set draws only from
//          ctx.rng.fork(`<set><i>`). `npm run atlas -- run fno data --opt '{"nTrain":4000,"nVal":400,"nTest":400}' --save`
//          writes the sets pipeline/fno/train.py reads (it checks every chunk's CRC-32). No Python reference: TS makes the data.
//   solve  the numerical solver on the learner's curve, one frame per time step (heat.ts heatStep). Python reference:
//          pipeline/fno/reference.py (the same arithmetic, bit for bit).
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { DT, FAMILIES, I, N, STEPS, crcBytes, heatStep, initial, solve as solveAll } from './heat.ts';

type I32 = { shape: number[]; data: Int32Array; dtype: 'i32' };
export type DataOpt = { nTrain: number; nVal: number; nTest: number; chunk: number };
type DataState = { u0: Tensor; uT: Tensor; fam: I32 };
const SETS = ['train', 'val', 'test'] as const;

export const data: ProcessDef<unknown, DataOpt, DataState> = {
	id: 'data',
	kind: 'algorithm',
	title: I('Make the training pairs', '生成训练样本对'),
	phases: { chunk: { label: I('A batch of starting curves and the solver’s answers', '一批起始曲线和求解器的答案'), kind: 'input' } },
	schema: {
		u0: { kind: 'tensor', label: I('Starting curves (64 heights each)', '起始曲线（每条 64 个高度）'), role: 'data', tol: 'exact' },
		uT: { kind: 'tensor', label: I('The solver’s curves after 50 steps', '求解器算 50 步后的曲线'), role: 'target', tol: 'exact' },
		fam: { kind: 'tensor', label: I('Family of each curve', '每条曲线的类型'), role: 'data', tol: 'exact' }
	},
	defaults: { nTrain: 8, nVal: 0, nTest: 0, chunk: 500 },
	*run(_input, ctx): ProcessRun<DataState, DataOpt> {
		const { chunk } = ctx.opt, sizes = [ctx.opt.nTrain, ctx.opt.nVal, ctx.opt.nTest];
		let k = 0;
		for (const [s, set] of SETS.entries()) {
			for (let from = 0; from < sizes[s] && !ctx.signal.cancelled; from += chunk) {
				const c = Math.min(chunk, sizes[s] - from), u0 = new Float32Array(c * N), uT = new Float32Array(c * N), fam = new Int32Array(c);
				for (let i = 0; i < c; i++) {
					fam[i] = (from + i) % FAMILIES.length;
					u0.set(initial(FAMILIES[fam[i]], ctx.rng.fork(`${set}${from + i}`)), i * N);
					uT.set(solveAll(u0.subarray(i * N, (i + 1) * N)), i * N);
				}
				const st = { u0: { shape: [c, N], data: u0 }, uT: { shape: [c, N], data: uT }, fam: { shape: [c], data: fam, dtype: 'i32' as const } };
				yield { phase: 'chunk', step: k++, state: st, metrics: { set: s, from, count: c, hash: crcBytes(u0), hashT: crcBytes(uT) } };
			}
		}
	}
};

type SolveState = { u: Tensor };
const stats = (u: Float64Array) => {
	let top = 0, lo = 0, steep = 0;
	for (let j = 0; j < N; j++) {
		if (u[j] > u[top]) top = j;
		if (u[j] < u[lo]) lo = j;
		steep = Math.max(steep, Math.abs(u[(j + 1) % N] - u[j]));
	}
	return { max: u[top], min: u[lo], top, spread: u[top] - u[lo], steep };
};

export const solve: ProcessDef<number[] | null, Record<string, never>, SolveState> = {
	id: 'solve',
	kind: 'algorithm',
	title: I('The numerical solver: 50 small time steps', '数值求解器：50 个小时间步'),
	phases: {
		start: { label: I('The starting curve, time 0', '起始曲线，时间 0'), kind: 'input' },
		step: {
			label: I('One time step: every point moves toward the average of its neighbours', '一个时间步：每个点都向两边邻居的平均值靠拢'),
			kind: 'update',
			narration: { what: I('After step {frame.step} the steepest rise between two neighbouring points is {frame.steep}.', '第 {frame.step} 步之后，相邻两点之间最陡的落差是 {frame.steep}。') },
			formula: { tex: 'u_j \\leftarrow u_j + r\\,(u_{j-1} - 2u_j + u_{j+1}),\\quad r = {{r}}', values: () => ({ r: '0.1' }) }
		}
	},
	schema: { u: { kind: 'tensor', label: I('The curve: 64 heights', '曲线：64 个高度'), tol: 1e-12 } },
	defaults: {},
	*run(u0): ProcessRun<SolveState, Record<string, never>> {
		if (!u0 || u0.length !== N) return;
		let u: Float64Array = Float64Array.from(u0);
		yield { phase: 'start', step: 0, state: { u: { shape: [N], data: u } }, metrics: { t: 0, ...stats(u) } };
		for (let s = 1; s <= STEPS; s++) {
			u = heatStep(u);
			yield { phase: 'step', step: s, state: { u: { shape: [N], data: u } }, metrics: { t: s * DT, ...stats(u) } };
		}
	}
};
