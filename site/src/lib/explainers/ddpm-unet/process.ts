// ddpm-unet processes (kind C). Every number comes from the schedule (ddpm.ts) and the exported U-Net, run by the kit's
// executor (in the package worker in the browser):
//   forward   the learner's drawing, stretched to −1 … 1, noised at t = 0, 1, …, T with the one-jump formula and the trace's ε
//   sample    DDPM sampling: x_T = noise drawn from the seed, then for each kept step t: ε̂ = UNet(x_t, t),
//             x_prev = (x_t − β/√(1 − ᾱ_t) · ε̂) / √α + σ·z (σ = √β, no noise on the last step). opt.stride 1 visits all
//             T steps; stride s keeps t = T, T − s, …, s with ᾱ of the kept steps (α = ᾱ_t / ᾱ_prev, β = 1 − α).
//             Draws: ctx.rng.fork('seed<k>') (k = the learner's seed), recorded so pipeline/ddpm-unet/reference.py replays them.
//   gallery   `sample` (stride 1) for seeds 0 … opt.n − 1, the picture at every FILM-th step (t = 100, 80, …, 0); its
//             1,200 network runs are precomputed (tapes/gallery)
import type { Frame, ProcessDef, ProcessRun, Rng, Tensor } from '../../atlas/types/index.ts';
import { FILM, N, PROBE, SCHED, SIDE, SEEDS, T, predictNoise, stepsOf, unetOf } from './ddpm.ts';

const L = (en: string, zh: string) => ({ en, zh });
const pic = (a: ArrayLike<number>): Tensor => ({ shape: [SIDE, SIDE], data: Float32Array.from(a) });
const PIC = { kind: 'tensor' as const, shape: [SIDE, SIDE], scale: 'diverging' as const, role: 'output' as const };

export interface ModelIn { seed: number; spec: unknown; weights: Record<string, Tensor> }
export type PicState = { x: Tensor; eps?: Tensor };
type SOpt = { stride: number; probe: number; n: number };

/** one sampling run: frame 0 is the noise, then one frame per kept step */
export function* chain(input: ModelIn, rng: Rng, stride: number, probe: number, cancel: { cancelled: boolean }): Generator<Frame<PicState>> {
	const m = unetOf(input.spec, input.weights);
	const r = rng.fork(`seed${input.seed}`);
	let x = Float64Array.from({ length: N }, () => r.normal());
	yield { phase: 'noise', step: 0, state: { x: pic(x) }, metrics: { t: T, seed: input.seed, K: stepsOf(stride).length } };
	const steps = stepsOf(stride);
	for (let i = 0; i < steps.length && !cancel.cancelled; i++) {
		const t = steps[i], last = i === steps.length - 1;
		const ab = SCHED.abar[t - 1], prev = last ? 1 : SCHED.abar[steps[i + 1] - 1];
		const a = ab / prev, b = 1 - a, coef = b / Math.sqrt(1 - ab), inv = 1 / Math.sqrt(a), sigma = last ? 0 : Math.sqrt(b);
		const eps = predictNoise(m, x, t);
		const next = new Float64Array(N);
		let zp = 0, mp = 0;
		for (let j = 0; j < N; j++) {
			const mean = (x[j] - coef * eps[j]) * inv, z = last ? 0 : r.normal();
			next[j] = mean + sigma * z;
			if (j === probe) (zp = z), (mp = mean);
		}
		yield {
			phase: 'step', step: i, state: { x: pic(next), eps: pic(eps) },
			metrics: { t, tNext: last ? 0 : steps[i + 1], k: i + 1, K: steps.length, abar: ab, alpha: a, beta: b, coef, inv, sigma, xt: x[probe], eps: eps[probe], mean: mp, z: zp, x: next[probe] }
		};
		x = next;
	}
}

const PHASES = {
	noise: { label: L('Start from pure noise', '从纯噪声出发'), kind: 'input' as const },
	step: {
		label: L('One denoising step', '去噪一步'),
		kind: 'update' as const,
		formula: {
			tex: '\\begin{aligned} x_{t-1} &= \\tfrac{1}{\\sqrt{\\alpha_t}}\\Big(x_t - \\tfrac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\,\\hat\\varepsilon\\Big) + \\sigma_t z \\\\ &= {{inv}} \\times \\big({{xt}} \\\\ &\\qquad - {{coef}} \\times {{eps}}\\big) \\\\ &\\quad + {{sigma}} \\times {{z}} \\\\ &= {{x}} \\end{aligned}',
			values: (f: Frame) => f.metrics ?? {}
		}
	}
};

export const sample: ProcessDef<ModelIn | null, SOpt, PicState> = {
	id: 'sample',
	kind: 'inference',
	title: L('DDPM sampling: noise to a drawing', 'DDPM 采样：从噪声到一幅画'),
	recordDraws: true,
	phases: PHASES,
	schema: { x: { ...PIC, label: L('The picture x_t', '图 x_t'), tol: 1e-3 }, eps: { ...PIC, label: L('Predicted noise ε̂', '预测的噪声 ε̂'), tol: 1e-3 } },
	defaults: { stride: 1, probe: PROBE, n: SEEDS },
	metricTol: 1e-3,
	*run(input, ctx): ProcessRun<PicState, SOpt> {
		if (!input) return;
		yield* chain(input, ctx.rng, Math.max(1, Math.floor(ctx.opt.stride)), ctx.opt.probe, ctx.signal);
	}
};

export const gallery: ProcessDef<ModelIn | null, SOpt, PicState> = {
	id: 'gallery',
	kind: 'inference',
	title: L('Twelve seeds, twelve drawings', '十二个种子，十二幅画'),
	recordDraws: true,
	phases: { film: { label: L('One seed’s run at t = 100, 80, …, 0', '一个种子的采样在 t = 100, 80, …, 0 时'), kind: 'output' } },
	schema: { x: { ...PIC, label: L('The picture x_t', '图 x_t'), tol: 1e-3 } },
	defaults: { stride: 1, probe: PROBE, n: SEEDS },
	*run(input, ctx): ProcessRun<PicState, SOpt> {
		if (!input) return;
		for (let s = 0; s < ctx.opt.n && !ctx.signal.cancelled; s++) {
			let j = 0;
			for (const f of chain({ ...input, seed: s }, ctx.rng, 1, ctx.opt.probe, ctx.signal)) {
				if (j % FILM === 0) yield { phase: 'film', step: s, sub: j / FILM, state: { x: f.state.x }, metrics: { seed: s, t: T - j } };
				j++;
			}
		}
	}
};

export const forward: ProcessDef<{ x0: ArrayLike<number>; eps: ArrayLike<number> } | null, { probe: number }, PicState> = {
	id: 'forward',
	kind: 'algorithm',
	title: L('Noising your drawing, t = 0 … T', '给你的画加噪，t = 0 … T'),
	phases: { noisy: { label: L('Your drawing at step t', '第 t 步的画'), kind: 'forward' } },
	schema: { x: { ...PIC, label: L('The picture x_t', '图 x_t'), tol: 1e-12 } },
	defaults: { probe: PROBE },
	*run(input, ctx): ProcessRun<PicState, { probe: number }> {
		if (!input) return;
		const p = ctx.opt.probe;
		for (let t = 0; t <= T; t++) {
			const ab = t ? SCHED.abar[t - 1] : 1, a = Math.sqrt(ab), b = Math.sqrt(1 - ab);
			const x = Float64Array.from({ length: N }, (_, j) => a * (2 * input.x0[j] - 1) + b * input.eps[j]);
			yield { phase: 'noisy', step: t, state: { x: { shape: [SIDE, SIDE], data: x } }, metrics: { t, abar: ab, a, b, x0: input.x0[p], eps: input.eps[p], x: x[p] } };
		}
	}
};
