// Euler integration of particles through a velocity field (flow-2d sampling, design §7):
//   x_{k+1} = x_k + h · v(x_k, t_k),  h = (t1 − t0) / steps,  t_k = t0 + k·h.
// Coarse: 'start', then one 'move' frame per step. Fine: 'velocity' (v at x_k) before each 'move'.
import type { I18n, Tensor } from '../../types/core.ts';
import type { Control, Frame, ProcessDef, RunCtx } from '../../types/process.ts';

export type EulerOpt = { steps: number; t0: number; t1: number };
export type EulerInput = { points: Tensor };
export type EulerState = { x: Tensor; v?: Tensor; t: number; k: number };

const L = (en: string, zh: string): I18n => ({ en, zh });

export function eulerProcess(def: {
	id: string;
	title?: I18n;
	/** velocities [N, D] for positions x [N, D] (row-major) at time t */
	field: (x: Float64Array, t: number, n: number, d: number) => ArrayLike<number>;
	defaults?: Partial<EulerOpt>;
}): ProcessDef<EulerInput, EulerOpt, EulerState> {
	return {
		id: def.id,
		kind: 'algorithm',
		title: def.title ?? L('Follow the arrows in small steps', '沿着箭头一小步一小步地走'),
		phases: {
			start: { label: L('Start from noise', '从噪声出发'), kind: 'input' },
			velocity: { label: L('Read the arrow at each point', '读出每个点上的箭头'), kind: 'forward' },
			move: {
				label: L('Take one step', '走一步'),
				kind: 'update',
				formula: { tex: 'x_{k+1} = x_k + h\\,v(x_k, t_k),\\quad h = {{h}}', values: (f) => ({ h: (f.metrics?.h as number) ?? NaN }) }
			}
		},
		schema: {
			x: { kind: 'tensor', label: L('Positions', '位置'), shape: ['N', 'D'], role: 'data', tol: 1e-9 },
			v: { kind: 'tensor', label: L('Velocities', '速度'), shape: ['N', 'D'], role: 'activation', tol: 1e-9 },
			t: { kind: 'scalar', label: L('Time', '时间'), tol: 1e-12 },
			k: { kind: 'int', label: L('Step', '步数'), tol: 'exact' }
		},
		defaults: { steps: 50, t0: 0, t1: 1, ...(def.defaults ?? {}) },
		*run(input: EulerInput, ctx: RunCtx<EulerOpt>): Generator<Frame<EulerState>, void, Control<EulerOpt> | undefined> {
			const [n, d] = input.points.shape.length === 2 ? input.points.shape : [input.points.data.length / 2, 2];
			const x = Float64Array.from(input.points.data);
			const K = Math.max(1, Math.floor(ctx.opt.steps));
			const h = (ctx.opt.t1 - ctx.opt.t0) / K;
			yield { phase: 'start', state: { x: { shape: [n, d], data: x.slice() }, t: ctx.opt.t0, k: 0 }, metrics: { t: ctx.opt.t0, h } };
			for (let k = 0; k < K && !ctx.signal.cancelled; k++) {
				const t = ctx.opt.t0 + k * h;
				const v = Float64Array.from(def.field(x, t, n, d));
				if (v.length !== n * d) throw new Error(`${def.id}: field returned ${v.length} numbers for ${n}×${d} points`);
				if (ctx.detail === 'fine') yield { phase: 'velocity', step: k, state: { v: { shape: [n, d], data: v.slice() }, t, k }, metrics: { t, h } };
				let speed = 0;
				for (let i = 0; i < n; i++) {
					let s2 = 0;
					for (let j = 0; j < d; j++) {
						x[i * d + j] += h * v[i * d + j];
						s2 += v[i * d + j] * v[i * d + j];
					}
					speed += Math.sqrt(s2);
				}
				yield { phase: 'move', step: k, state: { x: { shape: [n, d], data: x.slice() }, t: t + h, k: k + 1 }, metrics: { t: t + h, h, meanSpeed: n ? speed / n : 0 } };
			}
		}
	};
}
