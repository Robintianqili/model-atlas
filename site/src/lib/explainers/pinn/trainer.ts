// pinn: trainers 'plain' and 'pinn' (kind B). The same network, 1 input (t) -> 32 tanh -> 32 tanh -> 1 output (u): 1,153
// weights and biases, Adam lr 0.01, 1,000 steps, a snapshot every 20. Same seed = same start: model() gets the run's
// rng.fork('init'), so 'plain' and 'pinn' begin with identical weights (claims.test.ts checks it).
//   plain: loss = data                                  (the points only)
//   pinn:  loss = data + λ·physics + start              (the law on 200 times, and "let go from rest": u′(0) = 0)
// data = mean (u(tᵢ) − uᵢ)² over the points; physics = mean ((u″ + c·u′ + k·u) / k)² over the 200 times, with u′ and u″ by
// autograd (grad twice, createGraph); start = u′(0)² / k. Options: lam (λ), c, k, start (weight), physics (computed at all).
// Twin: pipeline/pinn/twin.py (torch.autograd.grad(create_graph=True)).   npm run atlas -- run pinn pinn --frames 0..20
import type { Tensor, TrainOpt, TrainerDef } from '../../atlas/types/index.ts';
import { MLP, loadParams } from '../../atlas/autograd/nn.ts';
import { mse } from '../../atlas/autograd/loss.ts';
import { grad, input, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { C, K, N, SPLIT, T, TRUTH, curves, derivs, gap, pointsOf, residual, valuesAt } from './osc.ts';

export type PinnOpt = TrainOpt & { hidden: number; lam: number; start: number; physics: boolean; c: number; k: number };
export type PinnBatch = { to: Tensor; uo: Tensor; tc: Tensor };

const TC: Tensor = { shape: [N, 1], data: T };
const batches = new WeakMap<object, PinnBatch>();
function batchOf(raw: unknown): PinnBatch {
	const hit = raw && typeof raw === 'object' ? batches.get(raw) : undefined;
	if (hit) return hit;
	const p = pointsOf(raw);
	const b = { to: { shape: [p.n, 1], data: p.t }, uo: { shape: [p.n, 1], data: p.u }, tc: TC };
	if (raw && typeof raw === 'object') batches.set(raw, b);
	return b;
}

export const netOf = (opt: { hidden?: number }, rng = createRng(0)) => new MLP('net', [1, opt.hidden ?? 32, opt.hidden ?? 32, 1], 'tanh', rng);

/** the network of a recorded state (its params), e.g. for the derived curves */
const nets = new WeakMap<object, MLP>();
export function netAt(params: Record<string, Tensor>, opt: { hidden?: number }): MLP {
	let m = nets.get(params);
	if (!m) {
		m = netOf(opt);
		loadParams(m, params);
		nets.set(params, m);
	}
	return m;
}

type S = { params?: Record<string, Tensor> };
const EMPTY: Tensor = { shape: [4, 0], data: new Float64Array(0) };
const memo = new WeakMap<object, Tensor>();
/** u, u′, u″ and the residual of a state's weights on the 200 times, against the run's own law (opt c, k) */
export function curvesAt(s: S, opt?: Partial<PinnOpt>): Tensor {
	if (!s.params) return EMPTY;
	let cv = memo.get(s.params);
	if (!cv) memo.set(s.params, (cv = curves(netAt(s.params, opt ?? {}), opt?.c ?? C, opt?.k ?? K)));
	return cv;
}

/** rows (t, |u(t) − truth(t)|) on the 200 times: the gap to the true curve */
export function gapRows(cv: Tensor): Tensor {
	if (cv.data.length !== 4 * N) return { shape: [0, 2], data: new Float64Array(0) };
	const out = new Float64Array(2 * N);
	for (let i = 0; i < N; i++) (out[2 * i] = T[i]), (out[2 * i + 1] = Math.abs(cv.data[i] - TRUTH[i]));
	return { shape: [N, 2], data: out };
}

const defaults: PinnOpt = { seed: 1, lr: 0.01, batch: 0, steps: 1000, snapshotEvery: 20, optimizer: 'adam', hidden: 32, lam: 1, start: 1, physics: true, c: C, k: K };

function def(id: string, title: { en: string; zh: string }, extra: Partial<PinnOpt>): TrainerDef<unknown, MLP, PinnBatch, PinnOpt> {
	return {
		id,
		title,
		defaults: { ...defaults, ...extra },
		model: (rng, opt) => netOf(opt, rng),
		batch: (raw) => batchOf(raw),
		loss(m, b, rec, opt) {
			// data: 0 (still tied to the weights, so backward runs) when there are no points
			const data = b.to.shape[0] ? mse(m.forward(input(b.to), (id, v) => rec('obs.' + id, v)), input(b.uo)) : op.scale(op.sum(m.layers[0].weight), 0);
			const parts: Record<string, Var> = { data };
			let loss = data;
			if (opt.physics) {
				parts.physics = op.mean(op.square(residual(derivs(m, b.tc.data), opt.c, opt.k)));
				loss = op.add(loss, op.scale(parts.physics, opt.lam));
			}
			if (opt.start > 0) {
				const t0 = input({ shape: [1, 1], data: [0] }, true);
				parts.start = op.scale(op.sum(op.square(grad(m.forward(t0), t0, { createGraph: true }))), 1 / opt.k);
				loss = op.add(loss, op.scale(parts.start, opt.start));
			}
			return { loss, parts };
		},
		derived: {
			curves: { kind: 'tensor', label: { en: 'u, u′, u″ and the residual on 200 times', zh: '200 个时刻上的 u、u′、u″ 和残差' }, shape: [4, N], role: 'output', derive: (s: S, o?: unknown) => curvesAt(s, o as Partial<PinnOpt>) },
			gap: { kind: 'tensor', label: { en: 'The gap to the true curve on 200 times', zh: '200 个时刻上与真实曲线的差距' }, shape: [N, 2], role: 'output', derive: (s: S, o?: unknown) => gapRows(curvesAt(s, o as Partial<PinnOpt>)) }
		},
		// mean gap to the true curve before and after t = 0.4, on the 200 times
		metrics(m) {
			const u = valuesAt(m, T);
			return { errIn: gap(u, -1, SPLIT), err: gap(u, SPLIT, 1) };
		}
	};
}

export const pinn = def('pinn', { en: 'Train the PINN: points + the law', zh: '训练 PINN：数据点 + 物理定律' }, {});
export const plain = def('plain', { en: 'Train the plain network: points only', zh: '训练普通网络：只用数据点' }, { lam: 0, start: 0, physics: false });
// the λ task and the wrong law: the same loss with other defaults, one trainer each, so `atlas parity` runs the twin on
// each one's own options (it runs one twin per trainer and records it for every source of that trainer). λ = 100 stays
// a 'pinn' source with opt { lam: 100 }: the parity gate's extra plain-SGD run diverges at lr 0.01 with λ = 100 (notes)
export const pinnLam0 = def('pinnLam0', { en: 'Train the PINN with λ = 0', zh: '训练 λ = 0 的 PINN' }, { lam: 0 });
export const pinnLam001 = def('pinnLam001', { en: 'Train the PINN with λ = 0.01', zh: '训练 λ = 0.01 的 PINN' }, { lam: 0.01 });
export const pinnWrongC = def('pinnWrongC', { en: 'Train the PINN with a wrong law: c = 0', zh: '用错误的定律训练 PINN：c = 0' }, { c: 0 });
