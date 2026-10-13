// lora: the processes (kind C). Every loss and gradient comes from the kit's autograd (atlas/autograd) and every update
// from its Adam (PyTorch's formula); the world (W0, the needed change, the inputs) is data.ts.
//   lora   LoRA on the toy layer at the learner's rank r: W = W0 + (α/r)·B·A with W0 frozen, A (r × 16) drawn uniform in
//          ±1/√16 (PyTorch's default for a 16-input layer, as LoRA's code does), B (16 × r) all zeros. Frame 'start'
//          (step 0: B = 0, so the layer is exactly the base layer), then one 'step' frame per Adam update (its `step` =
//          updates done). State: A, B and the change dW = (α/r)·B·A; metrics: the loss, the gradient sizes at the
//          frame's weights, the largest change of A and B in the update just made, and the analytic floor.
//   ranks  the same training at r = 1, 2, 4 and 8, and full fine-tuning (all 256 numbers of the change trained), in
//          lockstep; the losses as metrics l1, l2, l4, l8 and lfull. Its input is ignored, so a new rank keeps its run.
// Each run draws A from ctx.rng.fork('r<r>'), so the lora run at rank 2 and the ranks run's l2 are the same run.
//   npm run atlas -- run lora fit --frames 0..5
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { backward, input, param, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { Adam } from '../../atlas/autograd/optim.ts';
import { ALPHA, D, N, WORLD, floorOf, loraCount, rankOf } from './data.ts';

export type LoraOpt = { steps: number; lr: number; alpha: number };
export type LoraState = { A: Tensor; B: Tensor; dW: Tensor };

const X = () => input({ shape: [N, D], data: WORLD.X });
const Y = () => input({ shape: [N, D], data: WORLD.Y });
/** the frozen layer's outputs X·W0ᵀ: a constant, computed once per run */
const baseOut = () => op.linear(X(), input({ shape: [D, D], data: WORLD.W0 }));

const norm = (g: Var | null) => (g ? Math.sqrt(g.data.reduce((a, v) => a + v * v, 0)) : 0);
const maxDiff = (a: Float64Array, b: Float64Array) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

/** (α/r)·B·A as a [D, D] tensor */
function deltaOf(A: Float64Array, B: Float64Array, r: number, scale: number): Float64Array {
	const out = new Float64Array(D * D);
	for (let o = 0; o < D; o++)
		for (let i = 0; i < D; i++) {
			let s = 0;
			for (let k = 0; k < r; k++) s += B[o * r + k] * A[k * D + i];
			out[o * D + i] = scale * s;
		}
	return out;
}

interface Snap {
	step: number;
	loss: number;
	gA: number;
	gB: number;
	dA: number;
	dB: number;
	A: Float64Array;
	B: Float64Array;
	dW: Float64Array;
}

/** One training run: r adapters (or r = 'full': the whole change trained), one snapshot per update (step 0 first). */
function* fitRun(r: number | 'full', o: LoraOpt, next: () => number): Generator<Snap> {
	const x = X();
	const y = Y();
	const base = baseOut();
	const full = r === 'full';
	const k = full ? 0 : r;
	const scale = full ? 1 : o.alpha / k;
	const bound = 1 / Math.sqrt(D);
	const A = full ? null : param('lora_A', { shape: [k, D], data: Array.from({ length: k * D }, () => (2 * next() - 1) * bound) });
	const B = full ? null : param('lora_B', { shape: [D, k], data: new Float64Array(D * k) });
	const dW = full ? param('delta', { shape: [D, D], data: new Float64Array(D * D) }) : null;
	const params: Record<string, Var> = full ? { delta: dW! } : { lora_A: A!, lora_B: B! };
	const adam = new Adam(params, { lr: o.lr });
	const forward = () => {
		const change = full ? op.linear(x, dW!) : op.scale(op.linear(op.linear(x, A!), B!), scale);
		return op.mean(op.square(op.sub(op.add(base, change), y)));
	};
	const grads = () => {
		for (const p of Object.values(params)) p.grad = null;
		const L = forward();
		backward(L);
		return L.item();
	};
	const snap = (step: number, loss: number, dA: number, dB: number): Snap => {
		const a = A ? A.data.slice() : new Float64Array(0);
		const b = B ? B.data.slice() : new Float64Array(0);
		return { step, loss, gA: norm(A?.grad ?? null), gB: norm(full ? dW!.grad : B!.grad), dA, dB, A: a, B: b, dW: full ? dW!.data.slice() : deltaOf(a, b, k, scale) };
	};
	let loss = grads();
	yield snap(0, loss, 0, 0);
	for (let s = 1; s <= o.steps; s++) {
		const a0 = A ? A.data.slice() : null;
		const b0 = (full ? dW! : B!).data.slice();
		adam.step();
		const dA = a0 ? maxDiff(A!.data, a0) : 0;
		const dB = maxDiff((full ? dW! : B!).data, b0);
		loss = grads();
		yield snap(s, loss, dA, dB);
	}
}

const PH = {
	start: {
		label: { en: 'Before training: B = 0', zh: '训练之前：B = 0' },
		kind: 'input' as const,
		formula: {
			tex: 'W = W_0 + \\tfrac{\\alpha}{r}\\,B A = W_0 + \\tfrac{{{alpha}}}{{{r}}} \\cdot 0 \\cdot A = W_0',
			values: (f: { metrics?: Record<string, number> }) => ({ alpha: f.metrics!.alpha, r: f.metrics!.r })
		},
		narration: { what: { en: 'B starts at zero, so W = W0 and the loss is the base layer\'s, {frame.loss}.', zh: 'B 从零开始，所以 W = W0，损失就是基础层的损失 {frame.loss}。' } }
	},
	step: {
		label: { en: 'One Adam update of A and B', zh: 'A 和 B 的一次 Adam 更新' },
		kind: 'update' as const,
		formula: {
			tex: 'L = \\tfrac{1}{64 \\cdot 16} \\sum_{n,o} \\big( (W_0 + \\tfrac{\\alpha}{r} B A)\\,x_n - y_n \\big)_o^2 = {{loss}}',
			values: (f: { metrics?: Record<string, number> }) => ({ loss: f.metrics!.loss })
		},
		narration: { what: { en: 'After {frame.step} updates the loss is {frame.loss}.', zh: '更新 {frame.step} 次之后，损失是 {frame.loss}。' } }
	}
};

export const lora: ProcessDef<unknown, LoraOpt, LoraState> = {
	id: 'lora',
	kind: 'algorithm',
	title: { en: 'LoRA on one layer', zh: '在一个层上做 LoRA' },
	phases: PH,
	schema: {
		A: { kind: 'tensor', label: { en: 'A (r × 16), trainable', zh: 'A（r × 16），可训练' }, shape: ['r', D], role: 'param', scale: 'diverging', tol: 1e-7 },
		B: { kind: 'tensor', label: { en: 'B (16 × r), trainable, starts at 0', zh: 'B（16 × r），可训练，从 0 开始' }, shape: [D, 'r'], role: 'param', scale: 'diverging', tol: 1e-7 },
		dW: { kind: 'tensor', label: { en: 'The change (α/r)·B·A', zh: '改变量 (α/r)·B·A' }, shape: [D, D], role: 'param', scale: 'diverging', tol: 1e-7 }
	},
	defaults: { steps: 100, lr: 0.02, alpha: ALPHA },
	metricTol: 1e-7,
	*run(raw, ctx): ProcessRun<LoraState, LoraOpt> {
		const o = { ...lora.defaults, ...ctx.opt };
		const r = rankOf(raw);
		const fr = ctx.rng.fork(`r${r}`);
		const fixed = { r, alpha: o.alpha, scale: o.alpha / r, lr: o.lr, floor: floorOf(r), base: floorOf(0), trainable: loraCount(r, D, D), full: D * D };
		for (const s of fitRun(r, o, () => fr.next())) {
			const state = { A: { shape: [r, D], data: s.A }, B: { shape: [D, r], data: s.B }, dW: { shape: [D, D], data: s.dW } };
			const normB = Math.sqrt(s.B.reduce((a, v) => a + v * v, 0));
			yield { phase: s.step === 0 ? 'start' : 'step', step: s.step, state, metrics: { step: s.step, loss: s.loss, gA: s.gA, gB: s.gB, dA: s.dA, dB: s.dB, normB, ...fixed } };
		}
	}
};

/** the ranks the comparison trains, next to full fine-tuning */
export const SWEEP = [1, 2, 4, 8];

export const ranks: ProcessDef<unknown, LoraOpt, Record<string, never>> = {
	id: 'ranks',
	kind: 'algorithm',
	title: { en: 'LoRA at four ranks and full fine-tuning', zh: '四种秩的 LoRA 与全量微调' },
	phases: {
		start: { label: { en: 'Before training', zh: '训练之前' }, kind: 'input', narration: { what: { en: 'Every run starts at the base layer\'s loss, {frame.l1}.', zh: '每次运行都从基础层的损失 {frame.l1} 出发。' } } },
		step: { label: { en: 'Five runs, one update each', zh: '五次运行，各更新一次' }, kind: 'update', narration: { what: { en: 'After {frame.step} updates: rank 1 {frame.l1}, rank 2 {frame.l2}, full fine-tuning {frame.lfull}.', zh: '更新 {frame.step} 次之后：秩 1 为 {frame.l1}，秩 2 为 {frame.l2}，全量微调为 {frame.lfull}。' } } }
	},
	schema: {},
	defaults: { steps: 100, lr: 0.02, alpha: ALPHA },
	metricTol: 1e-7,
	*run(_raw, ctx): ProcessRun<Record<string, never>, LoraOpt> {
		const o = { ...ranks.defaults, ...ctx.opt };
		const runs = SWEEP.map((r) => {
			const fr = ctx.rng.fork(`r${r}`);
			return fitRun(r, o, () => fr.next());
		});
		const fullRun = fitRun('full', o, () => 0);
		for (let f = fullRun.next(); !f.done; f = fullRun.next()) {
			const m: Record<string, number> = { step: f.value.step, lfull: f.value.loss };
			runs.forEach((g, i) => {
				const s = g.next();
				m[`l${SWEEP[i]}`] = s.done ? NaN : s.value.loss;
			});
			yield { phase: f.value.step === 0 ? 'start' : 'step', step: f.value.step, state: {}, metrics: m };
		}
	}
};
