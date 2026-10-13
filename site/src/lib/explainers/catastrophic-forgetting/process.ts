// catastrophic-forgetting: the processes (kind C; every number from the kit's autograd and plain gradient descent).
//   sequence  one 2-16-16-2 tanh network (opt.hidden) trained with full-batch gradient descent on task A for opt.stepsA
//             updates, then on task B for opt.stepsB updates with the learner's settings (data.ts Settings; a source
//             whose input is null takes them from its opt): replay mixes stored task-A points into B's batch, pull adds
//             λ/2 · Σ (w − w_A)² to B's loss, freeze skips the first layer's update, slow divides the learning rate by 10.
//             Frames every opt.every updates: 'start', 'learnA', 'switch' (the end of task A, with what B's first update
//             does to both tasks), 'learnB'. state.grid is the network's share for class 1 over the plane (the plane2d
//             view's background); metrics hold both tasks' accuracy and loss and, during B, one weight's update.
//   remedies  task A once, then six copies of that network trained on B in lockstep: plain, replay 10%, pull λ = 0.3 and
//             λ = 0.6, frozen first layer, learning rate / 10. One frame every opt.every B updates (step = B updates so far),
//             with accA_<id> and accB_<id> per copy.
//   npm run atlas -- run catastrophic-forgetting plain --frames 28..34
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { MLP } from '../../atlas/autograd/nn.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { backward, input, noGrad, zeroGrad, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { accuracy, decisionGrid } from '../../atlas/train/metrics.ts';
import { createRng } from '../../atlas/core/rng.ts';
import { TASK_A, TASK_B, replayCount, settingsOf, type Settings, type Task } from './data.ts';

export type SeqOpt = { stepsA: number; stepsB: number; every: number; lr: number; hidden: number; res: number; seed: number } & Partial<Settings>;
export type SeqState = { grid: Tensor };

export const BOUNDS = { x0: -1.25, x1: 1.25, y0: -1.25, y1: 1.25 };
/** The remedies the comparison runs, in order. */
export const REMEDIES: { id: string; set: Settings }[] = [
	{ id: 'plain', set: { replay: 0, pull: 0, freeze: false, slow: false } },
	{ id: 'replay', set: { replay: 0.1, pull: 0, freeze: false, slow: false } },
	{ id: 'pullWeak', set: { replay: 0, pull: 0.3, freeze: false, slow: false } },
	{ id: 'pull', set: { replay: 0, pull: 0.6, freeze: false, slow: false } },
	{ id: 'freeze', set: { replay: 0, pull: 0, freeze: true, slow: false } },
	{ id: 'slow', set: { replay: 0, pull: 0, freeze: false, slow: true } }
];

const tensorsOf = (t: Task) => ({ x: input({ shape: [t.n, 2], data: t.xy }), y: t.label });

/** Weights and biases of the 2-h-h-2 network. */
export const paramsOf = (h: number): number => 2 * h + h + h * h + h + h * 2 + 2;

/** Task B's points plus the first k task-A points (the replay memory). */
export function batchOf(k: number): Task {
	if (k <= 0) return TASK_B;
	const n = TASK_B.n + k;
	const xy = new Float64Array(2 * n);
	const label = new Int32Array(n);
	xy.set(TASK_B.xy);
	label.set(TASK_B.label);
	xy.set(TASK_A.xy.subarray(0, 2 * k), 2 * TASK_B.n);
	label.set(TASK_A.label.subarray(0, k), TASK_B.n);
	return { xy, label, n };
}

/** One network with its training rule (exported for the claim tests). */
export class Learner {
	readonly m: MLP;
	readonly ps: Var[];
	anchor: Float64Array[] | null = null;
	constructor(hidden: number, seed: number) {
		this.m = new MLP('net', [2, hidden, hidden, 2], 'tanh', createRng(seed));
		this.ps = this.m.params();
	}
	clone(from: Learner): this {
		this.ps.forEach((p, j) => p.data.set(from.ps[j].data));
		return this;
	}
	measure(t: Task): { acc: number; loss: number } {
		return noGrad(() => {
			const { x, y } = tensorsOf(t);
			const z = this.m.forward(x);
			return { acc: accuracy(z, y), loss: crossEntropy(z, y).item() };
		});
	}
	/** Fill every parameter's grad with the gradient of t's loss (+ the pull, when there is an anchor and λ > 0). */
	grad(t: Task, pull = 0): number {
		zeroGrad(this.ps);
		const { x, y } = tensorsOf(t);
		let loss = crossEntropy(this.m.forward(x), y);
		const anchor = this.anchor;
		if (anchor && pull > 0) {
			const parts = this.ps.map((p, j) => op.sum(op.square(op.sub(p, input({ shape: p.shape, data: anchor[j] })))));
			loss = op.add(loss, op.scale(parts.reduce((a, b) => op.add(a, b)), pull / 2));
		}
		backward(loss);
		return loss.item();
	}
	/** w ← w − lr · grad, except the first layer (net.0.weight, net.0.bias) when frozen. */
	update(lr: number, freeze: boolean): void {
		this.ps.forEach((p, j) => {
			if (freeze && j < 2) return;
			const w = p.data;
			const g = p.grad!.data;
			for (let i = 0; i < w.length; i++) w[i] = w[i] - lr * g[i];
		});
	}
	grid(res: number): Tensor {
		return decisionGrid((x) => this.m.forward(x), BOUNDS, res);
	}
	/** The gradient as one flat vector (after grad()). */
	flatGrad(): Float64Array {
		return Float64Array.from(this.ps.flatMap((p) => Array.from(p.grad!.data)));
	}
	/** The weight that has moved furthest from w_A (ties: the first), as [parameter index, element index]. */
	furthest(): [number, number] {
		const a = this.anchor!;
		let pick: [number, number] = [0, 0];
		let far = -1;
		this.ps.forEach((p, j) =>
			p.data.forEach((v, i) => {
				const d = Math.abs(v - a[j][i]);
				if (d > far) {
					far = d;
					pick = [j, i];
				}
			})
		);
		return pick;
	}
}

const dot = (a: Float64Array, b: Float64Array) => a.reduce((s, v, i) => s + v * b[i], 0);

function scores(L: Learner): Record<string, number> {
	const a = L.measure(TASK_A);
	const b = L.measure(TASK_B);
	return { accA: a.acc, lossA: a.loss, accB: b.acc, lossB: b.loss };
}

export const sequence: ProcessDef<unknown, SeqOpt, SeqState> = {
	id: 'sequence',
	kind: 'algorithm',
	title: { en: 'Task A, then task B', zh: '先学任务 A，再学任务 B' },
	phases: {
		start: { label: { en: 'Untrained', zh: '未训练' }, kind: 'input' },
		learnA: { label: { en: 'Training on task A', zh: '在任务 A 上训练' }, kind: 'update' },
		switch: {
			label: { en: 'Task A done; B begins', zh: '任务 A 学完，开始任务 B' },
			kind: 'backward',
			formula: {
				tex: '\\cos(g_A, g_B) = \\frac{g_A \\cdot g_B}{\\lVert g_A \\rVert \\, \\lVert g_B \\rVert} = {{cos}}',
				values: (f) => ({ cos: f.metrics!.cos })
			}
		},
		learnB: {
			label: { en: 'Training on task B', zh: '在任务 B 上训练' },
			kind: 'update',
			formula: {
				// short lines (a phone is narrow): the slope with the pull, its value, then the update
				tex: '\\begin{aligned} & g_B + \\lambda (w - w_A) \\\\ &= {{g}} + {{lam}} \\times ({{d}}) \\\\ &= {{gt}} \\\\ w &\\leftarrow {{w}} - {{lr}} \\times ({{gt}}) \\\\ &= {{w1}} \\end{aligned}',
				values: (f) => {
					const m = f.metrics!;
					return { g: m.pg, lam: String(m.pull), d: m.pw - m.pa, gt: m.pg + m.pp, w: m.pw, lr: String(m.lrB), w1: m.pw1 };
				}
			}
		}
	},
	schema: {
		grid: { kind: 'tensor', label: { en: "The network's share for class 1 over the plane", zh: '网络在平面各处给第 1 类的比例' }, shape: ['r', 'r'], scale: 'sequential', role: 'output', tol: 1e-9 }
	},
	defaults: { stepsA: 300, stepsB: 300, every: 10, lr: 0.5, hidden: 16, res: 32, seed: 7 },
	*run(raw, ctx): ProcessRun<SeqState, SeqOpt> {
		const o = { ...sequence.defaults, ...ctx.opt };
		const set = settingsOf(raw ?? o);
		const k = replayCount(set.replay);
		const lrB = set.slow ? o.lr / 10 : o.lr;
		const knobs = { replay: set.replay, k, pull: set.pull, freeze: set.freeze ? 1 : 0, slow: set.slow ? 1 : 0, lr: o.lr, lrB, stepsA: o.stepsA, stepsB: o.stepsB, every: o.every };
		const L = new Learner(o.hidden, o.seed);
		yield { phase: 'start', step: 0, state: { grid: L.grid(o.res) }, metrics: { step: 0, task: 0, ...knobs, ...scores(L) } };
		for (let s = 1; s <= o.stepsA; s++) {
			L.grad(TASK_A);
			L.update(o.lr, false);
			if (s % o.every === 0 && s < o.stepsA) yield { phase: 'learnA', step: s, state: { grid: L.grid(o.res) }, metrics: { step: s, task: 0, ...knobs, ...scores(L) } };
		}
		// the end of task A: the anchor w_A, and what B's first update does (the cosine of the two tasks' slopes, and both
		// tasks before and after that update, which is this run's real first update on B)
		L.anchor = L.ps.map((p) => Float64Array.from(p.data));
		const before = scores(L);
		const grid = L.grid(o.res);
		L.grad(TASK_A);
		const gA = L.flatGrad();
		L.grad(TASK_B);
		const gB = L.flatGrad();
		const cos = dot(gA, gB) / Math.sqrt(dot(gA, gA) * dot(gB, gB));
		const batch = batchOf(k);
		L.grad(batch, set.pull);
		L.update(lrB, set.freeze);
		const after = scores(L);
		yield {
			phase: 'switch',
			step: o.stepsA,
			state: { grid },
			metrics: { step: o.stepsA, stepB: 0, task: 1, ...knobs, ...before, cos, lossA1: after.lossA, lossB1: after.lossB, accA1: after.accA, accB1: after.accB }
		};
		for (let s = 2; s <= o.stepsB; s++) {
			const shown = s % o.every === 0;
			// on a shown update: the weight that has moved furthest from w_A, its value, its anchor and B's slope on it
			const [j, i] = shown ? L.furthest() : [0, 0];
			const pw = L.ps[j].data[i];
			const pa = L.anchor[j][i];
			let pg = NaN;
			if (shown && set.pull > 0) {
				L.grad(batch, 0);
				pg = L.ps[j].grad!.data[i];
			}
			L.grad(batch, set.pull);
			if (shown && !(set.pull > 0)) pg = L.ps[j].grad!.data[i];
			L.update(lrB, set.freeze);
			if (!shown) continue;
			yield {
				phase: 'learnB',
				step: o.stepsA + s,
				state: { grid: L.grid(o.res) },
				metrics: { step: o.stepsA + s, stepB: s, task: 1, ...knobs, ...scores(L), pw, pa, pg, pp: set.pull * (pw - pa), pw1: L.ps[j].data[i] }
			};
		}
	}
};

export type RemOpt = Omit<SeqOpt, keyof Settings>;

export const remedies: ProcessDef<unknown, RemOpt, Record<string, never>> = {
	id: 'remedies',
	kind: 'algorithm',
	title: { en: 'Six ways to train on task B', zh: '在任务 B 上训练的六种做法' },
	phases: {
		learnB: { label: { en: 'Training on task B', zh: '在任务 B 上训练' }, kind: 'update' }
	},
	schema: {},
	defaults: { stepsA: 300, stepsB: 300, every: 10, lr: 0.5, hidden: 16, res: 32, seed: 7 },
	*run(_raw, ctx): ProcessRun<Record<string, never>, RemOpt> {
		const o = { ...remedies.defaults, ...ctx.opt };
		const base = new Learner(o.hidden, o.seed);
		for (let s = 1; s <= o.stepsA; s++) {
			base.grad(TASK_A);
			base.update(o.lr, false);
		}
		const runs = REMEDIES.map((r) => {
			const L = new Learner(o.hidden, o.seed).clone(base);
			L.anchor = L.ps.map((p) => Float64Array.from(p.data));
			return { ...r, L, batch: batchOf(replayCount(r.set.replay)), lr: r.set.slow ? o.lr / 10 : o.lr };
		});
		const frame = (s: number) => {
			const m: Record<string, number> = { step: s };
			for (const r of runs) {
				m[`accA_${r.id}`] = r.L.measure(TASK_A).acc;
				m[`accB_${r.id}`] = r.L.measure(TASK_B).acc;
			}
			return { phase: 'learnB', step: s, state: {}, metrics: m };
		};
		yield frame(0);
		for (let s = 1; s <= o.stepsB; s++) {
			for (const r of runs) {
				r.L.grad(r.batch, r.set.pull);
				r.L.update(r.lr, r.set.freeze);
			}
			if (s % o.every === 0) yield frame(s);
		}
	}
};
