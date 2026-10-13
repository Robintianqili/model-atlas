// trainingProcess (design §4.5, W1-B1): turns a TrainerDef into a ProcessDef that records a training tape.
//
// Step numbering: frame `step` k is training step k (0-based): its batch, the loss L_k of the weights θ_k on that batch,
// the gradient, and the update θ_k → θ_{k+1}.
// - coarse detail: one frame per step, recorded after the backward pass and before the update. Phase 'snapshot' every
//   snapshotEvery steps (k % snapshotEvery === 0) and at the end: it also carries `params` = θ_k (the weights that
//   produced its loss). Other steps are phase 'step' frames with metrics only (loss on every step). A final frame at
//   k = steps shows θ_steps (no update follows it). So `{ step: 300 }` = the weights after 300 updates.
// - fine detail: per step k, 'forward' (batch, recorded activations) → 'loss' (loss, parts) → 'backward' (grads) →
//   'update' (params = θ_{k+1}, prev = θ_k, the grads again, optimizer moments): 4 frames per step and no final frame.
// - metrics: step, loss, lr (+ gradNorm after a backward pass, loss parts by name, def.metrics() at init and snapshot
//   frames). A part or metric that reuses a reserved name (step, loss, lr, gradNorm, weightDecay, diverged) gets a
//   'part.' or 'metric.' prefix.
// - 'init' frames start every segment: at the start, after { type: 'input' } (new data) and after { type: 'reset' }.
// Controls arrive through gen.next(control) and take effect at the start of the next step:
//   { type: 'set', opt } (lr, momentum, betas, weightDecay act on the optimizer; a new `optimizer` kind rebuilds it),
//   { type: 'input', input } (swaps the data; the weights are kept unless opt.resetOnInput), { type: 'reset', seed? }.
//   'pause' / 'resume' / 'step' belong to the runner.
// Randomness: model() gets ctx.rng.fork('init') and batch() gets one ctx.rng.fork('batch') stream for the whole run,
// so two trainers with the same seed start from the same weights. Draw from the given rng; do not fork it per step.
// Every tensor in a frame is a fresh copy (parameters are updated in place, so references would change later).
import type {
	Control,
	Frame,
	I18n,
	Module,
	Optimizer,
	PhaseDef,
	ProcessDef,
	Rec,
	Rng,
	RunCtx,
	Schema,
	Tensor,
	TrainOpt,
	TrainState,
	TrainerDef
} from '../types/index.ts';
import { Var, backward, noGrad, zeroGrad } from '../autograd/var.ts';
import { makeOptimizer, type OptimizerKind } from '../autograd/optim.ts';
import { createRng } from '../core/rng.ts';
import type { AtlasSpec } from '../types/index.ts';

/** What a 'live-model' source needs from a train process (runtime/runtime.ts): the kind-A spec of the model at a
 *  recorded state. Added at integration 1a; not part of the frozen ProcessDef type. */
export interface LiveSpecFn<O> {
	liveSpec(state: Partial<TrainState>, opt?: Partial<O>): { spec: AtlasSpec; weights: Record<string, Tensor> };
}

export const TRAIN_PHASES: Record<string, PhaseDef> = {
	init: {
		label: { en: 'Start', zh: '开始' },
		kind: 'other',
		narration: { what: { en: 'Training starts from the current weights and data.', zh: '训练从当前的权重和数据开始。' } }
	},
	step: { label: { en: 'Training step', zh: '训练步' }, kind: 'update' },
	snapshot: {
		label: { en: 'Snapshot', zh: '快照' },
		kind: 'observe',
		narration: { what: { en: 'A copy of every weight at this step.', zh: '这一步所有权重的一份副本。' } }
	},
	forward: {
		label: { en: 'Forward pass', zh: '前向计算' },
		kind: 'forward',
		narration: { what: { en: 'The batch goes through the network with the current weights.', zh: '这一批数据用当前的权重通过网络。' } }
	},
	loss: {
		label: { en: 'Loss', zh: '损失' },
		kind: 'loss',
		narration: { what: { en: 'One number says how wrong the predictions are on this batch.', zh: '用一个数衡量这一批的预测错得有多厉害。' } }
	},
	backward: {
		label: { en: 'Backward pass', zh: '反向传播' },
		kind: 'backward',
		narration: { what: { en: 'Backpropagation finds the gradient of the loss for every weight.', zh: '反向传播求出损失对每个权重的梯度。' } }
	},
	update: {
		label: { en: 'Update', zh: '更新' },
		kind: 'update',
		narration: { what: { en: 'The optimizer moves every weight a little, using its gradient.', zh: '优化器根据梯度把每个权重移动一点点。' } }
	}
};

const TRAIN_SCHEMA: Schema = {
	step: { kind: 'int', label: { en: 'Step', zh: '步数' } },
	loss: { kind: 'scalar', label: { en: 'Loss', zh: '损失' }, role: 'output' },
	parts: { kind: 'json', label: { en: 'Loss parts', zh: '损失分项' }, role: 'output' },
	lr: { kind: 'scalar', label: { en: 'Learning rate', zh: '学习率' }, role: 'optimizer' },
	optimizer: { kind: 'text', label: { en: 'Optimizer', zh: '优化器' }, role: 'optimizer' },
	params: { kind: 'tensor', label: { en: 'Weights', zh: '权重' }, role: 'param', scale: 'diverging' },
	prev: { kind: 'tensor', label: { en: 'Weights before the update', zh: '更新前的权重' }, role: 'param', scale: 'diverging' },
	grads: { kind: 'tensor', label: { en: 'Gradients', zh: '梯度' }, role: 'grad', scale: 'diverging' },
	opt: { kind: 'tensor', label: { en: 'Optimizer state (m, v)', zh: '优化器状态（m, v）' }, role: 'optimizer' },
	batch: { kind: 'tensor', label: { en: 'Batch', zh: '这一批数据' }, role: 'data' },
	acts: { kind: 'tensor', label: { en: 'Activations', zh: '激活值' }, role: 'activation' }
};

const CAP_START: I18n = { en: 'Start: the first weights', zh: '开始：最初的权重' };
const CAP_DATA: I18n = { en: 'New data', zh: '换了新数据' };
const CAP_RESET: I18n = { en: 'Restarted from fresh weights', zh: '从新的初始权重重新开始' };
const CAP_DIVERGED: I18n = { en: 'The loss is no longer a finite number: training diverged.', zh: '损失不再是有限的数：训练发散了。' };

const RESERVED = new Set(['step', 'loss', 'lr', 'gradNorm', 'weightDecay', 'diverged']);

function copyT(t: { shape: number[]; data: ArrayLike<number> }): Tensor {
	return { shape: t.shape.slice(), data: Float64Array.from(t.data) };
}

function asVar(v: unknown, what: string): Var {
	if (!(v instanceof Var)) throw new Error(`trainingProcess: ${what} must be an autograd Var (atlas/autograd)`);
	return v;
}

function isTensorLike(v: unknown): v is { shape: number[]; data: ArrayLike<number> } {
	return !!v && typeof v === 'object' && Array.isArray((v as Tensor).shape) && ArrayBuffer.isView((v as Tensor).data);
}

/** Best-effort copy of a batch into named tensors (Vars, tensors, typed arrays and number arrays). */
export function batchTensors(b: unknown): Record<string, Tensor> {
	const out: Record<string, Tensor> = {};
	if (!b || typeof b !== 'object') return out;
	for (const [k, v] of Object.entries(b as Record<string, unknown>)) {
		if (v instanceof Var) out[k] = v.toTensor();
		else if (isTensorLike(v)) out[k] = copyT(v);
		else if (ArrayBuffer.isView(v) && !(v instanceof DataView)) {
			const arr = v as unknown as ArrayLike<number>;
			out[k] = { shape: [arr.length], data: Float64Array.from(arr) };
		} else if (Array.isArray(v) && v.length && v.every((n) => typeof n === 'number')) out[k] = { shape: [v.length], data: Float64Array.from(v) };
	}
	return out;
}

function namedVars(m: Module): [string, Var][] {
	return Object.entries(m.named()).map(([k, v]) => [k, asVar(v, `parameter '${k}'`)]);
}

function copyParams(ps: [string, Var][]): Record<string, Tensor> {
	const out: Record<string, Tensor> = {};
	for (const [k, v] of ps) out[k] = v.toTensor();
	return out;
}

function copyGrads(ps: [string, Var][]): Record<string, Tensor> {
	const out: Record<string, Tensor> = {};
	for (const [k, v] of ps) out[k] = v.grad ? v.grad.toTensor() : { shape: v.shape.slice(), data: new Float64Array(v.size) };
	return out;
}

function copyOptState(s: Record<string, { m?: Tensor; v?: Tensor }>): Record<string, { m?: Tensor; v?: Tensor }> {
	const out: Record<string, { m?: Tensor; v?: Tensor }> = {};
	for (const [k, e] of Object.entries(s)) {
		out[k] = {};
		if (e.m) out[k].m = copyT(e.m);
		if (e.v) out[k].v = copyT(e.v);
	}
	return out;
}

function gradNorm(ps: [string, Var][]): number {
	let s = 0;
	for (const [, p] of ps) {
		if (!p.grad) continue;
		const g = p.grad.data;
		for (let i = 0; i < g.length; i++) s += g[i] * g[i];
	}
	return Math.sqrt(s);
}

function defaultOptimizer(ps: [string, Var][], opt: TrainOpt): Optimizer {
	const named: Record<string, Var> = Object.fromEntries(ps);
	const kind: OptimizerKind = opt.optimizer === 'adam' && opt.decoupled === true ? 'adamw' : (opt.optimizer as OptimizerKind);
	return makeOptimizer(kind, named, {
		lr: opt.lr,
		momentum: typeof opt.momentum === 'number' ? opt.momentum : undefined,
		betas: Array.isArray(opt.betas) ? (opt.betas as [number, number]) : undefined,
		eps: typeof opt.eps === 'number' ? opt.eps : undefined,
		weightDecay: typeof opt.weightDecay === 'number' ? opt.weightDecay : undefined
	});
}

function partValues(parts: Record<string, unknown> | undefined): Record<string, number> | undefined {
	if (!parts) return undefined;
	const out: Record<string, number> = {};
	for (const [k, v] of Object.entries(parts)) out[k] = asVar(v, `loss part '${k}'`).item();
	return out;
}

function partMetrics(parts: Record<string, number> | undefined): Record<string, number> {
	const out: Record<string, number> = {};
	if (parts) for (const [k, v] of Object.entries(parts)) out[RESERVED.has(k) ? `part.${k}` : k] = v;
	return out;
}

/**
 * Wrap a TrainerDef as a ProcessDef (kind 'training'). The loaders register it as process 'train:<name>'.
 * TrainOpt extras read here: momentum, betas, eps, weightDecay, decoupled (optimizer 'adam' + decoupled: true = AdamW),
 * resetOnInput (default false),
 * stopOnDiverge (default true: end the run after a frame whose loss is not finite).
 */
export function trainingProcess<D, M extends Module, B, O extends TrainOpt>(def: TrainerDef<D, M, B, O>): ProcessDef<D, O, TrainState> & LiveSpecFn<O> & { twin?: TrainerDef<D, M, B, O>['twin']; trainer: string } {
	const schema: Schema = { ...TRAIN_SCHEMA, ...(def.derived ?? {}) };
	return {
		/** K2: the trainer these frames come from; a process that spreads `...trainingProcess(t)` inherits it, so atlas
		 *  parity can give the trainer's twin to it when its frames are the trainer's (train/wraps.ts) */
		trainer: def.id,
		/** K1: atlas parity's options for the twin runs (TrainerDef.twin) */
		twin: def.twin,
		/** The model rebuilt by def.model() (the architecture, with opt as the source ran it), its parameters set from
		 *  state.params, then Module.toSpec(). A module class can override toSpec() for its own inputs and axes. */
		liveSpec(state, opt) {
			if (!state.params) throw new Error(`trainer '${def.id}': this state has no params yet (no snapshot recorded)`);
			const o = { ...def.defaults, ...(opt ?? {}) } as O;
			const m = def.model(createRng(Number(o.seed) || 0).fork('init'), o);
			if (typeof m.toSpec !== 'function') throw new Error(`trainer '${def.id}': its model has no toSpec(), so a live-model source cannot trace it`);
			const named = m.named();
			for (const [k, t] of Object.entries(state.params)) {
				const v = named[k];
				if (!v || v.value.data.length !== t.data.length) throw new Error(`trainer '${def.id}': parameter '${k}' does not fit the rebuilt model (opt differs from the run?)`);
				v.value.data.set(t.data);
			}
			const { spec, weights } = m.toSpec({ slug: def.id });
			const w: Record<string, Tensor> = {};
			for (const t of spec.tensors) w[t.name] = { shape: t.shape.slice(), data: weights.slice(t.offset, t.offset + t.length) };
			return { spec, weights: w };
		},
		id: def.id,
		kind: 'training',
		title: def.title,
		phases: TRAIN_PHASES,
		schema,
		defaults: def.defaults,
		*run(input: D, ctx: RunCtx<O>): Generator<Frame<TrainState>, void, Control<O> | undefined> {
			let opt: O = { ...def.defaults, ...ctx.opt };
			let data = input;
			const fine = ctx.detail === 'fine';
			let root: Rng = ctx.rng;
			let model!: M;
			let params!: [string, Var][];
			let optim!: Optimizer;
			let batchRng!: Rng;
			const build = () => {
				model = def.model(root.fork('init'), opt);
				params = namedVars(model);
				optim = def.optimizer ? def.optimizer(model, opt) : defaultOptimizer(params, opt);
				batchRng = root.fork('batch');
			};
			build();
			let k = 0;
			const pending: Control<O>[] = [];
			const take = (c: Control<O> | undefined) => {
				if (c) pending.push(c);
			};
			const userMetrics = (): Record<string, number> => {
				if (!def.metrics) return {};
				const out: Record<string, number> = {};
				for (const [key, v] of Object.entries(noGrad(() => def.metrics!(model, data, opt)))) out[RESERVED.has(key) ? `metric.${key}` : key] = v;
				return out;
			};
			const optName = (): string => (optim as { kind?: string }).kind ?? opt.optimizer;
			const initFrame = (caption: I18n): Frame<TrainState> => ({
				phase: 'init',
				state: { step: k, params: copyParams(params), lr: optim.lr, optimizer: optName() },
				metrics: { step: k, lr: optim.lr, ...userMetrics() },
				caption
			});

			take(yield initFrame(CAP_START));

			while (!ctx.signal.cancelled) {
				let segment: I18n | null = null;
				while (pending.length) {
					const c = pending.shift()!;
					if (c.type === 'set') {
						const before = opt.optimizer;
						opt = { ...opt, ...c.opt };
						if (opt.optimizer !== before && !def.optimizer) optim = defaultOptimizer(params, opt);
						else {
							const s: Parameters<Optimizer['set']>[0] = {};
							if (c.opt.lr !== undefined) s.lr = c.opt.lr as number;
							if (typeof c.opt.momentum === 'number') s.momentum = c.opt.momentum;
							if (Array.isArray(c.opt.betas)) s.betas = c.opt.betas as [number, number];
							if (typeof c.opt.weightDecay === 'number') s.weightDecay = c.opt.weightDecay;
							optim.set(s);
						}
					} else if (c.type === 'input') {
						data = c.input as D;
						if (opt.resetOnInput === true) {
							build();
							k = 0;
						}
						segment = CAP_DATA;
					} else if (c.type === 'reset') {
						// RunCtx carries no Rng factory: a new seed becomes a labelled fork of the run's rng.
						// A runner that wants "same as a fresh run with seed s" re-opens the source instead.
						root = c.seed !== undefined && c.seed !== ctx.rng.seed ? ctx.rng.fork(`seed:${c.seed}`) : ctx.rng;
						build();
						k = 0;
						segment = CAP_RESET;
					}
				}
				if (segment) {
					take(yield initFrame(segment));
					continue;
				}
				if (fine ? k >= opt.steps : k > opt.steps) break;

				// K2: the batch may depend on the current model (an RL round played with the current policy)
				const b = def.batch(data, k, batchRng, opt, model);
				const acts: Record<string, Tensor> = {};
				const rec: Rec = fine ? (id, v) => (acts[id] = copyT(v.value)) : () => {};
				zeroGrad(params.map((p) => p[1]));
				const out = def.loss(model, b, rec, opt);
				const loss = asVar(out.loss, 'loss()');
				const L = loss.item();
				const parts = partValues(out.parts as Record<string, unknown> | undefined);
				const finite = Number.isFinite(L);
				const stop = !finite && opt.stopOnDiverge !== false;
				const lr = optim.lr;

				if (!fine) {
					const last = k >= opt.steps;
					const metrics: Record<string, number> = { step: k, loss: L, lr, ...partMetrics(parts) };
					if (!last && finite) {
						backward(loss);
						metrics.gradNorm = gradNorm(params);
					}
					const every = Math.max(1, Math.floor(Number(opt.snapshotEvery) || 1));
					const snap = last || stop || k % every === 0;
					const state: Partial<TrainState> = { step: k, loss: L };
					if (parts) state.parts = parts;
					if (snap) {
						state.params = copyParams(params);
						Object.assign(metrics, userMetrics());
					}
					const frame: Frame<TrainState> = { phase: snap ? 'snapshot' : 'step', step: k, state, metrics };
					if (!finite) {
						metrics.diverged = 1;
						frame.caption = CAP_DIVERGED;
					}
					take(yield frame);
					if (last || stop) break;
					if (finite) {
						optim.step();
						def.project?.(model, opt);
					}
					k++;
				} else {
					take(yield { phase: 'forward', step: k, state: { step: k, batch: batchTensors(b), acts }, metrics: { step: k, lr } });
					const lossState: Partial<TrainState> = { step: k, loss: L };
					if (parts) lossState.parts = parts;
					const lossFrame: Frame<TrainState> = { phase: 'loss', step: k, state: lossState, metrics: { step: k, loss: L, lr, ...partMetrics(parts) } };
					if (!finite) {
						lossFrame.metrics!.diverged = 1;
						lossFrame.caption = CAP_DIVERGED;
					}
					take(yield lossFrame);
					if (stop) break;
					backward(loss);
					const grads = copyGrads(params);
					take(yield { phase: 'backward', step: k, state: { step: k, grads }, metrics: { step: k, loss: L, lr, gradNorm: gradNorm(params) } });
					const prev = copyParams(params);
					optim.step();
					// K4: a constraint after the step (TransE's unit length); the frame then says the update was projected
					if (def.project) def.project(model, opt);
					const wd = (optim as { weightDecay?: number }).weightDecay ?? 0;
					take(
						yield {
							phase: 'update',
							step: k,
							state: { step: k, params: copyParams(params), prev, grads, opt: copyOptState(optim.state()), lr, optimizer: optName() },
							metrics: { step: k, loss: L, lr, weightDecay: wd, ...(def.project ? { projected: 1 } : {}) }
						}
					);
					k++;
				}
			}
		}
	};
}
