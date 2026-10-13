// residual-connections: the processes (kind C). Every number comes from the kit's autograd (atlas/autograd) on the deep
// MLP of net.ts; the weights are the run's rng.fork('init') draws, the same as the first weights of a training run
// with the same seed.
//   grads   one forward and one backward pass on all 120 points at the starting weights, for the learner's setup
//           ({ depth, skip }). 'forward': the loss and the size (RMS) of every hidden state. 'backward': the size
//           (Frobenius norm) of every layer's weight gradient, of the gradient at every hidden state, and the same
//           numbers for the other setting of the skip switch (same weights). 'skip' (with skips) or 'block' (without):
//           the backward pass through the middle layer, one number at a time: the gradient arriving at its output,
//           the share its tanh block sends back (the block's own backward rule run on it), the share the skip sends
//           back (the same number, untouched) and their sum, at the point and unit where the arriving gradient is
//           largest.
//   train   the learner's setup trained with plain gradient descent (trainingProcess frames of net.ts 'deep').
//   race    20 layers plain and 20 layers with skips, same weights, trained in lockstep (input ignored): the skip
//           run's frames, with the plain run's loss (lossPlain) every step, and at snapshots its accuracy (accPlain)
//           and how far each run's layer-0 weights have moved from their start (move, movePlain: Frobenius norm).
//   depths  the same training at each depth of data.ts DEPTHS, plain and with skips (input ignored): one frame per
//           depth with the final loss and accuracy of both.
//   npm run atlas -- run residual-connections bars --frames 0..2
import type { ProcessDef, ProcessRun, Rng, RunCtx, Tensor, TrainState } from '../../atlas/types/index.ts';
import { grad, noGrad } from '../../atlas/autograd/var.ts';
import { crossEntropy } from '../../atlas/autograd/loss.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { DEPTHS, MOONS, midLayer, setupOf } from './data.ts';
import { DeepNet, deep, pointsVar, type DeepOpt } from './net.ts';

const vec = (a: ArrayLike<number>): Tensor => ({ shape: [a.length], data: Float64Array.from(a) });
const norm = (a: ArrayLike<number>): number => {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * a[i];
	return Math.sqrt(s);
};

/** One forward and backward pass of a deep MLP at its starting weights. */
export interface Analysis {
	depth: number;
	skip: number;
	loss: number;
	/** |dL/dW_l| for every layer l = 0 .. depth − 1 */
	gW: number[];
	/** |dL/dh_l| for every hidden state l = 0 .. depth − 2 */
	gH: number[];
	/** RMS of every hidden state h_l over the 120 points and the units */
	rms: number[];
	/** the middle layer's backward step, one number: point p, unit j */
	mid: { layer: number; p: number; j: number; up: number; block: number; skipPart: number; total: number; upN: number; blockN: number; totalN: number };
}

export function analyse(depth: number, skip: number, rng: Rng): Analysis {
	const net = new DeepNet(depth, !!skip, rng);
	const tr = net.trace(pointsVar());
	const L = crossEntropy(tr.scores, MOONS.label);
	const g = grad(L, [...net.layers.map((l) => l.weight), ...tr.h]);
	const gW = g.slice(0, depth).map((v) => norm(v.data));
	const gHv = g.slice(depth);
	const gH = gHv.map((v) => norm(v.data));
	const rms = tr.h.map((h) => norm(h.data) / Math.sqrt(h.data.length));
	// the middle layer l: its output h_l receives up = dL/dh_l; its block a_l = tanh(W_l·h_{l−1} + b_l) sends back the
	// block's own backward of up; the skip (when there is one) sends back up itself; dL/dh_{l−1} is their sum
	const l = midLayer(depth);
	const up = gHv[l].data;
	const block = noGrad(() => grad(tr.a[l], tr.h[l - 1], { seeds: gHv[l] })).data;
	const total = gHv[l - 1].data;
	let k = 0;
	for (let i = 1; i < up.length; i++) if (Math.abs(up[i]) > Math.abs(up[k])) k = i;
	const W = net.width;
	return {
		depth,
		skip,
		loss: L.item(),
		gW,
		gH,
		rms,
		mid: { layer: l, p: Math.floor(k / W), j: k % W, up: up[k], block: block[k], skipPart: net.skips(l) ? up[k] : 0, total: total[k], upN: norm(up), blockN: norm(block), totalN: norm(total) }
	};
}

export type GradsOpt = Record<string, never>;
export type GradsState = { gW?: Tensor; gH?: Tensor; rms: Tensor; other?: Tensor; otherRms?: Tensor };

const ALIGN = (...rows: string[]) => `\\begin{aligned} ${rows.join(' \\\\ ')} \\end{aligned}`;
/** g_i = ∂L/∂h_i, the gradient of the loss at layer i's output */
const G = (i: string) => `g_{ ${i} }`;
const BLOCK = `W_{ {{layer}} }^{\\top}(\\tanh'(z_{ {{layer}} }) \\odot ${G('{{layer}}')})`;
// four significant digits, as the narration prints them: with three, the shown terms need not add up to the shown sum
// (0.0196 + 0.00976 printed as 0.0293)
const MID_VALUES = (f: { metrics?: Record<string, number> }) => ({ up: fmt3(f.metrics!.up, 4), block: fmt3(f.metrics!.block, 4), total: fmt3(f.metrics!.total, 4), layer: String(f.metrics!.layer), prev: String(f.metrics!.layer - 1) });

export const grads: ProcessDef<unknown, GradsOpt, GradsState> = {
	id: 'grads',
	kind: 'algorithm',
	title: { en: 'Gradients through a deep MLP at its starting weights', zh: '深层 MLP 在初始权重下的梯度' },
	phases: {
		forward: {
			label: { en: 'Forward pass', zh: '前向传播' },
			kind: 'forward',
			narration: { what: { en: 'The 120 points go through all {frame.depth} layers: the loss is {frame.loss}.', zh: '120 个点穿过全部 {frame.depth} 层：损失是 {frame.loss}。' } }
		},
		backward: {
			label: { en: 'Backward pass', zh: '反向传播' },
			kind: 'backward',
			narration: { what: { en: 'Every layer gets its gradient: {frame.gLast} at the last layer, {frame.g0} at layer 0.', zh: '每一层都得到自己的梯度：最后一层 {frame.gLast}，第 0 层 {frame.g0}。' } }
		},
		skip: {
			label: { en: 'Back through one layer with a skip', zh: '反向穿过一个带跳跃连接的层' },
			kind: 'backward',
			formula: {
				// the sum on a row of its own: with four digits one row does not fit a 390 px stage
				tex: ALIGN(`${G('{{prev}}')} &= ${G('{{layer}}')} + ${BLOCK}`, '&= {{up}} + ({{block}})', '&= {{total}}'),
				values: MID_VALUES
			},
			narration: { what: { en: 'Back through layer {frame.layer}: the skip passes {frame.up} on unchanged, the block adds {frame.block}.', zh: '反向穿过第 {frame.layer} 层：跳跃连接原样传回 {frame.up}，块再加上 {frame.block}。' } }
		},
		block: {
			label: { en: 'Back through one plain layer', zh: '反向穿过一个普通层' },
			kind: 'backward',
			formula: {
				tex: ALIGN(`${G('{{prev}}')} &= ${BLOCK}`, '{{up}} &\\;\\to\\; {{total}}'),
				values: MID_VALUES
			},
			narration: { what: { en: 'Back through layer {frame.layer}: only the block, so {frame.up} becomes {frame.total}.', zh: '反向穿过第 {frame.layer} 层：只有块这一条路，{frame.up} 变成 {frame.total}。' } }
		}
	},
	schema: {
		gW: { kind: 'tensor', label: { en: "Size of each layer's weight gradient", zh: '每层权重梯度的大小' }, shape: ['layers'], role: 'grad', tol: 1e-12 },
		gH: { kind: 'tensor', label: { en: 'Size of the gradient at each hidden state', zh: '每个隐藏状态处梯度的大小' }, shape: ['hidden'], role: 'grad', tol: 1e-12 },
		rms: { kind: 'tensor', label: { en: 'Size (RMS) of each hidden state', zh: '每个隐藏状态的大小（均方根）' }, shape: ['hidden'], role: 'activation', tol: 1e-12 },
		other: { kind: 'tensor', label: { en: 'The weight-gradient sizes with the skip switch flipped', zh: '跳跃连接开关反过来时的权重梯度大小' }, shape: ['layers'], role: 'grad', tol: 1e-12 },
		otherRms: { kind: 'tensor', label: { en: 'The hidden-state sizes with the skip switch flipped', zh: '跳跃连接开关反过来时的隐藏状态大小' }, shape: ['hidden'], role: 'activation', tol: 1e-12 }
	},
	defaults: {},
	*run(raw, ctx): ProcessRun<GradsState, GradsOpt> {
		const s = setupOf(raw);
		const a = analyse(s.depth, s.skip, ctx.rng.fork('init'));
		const o = analyse(s.depth, 1 - s.skip, ctx.rng.fork('init'));
		const D = s.depth;
		const base = { depth: D, skip: s.skip, loss: a.loss, oLoss: o.loss };
		yield { phase: 'forward', step: 0, state: { rms: vec(a.rms) }, metrics: base };
		const state: GradsState = { gW: vec(a.gW), gH: vec(a.gH), rms: vec(a.rms), other: vec(o.gW), otherRms: vec(o.rms) };
		const m = {
			...base,
			g0: a.gW[0],
			gLast: a.gW[D - 1],
			ratio: a.gW[D - 1] / a.gW[0],
			o0: o.gW[0],
			oLast: o.gW[D - 1],
			oRatio: o.gW[D - 1] / o.gW[0],
			h0: a.gH[0],
			hLast: a.gH[D - 2],
			rms0: a.rms[0],
			rmsLast: a.rms[D - 2],
			oRms0: o.rms[0],
			oRmsLast: o.rms[D - 2]
		};
		yield { phase: 'backward', step: 0, state, metrics: m };
		const mid = a.mid;
		yield {
			phase: s.skip ? 'skip' : 'block',
			step: 0,
			state,
			metrics: { ...m, layer: mid.layer, point: mid.p, unit: mid.j, up: mid.up, block: mid.block, skipPart: mid.skipPart, total: mid.total, upN: mid.upN, blockN: mid.blockN, totalN: mid.totalN, factor: mid.totalN / mid.upN }
		};
	}
};

// ---------------------------------------------------------------- training

const inner = trainingProcess(deep);

/** The learner's setup trained: the trainer's own frames, with depth and skip taken from the input. */
export const train: ProcessDef<unknown, DeepOpt, TrainState> = {
	...inner,
	id: 'train',
	*run(raw, ctx): ProcessRun<TrainState, DeepOpt> {
		const s = setupOf(raw);
		return yield* inner.run(raw, { ...ctx, opt: { ...ctx.opt, depth: s.depth, skip: s.skip } });
	}
};

const first = (st: Partial<TrainState> | undefined): ArrayLike<number> | undefined => st?.params?.['net.0.weight']?.data;
const dist = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
	return Math.sqrt(s);
};

export const race: ProcessDef<unknown, DeepOpt, TrainState> = {
	...inner,
	id: 'race',
	title: { en: 'Twenty layers, plain and with skips, trained side by side', zh: '二十层网络：无跳跃连接与有跳跃连接并排训练' },
	*run(_raw, ctx): ProcessRun<TrainState, DeepOpt> {
		const res = inner.run(null, { ...ctx, detail: 'coarse', opt: { ...ctx.opt, skip: 1 } });
		const pl = inner.run(null, { ...ctx, detail: 'coarse', opt: { ...ctx.opt, skip: 0 } });
		let w0: Float64Array | undefined;
		let p0: Float64Array | undefined;
		for (let r = res.next(); !r.done; r = res.next()) {
			const q = pl.next();
			const m: Record<string, number> = { ...r.value.metrics };
			const qm = q.done ? undefined : q.value.metrics;
			if (qm?.loss !== undefined) m.lossPlain = qm.loss;
			if (qm?.accuracy !== undefined) m.accPlain = qm.accuracy;
			const wa = first(r.value.state);
			const wb = q.done ? undefined : first(q.value.state);
			if (wa && !w0) w0 = Float64Array.from(wa);
			if (wb && !p0) p0 = Float64Array.from(wb);
			if (wa && w0) m.move = dist(wa, w0);
			if (wb && p0) m.movePlain = dist(wb, p0);
			yield { ...r.value, metrics: m };
		}
	}
};

/** The last numbers of one training run: the final loss and the accuracy of the last snapshot. */
function trained(depth: number, skip: number, ctx: RunCtx<DeepOpt>): { loss: number; acc: number } {
	const run = inner.run(null, { ...ctx, detail: 'coarse', opt: { ...ctx.opt, depth, skip } });
	let loss = NaN;
	let acc = NaN;
	for (let r = run.next(); !r.done; r = run.next()) {
		const m = r.value.metrics ?? {};
		if (typeof m.loss === 'number') loss = m.loss;
		if (typeof m.accuracy === 'number') acc = m.accuracy;
	}
	return { loss, acc };
}

export type DepthsState = Record<string, never>;

export const depths: ProcessDef<unknown, DeepOpt, DepthsState> = {
	id: 'depths',
	kind: 'algorithm',
	title: { en: 'The same training at four depths', zh: '四种深度下的同样训练' },
	phases: {
		depth: {
			label: { en: 'One depth', zh: '一种深度' },
			kind: 'observe',
			narration: { what: { en: '{frame.depth} layers: plain {frame.lossPlain}, with skips {frame.loss}.', zh: '{frame.depth} 层：无跳跃连接 {frame.lossPlain}，有跳跃连接 {frame.loss}。' } }
		}
	},
	schema: {},
	defaults: deep.defaults,
	*run(_raw, ctx): ProcessRun<DepthsState, DeepOpt> {
		const o = { ...deep.defaults, ...ctx.opt };
		for (let i = 0; i < DEPTHS.length; i++) {
			const d = DEPTHS[i];
			const p = trained(d, 0, ctx);
			const r = trained(d, 1, ctx);
			yield { phase: 'depth', step: i, state: {}, metrics: { depth: d, loss: r.loss, acc: r.acc, lossPlain: p.loss, accPlain: p.acc, steps: o.steps, lr: o.lr } };
		}
	}
};
