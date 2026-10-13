// backprop: the processes (kind C). Every value and gradient comes from the kit's autograd (atlas/autograd).
//   graph  the toy graph of data.ts on the learner's four leaves. 'forward': every node's value, no gradient known yet.
//          'seed': dL/dL = 1. 'edge' (step k = 0..8, counting from 0): the backward pass crosses edge k of data.ts EDGES,
//          child -> parent: the child's recorded backward rule (its GradFn, the same code backward() runs) turns the
//          child's gradient into the parent's share, which is ADDED to what the parent already has (L = e·e reaches e
//          twice). 'check': every leaf nudged by ±h and the graph run forward again: (L(v + h) − L(v − h)) / 2h.
//   net    the 2-2-1 network of data.ts on its one example (input ignored). 'forward', then the backward pass one layer at
//          a time ('out': dL/dŷ and the output layer's parameters; 'hidden': dL/dh and dL/dz; 'first': the first layer's
//          parameters and the inputs), then 'check' (the nine parameters nudged by ±h) and 'update' (one gradient-descent
//          step with η = NET_LR and the new loss).
//   npm run atlas -- run backprop graph --frames 0..12
import type { ProcessDef, ProcessRun, Schema, Tensor } from '../../atlas/types/index.ts';
import { grad, input, noGrad, param, type Var } from '../../atlas/autograd/var.ts';
import * as op from '../../atlas/autograd/ops.ts';
import { CHECK_H, EDGES, LEAVES, NET_LR, NET_NODES, NET_P0, NET_PARAMS, NET_T, NET_X, NODES, NODE_INDEX, leavesOf, type LeafId, type Leaves } from './data.ts';

const vec = (a: ArrayLike<number>): Tensor => ({ shape: [a.length], data: Float64Array.from(a) });
/** The numerical slope of f at 0 by a central difference: (f(h) − f(−h)) / 2h. */
const central = (f: (d: number) => number, h: number): number => (f(h) - f(-h)) / (2 * h);

// ---------------------------------------------------------------- the toy graph

/** The toy graph on the kit's autograd: four scalar leaves that require a gradient, five operations. */
export function buildGraph(l: Leaves): Record<string, Var> {
	const leaf = (v: number) => input({ shape: [1], data: [v] }, true);
	const x = leaf(l.x);
	const w = leaf(l.w);
	const b = leaf(l.b);
	const y = leaf(l.y);
	const p = op.mul(w, x);
	const z = op.add(p, b);
	const a = op.tanh(z);
	const e = op.sub(a, y);
	const L = op.mul(e, e);
	return { x, w, b, y, p, z, a, e, L };
}

/** L alone, without recording a graph (the numerical check). */
export const lossOf = (l: Leaves): number => noGrad(() => buildGraph(l).L.item());

export type GraphOpt = { h: number };
export type GraphState = { vals: Tensor; grads: Tensor; known: Tensor; locals: Tensor };

const N = NODES.length;
const GRAPH_SCHEMA: Schema = {
	vals: { kind: 'tensor', label: { en: 'Value of each node', zh: '每个节点的值' }, shape: [N], role: 'activation', tol: 1e-12 },
	grads: { kind: 'tensor', label: { en: 'Gradient of L at each node (0 until reached)', zh: '每个节点上 L 的梯度（到达之前为 0）' }, shape: [N], role: 'grad', tol: 1e-12 },
	known: { kind: 'tensor', label: { en: 'Reached by the backward pass (1) or not yet (0)', zh: '反向传播已到达（1）或尚未到达（0）' }, shape: [N], role: 'data', tol: 'exact' },
	locals: { kind: 'tensor', label: { en: 'Local derivative of each edge', zh: '每条边的局部导数' }, shape: [EDGES.length], role: 'activation', tol: 1e-12 }
};

export const graph: ProcessDef<unknown, GraphOpt, GraphState> = {
	id: 'graph',
	kind: 'algorithm',
	title: { en: 'Backpropagation on a small computation graph', zh: '小计算图上的反向传播' },
	phases: {
		forward: { label: { en: 'Forward pass', zh: '前向传播' }, kind: 'forward', narration: { what: { en: 'Every node computed from its inputs: L = {frame.L}.', zh: '每个节点都由它的输入算出：L = {frame.L}。' } } },
		seed: { label: { en: 'Start at L', zh: '从 L 开始' }, kind: 'backward', narration: { what: { en: 'The backward pass starts at L with dL/dL = 1.', zh: '反向传播从 L 开始，dL/dL = 1。' } } },
		edge: {
			label: { en: 'One edge back', zh: '倒退一条边' },
			kind: 'backward',
			formula: { tex: 'g_{\\text{parent}} \\mathrel{+}= g_{\\text{child}} \\times \\text{local} = {{before}} + {{up}} \\times {{local}} = {{after}}', values: (f) => ({ before: f.metrics!.before, up: f.metrics!.up, local: f.metrics!.local, after: f.metrics!.after }) },
			narration: { what: { en: 'Edge {frame.edge} (counting from 0): the gradient of the node it comes from becomes {frame.after}.', zh: '编号 {frame.edge} 的边（从 0 数起）：箭头出发的节点，梯度变成 {frame.after}。' } }
		},
		check: { label: { en: 'Numerical check', zh: '数值检验' }, kind: 'observe', narration: { what: { en: 'Nudging each leaf by ±{frame.h} gives the same slopes, within {frame.maxDiff}.', zh: '把每个叶子节点加减 {frame.h}，得到的斜率与反向传播相差不超过 {frame.maxDiff}。' } } }
	},
	schema: GRAPH_SCHEMA,
	defaults: { h: CHECK_H },
	metricTol: { nx: 1e-8, nw: 1e-8, nb: 1e-8, ny: 1e-8, maxDiff: 1e-8 },
	*run(raw, ctx): ProcessRun<GraphState, GraphOpt> {
		const l = leavesOf(raw);
		const h = ctx.opt.h ?? CHECK_H;
		const v = buildGraph(l);
		const vals = Float64Array.from(NODES, (n) => v[n.id].item());
		const grads = new Float64Array(N);
		const known = new Float64Array(N);
		const one = input({ shape: [1], data: [1] });
		const need = (id: string) => v[id].fn!.parents.map(() => true);
		// each edge's local derivative: the child's own backward rule run on a gradient of 1
		const locals = Float64Array.from(EDGES, (ed) => noGrad(() => v[ed.to].fn!.backward(one, need(ed.to)))[ed.slot]!.item());
		const snap = (): GraphState => ({ vals: vec(vals), grads: vec(grads), known: vec(known), locals: vec(locals) });
		const L = vals[NODE_INDEX.L];
		yield { phase: 'forward', step: 0, state: snap(), metrics: { L } };
		grads[NODE_INDEX.L] = 1;
		known[NODE_INDEX.L] = 1;
		yield { phase: 'seed', step: 0, state: snap(), metrics: { L } };
		for (let k = 0; k < EDGES.length; k++) {
			const ed = EDGES[k];
			const ci = NODE_INDEX[ed.to];
			const pi = NODE_INDEX[ed.from];
			// the child's own backward rule run on the child's gradient: the parent's share
			const share = noGrad(() => v[ed.to].fn!.backward(input({ shape: [1], data: [grads[ci]] }), need(ed.to)))[ed.slot]!.item();
			const local = locals[k];
			const acc = known[pi];
			const before = acc ? grads[pi] : 0;
			grads[pi] = before + share;
			known[pi] = 1;
			yield { phase: 'edge', step: k, state: snap(), metrics: { L, edge: k, up: grads[ci], local, share, before, after: grads[pi], acc } };
		}
		const m: Record<string, number> = { L, h };
		let maxDiff = 0;
		for (const { id } of LEAVES) {
			const num = central((d) => lossOf({ ...l, [id]: l[id] + d }), h);
			m[`n${id}`] = num;
			maxDiff = Math.max(maxDiff, Math.abs(num - grads[NODE_INDEX[id]]));
		}
		yield { phase: 'check', step: 0, state: snap(), metrics: { ...m, maxDiff } };
	}
};

/** The gradients backward() gives the four leaves (the engine's own walk; the claim tests compare it with the frames). */
export function leafGrads(l: Leaves): Record<LeafId, number> {
	const v = buildGraph(l);
	const g = grad(v.L, [v.x, v.w, v.b, v.y]);
	return { x: g[0].item(), w: g[1].item(), b: g[2].item(), y: g[3].item() };
}

// ---------------------------------------------------------------- the 2-2-1 network

export type NetOpt = { h: number; lr: number };
export type NetState = { vals: Tensor; grads: Tensor; known: Tensor; params: Tensor; pgrads: Tensor; pknown: Tensor; pnum: Tensor };

/** The network on the kit's autograd, from nine parameter values (state_dict order of data.ts NET_PARAMS). */
export function buildNet(p: ArrayLike<number>) {
	const x = input({ shape: [1, 2], data: NET_X }, true);
	const W1 = param('net.0.weight', { shape: [2, 2], data: [p[0], p[1], p[2], p[3]] });
	const b1 = param('net.0.bias', { shape: [2], data: [p[4], p[5]] });
	const W2 = param('net.2.weight', { shape: [1, 2], data: [p[6], p[7]] });
	const b2 = param('net.2.bias', { shape: [1], data: [p[8]] });
	const z = op.linear(x, W1, b1);
	const h = op.tanh(z);
	const yh = op.linear(h, W2, b2);
	const e = op.sub(yh, input({ shape: [1, 1], data: [NET_T] }));
	const L = op.sum(op.mul(e, e));
	return { x, W1, b1, W2, b2, z, h, yh, L };
}
export const netLoss = (p: ArrayLike<number>): number => noGrad(() => buildNet(p).L.item());

const K = NET_NODES.length;
const P = NET_PARAMS.length;

export const net: ProcessDef<unknown, NetOpt, NetState> = {
	id: 'net',
	kind: 'algorithm',
	title: { en: 'Backpropagation through a 2-2-1 network', zh: '2-2-1 网络中的反向传播' },
	phases: {
		forward: { label: { en: 'Forward pass', zh: '前向传播' }, kind: 'forward', narration: { what: { en: 'The network answers {frame.yh} for a target of {frame.t}: L = {frame.L}.', zh: '网络的回答是 {frame.yh}，目标是 {frame.t}：L = {frame.L}。' } } },
		out: { label: { en: 'Output layer', zh: '输出层' }, kind: 'backward', narration: { what: { en: 'dL/dŷ = {frame.gy}, and the output layer\'s three parameters get their gradients.', zh: 'dL/dŷ = {frame.gy}，输出层的三个参数得到各自的梯度。' } } },
		hidden: { label: { en: 'Hidden units', zh: '隐藏单元' }, kind: 'backward', narration: { what: { en: 'Through the output weights and each tanh: dL/dz₁ = {frame.gz1}, dL/dz₂ = {frame.gz2}.', zh: '经过输出权重和每个 tanh：dL/dz₁ = {frame.gz1}，dL/dz₂ = {frame.gz2}。' } } },
		first: { label: { en: 'First layer', zh: '第一层' }, kind: 'backward', narration: { what: { en: 'The first layer\'s six parameters and both inputs get their gradients.', zh: '第一层的六个参数和两个输入得到各自的梯度。' } } },
		check: { label: { en: 'Numerical check', zh: '数值检验' }, kind: 'observe', narration: { what: { en: '{frame.passes} extra forward passes agree within {frame.maxDiff}.', zh: '额外的 {frame.passes} 次前向传播与反向传播相差不超过 {frame.maxDiff}。' } } },
		update: { label: { en: 'One update', zh: '一次更新' }, kind: 'update', narration: { what: { en: 'One gradient-descent step: L goes from {frame.L} to {frame.L1}.', zh: '走一步梯度下降：L 从 {frame.L} 变成 {frame.L1}。' } } }
	},
	schema: {
		vals: { kind: 'tensor', label: { en: 'Value of each node', zh: '每个节点的值' }, shape: [K], role: 'activation', tol: 1e-12 },
		grads: { kind: 'tensor', label: { en: 'Gradient of L at each node (0 until reached)', zh: '每个节点上 L 的梯度（到达之前为 0）' }, shape: [K], role: 'grad', tol: 1e-12 },
		known: { kind: 'tensor', label: { en: 'Node reached (1) or not yet (0)', zh: '节点已到达（1）或尚未到达（0）' }, shape: [K], role: 'data', tol: 'exact' },
		params: { kind: 'tensor', label: { en: 'The nine parameters', zh: '九个参数' }, shape: [P], role: 'param', tol: 1e-12 },
		pgrads: { kind: 'tensor', label: { en: 'Their gradients (0 until reached)', zh: '它们的梯度（到达之前为 0）' }, shape: [P], role: 'grad', tol: 1e-12 },
		pknown: { kind: 'tensor', label: { en: 'Parameter reached (1) or not yet (0)', zh: '参数已到达（1）或尚未到达（0）' }, shape: [P], role: 'data', tol: 'exact' },
		pnum: { kind: 'tensor', label: { en: 'Numerical gradients (check frame)', zh: '数值梯度（检验帧）' }, shape: [P], role: 'grad', tol: 1e-8 }
	},
	defaults: { h: CHECK_H, lr: NET_LR },
	metricTol: { maxDiff: 1e-8 },
	*run(_raw, ctx): ProcessRun<NetState, NetOpt> {
		const h = ctx.opt.h ?? CHECK_H;
		const lr = ctx.opt.lr ?? NET_LR;
		const n = buildNet(NET_P0);
		const [gx, gz, gh, gy, gW1, gb1, gW2, gb2] = grad(n.L, [n.x, n.z, n.h, n.yh, n.W1, n.b1, n.W2, n.b2]).map((g) => g.data);
		const L = n.L.item();
		const vals = Float64Array.from([...n.x.data, ...n.z.data, ...n.h.data, n.yh.item(), L]);
		const [grads, known] = [new Float64Array(K), new Float64Array(K)];
		const [pgrads, pknown, pnum] = [new Float64Array(P), new Float64Array(P), new Float64Array(P)];
		const params = Float64Array.from(NET_P0);
		const snap = (): NetState => ({ vals: vec(vals), grads: vec(grads), known: vec(known), params: vec(params), pgrads: vec(pgrads), pknown: vec(pknown), pnum: vec(pnum) });
		const set = (node: number[], gs: ArrayLike<number>) => node.forEach((k, i) => ((grads[k] = gs[i]), (known[k] = 1)));
		const setP = (from: number, gs: ArrayLike<number>) => Array.from(gs).forEach((g, i) => ((pgrads[from + i] = g), (pknown[from + i] = 1)));
		const yh = n.yh.item();
		const base = { L, yh, t: NET_T };
		yield { phase: 'forward', step: 0, state: snap(), metrics: base };
		set([7, 6], [1, gy[0]]);
		setP(6, gW2);
		setP(8, gb2);
		yield { phase: 'out', step: 1, state: snap(), metrics: { ...base, gy: gy[0] } };
		set([4, 5, 2, 3], [gh[0], gh[1], gz[0], gz[1]]);
		yield { phase: 'hidden', step: 2, state: snap(), metrics: { ...base, gy: gy[0], gz1: gz[0], gz2: gz[1] } };
		setP(0, gW1);
		setP(4, gb1);
		set([0, 1], gx);
		yield { phase: 'first', step: 3, state: snap(), metrics: { ...base, gy: gy[0], gz1: gz[0], gz2: gz[1] } };
		let maxDiff = 0;
		for (let i = 0; i < P; i++) {
			pnum[i] = central((d) => netLoss(NET_P0.map((p, k) => (k === i ? p + d : p))), h);
			maxDiff = Math.max(maxDiff, Math.abs(pnum[i] - pgrads[i]));
		}
		yield { phase: 'check', step: 4, state: snap(), metrics: { ...base, maxDiff, passes: 2 * P, params: P, h } };
		const next = NET_P0.map((p, i) => p - lr * pgrads[i]);
		params.set(next);
		yield { phase: 'update', step: 5, state: snap(), metrics: { ...base, lr, L1: netLoss(next), maxDiff, passes: 2 * P, params: P } };
	}
};
