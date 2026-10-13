// backprop: the two graphs of the page and their fixed numbers, shared by the processes, the leaves widget, the views
// and the claim tests. No third-party data.
//   The toy graph: one neuron with a squared error, L = (tanh(w·x + b) − y)², written as five operations on four
//   leaves (x, w, b, y). The last one, L = e·e, uses e twice: its gradient arrives along two edges and is added up.
//   The 2-2-1 network: two inputs, two tanh units, one linear output and a squared error, with weights written out here
//   (as nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1)) names them), so every learner sees the same numbers.
import type { I18n } from '../../atlas/types/core.ts';
import { fmt3 } from '../../atlas/core/format.ts';

// ---------------------------------------------------------------- the toy graph

export type LeafId = 'x' | 'w' | 'b' | 'y';
export type Leaves = Record<LeafId, number>;

/** The four leaves the learner sets: two data values (the input x, the target y) and two parameters (w, b). */
export const LEAVES: { id: LeafId; range: [number, number]; name: I18n; role: 'data' | 'param' }[] = [
	{ id: 'x', range: [-2, 2], name: { en: 'Input x', zh: '输入 x' }, role: 'data' },
	{ id: 'w', range: [-3, 3], name: { en: 'Weight w', zh: '权重 w' }, role: 'param' },
	{ id: 'b', range: [-2, 2], name: { en: 'Bias b', zh: '偏置 b' }, role: 'param' },
	{ id: 'y', range: [-1, 1], name: { en: 'Target y', zh: '目标 y' }, role: 'data' }
];

/** Where the page starts. */
export const START: Leaves = { x: 1.5, w: 0.8, b: -0.5, y: 0.9 };
/** A saturated neuron: w·x + b = 6, where tanh is almost flat (the limits step and the comparison). */
export const SAT: Leaves = { x: 2, w: 2.5, b: 1, y: -0.8 };

const clamp = (v: number, [lo, hi]: [number, number]) => Math.min(hi, Math.max(lo, v));

/** Any raw input -> four leaves inside their ranges ({ x, w, b, y }; a missing or non-numeric leaf takes START's value). */
export function leavesOf(raw: unknown): Leaves {
	const o = (raw && typeof raw === 'object' ? raw : {}) as Partial<Record<LeafId, unknown>>;
	const out = { ...START };
	for (const l of LEAVES) {
		const v = o[l.id];
		if (typeof v === 'number' && Number.isFinite(v)) out[l.id] = clamp(v, l.range);
	}
	return out;
}

/** A node of the toy graph. `depth` runs along the flow (leaves first), `lane` across it; the view lays them out. */
export interface GNode {
	id: string;
	/** how the node is computed from its parents (null for a leaf) */
	formula: string | null;
	kind: 'data' | 'param' | 'op';
	depth: number;
	lane: number;
}

export const NODES: GNode[] = [
	{ id: 'x', formula: null, kind: 'data', depth: 0, lane: 0 },
	{ id: 'w', formula: null, kind: 'param', depth: 0, lane: 1 },
	{ id: 'b', formula: null, kind: 'param', depth: 1, lane: 1.75 },
	{ id: 'y', formula: null, kind: 'data', depth: 3, lane: 2.1 },
	{ id: 'p', formula: 'w·x', kind: 'op', depth: 1, lane: 0.5 },
	{ id: 'z', formula: 'p + b', kind: 'op', depth: 2, lane: 1 },
	{ id: 'a', formula: 'tanh(z)', kind: 'op', depth: 3, lane: 1 },
	{ id: 'e', formula: 'a − y', kind: 'op', depth: 4, lane: 1.5 },
	{ id: 'L', formula: 'e·e', kind: 'op', depth: 5, lane: 1.5 }
];
export const NODE_INDEX: Record<string, number> = Object.fromEntries(NODES.map((n, i) => [n.id, i]));

export type Edge = { from: string; to: string; slot: number };
/** 'from>to:slot' entries, space-separated -> edges */
function edgesOf(list: string): Edge[] {
	return list.split(' ').map((e) => {
		const [from, rest] = e.split('>');
		const [to, slot] = rest.split(':');
		return { from, to, slot: Number(slot) };
	});
}
/** The graph's edges, parent -> child, with the parent's place among the child's inputs (slot 0 or 1). L = e·e has two
 *  edges from e. Listed in the order the backward pass walks them (children before parents: a reverse topological order). */
export const EDGES = edgesOf('e>L:0 e>L:1 a>e:0 y>e:1 z>a:0 p>z:0 b>z:1 w>p:0 x>p:1');

/** The local derivative of each edge, as a formula in the child's inputs (what the view prints next to "×"). */
export const LOCAL_TEXT: string[] = ['e', 'e', '1', '−1', '1 − a²', '1', '1', 'x', 'w'];

/** The step of the numerical check: (L(v + h) − L(v − h)) / 2h. */
export const CHECK_H = 1e-5;

// ---------------------------------------------------------------- the 2-2-1 network

/** One example (x1, x2) with target t, and the network's nine parameters, written out in state_dict order: the first layer's
 *  weights W (row by row: W₁₁ W₁₂ W₂₁ W₂₂) and biases b₁ b₂, the output layer's weights v₁ v₂ and bias c. */
export const NET_X = [1.0, -0.5];
export const NET_T = 0.8;
export const NET_P0 = [0.7, -0.4, -0.5, 0.9, 0.1, -0.2, 0.8, -0.6, 0.05];
export const NET_PARAMS = ['W₁₁', 'W₁₂', 'W₂₁', 'W₂₂', 'b₁', 'b₂', 'v₁', 'v₂', 'c'];
/** the learning rate of the one update the 'net' process ends with */
export const NET_LR = 0.1;


/** The network's nodes: two inputs, two hidden units (z = weighted sum, h = tanh z), the output ŷ and the loss. */
export const NET_NODES = ['x₁', 'x₂', 'z₁', 'z₂', 'h₁', 'h₂', 'ŷ', 'L'];

// ---------------------------------------------------------------- narration helpers

/** A number as the narration prints it, in parentheses when negative (for a product: "2 × (−0.296)"). */
export const par = (v: number): string | number => (Number.isFinite(v) ? (v < 0 ? `(${fmt3(v)})` : fmt3(v)) : NaN);

/** The task's goal: a loss above TASK_L while |dL/dw| is below TASK_G. */
export const TASK_L = 0.5;
export const TASK_G = 0.01;

type GraphLike = { vals?: { data: ArrayLike<number> }; grads?: { data: ArrayLike<number> }; known?: { data: ArrayLike<number> } };
/** A node's value in a frame's state (the toy graph: an id of NODES; the network: an index of NET_NODES). */
export function nodeVal(state: unknown, id: string | number): number {
	const s = (state ?? {}) as GraphLike;
	const k = typeof id === 'number' ? id : NODE_INDEX[id];
	return s.vals && k !== undefined ? s.vals.data[k] : NaN;
}
/** A node's gradient in a frame's state: NaN ("?") until the backward pass has reached it. */
export function nodeGrad(state: unknown, id: string | number): number {
	const s = (state ?? {}) as GraphLike;
	const k = typeof id === 'number' ? id : NODE_INDEX[id];
	return s.grads && s.known && k !== undefined && s.known.data[k] ? s.grads.data[k] : NaN;
}

// ---------------------------------------------------------------- the two graphs as the views draw them

/** How an edge is drawn: not yet crossed by the backward pass, crossed, or crossed in this frame. */
export type EdgeState = 'todo' | 'done' | 'active';

export interface GraphDef {
	nodes: GNode[];
	edges: Edge[];
}
export const NEURON: GraphDef = { nodes: NODES, edges: EDGES };

/** The 2-2-1 network drawn as a graph of its units (the weights are its edges; the table lists all nine parameters). */
export const NET_GRAPH: GraphDef = {
	nodes: [
		{ id: 'x₁', formula: null, kind: 'data', depth: 0, lane: 0 }, { id: 'x₂', formula: null, kind: 'data', depth: 0, lane: 1 },
		{ id: 'z₁', formula: 'W₁·x + b₁', kind: 'op', depth: 1, lane: 0 }, { id: 'z₂', formula: 'W₂·x + b₂', kind: 'op', depth: 1, lane: 1 },
		{ id: 'h₁', formula: 'tanh(z₁)', kind: 'op', depth: 2, lane: 0 }, { id: 'h₂', formula: 'tanh(z₂)', kind: 'op', depth: 2, lane: 1 },
		{ id: 'ŷ', formula: 'v·h + c', kind: 'op', depth: 3, lane: 0.5 }, { id: 'L', formula: '(ŷ − t)²', kind: 'op', depth: 4, lane: 0.5 }
	],
	edges: edgesOf('x₁>z₁:0 x₂>z₁:1 x₁>z₂:0 x₂>z₂:1 z₁>h₁:0 z₂>h₂:0 h₁>ŷ:0 h₂>ŷ:1 ŷ>L:0')
};
/** The edges of NET_GRAPH each backward phase of the 'net' process crosses, in order. */
export const NET_PHASES = ['out', 'hidden', 'first'];
export const NET_PHASE_EDGES: Record<string, number[]> = { out: [8], hidden: [6, 7, 4, 5], first: [0, 1, 2, 3] };
