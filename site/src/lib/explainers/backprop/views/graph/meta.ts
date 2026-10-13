// View 'backprop/graph': the toy computation graph of data.ts at a frame of the 'graph' process: every node's value and
// gradient ("?" until the backward pass reaches it), the edges already crossed, the edge crossed in this frame with its
// multiplication written out under the graph, a "grad +=" mark when that edge adds to a gradient that was already
// there, and at the 'check' frame a table of backpropagation against the numerical estimate. Props:
//   locals?: boolean    label every edge with its local derivative (state.locals)
//   stepper?: boolean   buttons that walk the source's frames (forward, start, one edge at a time, the check) whatever
//                       the step's cursor; the page's input edits keep the position
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { EDGES, LEAVES, LOCAL_TEXT, NODES, nodeGrad, nodeVal, type EdgeState } from '../../data.ts';

/** What the view draws (graphOf's result). */
export type GraphPic = NonNullable<ReturnType<typeof graphOf>>;

/** What the view draws for one frame (pure; also the text form). Null when the frame has no graph. */
export function graphOf(frame: { phase?: string; step?: number; metrics?: Record<string, number> } | undefined, state: unknown) {
	const s = state as { vals?: { data: ArrayLike<number> }; locals?: { data: ArrayLike<number> } } | undefined;
	if (!frame || !s?.vals) return null;
	const m = frame.metrics ?? {};
	const phase = frame.phase ?? '';
	const edge = phase === 'edge' ? (frame.step ?? -1) : -1;
	const crossed = phase === 'check' ? EDGES.length : phase === 'edge' ? edge : 0;
	return {
		phase,
		vals: NODES.map((n) => nodeVal(state, n.id)),
		grads: NODES.map((n) => nodeGrad(state, n.id)),
		locals: Array.from(s.locals?.data ?? [], Number),
		edges: EDGES.map((_, k): EdgeState => (k === edge ? 'active' : k < crossed ? 'done' : 'todo')),
		edge,
		up: m.up ?? NaN,
		local: m.local ?? NaN,
		before: m.before ?? NaN,
		after: m.after ?? NaN,
		acc: m.acc === 1,
		check: phase === 'check' ? LEAVES.map((l) => ({ id: l.id, g: nodeGrad(state, l.id), n: m[`n${l.id}`] ?? NaN })) : null,
		h: m.h ?? NaN,
		maxDiff: m.maxDiff ?? NaN
	};
}

const pn = (v: number) => (v < 0 ? `(${fmt3(v)})` : fmt3(v));

/** The line under the graph: what this frame did. */
export function captionOf(p: GraphPic): I18n {
	if (p.phase === 'forward') return { en: `Forward pass done: L = ${fmt3(p.vals[NODES.length - 1])}. No gradient is known yet.`, zh: `前向传播完成：L = ${fmt3(p.vals[NODES.length - 1])}。还没有任何梯度。` };
	if (p.phase === 'seed') return { en: 'The backward pass starts at L: dL/dL = 1.', zh: '反向传播从 L 开始：dL/dL = 1。' };
	if (p.phase === 'check') return { en: `Numerical check, h = ${fmt3(p.h)}: the largest difference from backpropagation is ${fmt3(p.maxDiff)}.`, zh: `数值检验，h = ${fmt3(p.h)}：与反向传播的最大差是 ${fmt3(p.maxDiff)}。` };
	if (p.edge < 0) return { en: '', zh: '' };
	const e = EDGES[p.edge];
	// "grad w = grad p × x = …" or, when the parent already had a gradient, "grad e += grad L × e = before + …"
	const rule = `grad ${e.from} ${p.acc ? '+=' : '='} grad ${e.to} × ${LOCAL_TEXT[p.edge]} = ${p.acc ? `${fmt3(p.before)} + ` : ''}${pn(p.up)} × ${pn(p.local)} = ${fmt3(p.after)}`;
	return { en: `Edge ${e.from} ← ${e.to}: ${rule}`, zh: `边 ${e.from} ← ${e.to}：${rule}` };
}

export function describeGraph(p: GraphPic | null): I18n {
	if (!p) return { en: 'No graph to draw yet.', zh: '还没有可画的计算图。' };
	const list = (lang: 'en' | 'zh') =>
		NODES.map((n, i) => `${n.id}${n.formula ? ` = ${n.formula}` : ''} = ${fmt3(p.vals[i])}, ${lang === 'en' ? 'gradient' : '梯度'} ${Number.isFinite(p.grads[i]) ? fmt3(p.grads[i]) : '?'}`).join('; ');
	const cap = captionOf(p);
	return {
		en: `A computation graph of ${NODES.length} nodes, L = (tanh(w·x + b) − y)²: ${list('en')}. ${cap.en}`,
		zh: `一张有 ${NODES.length} 个节点的计算图，L = (tanh(w·x + b) − y)²：${list('zh')}。${cap.zh}`
	};
}

const meta = {
	key: 'backprop/graph',
	title: { en: 'A computation graph, forward and backward', zh: '计算图的前向与反向' },
	kinds: ['process'],
	describe: (p: StageProps) => describeGraph(graphOf(p.at?.frame, p.at?.state)),
	demo: async () => {
		const [{ record }, { graph }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { x: 1.5, w: 0.8, b: -0.5, y: 0.9 };
		const tape = record(graph, input, { h: 1e-5 });
		const rt = fakeRuntime({ slug: 'backprop', input, tapes: { graph: tape } });
		const i = tape.find({ phase: 'edge', step: 1 });
		return { rt, source: 'graph', at: { index: i, frame: tape.frame(i), state: tape.state(i), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
