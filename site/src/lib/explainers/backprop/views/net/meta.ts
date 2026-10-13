// View 'backprop/net': the 2-2-1 network of data.ts at a frame of the 'net' process, drawn as a graph of its units
// (values, gradients "?" until the backward pass reaches them; the edges of the layer crossed in this frame highlighted)
// and a table of its nine parameters: value, gradient, and from the check frame on the numerical estimate; at the
// 'update' frame also each parameter after one gradient-descent step. Props:
//   check?: boolean   show the numerical column (default: from the check frame on)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { NET_GRAPH, NET_NODES, NET_P0, NET_PARAMS, NET_PHASES, NET_PHASE_EDGES, nodeGrad, nodeVal, type EdgeState } from '../../data.ts';

/** What the view draws (netOf's result). */
export type NetPic = NonNullable<ReturnType<typeof netOf>>;

type Arr = { data: ArrayLike<number> };

/** What the view draws (pure; also the text form). Null when the frame has no network. */
export function netOf(frame: { phase?: string; metrics?: Record<string, number> } | undefined, state: unknown, props: Record<string, unknown> = {}) {
	const s = state as { params?: Arr; pgrads?: Arr; pknown?: Arr; pnum?: Arr } | undefined;
	if (!frame || !s?.params) return null;
	const phase = frame.phase ?? '';
	const k = NET_PHASES.indexOf(phase);
	const doneUpTo = phase === 'forward' ? 0 : k < 0 ? NET_PHASES.length : k;
	const done = new Set(NET_PHASES.slice(0, doneUpTo).flatMap((p) => NET_PHASE_EDGES[p]));
	const active = new Set(k >= 0 ? NET_PHASE_EDGES[phase] : []);
	const late = phase === 'check' || phase === 'update';
	return {
		phase,
		vals: NET_NODES.map((_, i) => nodeVal(state, i)),
		grads: NET_NODES.map((_, i) => nodeGrad(state, i)),
		edges: NET_GRAPH.edges.map((_, e): EdgeState => (active.has(e) ? 'active' : done.has(e) ? 'done' : 'todo')),
		rows: NET_PARAMS.map((name, i) => ({
			name,
			value: NET_P0[i],
			grad: s.pknown?.data[i] ? (s.pgrads?.data[i] ?? NaN) : NaN,
			num: late ? (s.pnum?.data[i] ?? NaN) : NaN,
			after: phase === 'update' ? s.params!.data[i] : NaN
		})),
		showNum: late || props.check === true,
		showAfter: phase === 'update',
		maxDiff: frame.metrics?.maxDiff ?? NaN
	};
}

export function describeNet(p: NetPic | null): I18n {
	if (!p) return { en: 'No network to draw yet.', zh: '还没有可画的网络。' };
	const nodes = NET_NODES.map((n, i) => `${n} = ${fmt3(p.vals[i])} (${Number.isFinite(p.grads[i]) ? fmt3(p.grads[i]) : '?'})`).join(', ');
	const rows = p.rows.map((r) => `${r.name} ${fmt3(r.value)}: ${Number.isFinite(r.grad) ? fmt3(r.grad) : '?'}${p.showNum && Number.isFinite(r.num) ? ` / ${fmt3(r.num)}` : ''}${p.showAfter ? ` → ${fmt3(r.after)}` : ''}`).join('; ');
	return {
		en: `A 2-2-1 network as a graph of its units, each with its value and, in parentheses, its gradient: ${nodes}. Its nine parameters, value: gradient${p.showNum ? ' / numerical estimate' : ''}${p.showAfter ? ' → after one update' : ''}: ${rows}.`,
		zh: `把 2-2-1 网络画成由单元组成的图，每个单元给出数值，括号里是梯度：${nodes}。九个参数，数值：梯度${p.showNum ? ' / 数值估计' : ''}${p.showAfter ? ' → 一次更新之后' : ''}：${rows}。`
	};
}

const meta = {
	key: 'backprop/net',
	title: { en: 'Backpropagation in a 2-2-1 network', zh: '2-2-1 网络中的反向传播' },
	kinds: ['process'],
	describe: (p: StageProps) => describeNet(netOf(p.at?.frame, p.at?.state, p.props)),
	demo: async () => {
		const [{ record }, { net }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const tape = record(net, null, { h: 1e-5, lr: 0.1 });
		const rt = fakeRuntime({ slug: 'backprop', input: null, tapes: { net: tape } });
		const i = tape.find({ phase: 'hidden' });
		return { rt, source: 'net', at: { index: i, frame: tape.frame(i), state: tape.state(i), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
