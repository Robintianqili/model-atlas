// View 'network' (kind B): an MLP as a node-link diagram on the recorded weights. Edges are colored by sign (blue
// negative, orange positive) and drawn thicker for larger values; nodes show the bias, or a value for one example of the
// batch when the frame has activations (fine detail).
// Props (StageProps.props):
//   color?: 'weight' | 'grad' | 'update'   what the edges show (default 'weight'; 'grad' needs grads, 'update' prev)
//   field?: string            the record-valued state field with the weights (default 'params')
//   inputs?: I18n[] | string[]; outputs?: I18n[] | string[]   node labels at both ends
//   sample?: number           which example of the batch colors the nodes (default 0)
//   maxNodes?: number         nodes drawn per layer before eliding (default 12)
// Focus { field: 'params' | 'grads', key: '<prefix>.weight', index: [out, in] } highlights one edge.
import type { I18n, Tensor } from '../../../types/core.ts';
import type { StageProps, ViewMeta } from '../../../types/view.ts';
import { fmt3, fmtInt } from '../../../core/format.ts';
import { mlpLayers, mlpSizes } from '../../train/refs.ts';

export type EdgeColor = 'weight' | 'grad' | 'update';

export interface NetLayer {
	prefix: string;
	weight: Tensor;
	bias?: Tensor;
	/** what the edges show (weight, its gradient, or its change in this update) */
	edge: Float64Array;
}

export interface NetData {
	layers: NetLayer[];
	sizes: number[] | null;
	color: EdgeColor;
	/** node values per layer for one example (activations), when the frame has them */
	values: (Float64Array | null)[] | null;
	maxAbs: number;
	focus: { layer: number; o: number; i: number } | null;
	params: number;
	problem: I18n | null;
}

function rec(state: Record<string, unknown> | undefined, name: string): Record<string, Tensor> | undefined {
	const v = state?.[name];
	return v && typeof v === 'object' ? (v as Record<string, Tensor>) : undefined;
}

/** Everything the view draws (pure). */
export function netData(p: Pick<StageProps, 'at' | 'props' | 'focus'>): NetData {
	const props = p.props ?? {};
	const state = p.at?.state as Record<string, unknown> | undefined;
	const field = typeof props.field === 'string' ? props.field : 'params';
	const params = rec(state, field);
	const raw = mlpLayers(params);
	const sizes = mlpSizes(raw);
	let color: EdgeColor = props.color === 'grad' || props.color === 'update' ? props.color : 'weight';
	const grads = rec(state, 'grads');
	const prev = rec(state, 'prev');
	if (color === 'grad' && !grads) color = 'weight';
	if (color === 'update' && !prev) color = 'weight';
	let maxAbs = 0;
	let count = 0;
	const layers: NetLayer[] = raw.map((l) => {
		const key = `${l.prefix}.weight`;
		let edge: Float64Array;
		if (color === 'grad') edge = Float64Array.from(grads![key]?.data ?? new Float64Array(l.weight.data.length));
		else if (color === 'update') {
			const old = prev![key]?.data;
			edge = Float64Array.from(l.weight.data, (w, i) => (old ? w - old[i] : 0));
		} else edge = Float64Array.from(l.weight.data);
		for (const v of edge) maxAbs = Math.max(maxAbs, Math.abs(v));
		count += l.weight.data.length + (l.bias?.data.length ?? 0);
		return { prefix: l.prefix, weight: l.weight, bias: l.bias, edge };
	});
	let values: (Float64Array | null)[] | null = null;
	const acts = rec(state, 'acts');
	const batch = rec(state, 'batch');
	if (acts && sizes) {
		const s = Math.max(0, Math.floor(Number(props.sample ?? 0)) || 0);
		const pick = (t: Tensor | undefined, width: number): Float64Array | null => {
			if (!t || t.data.length < (s + 1) * width) return null;
			return Float64Array.from(t.data.subarray ? t.data.subarray(s * width, (s + 1) * width) : Array.from(t.data).slice(s * width, (s + 1) * width));
		};
		const inputT = batch ? Object.values(batch).find((t) => t.shape.length === 2 && t.shape[1] === sizes[0]) : undefined;
		values = [pick(inputT, sizes[0])];
		layers.forEach((l, k) => {
			// post-activation outputs are recorded under the next module index (net.1, net.3, …); the last layer's own
			const idx = Number(l.prefix.split('.').pop());
			const base = l.prefix.slice(0, l.prefix.lastIndexOf('.') + 1);
			const after = k + 1 < layers.length ? acts[`${base}${idx + 1}`] : acts[l.prefix];
			values!.push(pick(after ?? acts[l.prefix], sizes[k + 1]));
		});
		if (values.every((v) => v === null)) values = null;
	}
	let focus: NetData['focus'] = null;
	for (const f of p.focus ?? []) {
		if (!f.key || !Array.isArray(f.index) || f.index.length !== 2 || typeof f.index[0] !== 'number') continue;
		const li = layers.findIndex((l) => `${l.prefix}.weight` === f.key);
		if (li >= 0) focus = { layer: li, o: f.index[0] as number, i: f.index[1] as number };
	}
	const problem: I18n | null = !params ? { en: 'This frame has no weights yet.', zh: '这一帧还没有权重。' } : !sizes ? { en: 'These weights do not form a simple chain of layers.', zh: '这些权重不是一串简单相连的层。' } : null;
	return { layers, sizes, color, values, maxAbs, focus, params: count, problem };
}

/** The largest edge value: { layer, o, i, v }. */
export function strongest(d: NetData): { layer: number; o: number; i: number; v: number } | null {
	let best: { layer: number; o: number; i: number; v: number } | null = null;
	d.layers.forEach((l, li) => {
		const I = l.weight.shape[1];
		for (let k = 0; k < l.edge.length; k++) if (!best || Math.abs(l.edge[k]) > Math.abs(best.v)) best = { layer: li, o: Math.floor(k / I), i: k % I, v: l.edge[k] };
	});
	return best;
}

const WHAT: Record<EdgeColor, I18n> = {
	weight: { en: 'weight', zh: '权重' },
	grad: { en: 'gradient', zh: '梯度' },
	update: { en: 'change in this update', zh: '这次更新的变化量' }
};

const meta = {
	key: 'network',
	title: { en: 'The network', zh: '网络' },
	kinds: ['train'],
	accepts: (o) => o.field === 'params' || o.field === 'grads',
	describe(p: StageProps): I18n {
		const d = netData(p);
		if (d.problem || !d.sizes) return d.problem ?? { en: 'No network to show.', zh: '没有可显示的网络。' };
		const s = strongest(d);
		const what = WHAT[d.color];
		const big = s ? { en: ` The largest ${what.en} is ${fmt3(s.v)}, into node ${s.o} of layer ${s.layer + 1} from node ${s.i} before it (nodes count from 0).`, zh: `最大的${what.zh}是 ${fmt3(s.v)}，从前一层的节点 ${s.i} 连到第 ${s.layer + 1} 层的节点 ${s.o}（节点从 0 数起）。` } : { en: '', zh: '' };
		return {
			en: `A network with layers ${d.sizes.join(' → ')}: ${fmtInt(d.params)} weights and biases. Each line is one ${what.en}: blue below zero, orange above, thicker when larger.${big.en}`,
			zh: `一个各层大小为 ${d.sizes.join(' → ')} 的网络：一共 ${fmtInt(d.params)} 个权重和偏置。每条线是一个${what.zh}：小于零为蓝色，大于零为橙色，越大越粗。${big.zh}`
		};
	},
	// the demo helpers (training code) load only when a demo is asked for, so this meta stays light
	demo: async () => {
		const { classifierRun, demoProps } = await import('../../train/demo.ts');
		return demoProps(classifierRun('coarse', 300), 'last', { inputs: ['x', 'y'], outputs: [{ en: 'class 0', zh: '第 0 类' }, { en: 'class 1', zh: '第 1 类' }] }, [{ field: 'params', key: 'net.0.weight', index: [2, 1], role: 'param' }]);
	},
	minHeight: 280
} satisfies ViewMeta;

export default meta;
