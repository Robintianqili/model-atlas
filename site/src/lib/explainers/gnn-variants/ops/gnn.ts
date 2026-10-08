// gnn-variants local ops (design §5.6). Python mirrors: pipeline/gnn-variants/ops_gnn.py (the same arithmetic as model.py).
// Every op reads H [n, i] and the adjacency A [n, n] and explains one output number as a weighted sum, so
// zoom.weighted-sum lists what each sender (or each half of a concatenated row) contributes:
//   gcn_conv   Â (H Wᵀ) + b; subs ahat (Â), xw (H Wᵀ). Terms: one per sender j, Âᵢⱼ × (H Wᵀ)ⱼ
//   sage_conv  [H ‖ mean of the friends' rows] Wᵀ + b; sub mean. Terms: the non-zero numbers of the concatenated row × W
//   gin_agg    (1 + ε) H + A H. Terms: the member's own number × (1 + ε), each friend's × 1
//   gat_conv   Σⱼ αᵢⱼ zⱼ + b, z = H Wᵀ, α = softmax of LeakyReLU(a_dst·zᵢ + a_src·zⱼ) over the senders; subs z, e, alpha
// f64 accumulation, f32 storage (atlas/ops/common.ts conventions); explain() recomputes with the same helpers. A sub-step
// unit [i, u] may lie outside the output (ahat is [n, n]): the output column is clamped and values.<sub> is the sub's value.
import type { AnyTensor, Axis, Dim, OpDef, Tensor, UnitExplanation, UnitQuery } from '../../../atlas/types/index.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, store, t, unitBase, weightName } from '../../../atlas/ops/common.ts';
import { leaky, normalize, senders } from '../graph.ts';

type NoP = Record<string, never>;
type GatP = { slope: number };
type Terms = NonNullable<UnitExplanation['terms']>;
const T = (shape: number[], data: Float32Array | Float64Array): Tensor => ({ shape, data });
const f32 = (v: ArrayLike<number>, like: AnyTensor) => { const o = outArray(v.length, like); o.set(Array.from(v)); return o; }; // prettier-ignore
/** H Wᵀ in f64, stored like H: [n, O] */
function times(h: AnyTensor, W: Tensor) {
	const [O, I] = W.shape;
	const z = outArray(h.shape[0] * O, h, W);
	for (let k = 0; k < z.length; k++) {
		let s = 0;
		for (let c = 0; c < I; c++) s += h.data[Math.floor(k / O) * I + c] * W.data[(k % O) * I + c];
		z[k] = s;
	}
	return z;
}
/** out[i, o] = Σⱼ w[i, j] · m[j, o] + b[o]: GCN (w = Â, m = H Wᵀ) and GAT (w = α, m = z) */
function mix(w: ArrayLike<number>, m: ArrayLike<number>, b: ArrayLike<number>, h: AnyTensor) {
	const [n, O] = [h.shape[0], b.length];
	const out = outArray(n * O, h);
	for (let k = 0; k < n * O; k++) {
		let s = 0;
		for (let j = 0; j < n; j++) s += w[Math.floor(k / O) * n + j] * m[j * O + (k % O)];
		out[k] = s + b[k % O];
	}
	return T([n, O], out);
}
/** explain one output of mix: one term per sender j; `subs` = the message and weight sub-steps first, then any other */
function mixExplain<P>(q: UnitQuery<P>, w: ArrayLike<number>, m: ArrayLike<number>, subs: Record<string, ArrayLike<number>>): UnitExplanation {
	const [i, u] = q.unit;
	const [n, O] = [q.ins[0].shape[0], q.out.shape[1]];
	const o = Math.min(u, O - 1);
	const terms = senders(q.ins[1].data, n, i).map((j) => ({ x: m[j * O + o], w: w[i * n + j], p: w[i * n + j] * m[j * O + o], xAt: j * O + o, wAt: i * n + j }));
	let s = 0;
	for (let j = 0; j < n; j++) s += w[i * n + j] * m[j * O + o];
	const b = q.w('bias').data[o];
	const e = unitBase('weighted-sum', q.layer, q.unit, store(s + b, q.out), q.out.data[i * O + o]);
	const [mName, wName] = Object.keys(subs);
	e.reads = [{ tensor: `${q.layer.id}#${mName}`, cells: terms.map((x) => x.xAt) }, { tensor: `${q.layer.id}#${wName}`, cells: terms.map((x) => x.wAt) }];
	const at = Object.fromEntries(Object.entries(subs).map(([k, v]) => [k, v[i * (v.length / n) + u]]));
	return Object.assign(e, { terms, bias: b, values: { n: terms.length, s, b, y: e.result, node: i, out: o, ...at } });
}
function shape(ins: Dim[][], O: Dim, I: Dim, what: string): Dim[] {
	if (ins[1].length !== 2 || ins[0].length !== 2) throw new Error(`${what}: needs H [n, i] and A [n, n]`);
	if (typeof ins[0][1] === 'number' && ins[0][1] !== I) throw new Error(`${what}: H must have ${I} numbers per member`);
	return [ins[0][0], O];
}
const common = { version: 1, attrs: {}, category: 'graph' as const, axes: (): Axis[] => ['n', 'd'], unit: { scan: 'rows' as const, pick: 'argmax' as const, axes: (o: Axis[]) => o.slice() }, zoom: 'weighted-sum' as const, color: 'diverging' as const, parity: { bar: 1e-4 }, vars: baseVars };
const SUM = (w: string, m: string) => `${sym('out', 'y')}_{ {{node}},{{out}} } = \\sum_j ${sym('in', w)}_{ {{node}},j }\\, ${m} + ${sym('param', 'b')}_{ {{out}} } = {{s}} + {{b}} = {{y}}`;

const gcnParts = (h: AnyTensor, a: AnyTensor, W: Tensor) => ({ ahat: f32(normalize(a.data, h.shape[0]).ahat, h), xw: times(h, W) });
export const gcn_conv: OpDef<NoP> = {
	...common, op: 'gcn_conv', label: t('GCN layer: normalized sum', 'GCN 层：归一化求和'), weights: [{ slot: 'weight' }, { slot: 'bias' }],
	subs: [{ name: 'ahat', label: t('Â: the weight of every edge', 'Â：每条边的权重'), axes: ['n', 'k'] }, { name: 'xw', label: t('Every member’s message H Wᵀ', '每个成员的消息 H Wᵀ'), axes: ['n', 'd'] }],
	inferShape: (ins, _p, w) => shape(ins, w.weight[0], w.weight[1], 'gcn_conv'),
	forward([h, a], _p, ctx) {
		const { ahat, xw } = gcnParts(h, a, ctx.w('weight'));
		ctx.emit('ahat', T([h.shape[0], h.shape[0]], ahat));
		ctx.emit('xw', T([h.shape[0], ctx.w('bias').data.length], xw));
		return mix(ahat, xw, ctx.w('bias').data, h);
	},
	explain(q) {
		const { ahat, xw } = gcnParts(q.ins[0], q.ins[1], q.w('weight'));
		return mixExplain(q, ahat, xw, { xw, ahat });
	},
	formula: SUM('\\hat A', `(${sym('in', 'H')}${sym('param', 'W')}^\\top)_{j,{{out}} }`),
	narration: { what: t('Every member’s row is multiplied by the learned weights W; then each member adds up its own and its friends’ results, each times a fixed weight Âᵢⱼ = 1 / √(dᵢ dⱼ) from the two degrees, and adds a bias. Shape {outShape}.', '每个成员的行先乘以学来的权重 W；然后每个成员把自己和朋友们的结果加起来，每一项乘一个固定权重 Âᵢⱼ = 1 / √(dᵢ dⱼ)（由两端的度决定），再加上偏置。形状是 {outShape}。'), how: t('A friend with many friends of its own counts less.', '自己朋友很多的朋友，分量更小。'), unit: t('{n|int} weighted messages add up to {s}; plus the bias {b} gives {y}.', '{n|int} 条加权消息加起来是 {s}；加上偏置 {b} 得到 {y}。') },
	pytorch: () => 'normalize(adj) @ (h @ self.weight.T) + self.bias', terms: ['weight', 'bias']
};

/** the friends' mean of H, f64 then stored like H: [n, I] (0 for a member without friends) */
function meanOf(h: AnyTensor, a: AnyTensor) {
	const [n, I] = h.shape;
	const m = outArray(n * I, h);
	for (let k = 0; k < n * I; k++) {
		const f = senders(a.data, n, Math.floor(k / I)).slice(1);
		let s = 0;
		for (const j of f) s += h.data[j * I + (k % I)];
		m[k] = f.length ? s / f.length : 0;
	}
	return m;
}
export const sage_conv: OpDef<NoP> = {
	...common, op: 'sage_conv', label: t('GraphSAGE layer: mean + concat', 'GraphSAGE 层：平均 + 拼接'), weights: [{ slot: 'weight' }, { slot: 'bias' }],
	subs: [{ name: 'mean', label: t('The average of the friends’ rows', '朋友们的行的平均'), axes: ['n', 'f'] }],
	inferShape: (ins, _p, w) => shape(ins, w.weight[0], w.weight[1] / 2, 'sage_conv'),
	forward([h, a], _p, ctx) {
		const [n, I] = h.shape;
		const [W, b] = [ctx.w('weight').data, ctx.w('bias').data];
		const m = meanOf(h, a);
		ctx.emit('mean', T([n, I], m));
		const out = outArray(n * b.length, h);
		for (let k = 0; k < out.length; k++) {
			const [i, o] = [Math.floor(k / b.length), k % b.length];
			let s = 0;
			for (let c = 0; c < 2 * I; c++) s += (c < I ? h.data[i * I + c] : m[i * I + c - I]) * W[o * 2 * I + c];
			out[k] = s + b[o];
		}
		return T([n, b.length], out);
	},
	explain(q) {
		const [i, u] = q.unit;
		const [h, I, O] = [q.ins[0], q.ins[0].shape[1], q.out.shape[1]];
		const [m, W, o] = [meanOf(h, q.ins[1]), q.w('weight').data, Math.min(u, O - 1)];
		const terms: Terms = [];
		for (let c = 0; c < 2 * I; c++) {
			const x = c < I ? h.data[i * I + c] : m[i * I + c - I];
			if (x !== 0) terms.push({ x, w: W[o * 2 * I + c], p: x * W[o * 2 * I + c], xAt: c, wAt: o * 2 * I + c });
		}
		const k = terms.filter((x) => x.xAt! < I).length;
		const own = terms.slice(0, k).reduce((acc, x) => acc + x.p, 0);
		const s = terms.reduce((acc, x) => acc + x.p, 0);
		const b = q.w('bias').data[o];
		const e = unitBase('weighted-sum', q.layer, q.unit, store(s + b, q.out), q.out.data[i * O + o]);
		e.reads = [{ tensor: `${q.layer.id}#mean`, cells: terms.slice(k).map((x) => i * I + x.xAt! - I) }, { tensor: weightName(q.layer, 'weight'), cells: terms.map((x) => x.wAt!) }];
		e.groups = [{ label: t('own row', '自己的行'), from: 0, to: k, sum: own }, { label: t('friends’ average', '朋友的平均'), from: k, to: terms.length, sum: s - own }];
		return Object.assign(e, { terms, bias: b, values: { n: terms.length, s, b, y: e.result, node: i, out: o, own, nb: s - own, mean: m[i * I + u] } });
	},
	formula: `${sym('out', 'y')}_{ {{node}},{{out}} } = \\sum_k [${sym('in', 'h')}_{ {{node}} } \\,\\Vert\\, \\mathrm{mean}_j\\, ${sym('in', 'h')}_j]_k\\, ${sym('param', 'W')}_{ {{out}},k } + ${sym('param', 'b')}_{ {{out}} } = {{own}} + {{nb}} + {{b}} = {{y}}`,
	narration: { what: t('Each member averages its friends’ rows (every friend counts the same), places its own row beside the average (concatenation) and multiplies both halves by the learned weights W, then adds a bias. Shape {outShape}.', '每个成员先把朋友们的行取平均（每个朋友分量相同），把自己的行和这个平均并排拼起来（拼接），两半一起乘以学来的权重 W，再加上偏置。形状是 {outShape}。'), how: t('The first half of W reads the member itself, the second half its friends’ average.', 'W 的前一半读成员自己，后一半读朋友们的平均。'), unit: t('Its own row gives {own}, the friends’ average {nb}; plus the bias {b}: {y}.', '自己的行贡献 {own}，朋友的平均贡献 {nb}；加上偏置 {b}：{y}。') },
	pytorch: () => 'self.lin(torch.cat([h, (adj @ h) / adj.sum(-1, keepdim=True).clamp(min=1)], -1))', terms: ['weight', 'bias']
};

export const gin_agg: OpDef<NoP> = {
	...common, op: 'gin_agg', label: t('GIN sum: (1 + ε) h + friends', 'GIN 求和：(1 + ε) h + 朋友'), weights: [{ slot: 'eps' }], axes: () => ['n', 'f'],
	inferShape: (ins) => shape(ins, ins[0][1], ins[0][1], 'gin_agg'),
	forward([h, a], _p, ctx) {
		const [n, I] = h.shape;
		const out = outArray(n * I, h);
		for (let k = 0; k < n * I; k++) {
			let s = (1 + ctx.w('eps').data[0]) * h.data[k];
			for (const j of senders(a.data, n, Math.floor(k / I)).slice(1)) s += h.data[j * I + (k % I)];
			out[k] = s;
		}
		return T([n, I], out);
	},
	explain(q) {
		const [i, c] = q.unit;
		const [h, I] = [q.ins[0], q.ins[0].shape[1]];
		const k1 = 1 + q.w('eps').data[0];
		const terms = senders(q.ins[1].data, h.shape[0], i).map((j, k) => ({ x: h.data[j * I + c], w: k ? 1 : k1, p: (k ? 1 : k1) * h.data[j * I + c], xAt: j * I + c }));
		let s = 0;
		for (const x of terms) s += x.p;
		const e = unitBase('weighted-sum', q.layer, q.unit, store(s, q.out), q.out.data[i * I + c]);
		e.reads = [{ tensor: q.layer.in[0], cells: terms.map((x) => x.xAt) }];
		return Object.assign(e, { terms, values: { k: terms.length - 1, eps: k1 - 1, onePlus: k1, self: terms[0].x, friends: s - terms[0].p, y: e.result, node: i, out: c } });
	},
	formula: `${sym('out', 'y')}_{ {{node}},{{out}} } = (1 + ${sym('param', '\\varepsilon')})\\,${sym('in', 'h')}_{ {{node}},{{out}} } + \\sum_{j \\in N( {{node}} )} ${sym('in', 'h')}_{j,{{out}} } = {{onePlus}} \\times {{self}} + {{friends}} = {{y}}`,
	narration: { what: t('Each member adds up its friends’ rows, without dividing, plus its own row times 1 + ε (ε is learned). Shape {outShape}.', '每个成员把朋友们的行直接加起来（不除），再加上自己的行乘以 1 + ε（ε 是学来的）。形状是 {outShape}。'), how: t('A sum keeps how many friends sent the same numbers; an average forgets it.', '求和记得有多少个朋友发来了同样的数；取平均就忘了。'), unit: t('{onePlus} × {self} + {friends} from {k|int} friends = {y}.', '{onePlus} × {self} + 来自 {k|int} 个朋友的 {friends} = {y}。') },
	pytorch: () => '(1 + self.eps) * h + adj @ h'
};

/** z = H Wᵀ, the scores e (−∞ off the senders) and α, f64 then stored like H */
function attend(h: AnyTensor, a: AnyTensor, w: (s: string) => Tensor, slope: number) {
	const n = h.shape[0];
	const [z, O] = [times(h, w('weight')), w('bias').data.length];
	const dot = (v: ArrayLike<number>, j: number) => { let s = 0; for (let o = 0; o < O; o++) s += z[j * O + o] * v[o]; return s; }; // prettier-ignore
	const [e, al] = [outArray(n * n, h).fill(-Infinity), outArray(n * n, h)];
	for (let i = 0; i < n; i++) {
		const S = senders(a.data, n, i);
		const sc = S.map((j) => leaky(dot(w('att_dst').data, i) + dot(w('att_src').data, j), slope));
		const ex = sc.map((v) => Math.exp(v - Math.max(...sc)));
		const tot = ex.reduce((x, y) => x + y, 0);
		S.forEach((j, k) => ((e[i * n + j] = sc[k]), (al[i * n + j] = ex[k] / tot)));
	}
	return { z, e, al };
}
export const gat_conv: OpDef<GatP> = {
	...common, op: 'gat_conv', attrs: { slope: { type: 'float', default: 0.2, doc: 'LeakyReLU slope for negative scores' } },
	label: t('GAT layer: attention over the friends', 'GAT 层：对朋友的注意力'), weights: [{ slot: 'weight' }, { slot: 'att_src' }, { slot: 'att_dst' }, { slot: 'bias' }],
	subs: [ { name: 'z', label: t('Every member’s message z = H Wᵀ', '每个成员的消息 z = H Wᵀ'), axes: ['n', 'd'] }, { name: 'e', label: t('Scores e (−∞ where there is no edge)', '分数 e（没有边的地方是 −∞）'), axes: ['n', 'k'] }, { name: 'alpha', label: t('Attention α: softmax over each member’s senders', '注意力 α：对每个成员的发送者做 softmax'), axes: ['n', 'k'] } ],
	inferShape: (ins, _p, w) => shape(ins, w.weight[0], w.weight[1], 'gat_conv'),
	forward([h, a], p, ctx) {
		const n = h.shape[0];
		const { z, e, al } = attend(h, a, ctx.w, p.slope ?? 0.2);
		ctx.emit('z', T([n, ctx.w('bias').data.length], z));
		ctx.emit('e', T([n, n], e));
		ctx.emit('alpha', T([n, n], al));
		return mix(al, z, ctx.w('bias').data, h);
	},
	explain(q) {
		const { z, e, al } = attend(q.ins[0], q.ins[1], q.w, q.p.slope ?? 0.2);
		return mixExplain(q, al, z, { z, alpha: al, e });
	},
	formula: SUM('\\alpha', `${sym('in', 'z')}_{j,{{out}} }`),
	narration: { what: t('Every member’s row becomes a message z = h Wᵀ. Each pair gets a score e = LeakyReLU(a_dst·zᵢ + a_src·zⱼ) from two learned vectors; a softmax over the member’s friends and itself turns the scores into weights α that add up to 1, and the member adds up α × z plus a bias. Shape {outShape}.', '每个成员的行变成一条消息 z = h Wᵀ。每一对成员由两个学来的向量得到一个分数 e = LeakyReLU(a_dst·zᵢ + a_src·zⱼ)；对这个成员的朋友和他自己做 softmax，把分数变成加起来等于 1 的权重 α，成员再把 α × z 加起来并加上偏置。形状是 {outShape}。'), how: t('The weights are learned from the numbers, not fixed by who is friends with whom.', '权重由一条学来的打分规则按这些数算出，不是由谁和谁是朋友固定下来的。'), unit: t('{n|int} messages times their attention weights add up to {s}; plus the bias {b} gives {y}.', '{n|int} 条消息乘上各自的注意力权重，加起来是 {s}；加上偏置 {b} 得到 {y}。') },
	pytorch: () => 'alpha = softmax(leaky_relu(z @ a_dst [:, None] + z @ a_src [None, :]) masked to the edges); alpha @ z + bias', terms: ['attention', 'softmax']
};
