// embedding: ids [T] -> rows of a learned table W [V, D]: y[t] = W[id(t)] (a copy; tinygpt.ts tok.set(tokW.subarray(...))).
// position_embedding: y[t] = P[t] for t = 0..T−1 (only the length of the input is used).
import type { AnyTensor, OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { baseVars, outArray, specWeightShape, t, unitBase, weightName } from './common.ts';

type NoP = Record<string, never>;

function idAt(x: AnyTensor, i: number, V: number, layer: string): number {
	const id = x.data[i];
	if (!Number.isInteger(id) || id < 0 || id >= V) throw new Error(`embedding ${layer}: id ${id} at position ${i} is outside 0..${V - 1}`);
	return id;
}

export const embedding: OpDef<NoP> = {
	op: 'embedding',
	version: 1,
	attrs: {},
	category: 'embedding',
	label: t('Embedding', '嵌入'),
	weights: [{ slot: 'weight' }],
	inferShape(ins, _p, w) {
		const ws = w.weight;
		if (!ws || ws.length !== 2) throw new Error('embedding: weight must be [V, D]');
		return [...ins[0], ws[1]];
	},
	axes: (ins) => [...ins[0], 'd'],
	forward(xs, _p, ctx) {
		const ids = xs[0];
		const W = ctx.w('weight');
		const [V, D] = W.shape;
		const T = ids.data.length;
		const out = outArray(T * D, W);
		for (let i = 0; i < T; i++) {
			const id = idAt(ids, i, V, ctx.layer.id);
			out.set(W.data.subarray(id * D, id * D + D), i * D);
		}
		return { shape: [...ids.shape, D], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'last' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const W = q.w('weight');
		const [V, D] = W.shape;
		const d = q.unit[q.unit.length - 1];
		const pos = (f - d) / D;
		const id = idAt(q.ins[0], pos, V, q.layer.id);
		const y = W.data[id * D + d];
		const e = unitBase('lookup', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [
			{ tensor: q.layer.in[0], cells: [pos] },
			{ tensor: weightName(q.layer, 'weight'), cells: [id * D + d] }
		];
		e.values = { id, t: pos, d, y, row: id, tNo: pos + 1 };
		return e;
	},
	formula: `${sym('out', 'y')}_{t,d} = ${sym('param', 'E')}_{\\mathrm{id}(t),\\,d} = {{y}}`,
	narration: {
		what: t(
			'An embedding is a table with one row of {dim|int} learned numbers for each of the {vocab|int} possible ids. Each input id simply picks its row.',
			'嵌入是一张表：{vocab|int} 个可能的编号，每个编号对应一行 {dim|int} 个学来的数。每个输入编号只是取出自己的那一行。'
		),
		how: t(
			'Nothing is multiplied: the output at each position is an exact copy of the row of the id there. The rows are learned during training like any other weights.',
			'这里没有乘法：每个位置的输出就是那里的编号对应那一行的原样副本。这些行和其他权重一样，是训练学出来的。'
		),
		why: t('A row of numbers gives the next layers something to compute with; the size of a bare id number means nothing.', '一行数字可以让后面的层进行计算；而一个裸编号的大小本身没有意义。'),
		look: t('Repeated ids give identical rows.', '相同的编号得到完全相同的行。'),
		unit: t('Position {t|int} has id {id|int}, so it copies row {id|int}; entry {d|int} of that row is {y} (counting from 0).', '位置 {t|int} 的编号是 {id|int}，所以复制第 {id|int} 行；这一行的第 {d|int} 个数是 {y}（从 0 数起）。')
	},
	vars(layer, spec) {
		const ws = specWeightShape(layer, spec, 'weight');
		return { ...baseVars(layer, spec), vocab: ws?.[0] ?? '?', dim: ws?.[1] ?? layer.outShape[layer.outShape.length - 1] };
	},
	zoom: 'lookup',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch(layer) {
		const m = /^(\d+)$/.exec(String(layer.outShape[layer.outShape.length - 1]));
		return `nn.Embedding(num_embeddings, ${m ? m[1] : 'dim'})`;
	},
	terms: ['embedding', 'token', 'vocabulary']
};

export const position_embedding: OpDef<NoP> = {
	op: 'position_embedding',
	version: 1,
	attrs: {},
	category: 'embedding',
	label: t('Position embedding', '位置嵌入'),
	weights: [{ slot: 'weight' }],
	inferShape(ins, _p, w) {
		const ws = w.weight;
		if (!ws || ws.length !== 2) throw new Error('position_embedding: weight must be [P, D]');
		const T = ins[0][0];
		if (typeof T === 'number' && T > ws[0]) throw new Error(`position_embedding: ${T} positions but the table has only ${ws[0]} rows`);
		return [T, ws[1]];
	},
	axes: () => ['t', 'd'],
	forward(xs, _p, ctx) {
		const T = xs[0].shape[0];
		const W = ctx.w('weight');
		const [P, D] = W.shape;
		if (T > P) throw new Error(`position_embedding ${ctx.layer.id}: ${T} positions but the table has only ${P} rows`);
		const out = outArray(T * D, W);
		for (let i = 0; i < T; i++) out.set(W.data.subarray(i * D, i * D + D), i * D);
		return { shape: [T, D], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'rows', pick: 'last' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const W = q.w('weight');
		const D = W.shape[1];
		const [pos, d] = q.unit;
		const y = W.data[pos * D + d];
		const e = unitBase('lookup', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [
			{ tensor: q.layer.in[0], cells: [pos] },
			{ tensor: weightName(q.layer, 'weight'), cells: [pos * D + d] }
		];
		e.values = { t: pos, d, y, row: pos, tNo: pos + 1 };
		return e;
	},
	formula: `${sym('out', 'y')}_{t,d} = ${sym('param', 'P')}_{t,d} = {{y}}`,
	narration: {
		what: t(
			'A position embedding is a table with one learned row of {dim|int} numbers for each position, {positions|int} rows in all, so the model can read at most {positions|int} tokens at once. Position t gets row t, whatever the token there is.',
			'位置嵌入是一张表：每个位置一行 {dim|int} 个学来的数，一共 {positions|int} 行，所以模型一次最多读 {positions|int} 个词元。位置 t 取第 t 行，不管那里是什么词元。'
		),
		how: t('Usually it is added to the token embedding, so each row carries both what the token is and where it is.', '通常把它加到词元嵌入上，这样每一行既包含“是什么”，也包含“在哪里”。'),
		why: t('Attention by itself does not know the order of tokens; these rows tell the model where each token sits.', '注意力本身不知道词元的顺序；这些行告诉模型每个词元在什么位置。'),
		look: t('The rows depend only on the position, so the same word at another position gets a different row.', '这些行只和位置有关，所以同一个词放在另一个位置，就会得到不同的行。'),
		unit: t('Position {t|int} copies its own row, row {t|int}; entry {d|int} of that row is {y} (counting from 0).', '位置 {t|int} 复制自己的那一行，即第 {t|int} 行；这一行的第 {d|int} 个数是 {y}（从 0 数起）。')
	},
	vars(layer, spec) {
		const ws = specWeightShape(layer, spec, 'weight');
		return { ...baseVars(layer, spec), positions: ws?.[0] ?? '?', dim: ws?.[1] ?? layer.outShape[layer.outShape.length - 1] };
	},
	zoom: 'lookup',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'pos_emb(torch.arange(T))',
	terms: ['positional-encoding', 'embedding']
};

export default embedding;
