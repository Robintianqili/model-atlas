// add: a + b cell by cell (residual connections, token + position). Same shapes take the plain loop of tinygpt.ts
// add() (y[i] = a[i] + b[i]); otherwise numpy-style broadcasting of b (or a) over leading axes.
import type { Dim, OpDef, Tensor, UnitExplanation } from '../types/index.ts';
import { flatIndex, numel, shapeEq, unravel } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { baseVars, dependsOn, inputName, isResidualAdd, outArray, store, t, unitBase, words, type StructuralTerms } from './common.ts';

type NoP = Record<string, never>;

function broadcastShape(a: readonly Dim[], b: readonly Dim[]): Dim[] {
	const r = Math.max(a.length, b.length);
	const out: Dim[] = [];
	for (let i = 0; i < r; i++) {
		const x = a[a.length - r + i] ?? 1;
		const y = b[b.length - r + i] ?? 1;
		if (x === y || y === 1) out.push(x);
		else if (x === 1) out.push(y);
		else throw new Error(`add: shapes [${a}] and [${b}] do not broadcast`);
	}
	return out;
}

/** flat index into `src` for output multi-index `idx` under broadcasting */
function srcIndex(src: readonly number[], idx: readonly number[]): number {
	const off = idx.length - src.length;
	let f = 0;
	for (let i = 0; i < src.length; i++) f = f * src[i] + (src[i] === 1 ? 0 : idx[off + i]);
	return f;
}

export const add: OpDef<NoP> & StructuralTerms = {
	op: 'add',
	version: 1,
	attrs: {},
	category: 'merge',
	label: t('Add', '相加'),
	inferShape(ins) {
		if (ins.length !== 2) throw new Error('add: needs exactly two inputs');
		return broadcastShape(ins[0], ins[1]);
	},
	axes: (ins) => (ins[0].length >= ins[1].length ? ins[0] : ins[1]).slice(),
	forward(xs) {
		const [a, b] = xs;
		if (shapeEq(a.shape, b.shape)) {
			const y = outArray(a.data.length, a, b);
			for (let i = 0; i < y.length; i++) y[i] = a.data[i] + b.data[i];
			return { shape: a.shape.slice(), data: y };
		}
		const shape = broadcastShape(a.shape, b.shape) as number[];
		const n = numel(shape);
		const y = outArray(n, a, b);
		for (let i = 0; i < n; i++) {
			const idx = unravel(shape, i);
			y[i] = a.data[srcIndex(a.shape, idx)] + b.data[srcIndex(b.shape, idx)];
		}
		return { shape, data: y };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmaxAbs' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const [a, b] = q.ins as Tensor[];
		const ia = srcIndex(a.shape, q.unit);
		const ib = srcIndex(b.shape, q.unit);
		const av = a.data[ia];
		const bv = b.data[ib];
		const y = store(av + bv, q.out);
		const e = unitBase('elementwise', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [
			{ tensor: q.layer.in[0], cells: [ia] },
			{ tensor: q.layer.in[1], cells: [ib] }
		];
		e.terms = [
			{ x: av, w: 1, p: av, xAt: ia },
			{ x: bv, w: 1, p: bv, xAt: ib }
		];
		e.values = { a: av, b: bv, y };
		return e;
	},
	receptive: (box) => ({ ...box }),
	formula: `${sym('out', 'y')} = ${sym('in', 'a')} + ${sym('in', 'b')} = {{a}} + {{b}} = {{y}}`,
	narration: {
		// {aName}, {bName}: the labels of the two inputs; {addRole}: a residual connection (one input is computed from the
		// other) or an add of two independent inputs (TinyGPT's token + position embeddings). Chosen by vars().
		what: t(
			'Add sums two tensors cell by cell: each output number is a + b, with a taken from “{aName}” and b from “{bName}”. {addRole}',
			'相加把两个张量逐格相加：每个输出数等于 a + b，a 来自“{aName}”，b 来自“{bName}”。{addRole}'
		),
		how: t('The result has shape {outShape}.', '结果的形状是 {outShape}。'),
		why: t(
			'Adding keeps the information of both inputs in one tensor. In a residual connection it also keeps the earlier numbers available to later layers, which makes deep stacks easier to train.',
			'相加把两个输入的信息放进同一个张量。在残差连接里，它还让前面的数在后面的层里仍然可用，使很深的网络更容易训练。'
		),
		look: t('Where one input is close to 0, the output looks like the other input.', '某个输入接近 0 的地方，输出就和另一个输入差不多。'),
		unit: t('{a} + {b} = {y}.', '{a} + {b} = {y}。')
	},
	vars(layer, spec) {
		const [ia, ib] = layer.in;
		const role =
			ia !== undefined && ib !== undefined && dependsOn(spec, ib, ia)
				? words(
						'This is a residual connection: b was computed from a, and adding a back means that part of the network only has to learn a change to a.',
						'这是残差连接：b 是从 a 算出来的；把 a 加回去，这部分网络就只需要学习对 a 的“改动”。'
					)
				: ia !== undefined && ib !== undefined && dependsOn(spec, ia, ib)
					? words(
							'This is a residual connection: a was computed from b, and adding b back means that part of the network only has to learn a change to b.',
							'这是残差连接：a 是从 b 算出来的；把 b 加回去，这部分网络就只需要学习对 b 的“改动”。'
						)
					: words('Neither input is computed from the other: adding them puts both kinds of information into the same numbers.', '两个输入互不依赖：相加把两种信息放进同一组数里。');
		return {
			...baseVars(layer, spec),
			aName: inputName(spec, ia ?? 'a'),
			bName: inputName(spec, ib ?? 'b'),
			addRole: role
		};
	},
	zoom: 'elementwise',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: (layer) => `${layer.in[0] ?? 'a'} + ${layer.in[1] ?? 'b'}`,
	terms: ['residual-connection'],
	// "residual connection" only where one input is computed from the other (not on token + position)
	termsFor: (layer, spec) => (isResidualAdd(layer, spec) ? ['residual-connection'] : [])
};

export default add;
