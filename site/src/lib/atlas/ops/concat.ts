// concat: joins 2+ tensors along one axis (torch.cat; default axis 0). Values are copied unchanged.
import type { Dim, OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex, normAxis, numel } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { baseVars, int, outArray, t, unitBase } from './common.ts';

export type ConcatP = { axis: number };

export const concat: OpDef<ConcatP> = {
	op: 'concat',
	version: 1,
	attrs: { axis: { type: 'int', default: 0, doc: 'the axis to join along (negative counts from the end), as torch.cat(dim=…)' } },
	category: 'merge',
	label: t('Concatenate', '拼接'),
	inferShape(ins, p) {
		if (ins.length < 2) throw new Error('concat: needs at least two inputs');
		const rank = ins[0].length;
		const ax = normAxis(p.axis ?? 0, rank);
		let total: Dim = 0;
		for (const s of ins) {
			if (s.length !== rank) throw new Error(`concat: rank mismatch [${s}] vs [${ins[0]}]`);
			for (let d = 0; d < rank; d++) {
				if (d === ax) continue;
				if (typeof s[d] === 'number' && typeof ins[0][d] === 'number' && s[d] !== ins[0][d]) throw new Error(`concat: axis ${d} differs: [${s}] vs [${ins[0]}]`);
			}
			total = typeof total === 'number' && typeof s[ax] === 'number' ? total + (s[ax] as number) : `${total}+${s[ax]}`;
		}
		const out = ins[0].slice();
		out[ax] = total;
		return out;
	},
	axes: (ins) => ins[0].slice(),
	forward(xs, p) {
		const rank = xs[0].shape.length;
		const ax = normAxis(p.axis ?? 0, rank);
		const outer = numel(xs[0].shape.slice(0, ax));
		const inner = numel(xs[0].shape.slice(ax + 1));
		let total = 0;
		for (const x of xs) total += x.shape[ax];
		const out = outArray(outer * total * inner, ...xs);
		let off = 0;
		for (const x of xs) {
			const len = x.shape[ax] * inner;
			for (let o = 0; o < outer; o++) {
				const dst = o * total * inner + off * inner;
				for (let i = 0; i < len; i++) out[dst + i] = x.data[o * len + i];
			}
			off += x.shape[ax];
		}
		const shape = xs[0].shape.slice();
		shape[ax] = total;
		return { shape, data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'first' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const rank = q.out.shape.length;
		const ax = normAxis(int(q.p, 'axis', 0), rank);
		let pos = q.unit[ax];
		let part = 0;
		while (part < q.ins.length - 1 && pos >= q.ins[part].shape[ax]) {
			pos -= q.ins[part].shape[ax];
			part++;
		}
		const idx = q.unit.slice();
		idx[ax] = pos;
		const src = q.ins[part];
		const at = flatIndex(src.shape, idx);
		const v = src.data[at];
		const y = q.out.data instanceof Float64Array ? v : Math.fround(v);
		const e = unitBase('index-map', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [{ tensor: q.layer.in[part], cells: [at] }];
		e.values = { y, part: part + 1, index: at, parts: q.ins.length };
		return e;
	},
	formula: `${sym('out', 'y')} = [\\,${sym('in', 'a')}\\ ;\\ ${sym('in', 'b')}\\,] \\Rightarrow {{y}}`,
	narration: {
		what: t(
			'Concatenation joins {parts|int} tensors end to end along one axis. No number changes; the result has shape {outShape}.',
			'拼接把 {parts|int} 个张量沿一个轴首尾相接。数值不变，结果的形状是 {outShape}。'
		),
		how: t('The inputs must agree on every other axis; only the joined axis grows.', '除了被拼接的轴，其他轴的长度必须相同；只有被拼接的轴变长。'),
		why: t('It lets the next layer see several sources of information side by side.', '这样下一层可以同时看到几路信息。'),
		look: t('Each part keeps its values; look for where one part ends and the next begins.', '每一部分都保留原来的值；找找一部分在哪里结束、下一部分从哪里开始。'),
		unit: t('This value, {y}, comes unchanged from part {part|int}.', '这个值 {y} 原样来自第 {part|int} 部分。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), parts: layer.in.length, axis: Number((layer.params ?? {}).axis ?? 0) }),
	zoom: 'index-map',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: (layer) => `torch.cat([${layer.in.join(', ')}], dim=${Number((layer.params ?? {}).axis ?? 0)})`,
	terms: ['tensor', 'shape']
};

export default concat;
