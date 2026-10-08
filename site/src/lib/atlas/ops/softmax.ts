// softmax along one axis (default −1). Same arithmetic as tinycnn.ts softmax() and tinygpt.ts softmaxRows():
//   m = max (`if (x > m) m = x`); e_i = exp(x_i − m) in f64; s = Σ e_i; p_i = e_i / s (stored f32). −∞ entries get 0.
// explain(): terms[j] over the unit's row = { x: score, w: exp(score − max), p: probability }; values.e/sum/max/y.
import type { OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex, normAxis, numel } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { baseVars, intText, outputLayer, outArray, store, t, unitBase, words, type StructuralTerms } from './common.ts';

export type SoftmaxP = { axis: number };

export const softmax: OpDef<SoftmaxP> & StructuralTerms = {
	op: 'softmax',
	version: 1,
	attrs: { axis: { type: 'int', default: -1 } },
	category: 'output',
	label: t('Softmax', 'Softmax'),
	inferShape: (ins, p) => {
		normAxis(p.axis ?? -1, ins[0].length);
		return ins[0].slice();
	},
	axes: (ins) => ins[0].slice(),
	forward(xs, p) {
		const src = xs[0];
		const x = src.data;
		const rank = src.shape.length;
		const ax = normAxis(p.axis ?? -1, rank);
		const n = src.shape[ax];
		const inner = numel(src.shape.slice(ax + 1));
		const outer = numel(src.shape.slice(0, ax));
		const y = outArray(x.length, src);
		const tmp = new Float64Array(n);
		for (let a = 0; a < outer; a++) {
			for (let b = 0; b < inner; b++) {
				const o = a * n * inner + b;
				let m = -Infinity;
				for (let i = 0; i < n; i++) if (x[o + i * inner] > m) m = x[o + i * inner];
				let s = 0;
				for (let i = 0; i < n; i++) {
					const e = Math.exp(x[o + i * inner] - m);
					tmp[i] = e;
					s += e;
				}
				for (let i = 0; i < n; i++) y[o + i * inner] = tmp[i] / s;
			}
		}
		return { shape: src.shape.slice(), data: y };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const src = q.ins[0];
		const x = src.data;
		const ax = normAxis(q.p.axis ?? -1, src.shape.length);
		const n = src.shape[ax];
		const inner = numel(src.shape.slice(ax + 1));
		const k = q.unit[ax];
		const o = f - k * inner;
		let m = -Infinity;
		for (let i = 0; i < n; i++) if (x[o + i * inner] > m) m = x[o + i * inner];
		let s = 0;
		const es = new Float64Array(n);
		for (let i = 0; i < n; i++) {
			const e = Math.exp(x[o + i * inner] - m);
			es[i] = e;
			s += e;
		}
		const y = store(es[k] / s, q.out);
		const e = unitBase('softmax', q.layer, q.unit, y, q.out.data[f]);
		const cells = Array.from({ length: n }, (_, i) => o + i * inner);
		e.reads = [{ tensor: q.layer.in[0], cells }];
		e.terms = cells.map((c, i) => ({ x: x[c], w: es[i], p: store(es[i] / s, q.out), xAt: c }));
		e.steps = [
			{ label: t('largest score in the row', '这一行最大的分数'), value: m, tex: 'm' },
			{ label: t('exp(score − largest)', 'exp(分数 − 最大值)'), value: es[k], tex: 'e^{x_i - m}' },
			{ label: t('sum over the row', '这一行的总和'), value: s, tex: '\\sum_j e^{x_j - m}' },
			{ label: t('share', '份额'), value: y, tex: 'p_i' }
		];
		let top = 0;
		for (let i = 1; i < n; i++) if (es[i] > es[top]) top = i;
		e.values = { x: x[o + k * inner], max: m, e: es[k], sum: s, y, n, k, top };
		return e;
	},
	formula: `${sym('prob', 'p')}_i = \\frac{e^{${sym('in', 'x')}_i - m}}{\\sum_j e^{${sym('in', 'x')}_j - m}} = \\frac{ {{e}} }{ {{sum}} } = {{y}}`,
	narration: {
		// {softmaxRole}: what the shares mean here, chosen by vars() from the layer's place (attention weights over keys,
		// next-token probabilities over a vocabulary, class probabilities), or nothing.
		what: t(
			'Softmax turns a row of scores into shares between 0 and 1 that add up to 1: it takes e to the power of each score, then divides each result by their sum.',
			'Softmax 把一行分数变成介于 0 和 1 之间、总和为 1 的份额：先对每个分数取指数（e 的分数次方），再把每个结果除以它们的总和。'
		),
		how: t(
			'Subtracting the row’s largest score first does not change the result but keeps the powers from overflowing. A score of −∞ gets a share of exactly 0.{softmaxRole}',
			'先减去这一行最大的分数，结果不变，但能防止指数溢出。分数为 −∞ 的位置份额正好是 0。{softmaxRole}'
		),
		why: t(
			'The largest score gets the largest share, but the others keep some, so the result reads as how strongly each option is preferred.',
			'最大的分数得到最大的份额，但其他位置也保留一些，所以结果可以看成对每个选项的偏好程度。'
		),
		look: t('When one score is one point higher than another, its share is e times as large.', '一个分数比另一个高 1，它的份额就是另一个的 e 倍。'),
		unit: t('e^({x} − {max}) = {e}. Divided by the row sum {sum}, this gives {y}.', 'e^({x} − {max}) = {e}。除以这一行的总和 {sum}，得到 {y}。')
	},
	vars(layer, spec) {
		const p = (layer.params ?? {}) as Partial<SoftmaxP>;
		const ax = p.axis ?? -1;
		const n = layer.outShape[ax < 0 ? layer.outShape.length + ax : ax];
		const axes = layer.axes ?? [];
		const src = spec.layers.find((l) => l.id === layer.in[0]);
		const attention = axes.includes('tk') || (!!src && ['attention_scores', 'causal_mask'].includes(src.op));
		const isOutput = (spec.outputs?.[0] ?? spec.layers[spec.layers.length - 1]?.id) === layer.id;
		const count = n === undefined ? '?' : intText(n);
		const softmaxRole = attention
			? words(
					' Here each row belongs to one query: its shares are the attention weights, how much that query reads from each key.',
					'这里每一行属于一个查询：它的份额就是注意力权重，表示这个查询从每个键读取多少。'
				)
			: axes.includes('vocab')
				? words(` Here the shares are the model’s probabilities for each of its ${count} tokens to come next.`, `这里的份额就是模型给 ${count} 个词元中每一个作为下一个词元的概率。`)
				: axes.includes('class') || isOutput
					? words(` Here the shares are the model’s probabilities for its ${count} classes.`, `这里的份额就是模型给 ${count} 个类别的概率。`)
					: words('', '');
		return { ...baseVars(layer, spec), n: n ?? '?', axis: ax, softmaxRole };
	},
	zoom: 'softmax',
	color: 'sequential',
	parity: { bar: 1e-4 },
	pytorch: (layer) => `torch.softmax(x, dim=${Number((layer.params ?? {}).axis ?? -1)})`,
	terms: ['softmax', 'logit'],
	// attention weights are not made from logits; an output softmax is
	termsFor: (layer, spec) => {
		const src = spec.layers.find((l) => l.id === layer.in[0]);
		const attention = (layer.axes ?? []).includes('tk') || (!!src && ['attention_scores', 'causal_mask'].includes(src.op));
		return attention ? ['softmax', 'attention'] : outputLayer(spec)?.id === layer.id ? ['softmax', 'logit'] : ['softmax'];
	}
};

export default softmax;
