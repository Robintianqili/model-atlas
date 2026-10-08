// linear (aliases: dense, linear_tied): y[..., o] = Σ_i x[..., i] · W[o, i] + b[o], W in PyTorch [out, in] layout.
//   accumulate 'bias-last'  (default; tinycnn.ts dense): s = 0; s += w·x; y = s + b
//   accumulate 'bias-first' (tinygpt.ts linear):         s = b; s += x·w; y = s
// The bias is optional (TinyGPT's tied logits have none). nHead/dHead are display hints (q/k/v projections).
import type { AtlasLayer, AtlasSpec, OpDef, UnitExplanation } from '../types/index.ts';
import { flatIndex } from '../core/tensor.ts';
import { sym } from '../core/tex.ts';
import { accMode, baseVars, biasWords, givesLogits, intText, optW, outArray, specWeightShape, store, t, unitBase, weightName, words, type StructuralTerms } from './common.ts';

export type LinearP = {
	accumulate: 'bias-last' | 'bias-first';
	inFeatures?: number;
	outFeatures?: number;
	nHead?: number;
	dHead?: number;
};

function features(layer: AtlasLayer, spec: AtlasSpec): { I: number | string; O: number | string; bias: boolean } {
	const p = (layer.params ?? {}) as Partial<LinearP>;
	const ws = specWeightShape(layer, spec, 'weight');
	const last = layer.outShape[layer.outShape.length - 1];
	return { O: p.outFeatures ?? ws?.[0] ?? last, I: p.inFeatures ?? ws?.[1] ?? '?', bias: !!specWeightShape(layer, spec, 'bias') };
}

export const linear: OpDef<LinearP> & StructuralTerms = {
	op: 'linear',
	version: 1,
	attrs: {
		accumulate: { type: 'string', default: 'bias-last', doc: "'bias-last' (sum from 0, add the bias last; tinycnn.ts) or 'bias-first' (tinygpt.ts)" },
		inFeatures: { type: 'int', doc: 'informational; must equal weight.shape[1]' },
		outFeatures: { type: 'int', doc: 'informational; must equal weight.shape[0]' },
		nHead: { type: 'int', doc: 'display hint: the output columns are nHead heads of dHead numbers (q/k/v projections)' },
		dHead: { type: 'int', doc: 'display hint, with nHead' }
	},
	aliases: { dense: {}, linear_tied: {} },
	category: 'linear',
	label: t('Linear (fully connected)', '线性层（全连接）'),
	weights: [{ slot: 'weight' }, { slot: 'bias', optional: true }],
	inferShape(ins, p, w) {
		const x = ins[0];
		const ws = w.weight;
		if (!ws || ws.length !== 2) throw new Error('linear: weight must be [out, in]');
		const [O, I] = ws;
		const last = x[x.length - 1];
		if (x.length < 1) throw new Error('linear: input must have at least one axis');
		if (typeof last === 'number' && last !== I) throw new Error(`linear: input has ${last} features, weight expects ${I}`);
		if (p.inFeatures !== undefined && p.inFeatures !== I) throw new Error(`linear: inFeatures=${p.inFeatures} but weight has ${I}`);
		if (p.outFeatures !== undefined && p.outFeatures !== O) throw new Error(`linear: outFeatures=${p.outFeatures} but weight has ${O}`);
		if (w.bias && (w.bias.length !== 1 || w.bias[0] !== O)) throw new Error(`linear: bias must be [${O}]`);
		if (p.nHead !== undefined && p.dHead !== undefined && p.nHead * p.dHead !== O) throw new Error(`linear: nHead·dHead = ${p.nHead * p.dHead} but there are ${O} outputs`);
		return [...x.slice(0, -1), O];
	},
	axes: (ins) => (ins[0].length >= 2 ? [...ins[0].slice(0, -1), 'd'] : ['f']),
	forward(xs, p, ctx) {
		const src = xs[0];
		const W = ctx.w('weight');
		const B = optW(ctx, 'bias');
		const [O, I] = W.shape;
		const x = src.data;
		const w = W.data;
		const b = B ? B.data : null;
		if (src.shape[src.shape.length - 1] !== I) throw new Error(`linear ${ctx.layer.id}: input [${src.shape}] has ${src.shape[src.shape.length - 1]} features, weight expects ${I}`);
		const rows = x.length / I;
		const out = outArray(rows * O, src, W);
		if (accMode(p) === 'bias-first') {
			// tinygpt.ts linear(): let s = b ? b[o] : 0; for i: s += x[xo + i] * W[wo + i]; y = s
			for (let r = 0; r < rows; r++) {
				const xo = r * I;
				for (let o = 0; o < O; o++) {
					const wo = o * I;
					let s = b ? b[o] : 0;
					for (let i = 0; i < I; i++) s += x[xo + i] * w[wo + i];
					out[r * O + o] = s;
				}
			}
		} else {
			// tinycnn.ts dense(): let s = 0; for i: s += w[base + i] * x[i]; out = s + b[o]
			for (let r = 0; r < rows; r++) {
				const xo = r * I;
				for (let o = 0; o < O; o++) {
					const base = o * I;
					let s = 0;
					for (let i = 0; i < I; i++) s += w[base + i] * x[xo + i];
					out[r * O + o] = b ? s + b[o] : s;
				}
			}
		}
		return { shape: [...src.shape.slice(0, -1), O], data: out };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'argmax' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const W = q.w('weight');
		const B = optW(q, 'bias');
		const [O, I] = W.shape;
		const o = q.unit[q.unit.length - 1];
		const r = (f - o) / O;
		const x = q.ins[0].data;
		const xo = r * I;
		const first = accMode(q.p) === 'bias-first';
		const bias = B ? B.data[o] : 0;
		let s = first && B ? bias : 0;
		let sum = 0;
		const terms: NonNullable<UnitExplanation['terms']> = new Array(I);
		const xCells = new Array<number>(I);
		const wCells = new Array<number>(I);
		let zeros = 0;
		for (let i = 0; i < I; i++) {
			const xv = x[xo + i];
			const wv = W.data[o * I + i];
			const pr = first ? xv * wv : wv * xv;
			s += pr;
			sum += pr;
			if (xv === 0) zeros++;
			terms[i] = { x: xv, w: wv, p: pr, xAt: xo + i, wAt: o * I + i };
			xCells[i] = xo + i;
			wCells[i] = o * I + i;
		}
		const y = store(!first && B ? s + bias : s, q.out);
		const e = unitBase('weighted-sum', q.layer, q.unit, y, q.out.data[f]);
		e.reads = [
			{ tensor: q.layer.in[0], cells: xCells },
			{ tensor: weightName(q.layer, 'weight'), cells: wCells }
		];
		e.terms = terms;
		if (B) e.bias = bias;
		e.values = { n: I, s: sum, b: bias, y, out: o, zeros };
		// Group the products by input channel when the input is a flattened [C, H, W] map (TinyCNN dense).
		const src = q.spec.layers.find((l) => l.id === q.layer.in[0]);
		if (src && (src.op === 'flatten') && src.in.length) {
			const pre = q.trace.t[src.in[0]];
			if (pre && pre.shape.length === 3 && pre.shape[0] > 1) {
				const C = pre.shape[0];
				const per = I / C;
				e.groups = [];
				for (let c = 0; c < C; c++) {
					let cs = 0;
					for (let i = c * per; i < (c + 1) * per; i++) cs += terms[i].p;
					e.groups.push({ label: t(`channel ${c}`, `通道 ${c}`), from: c * per, to: (c + 1) * per, sum: cs });
				}
				e.layout = { rows: pre.shape[1], cols: pre.shape[2] };
			}
		}
		if (q.p.nHead !== undefined && q.p.dHead !== undefined) e.values.head = Math.floor(o / q.p.dHead);
		return e;
	},
	formula: `${sym('out', 'y')}_{k} = \\sum_{i} ${sym('param', 'W')}_{k,i}\\,${sym('in', 'x')}_{i} + ${sym('param', 'b')}_{k} = {{s}} + {{b}} = {{y}}`,
	narration: {
		// {biasClause} and {headsNote} are phrases chosen by vars(): a bias or none (TinyGPT's tied logits have none), and the
		// head layout when the layer is a q/k/v projection (nHead/dHead display hints).
		what: t(
			'Each of the {outFeatures|int} outputs is a weighted sum of all {inFeatures|int} inputs: every input is multiplied by its own learned weight, and the products are added up{biasClause}.',
			'这一层有 {outFeatures|int} 个输出，每个都是全部 {inFeatures|int} 个输入的加权和：每个输入乘以属于它的学来的权重，乘积全部相加{biasClause}。'
		),
		how: t(
			'The weights form a table with {outFeatures|int} rows and {inFeatures|int} columns: row k holds the weights of output k.{headsNote}',
			'权重是一张 {outFeatures|int} 行、{inFeatures|int} 列的表：第 k 行是第 k 个输出的权重。{headsNote}'
		),
		why: t(
			'A weighted sum can reward some inputs and penalize others, so each output measures how much the input looks like what its row of weights prefers.',
			'加权和可以奖励一些输入、惩罚另一些输入，所以每个输出衡量的是输入有多像它那一行权重所偏好的样子。'
		),
		look: t('Large positive products push this output up; large negative products pull it down.', '大的正乘积把这个输出往上推，大的负乘积把它往下拉。'),
		unit: t(
			'The {n|int} products add up to {s}; {biasIs} {b}. The output is {s} + ({b}) = {y}.',
			'{n|int} 个乘积加起来是 {s}；{biasIs} {b}。输出是 {s} + ({b}) = {y}。'
		)
	},
	vars(layer, spec) {
		const { I, O, bias } = features(layer, spec);
		const p = (layer.params ?? {}) as Partial<LinearP>;
		const heads = typeof p.nHead === 'number' && typeof p.dHead === 'number';
		return {
			...baseVars(layer, spec),
			inFeatures: I,
			outFeatures: O,
			weightCount: typeof I === 'number' && typeof O === 'number' ? I * O + (bias ? O : 0) : '?',
			...biasWords(bias),
			headsNote: heads
				? words(` Attention reads these ${intText(O)} outputs as ${p.nHead} heads of ${p.dHead} numbers each.`, `注意力把这 ${intText(O)} 个输出看成 ${p.nHead} 个头，每个头 ${p.dHead} 个数。`)
				: words('', '')
		};
	},
	zoom: 'weighted-sum',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch(layer) {
		const p = (layer.params ?? {}) as Partial<LinearP>;
		const wname = layer.w?.weight;
		if (wname && wname !== `${layer.id}.weight`) return `F.linear(x, ${wname})  # weight shared with ${wname.replace(/\.weight$/, '')}`;
		const last = layer.outShape[layer.outShape.length - 1];
		return `nn.Linear(${p.inFeatures ?? 'in_features'}, ${p.outFeatures ?? last})`;
	},
	terms: ['dense-layer', 'weight', 'bias', 'logit'],
	// 'bias' only with a bias (TinyGPT's tied logits have none); 'logit' only for the model's output scores
	termsFor: (layer, spec) => ['dense-layer', 'weight', ...(features(layer, spec).bias ? ['bias'] : []), ...(givesLogits(layer, spec) ? ['logit'] : [])]
};

export default linear;
