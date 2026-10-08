// Executor paths the two shipped models do not exercise: a spec 1.1 written by hand with an LSTM (sub-steps, layer.sub
// filtering, explain of a sub id, record: 'layers'), symbolic dims, f64 precision, and padded convolutions.
// Run from site/: node --test src/lib/atlas/graph/executor.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fromParts, traceInputs } from './executor.ts';
import { checkExplain, pickUnit } from './explain.ts';
import { CORE_OPS } from '../ops/index.ts';
import { createRng } from '../core/rng.ts';
import { tensor } from '../core/tensor.ts';
import { errorsOf, formatIssues, validateSpec } from '../schema/spec.ts';
import type { AtlasSpec, Tensor } from '../types/index.ts';

function weightsFor(spec: AtlasSpec, seed: number): Record<string, Tensor> {
	const rng = createRng(seed);
	const w: Record<string, Tensor> = {};
	for (const t of spec.tensors) w[t.name] = tensor(t.shape, Array.from({ length: t.length }, () => rng.next() - 0.5));
	return w;
}

const seqSpec = (precision: 'fp32' | 'f64' = 'fp32'): AtlasSpec => ({
	slug: 'demo-seq',
	name: { en: 'Demo sequence model', zh: '演示序列模型' },
	version: '1',
	params: 4 * 4 * 3 + 4 * 4 * 4 + 16 + 16 + 2 * 4 + 2,
	precision,
	atlas: { spec: '1.1', kinds: ['trace'], exporter: 'test' },
	inputs: [{ name: 'x', adapter: 'series', kind: 'series', shape: ['T', 3], dtype: 'f32', axes: ['t', 'd'] }],
	dims: { T: { min: 1, max: 8, label: { en: 'steps', zh: '步数' } } },
	tensors: [
		{ name: 'lstm.weight_ih_l0', shape: [16, 3], offset: 0, length: 48 },
		{ name: 'lstm.weight_hh_l0', shape: [16, 4], offset: 48, length: 64 },
		{ name: 'lstm.bias_ih_l0', shape: [16], offset: 112, length: 16 },
		{ name: 'lstm.bias_hh_l0', shape: [16], offset: 128, length: 16 },
		{ name: 'head.weight', shape: [2, 4], offset: 144, length: 8 },
		{ name: 'head.bias', shape: [2], offset: 152, length: 2 }
	],
	layers: [
		{
			id: 'lstm1',
			op: 'lstm',
			label: { en: 'LSTM', zh: 'LSTM' },
			in: ['x'],
			outShape: ['T', 4],
			params: { hidden: 4 },
			w: { weight_ih: 'lstm.weight_ih_l0', weight_hh: 'lstm.weight_hh_l0', bias_ih: 'lstm.bias_ih_l0', bias_hh: 'lstm.bias_hh_l0' },
			sub: ['f', 'c']
		},
		{ id: 'head', op: 'linear', label: { en: 'Head', zh: '输出头' }, in: ['lstm1'], outShape: ['T', 2] },
		{ id: 'probs', op: 'softmax', label: { en: 'Probabilities', zh: '概率' }, in: ['head'], outShape: ['T', 2] }
	],
	outputs: ['probs'],
	parity: { against: 'pytorch-fp32-cpu', max_abs_diff: null, checked: null }
});

test('spec 1.1 with an LSTM: validates, records chosen sub-steps, explains sub ids, recomputes without them', () => {
	const spec = seqSpec();
	const errs = errorsOf(validateSpec(spec, { ops: CORE_OPS }));
	assert.equal(errs.length, 0, formatIssues(errs));
	const m = fromParts(spec, weightsFor(spec, 3), CORE_OPS);
	const rng = createRng(11);
	const x = tensor([5, 3], Array.from({ length: 15 }, () => rng.normal()));
	const tr = m.forward({ x });
	assert.deepEqual(tr.order, ['lstm1', 'lstm1#f', 'lstm1#c', 'head', 'probs']);
	assert.deepEqual(tr.meta!.dims, { T: 5 });
	assert.equal(traceInputs(tr)!.x, x);
	const f = m.explain(tr, 'lstm1#f', [2, 1]);
	assert.equal(f.layer, 'lstm1#f');
	assert.equal(f.result, f.traced);
	assert.equal(f.kind, 'gate');
	const c = checkExplain(m, tr);
	assert.equal(c.failures.length, 0, JSON.stringify(c.failures.slice(0, 2)));
	assert.equal(c.units, 5 * 4 * 3 + 5 * 2 * 2);
	// without sub-steps the LSTM explanation recomputes the recurrence and still matches bit for bit
	const tl = m.forward(x, { record: 'layers' });
	assert.deepEqual(tl.order, ['lstm1', 'head', 'probs']);
	for (let t = 0; t < 5; t++) for (let j = 0; j < 4; j++) {
		const e = m.explain(tl, 'lstm1', [t, j]);
		assert.equal(e.result, e.traced, `[${t}, ${j}]`);
		assert.equal(e.values.c, tr.t['lstm1#c'].data[t * 4 + j]);
	}
	assert.throws(() => m.explain(tl, 'lstm1#f', [0, 0]), /no 'lstm1#f'/);
	assert.deepEqual(pickUnit(m, tr, 'probs', 'argmax').length, 2);
	assert.throws(() => m.forward(tensor([9, 3])), /outside 1\.\.8/);
	assert.throws(() => m.forward(tensor([4, 2])), /axis 1 is 2/);
});

test('f64 precision: weights and outputs widen to Float64Array, explain still equals the trace', () => {
	const spec = seqSpec('f64');
	const m = fromParts(spec, weightsFor(spec, 3), CORE_OPS);
	const m32 = fromParts(seqSpec(), weightsFor(spec, 3), CORE_OPS);
	const x = tensor([3, 3], [0.1, -0.2, 0.3, 0.5, 0.4, -0.6, 1, 0, -1]);
	const tr = m.forward(x);
	assert.ok(tr.t.probs.data instanceof Float64Array);
	assert.ok(tr.t['lstm1#c'].data instanceof Float64Array);
	const c = checkExplain(m, tr);
	assert.equal(c.failures.length, 0);
	const t32 = m32.forward(x);
	for (let i = 0; i < 6; i++) assert.ok(Math.abs(tr.t.probs.data[i] - t32.t.probs.data[i]) < 1e-6);
});

test('padded and strided convolutions: forward shape, receptive fields clamped to the image, explain windows', () => {
	const spec: AtlasSpec = {
		slug: 'demo-img',
		name: { en: 'Demo image model', zh: '演示图像模型' },
		version: '1',
		params: 2 * 1 * 9 + 2,
		precision: 'fp32',
		atlas: { spec: '1.1', kinds: ['trace'], exporter: 'test' },
		inputs: [{ name: 'img', adapter: 'drawing28', kind: 'drawing', shape: [1, 6, 6], dtype: 'f32', axes: ['c', 'y', 'x'] }],
		tensors: [
			{ name: 'c1.weight', shape: [2, 1, 3, 3], offset: 0, length: 18 },
			{ name: 'c1.bias', shape: [2], offset: 18, length: 2 }
		],
		layers: [
			{ id: 'in', op: 'input', label: { en: 'Input', zh: '输入' }, in: ['img'], outShape: [1, 6, 6] },
			{ id: 'c1', op: 'conv2d', label: { en: 'Conv', zh: '卷积' }, in: ['in'], outShape: [2, 6, 6], params: { kernel: 3, padding: 1 } },
			{ id: 'r1', op: 'relu', label: { en: 'ReLU', zh: 'ReLU' }, in: ['c1'], outShape: [2, 6, 6] },
			{ id: 'p1', op: 'maxpool2d', label: { en: 'Pool', zh: '池化' }, in: ['r1'], outShape: [2, 3, 3], params: { kernel: 2 } },
			{ id: 'f', op: 'flatten', label: { en: 'Flatten', zh: '展平' }, in: ['p1'], outShape: [18] }
		],
		parity: { against: 'none', max_abs_diff: null, checked: null }
	};
	assert.equal(errorsOf(validateSpec(spec, { ops: CORE_OPS })).length, 0);
	const m = fromParts(spec, weightsFor(spec, 5), CORE_OPS);
	const rng = createRng(2);
	const tr = m.forward(tensor([1, 6, 6], Array.from({ length: 36 }, () => rng.next())));
	assert.equal(checkExplain(m, tr).failures.length, 0);
	assert.deepEqual(m.receptiveField('p1', { y0: 1, x0: 1, y1: 1, x1: 1 }), { y0: 1, x0: 1, y1: 4, x1: 4 });
	assert.deepEqual(m.receptiveField('p1', { y0: 0, x0: 0, y1: 0, x1: 0 }), { y0: 0, x0: 0, y1: 2, x1: 2 });
	assert.deepEqual(m.receptiveField('f', { y0: 0, x0: 0, y1: 0, x1: 0 }), { y0: 0, x0: 0, y1: 5, x1: 5 });
	const corner = m.explain(tr, 'c1', [1, 0, 0]);
	assert.deepEqual(corner.reads[0].box, { y0: 0, x0: 0, y1: 1, x1: 1 });
	assert.equal(corner.reads[0].cells.length, 4, 'padding taps read nothing');
	assert.equal(corner.terms!.length, 9);
	assert.deepEqual(m.receptiveField('nope', { y0: 0, x0: 0, y1: 0, x1: 0 }), null);
});
