// 'fixture-seq': a dev-only package used by W1-C's tests. A tiny letter-sequence classifier on stub ops (kind A: three
// repeated blocks, so the planner's group rule runs) plus a letter-counting process (kind C). Its assets live in memory
// (memoryFetch), its weights are seeded, and every number it produces is illustrative: it is a test fixture, never shipped.
import type { FetchLike, I18n, IntTensor, Tensor } from '../../types/core.ts';
import type { AtlasSpec } from '../../types/spec.ts';
import type { ExplainerManifest, InputAdapter } from '../../types/explainer.ts';
import type { ProcessDef } from '../../types/process.ts';
import type { LessonDef } from '../../types/lesson.ts';
import type { PackageModules, Preprocessed } from '../../types/runtime.ts';
import { createRng } from '../../runtime/seams.ts';
import { STUB_OPS } from './stub-ops.ts';

export const CONTEXT = 12;
export const ALPHABET = 'abcdefghijklmnopqrstuvwxyz ';
const D = 4;
const V = ALPHABET.length;

export function encode(text: string): number[] {
	const ids: number[] = [];
	for (const ch of text.toLowerCase()) {
		const i = ALPHABET.indexOf(ch);
		if (i >= 0) ids.push(i);
	}
	return ids;
}

export const seqAdapter: InputAdapter<string> = {
	id: 'stub-seq',
	kind: 'text',
	preprocess(raw): Preprocessed {
		const all = encode(typeof raw === 'string' ? raw : '');
		const ids = all.slice(-CONTEXT);
		const t: IntTensor = { shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' };
		const notes: I18n[] = [];
		if (all.length > CONTEXT) notes.push({ en: `Only the last ${CONTEXT} letters fit.`, zh: `只放得下最后 ${CONTEXT} 个字母。` });
		if (!ids.length) notes.push({ en: 'Type some letters first.', zh: '先输入一些字母。' });
		return { tensors: { input: t }, steps: [{ id: 'letters', label: { en: 'Letters to numbers', zh: '字母变成数' }, tensor: { shape: [ids.length], data: Float32Array.from(ids) } }], empty: ids.length === 0, notes };
	},
	presets(assets) {
		const p = (assets.presets ?? assets['presets.json']) as { items: { id: string; en: string; zh: string; text: string }[] } | undefined;
		return (p?.items ?? []).map((x) => ({ id: x.id, label: { en: x.en, zh: x.zh }, raw: x.text }));
	},
	edgeFixtures: () => [
		{ id: 'empty', label: { en: 'No letters', zh: '没有字母' }, raw: '' },
		{ id: 'one', label: { en: 'One letter', zh: '一个字母' }, raw: 'a' },
		{ id: 'long', label: { en: 'Longer than the context', zh: '超过上下文长度' }, raw: 'the quick brown fox jumps over' }
	],
	layer0: {
		view: 'layer.tensor',
		narration: {
			what: { en: 'Each letter becomes one number: a is 0, b is 1, and so on.', zh: '每个字母变成一个数：a 是 0，b 是 1，依此类推。' },
			how: { en: 'For your text that is {n|int} in total, arranged as {shape}.', zh: '你的文字一共变成 {n|int} 个数，排成 {shape}。' }
		}
	}
};

const L = (en: string, zh: string): I18n => ({ en, zh });

function makeSpecAndWeights(): { spec: AtlasSpec; weights: Float32Array } {
	const rng = createRng(2026);
	const tensors: { name: string; shape: number[] }[] = [{ name: 'emb.weight', shape: [V, D] }];
	for (const b of ['B0', 'B1', 'B2']) tensors.push({ name: `${b}.lin.weight`, shape: [D, D] }, { name: `${b}.lin.bias`, shape: [D] });
	tensors.push({ name: 'head.weight', shape: [3, D] }, { name: 'head.bias', shape: [3] });
	let offset = 0;
	const entries = tensors.map((t) => {
		const length = t.shape.reduce((a, b) => a * b, 1);
		const e = { name: t.name, shape: t.shape, offset, length };
		offset += length;
		return e;
	});
	const weights = Float32Array.from({ length: offset }, () => rng.normal() * 0.8);
	const block = (b: string, input: string) => [
		{ id: `${b}.lin`, op: 'stub_linear', label: L(`${b} weighted sum`, `${b} 加权和`), in: [input], outShape: ['T', D] },
		{ id: `${b}.act`, op: 'stub_tanh', label: L(`${b} tanh`, `${b} tanh`), in: [`${b}.lin`], outShape: ['T', D] }
	];
	const spec: AtlasSpec = {
		slug: 'fixture-seq',
		name: L('Letter-sequence fixture', '字母序列测试模型'),
		version: 'fixture-1',
		params: offset,
		precision: 'fp32',
		atlas: { spec: '1.1', kinds: ['trace', 'process'], exporter: 'lesson/fixtures/seq.ts' },
		inputs: [{ name: 'input', adapter: 'stub-seq', kind: 'text', shape: ['T'], dtype: 'i32', axes: ['t'] }],
		outputs: ['prob'],
		dims: { T: { min: 1, max: CONTEXT, label: L('letters', '字母数') } },
		tensors: entries,
		layers: [
			{ id: 'tokens', op: 'input', label: L('Letters', '字母'), in: ['input'], outShape: ['T'], axes: ['t'] },
			{ id: 'emb', op: 'stub_embed', label: L('Letter vectors', '字母向量'), in: ['tokens'], outShape: ['T', D], axes: ['t', 'd'] },
			...block('B0', 'emb'),
			...block('B1', 'B0.act'),
			...block('B2', 'B1.act'),
			{ id: 'pool', op: 'stub_meanpool', label: L('Average', '平均'), in: ['B2.act'], outShape: [D], axes: ['d'] },
			{ id: 'head', op: 'stub_linear', label: L('Scores', '分数'), in: ['pool'], outShape: [3], axes: ['class'] },
			{ id: 'prob', op: 'stub_softmax', label: L('Chances', '概率'), in: ['head'], outShape: [3], axes: ['class'] }
		],
		parity: { against: 'none (test fixture)', max_abs_diff: null, checked: null },
		classes: [
			{ id: 'calm', en: 'calm', zh: '平静' },
			{ id: 'busy', en: 'busy', zh: '忙碌' },
			{ id: 'odd', en: 'odd', zh: '古怪' }
		]
	};
	return { spec, weights };
}

export const { spec: SEQ_SPEC, weights: SEQ_WEIGHTS } = makeSpecAndWeights();

export const SEQ_FILES: Record<string, unknown> = {
	'models/fixture-seq/spec.json': SEQ_SPEC,
	'models/fixture-seq/weights.bin': SEQ_WEIGHTS.buffer,
	'models/fixture-seq/facts.json': {
		format: 'atlas-facts/1',
		facts: {
			test_accuracy: { value: 0.9681, source: 'fixture (made up)', measured: '2026-10-06' },
			trained_steps: { value: 1200, source: 'fixture (made up)', measured: '2026-10-06' }
		}
	},
	'models/fixture-seq/presets.json': {
		items: [
			{ id: 'hello', en: 'hello world', zh: '你好世界', text: 'hello world' },
			{ id: 'zz', en: 'buzz', zh: '嗡嗡', text: 'buzz buzz' }
		]
	}
};

/** fetch() over an in-memory file table; urls may carry a base prefix. */
export function memoryFetch(files: Record<string, unknown>, base = ''): FetchLike {
	return async (url: string) => {
		const key = url.replace(/^\/+/, '').slice(base.replace(/^\/+|\/+$/g, '').length).replace(/^\/+/, '');
		const v = files[key];
		if (v === undefined) return { ok: false, status: 404, json: async () => null, arrayBuffer: async () => new ArrayBuffer(0), text: async () => '' };
		return {
			ok: true,
			status: 200,
			json: async () => JSON.parse(JSON.stringify(v)),
			arrayBuffer: async () => (v instanceof ArrayBuffer ? v.slice(0) : new TextEncoder().encode(JSON.stringify(v)).buffer),
			text: async () => JSON.stringify(v)
		};
	};
}

export type CountState = { t: number; char: string; count: Tensor };

/** Kind C: read the text one letter at a time and keep a count per letter (tracked by changes). */
export const unroll: ProcessDef<string, { stopAt: number }, CountState> = {
	id: 'unroll',
	kind: 'algorithm',
	title: L('Count the letters', '数字母'),
	phases: {
		read: {
			label: L('Read one letter', '读一个字母'),
			kind: 'observe',
			formula: { tex: '\\text{share} = \\frac{{{c}}}{{{n}}}', values: (f) => ({ c: f.metrics?.c ?? NaN, n: f.metrics?.n ?? NaN }) },
			narration: { what: L('The counter reads one letter and adds one to its count.', '计数器读入一个字母，把它的计数加一。') }
		}
	},
	schema: {
		t: { kind: 'int', label: L('Position', '位置'), tol: 'exact' },
		char: { kind: 'text', label: L('Letter', '字母'), tol: 'exact' },
		count: { kind: 'tensor', label: L('Count per letter', '每个字母的计数'), shape: [V], track: 'changes', tol: 'exact' }
	},
	defaults: { stopAt: 64 },
	*run(input) {
		const ids = encode(typeof input === 'string' ? input : '');
		const count: Tensor = { shape: [V], data: new Float64Array(V) };
		for (let t = 0; t < ids.length; t++) {
			const id = ids[t];
			const before = count.data[id];
			count.data[id] = before + 1;
			yield {
				phase: 'read',
				step: t,
				// the first frame gives the full count (already including this letter); later frames report their one edit
				state: t === 0 ? { t, char: ALPHABET[id], count } : { t, char: ALPHABET[id] },
				...(t === 0 ? {} : { changes: [{ field: 'count', index: [id], from: before, to: before + 1 }] }),
				metrics: { c: before + 1, n: t + 1, share: (before + 1) / (t + 1) }
			};
		}
	}
};

export const seqManifest: ExplainerManifest = {
	slug: 'fixture-seq',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'draft',
	devOnly: true,
	assets: { dir: 'models/fixture-seq', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'stub-seq', presets: 'presets.json' },
	output: { view: 'kit.probbars', layer: 'prob' },
	sources: { fwd: { kind: 'trace' }, unroll: { kind: 'process', process: 'unroll', run: 'main' } },
	page: {
		goals: [L('See a sequence turn into one answer', '看一串字母如何变成一个答案')],
		prequestion: { id: 'seq-q', question: L('Does the order of the letters matter here?', '这里字母的顺序重要吗？'), options: [L('Yes', '重要'), L('No', '不重要')], answer: 1, explain: L('The average over positions forgets the order.', '对位置求平均会忘掉顺序。') }
	},
	credits: []
};

export function seqPackage(lesson: LessonDef<any> | null): PackageModules {
	return { slug: 'fixture-seq', manifest: seqManifest, lesson, ops: STUB_OPS, processes: { unroll }, adapters: { 'stub-seq': seqAdapter as InputAdapter } };
}

export const SEQ_BASE = '';
export const seqFetch = memoryFetch(SEQ_FILES);
