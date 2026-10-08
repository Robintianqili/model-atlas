// tinyseq2seq: process 'decode' (kind C): greedy decoding one event at a time, in float64 from the exported float32
// weights (seq.ts run64), so pipeline/tinyseq2seq/reference.py (PyTorch float64) matches it frame by frame to 1e-12.
//   start   the decoder state is the encoder's last state; the first input symbol is <s>
//   attend  one per step: the weights over the input characters (the alignment matrix gets row t)
//   write   one per step: the chosen symbol and its chance; the run stops after the end symbol
// The input is decodeInput(rt): the learner's character ids and the weights (typed arrays, so they reach the package
// worker); as JSON (atlas parity hands it to reference.py) it is only { ids }, and the reference loads the weights.
//   npm run atlas -- run tinyseq2seq decode   (needs a runtime with the model: claims.test.ts runs it on every fixture)
import type { GraphModel, I18n, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { EOS, STEPS } from './data.ts';
import { argmax, run64, weightsOf, type Weights } from './seq.ts';

export type DecodeInput = { ids: number[]; w?: Weights | null };
export type DecodeState = { align: Tensor; out: Tensor };
const L = (en: string, zh: string): I18n => ({ en, zh });

export function decodeInput(rt: { pre: { tensors: Record<string, { data: ArrayLike<number> }> } | null; model: GraphModel | null }): DecodeInput {
	const o: DecodeInput = { ids: Array.from(rt.pre?.tensors.input?.data ?? []), w: rt.model ? weightsOf(rt.model) : null };
	Object.defineProperty(o, 'toJSON', { value: () => ({ ids: o.ids }), enumerable: false });
	return o;
}

export const decode: ProcessDef<DecodeInput, Record<string, never>, DecodeState> = {
	id: 'decode',
	kind: 'inference',
	title: L('Greedy decoding, one step at a time', '贪心解码，一次一步'),
	phases: {
		start: { label: L('Start', '开始'), kind: 'input', narration: { what: L('The decoder starts from the encoder’s last state and the start symbol; the input has {frame.T|int} characters.', '解码器从编码器的最后一个状态和起始符号出发；输入有 {frame.T|int} 个字符。') } },
		attend: { label: L('Look at the input', '看输入'), kind: 'forward', narration: { what: L('Step {frame.t|int} puts {frame.w|pct} of its attention on input position {frame.j|int}.', '第 {frame.t|int} 步把 {frame.w|pct} 的注意力放在输入位置 {frame.j|int} 上。') } },
		write: { label: L('Write a symbol', '写一个符号'), kind: 'output', narration: { what: L('Step {frame.t|int} writes the symbol with the largest chance, {frame.p|pct}.', '第 {frame.t|int} 步写下概率最大的符号，概率 {frame.p|pct}。') } }
	},
	schema: {
		align: { kind: 'tensor', label: L('attention weights, one row per step', '注意力权重，每步一行'), scale: 'sequential', tol: 1e-12 },
		out: { kind: 'tensor', label: L('symbols written (−1: not yet)', '写下的符号（−1：还没写）'), tol: 'exact' }
	},
	defaults: {},
	*run(input): ProcessRun<DecodeState, Record<string, never>> {
		if (!input.w) throw new Error('tinyseq2seq: the model is not loaded (export it first: npm run atlas -- export tinyseq2seq)');
		const T = input.ids.length;
		if (!T) return;
		const align = new Float64Array(STEPS * T);
		const out = new Float64Array(STEPS).fill(-1);
		const snap = () => ({ align: { shape: [STEPS, T], data: align.slice() }, out: { shape: [STEPS], data: out.slice() } });
		yield { phase: 'start', state: snap(), metrics: { T } };
		for (const [t, s] of run64(input.w, input.ids).steps.entries()) {
			align.set(s.a, t * T);
			const j = argmax(s.a);
			yield { phase: 'attend', step: t, state: { align: snap().align }, metrics: { t, j, w: s.a[j] } };
			out[t] = s.y;
			yield { phase: 'write', step: t, state: { out: snap().out }, metrics: { t, y: s.y, p: s.p[s.y] } };
			if (s.y === EOS) break;
		}
	}
};
