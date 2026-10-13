// tiny-encdec: process 'decode' (kind C): greedy decoding one event at a time, in float64 from the exported float32
// weights (tf.ts run64), so pipeline/tiny-encdec/reference.py (PyTorch float64) matches it frame by frame to 1e-12.
//   start   the encoder has read every character at once (its memory); the decoder's first input is <s>
//   attend  one per step: the cross-attention weights of both heads over the input (the matrix gets row t)
//   write   one per step: the chosen symbol and its chance; the run stops after the end symbol
// The input is decodeInput(rt): the learner's character ids and the weights (typed arrays, so they reach the package
// worker); as JSON (atlas parity hands it to reference.py) it is only { ids }, and the reference loads the weights.
import type { GraphModel, I18n, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { EOS, STEPS } from './data.ts';
import { argmax, run64, softmax, weightsOf, type Weights } from './tf.ts';

export type DecodeInput = { ids: number[]; w?: Weights | null };
export type DecodeState = { cross: Tensor; out: Tensor };
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
		start: { label: L('Start', '开始'), kind: 'input', narration: { what: L('The encoder has read all {frame.T|int} characters at once; the decoder starts from the start symbol.', '编码器一次读完全部 {frame.T|int} 个字符；解码器从起始符号出发。') } },
		attend: { label: L('Look at the input', '看输入'), kind: 'forward', narration: { what: L('Step {frame.t|int} puts {frame.w|pct} of its cross-attention (both heads averaged) on input position {frame.j|int}.', '第 {frame.t|int} 步把 {frame.w|pct} 的交叉注意力（两个头取平均）放在输入位置 {frame.j|int} 上。') } },
		write: { label: L('Write a symbol', '写一个符号'), kind: 'output', narration: { what: L('Step {frame.t|int} writes the symbol with the largest chance, {frame.p|pct}.', '第 {frame.t|int} 步写下概率最大的符号，概率 {frame.p|pct}。') } }
	},
	schema: {
		cross: { kind: 'tensor', label: L('cross-attention weights [head, step, character]', '交叉注意力权重 [头, 步, 字符]'), scale: 'sequential', tol: 1e-12 },
		out: { kind: 'tensor', label: L('symbols written (−1: not yet)', '写下的符号（−1：还没写）'), tol: 'exact' }
	},
	defaults: {},
	*run(input): ProcessRun<DecodeState, Record<string, never>> {
		if (!input.w) throw new Error('tiny-encdec: the model is not loaded (export it first: npm run atlas -- export tiny-encdec)');
		const T = input.ids.length;
		if (!T) return;
		const { H, V } = input.w.dec;
		const cross = new Float64Array(H * STEPS * T);
		const out = new Float64Array(STEPS).fill(-1);
		const snap = () => ({ cross: { shape: [H, STEPS, T], data: cross.slice() }, out: { shape: [STEPS], data: out.slice() } });
		yield { phase: 'start', state: snap(), metrics: { T } };
		const r = run64(input.w, input.ids);
		for (const [t, run] of r.runs.entries()) {
			const n = t + 1;
			const avg = new Float64Array(T);
			for (let h = 0; h < H; h++) {
				const row = run.cross.subarray((h * n + t) * T, (h * n + t + 1) * T);
				cross.set(row, (h * STEPS + t) * T);
				row.forEach((v, j) => (avg[j] += v / H));
			}
			const j = argmax(avg);
			yield { phase: 'attend', step: t, state: { cross: snap().cross }, metrics: { t, j, w: avg[j] } };
			const p = softmax(run.logits.subarray(t * V, n * V));
			out[t] = argmax(p);
			yield { phase: 'write', step: t, state: { out: snap().out }, metrics: { t, y: out[t], p: p[out[t]] } };
			if (out[t] === EOS) break;
		}
	}
};
