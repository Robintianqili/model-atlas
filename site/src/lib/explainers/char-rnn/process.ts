// char-rnn: two processes on the exported model (kind C), both in float64 from the float32 weights (rnn.ts), so the
// Python reference (pipeline/char-rnn/reference.py, PyTorch float64) matches them frame by frame to 1e-12.
//   unroll  the LSTM read one character at a time: gates -> cell -> hidden -> predict, four frames per character
//   sample  writing text one character at a time (the kit's sampling process, TinyGPT semantics: softmax in float64,
//           u = rng.next(), the first character whose running total passes u; temperature 0 = always the top one)
// Both get the learner's ids and the weights (rnn.ts rnnInput) and run in the package worker.
//   npm run atlas -- run char-rnn unroll --frames 0..12        npm run atlas -- run char-rnn sample --frames 0..20
import type { I18n, ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { argmax, samplingProcess, type SamplingOpt, type SamplingState } from '../../atlas/process/library/sampling.ts';
import { lstmStep, scores, scoresAfter, softmax, type RnnInput, type Weights } from './rnn.ts';

export type UnrollState = { i: Tensor; f: Tensor; g: Tensor; o: Tensor; c: Tensor; h: Tensor; probs: Tensor };
/** the window the model reads (as the page's trace: the adapter keeps the last 128 characters) */
export const CONTEXT = 128;

const L = (en: string, zh: string): I18n => ({ en, zh });
const vec = (a: Float64Array): Tensor => ({ shape: [a.length], data: a });
const need = (w: Weights | null | undefined): Weights => {
	if (!w) throw new Error('char-rnn: the model is not loaded (export it first: npm run atlas -- export char-rnn)');
	return w;
};
const T = (label: I18n, scale: 'diverging' | 'sequential' = 'diverging') => ({ kind: 'tensor' as const, label, scale, tol: 1e-12 });

export const unroll: ProcessDef<RnnInput, Record<string, never>, UnrollState> = {
	id: 'unroll',
	kind: 'inference',
	title: L('The LSTM reads your text one character at a time', 'LSTM 一次读一个字符'),
	phases: {
		gates: { label: L('Gates', '门'), kind: 'forward', narration: { what: L('Character {frame.t|int}: the three gates and the candidate.', '第 {frame.t|int} 个字符：三个门和候选值。') } },
		cell: { label: L('Cell state', '细胞状态'), kind: 'forward', narration: { what: L('Character {frame.t|int}: the new cell state c.', '第 {frame.t|int} 个字符：新的细胞状态 c。') } },
		hidden: { label: L('Hidden state', '隐藏状态'), kind: 'forward', narration: { what: L('Character {frame.t|int}: the new hidden state h.', '第 {frame.t|int} 个字符：新的隐藏状态 h。') } },
		predict: { label: L('Guess the next character', '猜下一个字符'), kind: 'output', narration: { what: L('After character {frame.t|int} the top guess has a chance of {frame.pTop|pct}.', '读完第 {frame.t|int} 个字符，排第一的猜测概率是 {frame.pTop|pct}。') } }
	},
	schema: {
		i: T(L('input gate i', '输入门 i'), 'sequential'),
		f: T(L('forget gate f', '遗忘门 f'), 'sequential'),
		g: T(L('candidate g', '候选值 g')),
		o: T(L('output gate o', '输出门 o'), 'sequential'),
		c: T(L('cell state c', '细胞状态 c')),
		h: T(L('hidden state h', '隐藏状态 h')),
		probs: T(L('chances of the next character', '下一个字符的概率'), 'sequential')
	},
	defaults: {},
	*run(input, _ctx): ProcessRun<UnrollState, Record<string, never>> {
		const w = need(input.w);
		const ids = input.ids.slice(-CONTEXT);
		let h: Float64Array = new Float64Array(w.H);
		let c: Float64Array = new Float64Array(w.H);
		let hits = 0;
		for (let t = 0; t < ids.length; t++) {
			const s = lstmStep(w, ids[t], h, c);
			const m = { t, ch: ids[t] };
			yield { phase: 'gates', step: t, state: { i: vec(s.i), f: vec(s.f), g: vec(s.g), o: vec(s.o) }, metrics: m };
			yield { phase: 'cell', step: t, state: { c: vec(s.c) }, metrics: m };
			yield { phase: 'hidden', step: t, state: { h: vec(s.h) }, metrics: m };
			const p = softmax(scores(w, s.h));
			const top = argmax(p);
			const next: Record<string, number> = t + 1 < ids.length ? { next: ids[t + 1], pNext: p[ids[t + 1]] } : {};
			if (t + 1 < ids.length && top === ids[t + 1]) hits++;
			yield { phase: 'predict', step: t, state: { probs: vec(p) }, metrics: { ...m, top, pTop: p[top], ...next, hits, guesses: Math.min(t + 1, ids.length - 1) } };
			h = s.h;
			c = s.c;
		}
	}
};

const shape = samplingProcess({ id: 'sample', logits: () => [] });

export const sample: ProcessDef<RnnInput, SamplingOpt, SamplingState> = {
	...shape,
	title: L('Writing one character at a time', '一次写一个字符'),
	defaults: { ...shape.defaults, maxNew: 60, temperature: 1 },
	phases: {
		...shape.phases,
		prompt: { ...shape.phases.prompt, narration: { what: L('The model starts from your text, {frame.length|int} characters.', '模型从你的文字出发，共 {frame.length|int} 个字符。') } },
		sample: { ...shape.phases.sample, narration: { what: L('A random number {frame.u|n3} picks a character whose chance was {frame.p|pct}.', '随机数 {frame.u|n3} 选出一个概率为 {frame.p|pct} 的字符。') } }
	},
	run(input, ctx) {
		const w = need(input.w);
		const p = samplingProcess({ id: 'sample', context: CONTEXT, logits: (ids) => scoresAfter(w, ids) });
		return p.run({ ids: input.ids.slice(-CONTEXT) }, ctx);
	}
};
