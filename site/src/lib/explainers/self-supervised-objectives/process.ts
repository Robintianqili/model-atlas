// self-supervised-objectives: process 'hide' (kind C), copied unchanged from tinybert (credit: the tinybert package):
// BERT's training-time masking, as TinyBERT was trained (pipeline/tinybert/data.py), replayed on the learner's text with the page's rng, one frame per token. Each token is chosen with probability rate (u < rate); a chosen token
// becomes [MASK] when v < pMask, a random text token when v < pMask + pRandom, and stays as it is otherwise. When no
// token was chosen, one is picked at random (training does the same, so every example teaches something).
// pipeline/self-supervised-objectives/reference.py (a copy of tinybert's) is the independent Python run (atlas parity: same
// frames).
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { contentOf, MASK, type MaskedRaw } from './adapters/masked-text/adapter.ts';

export type HideOpt = { rate: number; pMask: number; pRandom: number; vocab: number };
/** fate per text token: 0 not chosen, 1 [MASK], 2 a random token, 3 chosen but kept; shown: the id the model reads */
export type HideState = { fate: Tensor; shown: Tensor };
type Run = ProcessRun<HideState, HideOpt>;
export const FATE = { none: 0, mask: 1, random: 2, keep: 3 };

const what = (en: string, zh: string) => ({ what: { en, zh } });

export const hide: ProcessDef<MaskedRaw, HideOpt, HideState> = {
	id: 'hide',
	kind: 'algorithm',
	title: { en: 'Hide tokens for training', zh: '为训练藏起词元' },
	phases: {
		start: { label: { en: 'Your text', zh: '你的文字' }, kind: 'input', narration: what('Your text has {frame.n|int} tokens; none is hidden yet.', '你的文字有 {frame.n|int} 个词元；还没有藏起任何一个。') },
		draw: {
			label: { en: 'Draw for one token', zh: '为一个词元抽签' },
			kind: 'sample',
			narration: what('Token {frame.i|int} (counting from 0) draws u = {frame.u}; it is chosen when u is below {frame.rate}.', '第 {frame.i|int} 个词元（从 0 数起）抽到 u = {frame.u}；u 小于 {frame.rate} 时它被选中。')
		},
		force: {
			label: { en: 'None chosen: pick one', zh: '一个都没选中：随机挑一个' },
			kind: 'sample',
			narration: what('No token was chosen, so token {frame.i|int} (counting from 0) is picked at random.', '一个词元都没被选中，所以随机挑了第 {frame.i|int} 个词元（从 0 数起）。')
		},
		done: {
			label: { en: 'The training example', zh: '训练样本' },
			kind: 'output',
			narration: what('{frame.hidden|int} of {frame.n|int} tokens are hidden: {frame.masked|int} as [MASK], {frame.random|int} as a random token, {frame.kept|int} kept.', '{frame.n|int} 个词元中藏起了 {frame.hidden|int} 个：{frame.masked|int} 个换成 [MASK]，{frame.random|int} 个换成随机词元，{frame.kept|int} 个保持不变。')
		}
	},
	schema: {
		fate: { kind: 'tensor', label: { en: 'What happens to each token', zh: '每个词元的处理方式' }, track: 'changes', tol: 'exact' },
		shown: { kind: 'tensor', label: { en: 'The ids the model reads', zh: '模型读到的 id' }, track: 'changes', tol: 'exact' }
	},
	defaults: { rate: 0.15, pMask: 0.8, pRandom: 0.1, vocab: 1023 },
	*run(input, ctx): Run {
		const o = ctx.opt;
		const c = contentOf(input);
		const n = c.length;
		const fate: Tensor = { shape: [n], data: new Float64Array(n) };
		const shown: Tensor = { shape: [n], data: Float64Array.from(c) };
		const count = [0, 0, 0, 0];
		if (n > 0) yield { phase: 'start', state: { fate, shown }, metrics: { n } };
		function* treat(i: number, phase: 'draw' | 'force', u: number): Run {
			const v = ctx.rng.next();
			const f = v < o.pMask ? FATE.mask : v < o.pMask + o.pRandom ? FATE.random : FATE.keep;
			const rid = f === FATE.random ? ctx.rng.int(o.vocab) : -1;
			const to = f === FATE.mask ? MASK : f === FATE.random ? rid : c[i];
			fate.data[i] = f;
			shown.data[i] = to;
			count[f]++;
			yield { phase, step: i, state: { fate, shown }, changes: [{ field: 'fate', index: [i], from: 0, to: f }, ...(to !== c[i] ? [{ field: 'shown', index: [i], from: c[i], to }] : [])], metrics: { i, u, chosen: 1, v, fate: f, rid, rate: o.rate } };
		}
		for (let i = 0; i < n && !ctx.signal.cancelled; i++) {
			const u = ctx.rng.next();
			if (u < o.rate) yield* treat(i, 'draw', u);
			else yield { phase: 'draw', step: i, state: { fate, shown }, metrics: { i, u, chosen: 0, v: -1, fate: 0, rid: -1, rate: o.rate } };
		}
		if (n > 0 && count[1] + count[2] + count[3] === 0) yield* treat(ctx.rng.int(n), 'force', -1);
		if (n > 0) yield { phase: 'done', state: { fate, shown }, metrics: { n, hidden: count[1] + count[2] + count[3], masked: count[1], random: count[2], kept: count[3] } };
	}
};
