// masked-lm: process 'batch' (kind C), adapted from tinybert's 'hide' (credit: the tinybert package): BERT's
// training-time masking, as TinyBERT was trained (pipeline/tinybert/data.py), drawn on a small batch with the page's rng,
// one frame per token. Row 0 is the learner's text; the other rows are TinyBERT's example sentences (source_data.ts).
// Each token is chosen with probability rate (u < rate); a chosen token becomes [MASK] when v < pMask, a random text
// token when v < pMask + pRandom, and stays as it is otherwise. A row with no token chosen gets one picked at random
// (training does the same, so every example teaches something). After the batch, the same rule is counted (no frames)
// on a full training batch: `big` examples of `bigLen` tokens. pipeline/masked-lm/reference.py is the independent Python
// run (atlas parity: same frames).
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { contentOf, MASK, MAXC, TEXT_VOCAB, type MaskedRaw } from './adapters/masked-text/adapter.ts';
import { STORIES } from './source_data.ts';

export type BatchOpt = { rate: number; pMask: number; pRandom: number; vocab: number; big: number; bigLen: number };
/** over every token of the batch, row after row: fate 0 not chosen, 1 [MASK], 2 a random token, 3 chosen but kept;
 *  shown: the id the model reads */
export type BatchState = { fate: Tensor; shown: Tensor };
type Run = ProcessRun<BatchState, BatchOpt>;
export const FATE = { none: 0, mask: 1, random: 2, keep: 3 };

/** The batch: the learner's text first, then every example sentence that differs from it (none for an empty text). */
export function batchRows(raw: MaskedRaw): number[][] {
	const c = contentOf(raw);
	return c.length ? [c, ...STORIES.map((s) => s.ids).filter((ids) => ids.join() !== c.join())] : [];
}

const what = (en: string, zh: string) => ({ what: { en, zh } });

export const batch: ProcessDef<MaskedRaw, BatchOpt, BatchState> = {
	id: 'batch',
	kind: 'algorithm',
	title: { en: 'The 80/10/10 rule on a batch', zh: '在一个批次上运行 80/10/10 规则' },
	phases: {
		start: { label: { en: 'The batch', zh: '这个批次' }, kind: 'input', narration: what('The batch has {frame.rows|int} rows and {frame.n|int} tokens; none is chosen yet.', '这个批次有 {frame.rows|int} 行、{frame.n|int} 个词元；还没有选中任何一个。') },
		draw: {
			label: { en: 'Draw for one token', zh: '为一个词元抽签' },
			kind: 'sample',
			narration: what('Row {frame.row|int}, token {frame.i|int} (counting from 0) draws u = {frame.u}; it is chosen when u is below {frame.rate}.', '第 {frame.row|int} 行第 {frame.i|int} 个词元（从 0 数起）抽到 u = {frame.u}；u 小于 {frame.rate} 时它被选中。')
		},
		force: {
			label: { en: 'None chosen in a row: pick one', zh: '一行里一个都没选中：随机挑一个' },
			kind: 'sample',
			narration: what('No token of row {frame.row|int} was chosen, so token {frame.i|int} (counting from 0) is picked at random.', '第 {frame.row|int} 行一个词元都没被选中，所以随机挑了第 {frame.i|int} 个词元（从 0 数起）。')
		},
		done: {
			label: { en: 'The training batch', zh: '训练批次' },
			kind: 'output',
			narration: what('{frame.chosen|int} of {frame.n|int} tokens are chosen: {frame.masked|int} as [MASK], {frame.random|int} as a random token, {frame.kept|int} kept.', '{frame.n|int} 个词元中选中了 {frame.chosen|int} 个：{frame.masked|int} 个换成 [MASK]，{frame.random|int} 个换成随机词元，{frame.kept|int} 个保持不变。')
		}
	},
	schema: {
		fate: { kind: 'tensor', label: { en: 'What happens to each token', zh: '每个词元的处理方式' }, track: 'changes', tol: 'exact' },
		shown: { kind: 'tensor', label: { en: 'The ids the model reads', zh: '模型读到的 id' }, track: 'changes', tol: 'exact' }
	},
	defaults: { rate: 0.15, pMask: 0.8, pRandom: 0.1, vocab: TEXT_VOCAB, big: 512, bigLen: MAXC },
	*run(input, ctx): Run {
		const o = ctx.opt;
		const rows = batchRows(input);
		const flat = rows.flat();
		const N = flat.length;
		if (!N) return;
		const fate: Tensor = { shape: [N], data: new Float64Array(N) };
		const shown: Tensor = { shape: [N], data: Float64Array.from(flat) };
		const count = [0, 0, 0, 0];
		const tally = () => ({ chosen: count[1] + count[2] + count[3], masked: count[1], random: count[2], kept: count[3] });
		/** the fate of one chosen token: 1 [MASK], 2 random (with its id), 3 kept */
		const fateOf = () => {
			const v = ctx.rng.next();
			const f = v < o.pMask ? FATE.mask : v < o.pMask + o.pRandom ? FATE.random : FATE.keep;
			return { v, f, rid: f === FATE.random ? ctx.rng.int(o.vocab) : -1 };
		};
		yield { phase: 'start', state: { fate, shown }, metrics: { rows: rows.length, n: N } };
		let at = 0;
		for (let r = 0; r < rows.length && !ctx.signal.cancelled; r++) {
			const len = rows[r].length;
			const first = at;
			function* treat(i: number, phase: 'draw' | 'force', u: number): Run {
				const { v, f, rid } = fateOf();
				const g = first + i;
				const to = f === FATE.mask ? MASK : f === FATE.random ? rid : flat[g];
				fate.data[g] = f;
				shown.data[g] = to;
				count[f]++;
				yield { phase, step: g, state: { fate, shown }, changes: [{ field: 'fate', index: [g], from: 0, to: f }, ...(to !== flat[g] ? [{ field: 'shown', index: [g], from: flat[g], to }] : [])], metrics: { row: r, i, at: g, u, pick: 1, v, fate: f, rid, rate: o.rate, ...tally() } };
			}
			let inRow = 0;
			for (let i = 0; i < len; i++) {
				const u = ctx.rng.next();
				if (u < o.rate) {
					inRow++;
					yield* treat(i, 'draw', u);
				} else yield { phase: 'draw', step: first + i, state: { fate, shown }, metrics: { row: r, i, at: first + i, u, pick: 0, v: -1, fate: 0, rid: -1, rate: o.rate, ...tally() } };
			}
			if (inRow === 0) yield* treat(ctx.rng.int(len), 'force', -1);
			at += len;
		}
		// the same rule on a full training batch, counted only: big examples of bigLen tokens each
		const big = [0, 0, 0, 0];
		for (let e = 0; e < o.big; e++) {
			let inRow = 0;
			for (let i = 0; i < o.bigLen; i++) {
				if (ctx.rng.next() < o.rate) {
					inRow++;
					big[fateOf().f]++;
				}
			}
			if (inRow === 0) {
				ctx.rng.int(o.bigLen); // the forced token: which one does not change the counts
				big[fateOf().f]++;
			}
		}
		const bigChosen = big[1] + big[2] + big[3];
		yield { phase: 'done', state: { fate, shown }, metrics: { rows: rows.length, n: N, ...tally(), big: o.big, bigLen: o.bigLen, bigN: o.big * o.bigLen, bigChosen, bigMasked: big[1], bigRandom: big[2], bigKept: big[3] } };
	}
};
