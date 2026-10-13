// tiny-encdec: what the views and the narration read from the trace of the learner's date (pure TS). Decoder steps,
// heads and input positions count from 0, as the trace does; the narration says "(counting from 0)" where it quotes one.
import type { ExplainerRuntime, I18n, Tensor } from '../../atlas/types/index.ts';
import { EOS, TGT, iso, readDate } from './data.ts';
import { argmax } from './tf.ts';
import { textOf } from './adapters/date-text/adapter.ts';

export type RT = ExplainerRuntime;
export const tr = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
/** an output symbol as the page writes it (▶ start, ■ end) */
export const sym = (id: number) => (id === EOS ? '■' : id === 0 ? '▶' : (TGT[id] ?? '?'));
/** an input character as the page writes it (a space as ␣) */
export const chr = (c: string) => (c === ' ' ? '␣' : c);
export const q = (c: string) => `“${chr(c)}”`;
/** the decoder step that writes the year's last digit (YYYY-MM-DD: index 3), followed by steps 5 to 8 */
export const ROW = 3;

/** The learner's date, the characters the model read, the decoder's input and output, and the right answer. */
export function look(rt: RT) {
	const text = textOf(rt.input);
	const chars = rt.pre?.tokens ?? [];
	const P = tr(rt, 'probs');
	const V = P?.shape[1] ?? 0;
	const ids = P ? Array.from({ length: P.shape[0] }, (_, t) => argmax(P.data.subarray(t * V, (t + 1) * V))) : [];
	const yin = Array.from(tr(rt, 'greedy')?.data ?? []);
	const end = ids.indexOf(EOS);
	const written = (end < 0 ? ids : ids.slice(0, end)).map(sym).join('');
	const d = readDate(text);
	const want = d.valid ? iso(d.ymd!) : '';
	return { text, chars, T: chars.length, ids, yin, end, written, want, right: !!want && want === written, noDay: !!d.ymd && !d.valid, fmt: d.fmt };
}

/** After “the model wrote …”: right, wrong (and the right answer), no such day, or nothing to check. */
export function verdict(rt: RT): I18n {
	const L = look(rt);
	if (L.want) return L.right ? { en: ', which is right', zh: '，这是对的' } : { en: `, but the right answer is ${L.want}`, zh: `，但正确答案是 ${L.want}` };
	if (L.noDay) return { en: ', but no such day exists, so there is no right answer', zh: '，但这一天根本不存在，所以没有正确答案' };
	return { en: ' (the page cannot read your text as a date, so there is nothing to check)', zh: '（你的文字不是本页能读懂的日期格式，所以没有答案可以核对）' };
}

/** Row r of head h of an [H, R, C] attention tensor. */
export function head(t: Tensor | undefined, h: number, r: number): Float32Array | Float64Array | null {
	if (!t || t.shape.length !== 3) return null;
	const [H, R, C] = t.shape;
	return h >= 0 && h < H && r >= 0 && r < R ? t.data.subarray((h * R + r) * C, (h * R + r + 1) * C) : null;
}
/** The heads averaged: [R, C]. */
export function avgHeads(t: Tensor | undefined): Float64Array {
	if (!t || t.shape.length !== 3) return new Float64Array(0);
	const [H, R, C] = t.shape;
	const o = new Float64Array(R * C);
	t.data.forEach((v, i) => (o[i % (R * C)] += v / H));
	return o;
}
/** the largest entry of a row and where it is */
export const top = (r: ArrayLike<number> | null) => (r && r.length ? { j: argmax(r), w: r[argmax(r)] } : { j: -1, w: NaN });

/** Row r of an [H, R, C] attention layer: per head, the column with the largest weight and that weight, and the head
 *  with the sharper row (the one the worked steps follow). */
export function sharpest(rt: RT, id: string, r: number) {
	const A = tr(rt, id);
	const heads = Array.from({ length: A?.shape[0] ?? 0 }, (_, h) => top(head(A, h, r)));
	const h = heads.length ? heads.reduce((b, x, i) => (x.w > heads[b].w ? i : b), 0) : -1;
	return { heads, h, ...(heads[h] ?? { j: -1, w: NaN }) };
}
/** Cross-attention of decoder step t. */
export const crossAt = (rt: RT, t: number) => sharpest(rt, 'dec.cattn', t);

/** Encoder self-attention: the share of every query's weight that goes to characters on its right (later positions),
 *  averaged over queries and heads, and the strongest look to the right (query i, key j, head h, weight w). */
export function rightward(rt: RT) {
	const A = tr(rt, 'enc.attn');
	if (!A) return { share: NaN, i: -1, j: -1, h: -1, w: NaN };
	const [H, T] = A.shape;
	let s = 0;
	let best = { i: -1, j: -1, h: -1, w: -1 };
	for (let h = 0; h < H; h++)
		for (let i = 0; i < T; i++)
			for (let j = 0; j < T; j++) {
				const w = A.data[(h * T + i) * T + j];
				if (j > i) s += w;
				if (j > i && w > best.w) best = { i, j, h, w };
			}
	return { share: s / (H * T), ...best };
}
