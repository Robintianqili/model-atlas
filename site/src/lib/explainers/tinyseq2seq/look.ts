// tinyseq2seq: what the views and the narration read from the trace of the learner's date (pure TS). Decoder steps and
// input positions count from 0, as the trace does; the narration says "(counting from 0)" where it quotes one.
import type { ExplainerRuntime, I18n, Tensor } from '../../atlas/types/index.ts';
import { EOS, TGT, iso, readDate } from './data.ts';
import { argmax } from './seq.ts';
import { textOf } from './adapters/date-text/adapter.ts';
import { minus } from '../../atlas/core/format.ts';

export type RT = ExplainerRuntime;
export const tr = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
/** row r of a [rows, n] tensor */
export const row = (t: Tensor | undefined, r: number): Float32Array | Float64Array | null => (t && r >= 0 && r < t.shape[0] ? t.data.subarray(r * t.shape[1], (r + 1) * t.shape[1]) : null);
/** an output symbol as the page writes it */
export const sym = (id: number) => (id === EOS ? '■' : id === 0 ? '▶' : (TGT[id] ?? '?'));
/** an input character as the page writes it (a space as ␣) */
export const chr = (c: string) => (c === ' ' ? '␣' : c);
/** Numbers in worked sums, never in e-notation (the lesson's e is Euler's number): fx = fixed decimals, so the printed
 *  parts add up to the printed total; dec = k significant digits, small values as decimals (0.0004647). */
export const fx = (v: number, d = 4) => minus(v.toFixed(d));
/** the rest of a sum printed with fx: (printed total) − (printed part), so the printed numbers add up exactly */
export const restOf = (total: number, part: number) => (Math.round(total * 1e4) - Math.round(part * 1e4)) / 1e4;
export const dec = (v: number, k = 4) => (Math.abs(v) < 1e-6 ? `${v < 0 ? '−' : ''}<0.000001` : minus(v.toPrecision(k)));

/** The learner's date, the characters the model read, what it wrote, and the right answer when the text is a real date. */
export function look(rt: RT) {
	const text = textOf(rt.input);
	const chars = rt.pre?.tokens ?? [];
	const n = tr(rt, 'dec#next');
	const ids = n ? Array.from(n.data) : [];
	const end = ids.indexOf(EOS);
	const written = (end < 0 ? ids : ids.slice(0, end)).map(sym).join('');
	const d = readDate(text);
	const want = d.valid ? iso(d.ymd!) : '';
	const y = /\d{4}/.exec(chars.join(''));
	return { text, chars, T: chars.length, ids, end, written, want, right: !!want && want === written, noDay: !!d.ymd && !d.valid, fmt: d.fmt, year: y ? [y.index, y.index + 4] : null };
}

/** After “the model wrote …”: right, wrong (and the right answer), no such day, or nothing to check. */
export function verdict(rt: RT): I18n {
	const L = look(rt);
	if (L.want) return L.right ? { en: ', which is right', zh: '，这是对的' } : { en: `, but the right answer is ${L.want}`, zh: `，但正确答案是 ${L.want}` };
	if (L.noDay) return { en: ', but no such day exists, so there is no right answer', zh: '，但这一天根本不存在，所以没有正确答案' };
	return { en: ' (the page cannot read your text as a date, so there is nothing to check)', zh: '（你的文字不是本页能读懂的日期格式，所以没有答案可以核对）' };
}

/** Decoder step t: the state it starts from, the symbol it reads, the scores and weights over the input, the context,
 *  the new state and the chances of the 13 symbols; j = the input position with the largest weight. */
export function stepAt(rt: RT, t: number) {
	const L = look(rt);
	const a = row(tr(rt, 'dec#weights'), t);
	const j = a ? argmax(a) : -1;
	const p = row(tr(rt, 'probs'), t);
	const y = L.ids[t] ?? -1;
	const sorted = p ? Array.from(p, (v, i) => i).sort((u, v) => p[v] - p[u] || u - v) : [];
	const yearShare = a && L.year ? Array.from(a.slice(L.year[0], L.year[1])).reduce((s, v) => s + v, 0) : NaN;
	return {
		...L, t, j, a, e: row(tr(rt, 'dec#scores'), t), c: row(tr(rt, 'dec#context'), t), s: row(tr(rt, 'dec#state'), t), p, y, yearShare,
		prev: t === 0 ? 0 : (L.ids[t - 1] ?? 0),
		sPrev: t === 0 ? row(tr(rt, 'enc'), L.T - 1) : row(tr(rt, 'dec#state'), t - 1),
		runnerUp: sorted[1] ?? -1
	};
}

/** The score of input position j at step t, worked out: e = Σ_a v_a · tanh(q_a + k_a). */
export function scoreParts(rt: RT, t: number, j: number) {
	const q = row(tr(rt, 'dec#query'), t);
	const k = row(tr(rt, 'dec#keys'), j);
	const v = rt.model?.weights['dec.v']?.data ?? null;
	if (!q || !k || !v) return null;
	const th = Float64Array.from(q, (x, a) => Math.tanh(x + k[a]));
	const prod = th.map((x, a) => x * v[a]);
	return { q, k, v, th, prod, sum: prod.reduce((s, x) => s + x, 0) };
}

/** Softmax of step t for position j, worked out: α_j = e^(e_j − max) / Σ_i e^(e_i − max). */
export function softParts(e: ArrayLike<number> | null, j: number) {
	if (!e || j < 0) return null;
	const m = Math.max(...Array.from(e));
	const sum = Array.from(e).reduce((s, x) => s + Math.exp(x - m), 0);
	return { m, ex: Math.exp(e[j] - m), sum, w: Math.exp(e[j] - m) / sum };
}

/** The context vector of step t, worked out for its largest feature f: c_f = Σ_j α_j h_jf, largest parts first. */
export function mixParts(rt: RT, t: number) {
	const S = stepAt(rt, t);
	const h = tr(rt, 'enc');
	if (!S.c || !S.a || !h) return null;
	const f = argmax(Array.from(S.c, Math.abs));
	const H = h.shape[1];
	const parts = Array.from(S.a, (w, j) => ({ j, w, h: h.data[j * H + f], wh: w * h.data[j * H + f] })).sort((x, y) => Math.abs(y.wh) - Math.abs(x.wh));
	return { f, cf: S.c[f], parts, H };
}
