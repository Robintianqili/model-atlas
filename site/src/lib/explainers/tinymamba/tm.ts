// tinymamba: helpers shared by the manifest, the lesson, the views and the claim tests (pure TS). Everything is read
// from the runtime: the trace of the learner's row ('fwd': Δ is B<b>.scan#delta, the state B<b>.scan#h) and the spec.
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { ALPHABET, ASK, answersOf, isLetter } from './task.ts';
import type { ScanInput } from './process.ts';

export type RT = ExplainerRuntime;
export const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
export const idsOf = (rt: RT): number[] => Array.from((rt.pre?.tensors.input?.data ?? []) as ArrayLike<number>);
export const tok = (id: number) => ALPHABET[id] ?? '?';

/** A number as a plain decimal with `sig` significant digits (never "e−4": the lesson's formula already has an e in
 *  it), with the typographic minus: 0.0001270, −6.617, 0. */
export function dec(v: number, sig = 4): string {
	if (!Number.isFinite(v)) return 'NaN';
	if (v === 0) return '0';
	const r = Number(v.toPrecision(sig));
	const e = Math.floor(Math.log10(Math.abs(r)));
	return r.toFixed(Math.max(0, Math.min(20, sig - 1 - e))).replace('-', '−');
}

/** One scan step of one slot as a calculator sees it (steps 5 and 6): the inputs rounded to 4 significant digits and
 *  every later number worked out from the rounded ones (decay, write and kept part to 4 digits, the new state to 3), so
 *  each printed line can be checked by hand. claims.test.ts checks that the new state is the model's. */
export function worked(m: { delta: number; A: number; B: number; u: number; hOld: number }) {
	const r = (v: number, s: number) => (v === 0 ? 0 : Number(v.toPrecision(s)));
	const [delta, A, B, u, hOld] = [m.delta, m.A, m.B, m.u, m.hOld].map((v) => r(v, 4));
	const decay = r(Math.exp(delta * A), 4);
	const write = r(delta * B * u, 4);
	const kept = r(decay * hOld, 4);
	return { delta, A, B, u, hOld, decay, write, kept, hNew: r(kept + write, 3) };
}

/** Sizes from the spec: d (token vector), D (channels), N (slots per channel), L (blocks), V (alphabet), T (tokens). */
export function dims(rt: RT) {
	const s = rt.spec as { config?: { d: number; D: number; N: number; layers: number; vocab: number } } | null;
	const c = s?.config;
	return { d: c?.d ?? NaN, D: c?.D ?? NaN, N: c?.N ?? NaN, L: c?.layers ?? NaN, V: c?.vocab ?? NaN, T: idsOf(rt).length };
}

/** Δ of block b at token t, channel c (NaN when missing). */
export const deltaAt = (rt: RT, b: number, t: number, c: number) => T(rt, `B${b}.scan#delta`)?.data[t * dims(rt).D + c] ?? NaN;

/** Mean Δ of channel c (or of all channels when c < 0) over the tokens of one kind: dots, letters or question marks. */
export function meanDelta(rt: RT, b: number, kind: 'dot' | 'letter' | 'ask', c = -1) {
	const ids = idsOf(rt);
	const { D } = dims(rt);
	let s = 0;
	let n = 0;
	ids.forEach((id, t) => {
		if ((kind === 'dot' && id === 0) || (kind === 'letter' && isLetter(id)) || (kind === 'ask' && id === ASK))
			for (let j = c < 0 ? 0 : c; j < (c < 0 ? D : c + 1); j++) (s += deltaAt(rt, b, t, j), n++);
	});
	return n ? s / n : NaN;
}

/** The channel of block b that separates letters from dots most: the largest (mean Δ at letters − mean Δ at dots).
 *  Ties: the lower channel. Without both kinds of token: the channel with the largest Δ anywhere. */
export function pickCh(rt: RT, b = 1): number {
	const { D, T: n } = dims(rt);
	let best = 0;
	let bv = -Infinity;
	for (let c = 0; c < D; c++) {
		let v = meanDelta(rt, b, 'letter', c) - meanDelta(rt, b, 'dot', c);
		if (!Number.isFinite(v)) for (let t = 0, m = -Infinity; t < n; t++) v = m = Math.max(m, deltaAt(rt, b, t, c));
		if (v > bv) (bv = v, best = c);
	}
	return best;
}

/** The first letter of the row (else the first token). */
export const firstLetter = (rt: RT) => Math.max(0, idsOf(rt).findIndex(isLetter));

/** The slot of channel ch whose state the first letter changes most (ties: the lower slot). */
export function pickSlot(rt: RT, b = 1, ch = pickCh(rt, b)): number {
	const h = T(rt, `B${b}.scan#h`);
	const { D, N } = dims(rt);
	const t = firstLetter(rt);
	let best = 0;
	for (let n = 1; h && n < N; n++) {
		const at = (s: number, k: number) => (s < 0 ? 0 : h.data[(s * D + ch) * N + k]);
		if (Math.abs(at(t, n) - at(t - 1, n)) > Math.abs(at(t, best) - at(t - 1, best))) best = n;
	}
	return best;
}

/** The scan process's input: the row, the weights, and the channel and slot its metrics follow. JSON: no weights. */
export function scanInput(rt: RT, block = 1): ScanInput {
	const ch = pickCh(rt, block);
	const o: ScanInput = { ids: idsOf(rt), w: rt.model?.weights ?? null, block, ch, slot: pickSlot(rt, block, ch) };
	Object.defineProperty(o, 'toJSON', { value: () => ({ ids: o.ids, block, ch: o.ch, slot: o.slot }), enumerable: false });
	return o;
}

/** At every '?': the letter it asks for (-1: none), the model's top token and its chance, the chance of the right one. */
export function answers(rt: RT) {
	const p = T(rt, 'probs');
	const V = p?.shape[1] ?? 0;
	return answersOf(idsOf(rt)).asks.map((a) => {
		const row = p ? Array.from(p.data.subarray(a.at * V, (a.at + 1) * V)) : [];
		const top = row.reduce((bi, v, i) => (v > row[bi] ? i : bi), 0);
		return { ...a, top: row.length ? top : -1, p: row[top] ?? NaN, pWant: a.want >= 0 ? (row[a.want] ?? NaN) : NaN, ok: a.want >= 0 && top === a.want, row };
	});
}

/** Step 5: slot `slot` of the followed channel: the write at the first letter, the read at the first '?'. */
export function writeRead(rt: RT) {
	const { D, N } = dims(rt);
	const ids = idsOf(rt);
	const r = { ids, t1: firstLetter(rt), tq: ids.indexOf(ASK) };
	const ch = pickCh(rt, 1);
	const slot = pickSlot(rt, 1, ch);
	const v = (id: string, t: number, i: number, w: number) => T(rt, id)?.data[t * w + i] ?? NaN;
	const t = r.t1;
	const tq = Math.max(0, r.tq);
	return { ch, slot, N, t, tq, let: tok(r.ids[t]), B: v('B1.B', t, slot, N), u: v('B1.u', t, ch, D), d: deltaAt(rt, 1, t, ch), write: deltaAt(rt, 1, t, ch) * v('B1.B', t, slot, N) * v('B1.u', t, ch, D), C: v('B1.C', tq, slot, N) };
}

/** Step 8: the question marks (the same token every time) through block 1 and block 2: rows of a [T, w] trace tensor
 *  at each '?', and the distance between two rows. */
export function asks(rt: RT) {
	const at = idsOf(rt).flatMap((id, t) => (id === ASK ? [t] : []));
	const row = (id: string, t: number, w: number) => Array.from(T(rt, id)?.data.subarray(t * w, (t + 1) * w) ?? []);
	const dist = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
	const norm = (a: number[]) => Math.sqrt(a.reduce((s, v) => s + v * v, 0));
	return { at, row, dist, norm };
}
