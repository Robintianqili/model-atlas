// View 'char-rnn/timeline', the numbers behind it (pure TS: the view, its describe() and lesson.ts all read these, so the
// narration and the stage cannot disagree). Modes:
//   memory  the hidden state h of all units after every character of the learner's text (the fwd trace)
//   unroll  the same, filled one character at a time from the `unroll` process, with each guess for the next character
//   unit    one unit's cell value c under each character, and its top contexts on held-out stories (unitTop.json)
//   fade    how much of a memory survives d characters: products of forget gates (the fwd trace's lstm#f)
//   sample  the text one sampling source wrote so far (a step's compare shows `sample` and `greedy` side by side)
//   next    the chances of the next character after any one character (the softmax rows of the fwd trace)
import type { ExplainerRuntime, StageProps, Tape, Tensor } from '../../../../atlas/types/index.ts';
import { scoresAfter, softmax, weightsOf, type Weights } from '../../rnn.ts';
import { CONTEXT } from '../../process.ts';

/** the reference forget gate of the fade chart and of the predict question in step 11 */
export const REF_GATE = 0.9;
export type Mode = 'memory' | 'unroll' | 'unit' | 'fade' | 'sample' | 'next';
type Props = Omit<StageProps, 'ctx'>;
type RT = ExplainerRuntime;

export const show = (c: string | undefined) => (c === undefined ? '' : c === ' ' ? '␣' : c === '\n' ? '↵' : c);
export const isLetter = (c: string | undefined) => !!c && /^[A-Za-z]$/.test(c);
export const alphabetOf = (rt: RT): string[] => ((rt.spec as unknown as { alphabet?: string[] } | null)?.alphabet ?? []);
export const charsOf = (rt: RT): string[] => rt.pre?.tokens ?? [];
/** the unit the lesson follows: the letter counter that units.py found (facts.json) */
export const unitOf = (rt: RT): number => rt.num('fact:counterUnit');
const tr = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
/** column j of a [T, H] tensor */
export const col = (x: Tensor, j: number): Float64Array => Float64Array.from({ length: x.shape[0] }, (_, t) => x.data[t * x.shape[1] + j]);

/** The words of the text: letters only, with their first and last positions. */
export function words(chars: string[]): { word: string; from: number; to: number }[] {
	const out: { word: string; from: number; to: number }[] = [];
	for (let t = 0; t < chars.length; t++) {
		if (!isLetter(chars[t]) || isLetter(chars[t - 1])) continue;
		let e = t;
		while (isLetter(chars[e + 1])) e++;
		out.push({ word: chars.slice(t, e + 1).join(''), from: t, to: e });
	}
	return out;
}
/** The longest word (ties: the last one). */
export function longest(chars: string[]) {
	let b: { word: string; from: number; to: number } | null = null;
	for (const w of words(chars)) if (!b || w.word.length >= b.word.length) b = w;
	return b;
}
/** The moment the gate steps zoom into: the space right after the longest word that a space ends (ties: the last);
 *  without one, the last character. */
export function momentOf(chars: string[]): { t: number; word: string; space: boolean } {
	let b: { word: string; to: number } | null = null;
	for (const w of words(chars)) if (chars[w.to + 1] === ' ' && (!b || w.word.length >= b.word.length)) b = w;
	return b ? { t: b.to + 1, word: b.word, space: true } : { t: chars.length - 1, word: '', space: false };
}

/** Of a memory held at character s, the share still there d characters later through the forget gates alone:
 *  f(s+1) × … × f(s+d). keep[d] is the median over every unit and start; unit u's curve is its median over starts. */
/** The median of a (reordered in place): quickselect, linear time (a full sort of 8,000 numbers 41 times took ~100 ms). */
export function median(a: Float64Array): number {
	const n = a.length;
	if (!n) return NaN;
	const k = n >> 1;
	let l = 0;
	let r = n - 1;
	while (r > l) {
		const p = a[(l + r) >> 1];
		let i = l;
		let j = r;
		while (i <= j) {
			while (a[i] < p) i++;
			while (a[j] > p) j--;
			if (i <= j) {
				const x = a[i];
				a[i++] = a[j];
				a[j--] = x;
			}
		}
		if (k <= j) r = j;
		else if (k >= i) l = i;
		else break;
	}
	if (n % 2) return a[k];
	let lo = -Infinity;
	for (let i = 0; i < k; i++) if (a[i] > lo) lo = a[i];
	return (lo + a[k]) / 2;
}
export function fadeOf(F: Tensor, maxD = 40, unit = -1) {
	const [T, H] = F.shape;
	const D = Math.max(0, Math.min(maxD, T - 1));
	const prod = new Float64Array(H * T).fill(1); // prod[j, s]: f(s+1) × … × f(s+d) of unit j, grown one d at a time
	const keep: number[] = [];
	const per: number[][] = Array.from({ length: H }, () => []);
	for (let d = 0; d <= D; d++) {
		const n = T - d;
		const all = new Float64Array(H * n);
		for (let j = 0; j < H; j++) {
			const row = new Float64Array(n);
			for (let s = 0; s < n; s++) {
				if (d > 0) prod[j * T + s] *= F.data[(s + d) * H + j];
				row[s] = prod[j * T + s];
			}
			all.set(row, j * n);
			per[j].push(median(row));
		}
		keep.push(median(all));
	}
	let mean = 0;
	for (let k = 0; k < F.data.length; k++) mean += F.data[k];
	mean /= F.data.length;
	let patient = 0;
	for (let j = 1; j < H; j++) if (per[j][D] > per[patient][D]) patient = j;
	return { D, keep, patient, patientCurve: per[patient], unitCurve: unit >= 0 && unit < H ? per[unit] : [], mean };
}
const fadeMemo = new WeakMap<object, ReturnType<typeof fadeOf>>();
export const fadeAt = (rt: RT) => {
	const F = tr(rt, 'lstm#f');
	if (!F) return null;
	const hit = fadeMemo.get(F) ?? fadeOf(F, 40, unitOf(rt));
	fadeMemo.set(F, hit);
	return hit;
};

/** The h columns of the unroll tape up to frame `index`, with each guess and its frame. */
export function unrolled(tape: Tape | null | undefined, index: number) {
	const cols: (Float64Array | null)[] = [];
	const guesses: { top: number; pTop: number; next?: number }[] = [];
	if (!tape || index < 0) return { cols, guesses, t: -1, phase: '' };
	const f = tape.frame(Math.min(index, tape.length - 1));
	const t = f.step ?? 0;
	for (let s = 0; s <= t; s++) {
		const ih = tape.find({ phase: 'hidden', step: s });
		cols.push(ih >= 0 && ih <= index ? ((tape.state(ih) as { h: Tensor }).h.data as Float64Array) : null);
		const ip = tape.find({ phase: 'predict', step: s });
		if (ip >= 0 && ip <= index) {
			const m = tape.frame(ip).metrics ?? {};
			guesses.push({ top: m.top, pTop: m.pTop, next: m.next });
		}
	}
	return { cols, guesses, t, phase: f.phase };
}

/** One sampling run at the frame that matches `step` (the whole run without it). */
export function written(rt: RT, source: string, step?: number) {
	const tape = rt.sources[source]?.tape;
	if (!tape || !tape.length) return null;
	const i = step === undefined ? tape.length - 1 : tape.find({ phase: 'sample', step });
	const k = i < 0 ? tape.length - 1 : i;
	const s = tape.state(k) as { ids: number[]; top?: { id: number; p: number }[]; token?: number; u?: number; p?: number; pModel?: number };
	const prompt = (tape.state(0) as { ids: number[] }).ids.length;
	return { ids: s.ids, prompt, top: s.top ?? [], token: s.token, u: s.u, p: s.p, pModel: s.pModel, frame: tape.frame(k) };
}

/** The least likely character a sampling run drew (its chance), over the whole run, and how many it drew. */
export function riskiest(rt: RT, source: string) {
	const tape = rt.sources[source]?.tape;
	const ps = Array.from({ length: Math.max(0, (tape?.length ?? 0) - 1) }, (_, k) => (tape!.state(k + 1) as { p?: number }).p).filter((p): p is number => typeof p === 'number');
	return { p: ps.length ? Math.min(...ps) : NaN, n: ps.length };
}

/** The chances a sample frame's draw walked, in alphabet (id) order (the model's, temperature 1), and where each
 *  character's piece starts; memoized per frame, since the view and its describe() both ask. */
const drawMemo = new WeakMap<object, { draw: Float64Array; from: number[] }>();
function drawOf(w: Weights, ids: number[]) {
	const draw = softmax(scoresAfter(w, ids.slice(0, -1).slice(-CONTEXT)));
	let acc = 0;
	return { draw, from: Array.from(draw, (v) => ((acc += v), acc - v)) };
}

export function timelineOf(p: Props) {
	const rt = p.rt;
	const mode = (p.props.mode ?? 'memory') as Mode;
	const chars = charsOf(rt);
	const alphabet = alphabetOf(rt);
	const unit = unitOf(rt);
	const h = tr(rt, 'lstm');
	const c = tr(rt, 'lstm#c');
	const base = { mode, chars, alphabet, unit, moment: momentOf(chars) };
	if (mode === 'unroll') return { ...base, ...unrolled(rt.sources[p.source]?.tape, p.at.index) };
	if (mode === 'sample') {
		const a = written(rt, p.source, p.at.frame?.step);
		const d = a && Number.isFinite(a.u) && rt.model ? (drawMemo.get(a.frame) ?? drawOf(weightsOf(rt.model), a.ids)) : null;
		if (a && d) drawMemo.set(a.frame, d);
		return { ...base, a, draw: d?.draw ?? null, from: d?.from ?? [] };
	}
	if (!h || !c) return null;
	if (mode === 'fade') return { ...base, fade: fadeAt(rt) };
	if (mode === 'next') return { ...base, probs: tr(rt, 'probs') ?? null };
	if (mode === 'unit') {
		const top = (rt as unknown as { assets?: Record<string, unknown> }).assets?.['unitTop.json'] as
			| { chars: number; units: { unit: number; r: number; top: { before: string; at: string; after: string; c: number }[] }[] }
			| undefined;
		return { ...base, cu: Number.isFinite(unit) ? col(c, unit) : null, word: longest(chars), top: top?.units[unit]?.top ?? [], topChars: top?.chars ?? NaN };
	}
	return { ...base, h };
}
export type Timeline = NonNullable<ReturnType<typeof timelineOf>>;
