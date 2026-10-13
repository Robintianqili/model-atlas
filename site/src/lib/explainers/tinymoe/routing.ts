// tinymoe: helpers shared by the lesson, the process, the view and the claim tests (pure TS). Everything is read from the
// runtime: the trace of the learner's text ('fwd': B<b>.router scores, B<b>.route chances, B<b>.moe#gate gates,
// B<b>.moe#e expert outputs) and the spec (config, alphabet, the held-out routing of stats.py in `routing`).
import type { ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { topK } from './ops/moe.ts';

export type RT = ExplainerRuntime;
type Spec = { config?: { d: number; heads: number; layers: number; experts: number; k: number; hidden: number; denseHidden: number; vocab: number }; alphabet?: string[]; routing?: { moe: { share: number[]; top: { char: string; share: number; count: number }[][] }[] } };

export const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
export const idsOf = (rt: RT): number[] => Array.from((rt.pre?.tensors.input?.data ?? []) as ArrayLike<number>);
const spec = (rt: RT) => (rt.spec ?? {}) as Spec;
/** A character as the views print it: a space as ␣, a new line as ↵. */
export const show = (c: string | undefined) => (c === ' ' ? '␣' : c === '\n' ? '↵' : (c ?? '?'));
export const charOf = (rt: RT, t: number) => show(spec(rt).alphabet?.[idsOf(rt)[t]]);

/** Sizes from the spec: d, E experts, K per character, H hidden units per expert, Hd of the dense twin, L blocks, n chars. */
export function cfg(rt: RT) {
	const c = spec(rt).config;
	return { d: c?.d ?? NaN, E: c?.experts ?? NaN, K: c?.k ?? NaN, H: c?.hidden ?? NaN, Hd: c?.denseHidden ?? NaN, L: c?.layers ?? NaN, n: idsOf(rt).length };
}

/** Row t of a [T, w] trace tensor (empty when missing). */
export function row(rt: RT, id: string, t: number): number[] {
	const x = T(rt, id);
	if (!x || t < 0) return [];
	const w = x.shape[x.shape.length - 1];
	return Array.from(x.data.subarray(t * w, (t + 1) * w));
}

/** Block b's router chances, one row per character. */
export const chances = (rt: RT, b: number): number[][] => idsOf(rt).map((_, t) => row(rt, `B${b}.route`, t));

/** One character's route in block b: the 4 scores, the chances, the two experts kept (a, b) and their gates (the op's). */
export function routeOf(rt: RT, blk: number, t: number) {
	const z = row(rt, `B${blk}.router`, t);
	const p = row(rt, `B${blk}.route`, t);
	const g = row(rt, `B${blk}.moe#gate`, t);
	const { i } = topK(p, cfg(rt).K || 2);
	const [a, b] = [i[0] ?? -1, i[1] ?? -1];
	return { t, c: charOf(rt, t), z, p, g, a, b, pa: p[a] ?? NaN, pb: p[b] ?? NaN, ga: g[a] ?? NaN, gb: g[b] ?? NaN };
}

/** The character the lesson follows: the last one (the one whose next character the model guesses). */
export const focusT = (rt: RT) => idsOf(rt).length - 1;

/** The capacity rule the page plays (process.ts): characters in order, each one's experts in order of their chance; an
 *  expert with fewer than cap slots takes the slot, a full one drops it. cap = ceil(cf · k · T / E). */
export function dispatchAll(p: number[][], k: number, cf: number) {
	const n = p.length;
	const E = p[0]?.length ?? 0;
	const cap = Math.ceil((cf * k * n) / E - 1e-9);
	const load = new Array<number>(E).fill(0);
	const sent = new Array<number>(E).fill(0);
	let dropped = 0;
	let lost = 0;
	const steps = p.map((r) => {
		const { i, g } = topK(r, k);
		const ok = i.map((j) => (sent[j]++, load[j] < cap ? (load[j]++, 1) : 0));
		const miss = k - ok.reduce((a: number, b) => a + b, 0);
		dropped += miss;
		if (miss === k) lost++;
		return { i, g, ok, load: load.slice(), dropped, lost };
	});
	return { cap, steps, load, sent, dropped, lost, slots: k * n };
}

/** The expert of block blk that took spaces most often on held-out stories (stats.py, in the spec's `routing`). */
export function spaceExpert(rt: RT, blk: number) {
	const top = spec(rt).routing?.moe[blk - 1]?.top ?? [];
	let best = -1;
	let share = -1;
	top.forEach((list, e) => {
		const s = list.find((x) => x.char === ' ')?.share ?? 0;
		if (s > share) (best = e, share = s);
	});
	return { e: best, share, top: top[best] ?? [] };
}
