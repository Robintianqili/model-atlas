// node2vec: process 'walks' (kind C), the biased random walks one move at a time: WALKS rounds, a walk from every member
// in turn, LEN − 1 moves each (walks.ts walkEvents: the same generator walksOf() collects for the trainer, on the same
// seed). Each 'move' frame (sub: the move's number in its walk, from 1) carries the move's numbers: the members t, v, x, the weights by kind, the chances and the draw.
// fw = 1 marks the moves of the lesson's walk (walks.ts exampleWalk: the member in focus's first walk), ex = 1 the
// lesson's example move: that walk's first move with a previous member and the most kinds of candidates.
// Python reference: pipeline/node2vec/reference.py (ak.Rng, bit-identical draws; frames compared one by one).
import type { ProcessDef, ProcessRun } from '../../atlas/types/index.ts';
import { exampleWalk, inputOf, moveOf, N, walkEvents, walksOf, type Ev, type WalkRaw } from './walks.ts';

const L = (en: string, zh: string) => ({ en, zh });
type NoOpt = Record<string, never>;

/** a move's numbers: what each kind of candidate weighs and gets (kinds 0 back, 1 also t's friend, 2 farther) */
export function moveMetrics(raw: unknown, e: Ev) {
	const g = inputOf(raw), m = moveOf(walksOf(raw).nb, e.t, e.v, g.p, g.q);
	const by = (k: number) => {
		let n = 0, p = 0;
		m.kind.forEach((x, i) => x === k && (n++, (p += m.prob[i])));
		return [n, p];
	};
	const [n0, p0] = by(0), [n1, p1] = by(1), [n2, p2] = by(2);
	return { w: e.w, s: e.s, k: e.k, t: e.t, v: e.v, x: e.x, u: e.u, i: e.i, deg: m.x.length, total: m.total, ux: e.u * m.total, a: m.a[e.i], px: m.prob[e.i], kx: m.kind[e.i], n0, p0, n1, p1, n2, p2 };
}

export const walks: ProcessDef<WalkRaw, NoOpt, Record<string, never>> = {
	id: 'walks',
	kind: 'algorithm',
	title: L('Biased random walks', '有偏随机游走'),
	phases: {
		move: {
			label: L('One move of a walk', '游走的一步'),
			kind: 'sample',
			narration: { what: L('Walk {frame.w|int}, move {frame.k|int}: from member {frame.v|int} to member {frame.x|int}, which had a {frame.px|pct} chance.', '第 {frame.w|int} 条游走的第 {frame.k|int} 步：从成员 {frame.v|int} 走到成员 {frame.x|int}，这一步的概率是 {frame.px|pct}。') }
		}
	},
	schema: {},
	defaults: {},
	*run(input, ctx): ProcessRun<Record<string, never>, NoOpt> {
		const evs = [...walkEvents(input, ctx.rng)], lw = exampleWalk(input), g = inputOf(input);
		const kinds = (e: Ev) => new Set(moveOf(walksOf(input).nb, e.t, e.v, g.p, g.q).kind).size;
		const best = evs.filter((e) => e.w === lw && e.t >= 0).reduce<Ev | null>((b, e) => (!b || kinds(e) > kinds(b) ? e : b), null);
		let t = 0;
		for (const e of evs) yield { phase: 'move', step: t++, sub: e.k + 1, state: {}, metrics: { ...moveMetrics(input, e), fw: e.w === lw ? 1 : 0, ex: e === best ? 1 : 0 } };
	}
};

type F = { metrics?: Record<string, number> };
/** cursors: the lesson's example move, and the first and last moves of the lesson's walk */
export const EXAMPLE = { phase: 'move', where: (f: F) => f.metrics?.ex === 1 };
export const WALK_FIRST = { phase: 'move', where: (f: F) => f.metrics?.fw === 1 };
export const WALK_LAST = { ...WALK_FIRST, last: true };
