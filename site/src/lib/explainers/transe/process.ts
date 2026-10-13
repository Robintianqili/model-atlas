// transe: process 'rank' (kind C), link prediction one candidate at a time. Its input is the trained tables of a train
// source (source `after` it) and a question (h, r, ?): one 'score' frame per candidate tail x = 0 … 33 with its distance
// d = ‖h + r − x‖ and the best candidate so far, then one 'result' frame with the three best and the rank of the true
// tail, when the facts know one (other true tails of the question left out). Python reference:
// pipeline/transe/reference.py (the same sums in the same order; frames compared one by one).
import type { ProcessDef, ProcessRun, Tape } from '../../atlas/types/index.ts';
import { NE } from './kg.ts';
import { distFlat } from './space.ts';

const L = (en: string, zh: string) => ({ en, zh });
type NoOpt = Record<string, never>;
/** the trained tables (flat, row-major), their width, the question and its true tails among the facts */
export interface RankIn { E: number[]; R: number[]; dim: number; h: number; r: number; tails: number[] }
/** the process input from a train source's last frame */
export function rankInput(tape: Tape | null | undefined, h: number, r: number, tails: number[]): RankIn | null {
	const p = tape?.length ? (tape.state(tape.length - 1) as { params?: Record<string, { shape: number[]; data: ArrayLike<number> }> }).params : null;
	if (!p) return null;
	return { E: Array.from(p['ent.weight'].data), R: Array.from(p['rel.weight'].data), dim: p['rel.weight'].shape[1], h, r, tails };
}

export const rank: ProcessDef<RankIn | null, NoOpt, Record<string, never>> = {
	id: 'rank',
	kind: 'algorithm',
	title: L('Rank every candidate tail', '给每个候选尾实体排名'),
	phases: {
		score: {
			label: L('Score one candidate', '给一个候选打分'),
			kind: 'observe',
			narration: { what: L('Candidate {frame.x|int}: its distance from h + r is {frame.d}.', '候选 {frame.x|int}：它离 h + r 的距离是 {frame.d}。') }
		},
		result: { label: L('The ranking', '排名'), kind: 'output' }
	},
	schema: {},
	defaults: {},
	*run(input): ProcessRun<Record<string, never>, NoOpt> {
		if (!input) return;
		const { E, R, dim, h, r } = input, ds: number[] = [];
		let best = -1;
		for (let x = 0; x < NE; x++) {
			const d = distFlat(E, R, dim, h, r, x);
			ds.push(d);
			if (best < 0 || d < ds[best]) best = x;
			yield { phase: 'score', step: x, state: {}, metrics: { x, d, best, bestD: ds[best], truth: input.tails.includes(x) ? 1 : 0 } };
		}
		const order = ds.map((d, x) => ({ d, x })).sort((a, b) => a.d - b.d || a.x - b.x);
		const t0 = input.tails.length ? input.tails[0] : -1, others = new Set(input.tails.slice(1));
		// truthRank 0: the facts know no tail for this question
		const truthRank = t0 < 0 ? 0 : order.filter((c) => !others.has(c.x)).findIndex((c) => c.x === t0) + 1;
		yield { phase: 'result', step: NE, state: {}, metrics: { t1: order[0].x, d1: order[0].d, t2: order[1].x, d2: order[1].d, t3: order[2].x, d3: order[2].d, truth: t0, truthRank } };
	}
};
