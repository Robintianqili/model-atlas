// classic-ml: the pieces every file shares (pure TS). The learner's input is the kit's points2d raw value plus four
// settings the stage edits (the new point q, kNN's k, the tree's depth, k-means' k); dataOf() turns it into plain numbers,
// the input of every process (and, as JSON, of the Python reference). The grid covers the plane [−B, B]² in R × R cells,
// row 0 at the top, read at the cell centres.
import type { I18n } from '../../atlas/types/index.ts';
import { pointsOf, presetRaw, type PointsRaw } from '../../atlas/adapters/points2d/adapter.ts';

export interface MlRaw extends PointsRaw { q?: { x: number; y: number }; k?: number; depth?: number; kc?: number }
export interface MlData { xy: number[]; label: number[]; classes: number; q: number[]; k: number; depth: number; kc: number }

export const B = 1.25;
export const R = 40;
/** the k values of the decision-map replay, and the deepest tree the depth sweep grows */
export const KS = [1, 3, 5, 9, 15, 25];
export const DMAX = 8;
export const Q0 = { x: -0.2, y: 0.15 };
/** the start: two moons with noise 0.3 (seed 1), shrunk by SHRINK about the origin. Noise 0.3 throws 19 of the preset's
 *  200 points outside the plane [−B, B]² that the stage and the input widget draw; shrunk, the farthest lies at 1.18.
 *  A uniform shrink keeps every neighbour order, split and assignment: only lengths change. Painted points, so the
 *  input widget shows no preset chip as chosen (its "Two moons" chip is the noise-0.1 set). */
const SHRINK = 0.75;
const MOONS = pointsOf(presetRaw('moons', { noise: 0.3 }));
export const START: MlRaw = { points: Array.from({ length: MOONS.n }, (_, i) => ({ x: SHRINK * MOONS.xy[2 * i], y: SHRINK * MOONS.xy[2 * i + 1], c: MOONS.label[i] })), q: Q0, k: 5, depth: DMAX, kc: 3 };
/** the limits step's fixed data: a disc (class 1) inside a ring (class 0), clustered with k = 2 */
export const RINGS: MlRaw = { ...presetRaw('circles'), kc: 2 };
export const CENTROIDS = ['A', 'B', 'C', 'D', 'E', 'F'];

const clampInt = (v: unknown, lo: number, hi: number, d: number) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : d);
const clampB = (v: number) => Math.min(B, Math.max(-B, v));

export function rawOf(raw: unknown): MlRaw {
	return (raw && typeof raw === 'object' ? raw : {}) as MlRaw;
}
export function dataOf(raw: unknown): MlData {
	const r = rawOf(raw), p = pointsOf(r);
	const ok = r.q && Number.isFinite(r.q.x) && Number.isFinite(r.q.y);
	return { xy: Array.from(p.xy), label: Array.from(p.label), classes: Math.max(1, p.classes), q: ok ? [clampB(r.q!.x), clampB(r.q!.y)] : [Q0.x, Q0.y], k: clampInt(r.k, 1, 25, 5), depth: clampInt(r.depth, 1, DMAX, DMAX), kc: clampInt(r.kc, 1, 6, 3) };
}
/** the centre of grid column c and of grid row r */
export const cellX = (c: number) => -B + ((c + 0.5) * 2 * B) / R;
export const cellY = (r: number) => B - ((r + 0.5) * 2 * B) / R;

/** Gini impurity of class counts over m points: 1 − Σ p² */
export function gini(counts: number[], m: number): number {
	let s = 0;
	for (const c of counts) {
		const p = c / m;
		s += p * p;
	}
	return 1 - s;
}
/** the class with the most votes; ties go to the smaller class number */
export function majority(counts: number[]): number {
	let best = 0;
	for (let c = 1; c < counts.length; c++) if (counts[c] > counts[best]) best = c;
	return best;
}
/** separate patches of one class in a grid (cells joined through their four sides) */
export function patches(grid: ArrayLike<number>, n = R): number {
	const seen = new Uint8Array(n * n);
	let count = 0;
	for (let s = 0; s < n * n; s++) {
		if (seen[s]) continue;
		count++;
		const stack = [s];
		seen[s] = 1;
		while (stack.length) {
			const c = stack.pop()!, r = Math.floor(c / n), x = c % n;
			for (const [rr, xx] of [[r - 1, x], [r + 1, x], [r, x - 1], [r, x + 1]]) {
				const t = rr * n + xx;
				if (rr >= 0 && rr < n && xx >= 0 && xx < n && !seen[t] && grid[t] === grid[c]) (seen[t] = 1), stack.push(t);
			}
		}
	}
	return count;
}
export const I = (en: string, zh: string): I18n => ({ en, zh });
/** "class 1" / "第 1 类" (the input widget's names) */
export const className = (c: number) => I(`class ${c}`, `第 ${c} 类`);
