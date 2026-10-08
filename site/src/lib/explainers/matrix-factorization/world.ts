// matrix-factorization: the synthetic world (TS is the source of truth; no download). 20 made-up movies of four kinds
// and 24 made-up people, each with a hidden taste of two numbers. A person's rating of a movie is their taste dotted with
// the movie's hidden traits, plus noise, rounded to a whole number from −2 (hated it) to +2 (loved it). Each person rated
// 8 to 12 movies; the learner is one more row (YOU) with their own ratings. Training never sees the hidden numbers or the
// kinds: the kinds only color the map, so the learner can check what training found.
import type { I18n, Tensor } from '../../atlas/types/index.ts';
import { createRng, sampleIndices } from '../../atlas/core/rng.ts';

const I = (en: string, zh: string): I18n => ({ en, zh });
export const KINDS: I18n[] = [I('comedy', '喜剧'), I('fantasy', '奇幻'), I('sci-fi', '科幻'), I('crime', '犯罪')];
/** each kind's color, a site categorical color (--cat-k) far from the other three */
export const kindColor = (kind: number) => `var(--cat-${[0, 1, 2, 6][kind]})`;
/** Hidden traits: x = funny (+) or serious (−), y = fantastic (+) or everyday (−); kind k sits around 45° + 90°·k − 90°. */
const CENTER = [-45, 45, 135, 225];
const TITLES: I18n[][] = [
	[I('Office Pranks', '办公室整蛊'), I('Wedding Mix-Up', '婚礼大乌龙'), I('Grandpa Goes Viral', '爷爷成网红'), I('Food Truck Summer', '餐车之夏'), I('Dog Show Chaos', '狗狗选秀大乱斗')],
	[I('Dragon School', '龙族学院'), I('The Talking Cat', '会说话的猫'), I('Candy Kingdom', '糖果王国'), I('Wizard on Holiday', '度假的巫师'), I('Cloud Pirates', '云端海盗')],
	[I('Mars Signal', '火星信号'), I('Robot Uprising', '机器人起义'), I('The Last Ship', '最后的飞船'), I('Time Loop', '时间循环'), I('Deep Space Silence', '深空寂静')],
	[I('Midnight Detective', '午夜侦探'), I('The Heist', '大劫案'), I('Cold Case', '陈年悬案'), I('Courtroom', '法庭风云'), I('The Witness', '目击者')]
];
export type Movie = { title: I18n; kind: number; trait: [number, number] };
/** Movie j is kind j % 4 (the columns mix the kinds). */
export const MOVIES: Movie[] = Array.from({ length: 20 }, (_, j) => {
	const kind = j % 4, k = Math.floor(j / 4);
	const a = ((CENTER[kind] + (k - 2) * 14) * Math.PI) / 180, rad = 0.8 + 0.1 * ((k * 3) % 5);
	return { title: TITLES[kind][k], kind, trait: [rad * Math.cos(a), rad * Math.sin(a)] };
});
export const N_ITEMS = MOVIES.length;
export const N_OTHERS = 24;
/** the learner's row (counting from 0): the last one */
export const YOU = N_OTHERS;
export const N_USERS = N_OTHERS + 1;
export const SCALE = [-2, -1, 0, 1, 2];
export const SCALE_NAMES: I18n[] = [I('hated it', '讨厌'), I('did not like it', '不喜欢'), I('so-so', '一般'), I('liked it', '喜欢'), I('loved it', '超爱')];
const clampRating = (v: number) => Math.max(-2, Math.min(2, Math.round(v))) + 0;

export type Rating = { u: number; i: number; r: number };
/** The other people's ratings, the same for everyone (seed 7): hidden taste t, rating = round(1.6 t·trait + 0.5 noise). */
export const OTHERS: Rating[] = (() => {
	const rng = createRng(7);
	const out: Rating[] = [];
	for (let u = 0; u < N_OTHERS; u++) {
		const a = rng.next() * 2 * Math.PI, m = 0.7 + 0.6 * rng.next();
		const items = sampleIndices(rng, N_ITEMS, 8 + rng.int(5)).sort((x, y) => x - y);
		for (const i of items) out.push({ u, i, r: clampRating(1.6 * m * (Math.cos(a) * MOVIES[i].trait[0] + Math.sin(a) * MOVIES[i].trait[1]) + 0.5 * rng.normal()) });
	}
	return out;
})();

/** The learner's raw input: one entry per movie, a whole number from −2 to 2 or null (not rated). */
export type RatingsRaw = { ratings: (number | null)[] };
export function toRaw(raw: unknown): RatingsRaw {
	const a = raw && typeof raw === 'object' && Array.isArray((raw as RatingsRaw).ratings) ? (raw as RatingsRaw).ratings : [];
	return { ratings: MOVIES.map((_, j) => (typeof a[j] === 'number' && Number.isFinite(a[j]) ? clampRating(a[j] as number) : null)) };
}
const profile = (pairs: [number, number][]): RatingsRaw => {
	const ratings: (number | null)[] = MOVIES.map(() => null);
	for (const [j, r] of pairs) ratings[j] = r;
	return { ratings };
};
/** Profiles to start from; the first is the page's start input. */
export const PRESETS: { id: string; label: I18n; raw: RatingsRaw }[] = [
	{ id: 'sample', label: I('Laughs and magic', '欢笑与魔法'), raw: profile([[0, 2], [1, 2], [9, 1], [2, -1], [3, -2]]) },
	{ id: 'dark', label: I('Crime and suspense', '犯罪与悬疑'), raw: profile([[3, 2], [7, 2], [11, 1], [4, -2], [5, -1], [2, 1]]) },
	{ id: 'space', label: I('Space and magic', '太空与魔法'), raw: profile([[2, 2], [6, 2], [5, 1], [4, -1], [7, -1], [13, 1]]) },
	{ id: 'none', label: I('Nothing rated', '一部也没评'), raw: profile([]) }
];

export type Known = { u: Int32Array; i: Int32Array; r: Float64Array; n: number; yours: number };
let memo: { raw: unknown; cold: boolean; k: Known } | null = null;
/** Every known rating: the others' (in order), then yours (none when cold: a new user who rated nothing). */
export function known(raw: unknown, cold = false): Known {
	if (memo && memo.raw === raw && memo.cold === cold) return memo.k;
	const mine = cold ? [] : toRaw(raw).ratings.flatMap((r, i) => (r === null ? [] : [{ u: YOU, i, r }]));
	const all = [...OTHERS, ...mine];
	const k = { u: Int32Array.from(all, (x) => x.u), i: Int32Array.from(all, (x) => x.i), r: Float64Array.from(all, (x) => x.r), n: all.length, yours: mine.length };
	memo = { raw, cold, k };
	return k;
}

/** Factor tables U [users, 2] and V [movies, 2], uniform in ±init, drawn U first, row by row (the trainer and the
 *  'nudge' process both start here; pipeline/matrix-factorization/reference.py draws the same numbers). */
export function initFactors(rng: { next(): number }, init: number): { U: Float64Array; V: Float64Array } {
	const draw = (n: number) => Float64Array.from({ length: n }, () => (2 * rng.next() - 1) * init);
	const U = draw(N_USERS * 2);
	return { U, V: draw(N_ITEMS * 2) };
}

export type Factors = { U: ArrayLike<number>; V: ArrayLike<number> };
export const factorsOf = (params: Record<string, Tensor> | undefined): Factors | null =>
	params?.['user.weight'] && params['item.weight'] ? { U: params['user.weight'].data, V: params['item.weight'].data } : null;
/** predicted rating of person u for movie i: the dot product of their two numbers */
export const predict = (f: Factors, u: number, i: number) => f.U[2 * u] * f.V[2 * i] + f.U[2 * u + 1] * f.V[2 * i + 1];
export const vecU = (f: Factors, u: number): [number, number] => [f.U[2 * u], f.U[2 * u + 1]];
export const vecV = (f: Factors, i: number): [number, number] => [f.V[2 * i], f.V[2 * i + 1]];
/** mean squared error over the known ratings (all, or only the learner's) */
export function mse(f: Factors, k: Known, onlyYou = false): number {
	let s = 0, n = 0;
	for (let j = 0; j < k.n; j++) {
		if (onlyYou && k.u[j] !== YOU) continue;
		const e = predict(f, k.u[j], k.i[j]) - k.r[j];
		s += e * e;
		n++;
	}
	return n ? s / n : NaN;
}
/** the movies the learner did not rate, best predicted first (ties: the smaller index) */
export function ranked(f: Factors, raw: unknown, u = YOU): { i: number; p: number }[] {
	const mine = toRaw(raw).ratings;
	return MOVIES.map((_, i) => ({ i, p: predict(f, u, i) }))
		.filter((x) => u !== YOU || mine[x.i] === null)
		.sort((a, b) => b.p - a.p || a.i - b.i);
}
