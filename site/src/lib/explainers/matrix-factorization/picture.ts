// matrix-factorization: what the two views draw, as pure functions (the views, their describe() and the lesson share
// them, so the text and the picture always show the same numbers).
import type { At, I18n, StageProps, TrainState } from '../../atlas/types/index.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { MOVIES, N_ITEMS, N_USERS, YOU, factorsOf, predict, ranked, vecU, vecV, type Factors } from './world.ts';

/** 3 significant digits, as the narration prints them, but a small number as a plain decimal (0.000653, not 6.53e−4) */
export const num = (v: number) => (v !== 0 && Math.abs(v) < 1e-3 && Math.abs(v) >= 1e-6 ? String(Number(v.toPrecision(3))).replace('-', '−') : fmt3(v));
/** The factors at a frame: a train frame's params, or a 'nudge' frame (the pair it changed, on the start numbers). */
export function factorsAt(at: At | undefined): Factors | null {
	const s = at?.state as Partial<TrainState> | undefined;
	return factorsOf(s?.params) ?? null;
}
/** The factors at the last frame of a train source (null until it has a snapshot). */
export function lastFactors(p: Pick<StageProps, 'rt'>, source: string): Factors | null {
	const t = p.rt.sources[source]?.tape;
	return t && t.length ? factorsOf((t.state(t.length - 1) as Partial<TrainState>).params) : null;
}
/** The movie whose direction is closest to the person's (largest cosine), among the unrated movies that point less than
 *  a right angle away (unrated: true), else among all movies (unrated: false). */
export function closest(f: Factors, raw: unknown, u = YOU): { i: number; unrated: boolean } {
	const cos = (i: number) => predict(f, u, i) / (Math.hypot(...vecV(f, i)) || 1);
	const pool = ranked(f, raw, u).map((x) => x.i).filter((i) => cos(i) > 0);
	const best = (pool.length ? pool : MOVIES.map((_, i) => i)).reduce((b, i) => (cos(i) > cos(b) ? i : b));
	return { i: best, unrated: pool.length > 0 };
}
/** For each movie, is its nearest other movie on the map of the same kind? (how many are) */
export function sameKindNeighbours(f: Factors): number {
	const d = (a: number, b: number) => Math.hypot(vecV(f, a)[0] - vecV(f, b)[0], vecV(f, a)[1] - vecV(f, b)[1]);
	return MOVIES.filter((mv, i) => MOVIES[MOVIES.reduce((b, _, j) => (j !== i && (b === i || d(i, j) < d(i, b)) ? j : b), i)].kind === mv.kind).length;
}
/** a × b written with brackets around a negative b: "0.5 × (−0.3)" */
export const times = (a: number, b: number) => `${num(a)} × ${b < 0 ? `(${num(b)})` : num(b)}`;
export const dotText = (u: [number, number], v: [number, number]) => `${times(u[0], v[0])} + ${times(u[1], v[1])} = ${num(u[0] * v[0] + u[1] * v[1])}`;

export function describeMap(p: Omit<StageProps, 'ctx'>): I18n {
	const m = p.at.frame?.metrics;
	if (p.at.frame?.phase && m && 'u1' in m) {
		const after = 'nu1' in m ? { en: ` After the step: (${num(m.nu1)}, ${num(m.nu2)}) and (${num(m.nv1)}, ${num(m.nv2)}), prediction ${num(m.predAfter)}.`, zh: `这一步之后：(${num(m.nu1)}, ${num(m.nu2)}) 和 (${num(m.nv1)}, ${num(m.nv2)})，预测 ${num(m.predAfter)}。` } : { en: '', zh: '' };
		return { en: `One rating, ${m.r}, of ${MOVIES[m.item].title.en}: the person at (${num(m.u1)}, ${num(m.u2)}), the movie at (${num(m.v1)}, ${num(m.v2)}).${after.en}`, zh: `一个评分 ${m.r}，电影《${MOVIES[m.item].title.zh}》：这个人在 (${num(m.u1)}, ${num(m.u2)})，电影在 (${num(m.v1)}, ${num(m.v2)})。${after.zh}` };
	}
	const f = factorsAt(p.at);
	if (!f) return { en: 'The map: nothing to show yet.', zh: '地图：还没有可显示的内容。' };
	const u = vecU(f, YOU), k = sameKindNeighbours(f);
	return {
		en: `A plane with ${N_ITEMS} movies (dots colored by kind), ${N_USERS - 1} other people (gray dots) and an arrow to your two numbers (${num(u[0])}, ${num(u[1])}), after ${p.at.frame?.step ?? 0} training steps. For ${k} of the ${N_ITEMS} movies the nearest movie is of the same kind.`,
		zh: `一个平面，上面有 ${N_ITEMS} 部电影（按类型着色的点）、${N_USERS - 1} 个其他人（灰点），以及指向你的两个数 (${num(u[0])}, ${num(u[1])}) 的箭头；已训练 ${p.at.frame?.step ?? 0} 步。${N_ITEMS} 部电影里有 ${k} 部，离它最近的电影与它同类。`
	};
}

export function describeMatrix(p: Omit<StageProps, 'ctx'>, n: { known: number; yours: number }): I18n {
	const f = p.props.fill ? factorsAt(p.at) : null;
	const top = f ? ranked(f, p.rt.input).slice(0, 3) : [];
	const recs = top.length ? { en: ` Your highest predictions among the movies you did not rate: ${top.map((x) => `${MOVIES[x.i].title.en} ${num(x.p)}`).join(', ')}.`, zh: `在你没评过的电影里，预测最高的是：${top.map((x) => `《${MOVIES[x.i].title.zh}》${num(x.p)}`).join('，')}。` } : { en: '', zh: '' };
	return {
		en: `The rating matrix: ${N_ITEMS} movies (rows) by ${N_USERS} people (columns, you first). ${n.known} cells hold a rating from −2 to +2, ${n.yours} of them yours; ${f ? 'the empty cells are filled with predictions.' : 'the rest are empty.'}${recs.en}`,
		zh: `评分矩阵：${N_ITEMS} 部电影（行）× ${N_USERS} 个人（列，你在最前）。${n.known} 个格子里有 −2 到 +2 的评分，其中 ${n.yours} 个是你的；${f ? '空格里填上了预测值。' : '其余格子是空的。'}${recs.zh}`
	};
}
