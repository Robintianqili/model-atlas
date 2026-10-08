// two-tower: the synthetic world (TS is the source of truth; no download). 36 made-up movies of six genres, each
// described by three features (genre, year, length), and made-up viewers. A viewer has a hidden taste: one or two
// favourite genres, a liking for old or new movies and for short or long ones. Each movie also has a hidden appeal that
// no feature shows (the twins Orbit Nine and Nova Station share every feature, yet one is far better liked). The logs
// (process 'logs', process.ts) hold, per viewer, the genres they said they like, the movies they watched and liked
// (their history) and the other movies they liked (the training targets). pipeline/two-tower/train.py reads the
// logs' bytes from the TS tape; the towers never see the hidden numbers.
import type { I18n } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

const I = (en: string, zh: string): I18n => ({ en, zh });
export const GENRES: I18n[] = [I('action', '动作'), I('comedy', '喜剧'), I('drama', '剧情'), I('horror', '恐怖'), I('sci-fi', '科幻'), I('romance', '爱情')];
export const G = GENRES.length;
export const genreColor = (g: number) => `var(--cat-${g})`;
export type Movie = { title: I18n; genre: number; year: number; length: number };
const M = (en: string, zh: string, genre: number, year: number, length: number): Movie => ({ title: I(en, zh), genre, year, length });
export const MOVIES: Movie[] = [
	M('Iron Harbor', '钢铁港', 0, 1979, 118), M('Night Convoy', '夜行车队', 0, 1988, 104), M('Rooftop Run', '屋顶狂奔', 0, 1997, 96),
	M('Blackout Protocol', '断电行动', 0, 2009, 131), M('Steel Monsoon', '钢铁季风', 0, 2018, 142), M('Last Courier', '最后的信使', 0, 2023, 99),
	M('Uncle Barney’s Diner', '巴尼叔叔的餐馆', 1, 1974, 92), M('Wrong Wedding', '错嫁婚礼', 1, 1985, 101), M('Office Pranks', '办公室整蛊', 1, 1999, 88),
	M('Road Trip to Nowhere', '没有终点的公路旅行', 1, 2008, 97), M('Grandpa Goes Viral', '爷爷成网红', 1, 2019, 94), M('Dog Show Chaos', '狗狗选秀大乱斗', 1, 2024, 86),
	M('The Quiet Harvest', '静静的收获', 2, 1972, 151), M('Letters from Lisbon', '里斯本来信', 2, 1983, 128), M('The Long Winter', '漫长的冬天', 2, 1994, 163),
	M('Courtroom Seven', '第七法庭', 2, 2004, 137), M('Small Town Mayor', '小镇镇长', 2, 2015, 119), M('The Last Lighthouse', '最后的灯塔', 2, 2022, 146),
	M('The Basement Door', '地下室的门', 3, 1976, 98), M('Fog Hollow', '雾谷', 3, 1982, 91), M('Midnight Static', '午夜杂音', 3, 1996, 102),
	M('The Doll Shop', '玩偶店', 3, 2006, 94), M('Do Not Answer', '别接电话', 3, 2017, 89), M('Hollow Signal', '空洞信号', 3, 2023, 108),
	M('Mars Signal', '火星信号', 4, 1978, 124), M('Robot Uprising', '机器人起义', 4, 1991, 117), M('Time Loop', '时间循环', 4, 2002, 109),
	M('Deep Space Silence', '深空寂静', 4, 2014, 158), M('Orbit Nine', '九号轨道', 4, 2021, 121), M('Nova Station', '新星空间站', 4, 2021, 121),
	M('Rain over Paris', '巴黎的雨', 5, 1973, 107), M('Harbor Lights', '港口灯火', 5, 1987, 99), M('Second Chances', '第二次机会', 5, 1998, 113),
	M('The Planner’s Diary', '婚礼策划师日记', 5, 2007, 104), M('Notes on a Napkin', '餐巾上的字条', 5, 2016, 96), M('Stay a While', '再待一会儿', 5, 2024, 131)
];
export const N_ITEMS = MOVIES.length;
/** the twins of the limits step: the same genre, year and length, very different hidden appeal */
export const TWINS: [number, number] = [28, 29];
/** item features: the genre as 6 switches (one-hot), then the year and the length scaled to about −1 … 1 */
export const YEAR0 = 1998, YEAR_SPAN = 26, LEN0 = 125, LEN_SPAN = 40;
export const N_ITEM_F = G + 2;
export const N_USER_F = 2 * G + 3;
export const yearF = (y: number) => (y - YEAR0) / YEAR_SPAN;
export const lenF = (l: number) => (l - LEN0) / LEN_SPAN;
export function itemFeatures(m: { genre: number; year: number; length: number }): Float32Array {
	const f = new Float32Array(N_ITEM_F);
	f[m.genre] = 1;
	f[G] = yearF(m.year);
	f[G + 1] = lenF(m.length);
	return f;
}
/** user features: the genres you like (6 switches), the genre shares of the movies you picked (6), their mean year and
 *  mean length (scaled as above) and a 1 when you picked any (else the last 9 numbers are 0) */
export function userFeatures(genres: number[], picks: number[]): Float32Array {
	const f = new Float64Array(N_USER_F);
	for (const g of genres) f[g] = 1;
	if (picks.length) {
		for (const i of picks) {
			f[G + MOVIES[i].genre] += 1;
			f[2 * G] += yearF(MOVIES[i].year);
			f[2 * G + 1] += lenF(MOVIES[i].length);
		}
		for (let k = G; k < 2 * G + 2; k++) f[k] /= picks.length;
		f[2 * G + 2] = 1;
	}
	return Float32Array.from(f);
}
export const ITEM_F_NAMES: I18n[] = [...GENRES, I('year', '年份'), I('length', '片长')];
export const USER_F_NAMES: I18n[] = [...GENRES.map((g) => I(`likes ${g.en}`, `喜欢${g.zh}`)), ...GENRES.map((g) => I(`picked ${g.en}`, `所选${g.zh}`)),
	I('picks: year', '所选：年份'), I('picks: length', '所选：片长'), I('picked any', '选过电影')];

/** The learner's raw input: liked genres, liked movies (at most MAX_PICKS) and a new movie of their own. */
export type NewMovie = { genre: number; year: number; length: number };
export type Prefs = { genres: number[]; picks: number[]; item: NewMovie };
export const MAX_PICKS = 12;
export const NEW_RANGE = { year: [1960, 2030], length: [70, 200] } as const;
export const NEW_DEFAULT: NewMovie = { genre: 4, year: 2025, length: 112 };
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(v)));
const ids = (a: unknown, n: number) => (Array.isArray(a) ? [...new Set(a.filter((x): x is number => Number.isInteger(x) && x >= 0 && x < n))] : []);
export function toPrefs(raw: unknown): Prefs {
	const r = (raw && typeof raw === 'object' ? raw : {}) as Partial<Prefs>;
	const it = (r.item && typeof r.item === 'object' ? r.item : NEW_DEFAULT) as NewMovie;
	return {
		genres: ids(r.genres, G).sort((a, b) => a - b),
		picks: ids(r.picks, N_ITEMS).slice(0, MAX_PICKS),
		item: {
			genre: Number.isInteger(it.genre) ? clamp(it.genre, 0, G - 1) : NEW_DEFAULT.genre,
			year: clamp(Number(it.year) || NEW_DEFAULT.year, ...NEW_RANGE.year),
			length: clamp(Number(it.length) || NEW_DEFAULT.length, ...NEW_RANGE.length)
		}
	};
}
const P = (genres: number[], picks: number[], item = NEW_DEFAULT): Prefs => ({ genres, picks, item });
/** start profiles; the first is the page's start input */
export const PRESETS: { id: string; label: I18n; raw: Prefs }[] = [
	{ id: 'space', label: I('Space and speed', '太空与速度'), raw: P([0, 4], [25, 26, 4]) },
	{ id: 'laughs', label: I('New, short comedies', '新的短喜剧'), raw: P([1], [10, 11, 9]) },
	{ id: 'classic', label: I('Old dramas and romance', '老派剧情与爱情'), raw: P([2, 5], [12, 13, 30], { genre: 2, year: 1975, length: 140 }) },
	{ id: 'scary', label: I('Horror, from history only', '只看历史：恐怖片'), raw: P([], [19, 20, 21, 22]) },
	{ id: 'none', label: I('Nothing chosen', '什么都没选'), raw: P([], []) }
];
/** inputs every lesson step must survive (the adapter's edge fixtures) */
export const EDGES: { id: string; label: I18n; raw: Prefs }[] = [
	{ id: 'genres-only', label: I('every genre liked, no movie picked', '喜欢所有类型，一部电影也没选'), raw: P([0, 1, 2, 3, 4, 5], []) },
	{ id: 'one-pick', label: I('one horror movie picked, nothing else', '只选了一部恐怖片'), raw: P([], [21]) },
	{ id: 'twelve', label: I('twelve movies picked, all genres', '选了十二部，各种类型都有'), raw: P([3], [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33], { genre: 3, year: 1960, length: 200 }) },
	{ id: 'twin', label: I('a new movie identical to Orbit Nine', '一部和《九号轨道》完全相同的新电影'), raw: P([4], [26], { genre: 4, year: 2021, length: 121 }) }
];
/** the reference cases of export.py: the presets and the edges (frame 0 of the logs tape carries their features) */
export const FIXED = [...PRESETS, ...EDGES];

/** Hidden appeal of each movie (seed 11), the twins set apart: no feature shows it. */
export const APPEAL: number[] = (() => {
	const rng = createRng(11);
	const a = MOVIES.map(() => 0.6 * rng.normal());
	a[TWINS[0]] = 1.4;
	a[TWINS[1]] = -1.4;
	return a;
})();

export type LogUser = { genres: number[]; history: number[]; liked: number[] };
/** Viewer u of the logs (rng fork 'u<u>' of seed 2026, so the first viewers are the same whatever the log length):
 *  hidden taste -> the 5 to 9 movies they liked (Gumbel top-k on taste + appeal) -> 0 to 5 of them as the history and
 *  the rest as targets; the genres they say they like are their favourites, sometimes one fewer or one more. */
export function logUser(u: number): LogUser {
	const rng = createRng(2026).fork(`u${u}`);
	const fav1 = rng.int(G);
	const fav2 = rng.next() < 0.5 ? (fav1 + 1 + rng.int(G - 1)) % G : -1;
	const a = GENRES.map(() => -0.8 + 0.5 * rng.normal());
	a[fav1] += 2.5;
	if (fav2 >= 0) a[fav2] += 1.6;
	const era = 2 * rng.next() - 1, len = 2 * rng.next() - 1;
	const keys = MOVIES.map((m, i) => a[m.genre] + era * yearF(m.year) + 0.7 * len * lenF(m.length) + APPEAL[i] - Math.log(-Math.log(1 - rng.next())));
	const k = 5 + rng.int(5);
	const liked = keys.map((v, i) => [v, i]).sort((x, y) => y[0] - x[0] || x[1] - y[1]).slice(0, k).map((x) => x[1]);
	for (let i = liked.length - 1; i > 0; i--) {
		const j = rng.int(i + 1);
		[liked[i], liked[j]] = [liked[j], liked[i]];
	}
	const h = Math.min(k - 1, rng.int(6));
	let genres = rng.next() < 0.12 ? [] : [fav1, ...(fav2 >= 0 && rng.next() < 0.7 ? [fav2] : [])];
	if (genres.length && rng.next() < 0.1) genres.push((fav1 + 1 + rng.int(G - 1)) % G);
	if (!h && !genres.length) genres = [fav1];
	return { genres: [...new Set(genres)].sort((x, y) => x - y), history: liked.slice(0, h), liked: liked.slice(h) };
}
