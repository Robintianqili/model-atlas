// sasrec: the synthetic shop (TS is the source of truth; no download). 30 made-up products in six categories of five,
// plus one new arrival (the drone) that no session in the logs ever clicked. A session is a short run of clicks: one or
// two interests (categories); inside a category people tend to follow a typical order (a camera, then a lens, then a
// tripod …: each category's five products are listed in that order); sometimes they switch to their other interest and
// pick up where they left off there, sometimes they wander off. Popular products are clicked more often. Only plain
// arithmetic on small numbers is used (no exp or log), so every engine draws the same sessions. The logs (process 'logs',
// process.ts) hold the sessions as product ids; pipeline/sasrec/train.py reads their bytes from the TS tape.
import type { I18n, Rng } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

const I = (en: string, zh: string): I18n => ({ en, zh });
export const CATS: I18n[] = [I('photo', '摄影'), I('camping', '露营'), I('coffee', '咖啡'), I('fitness', '健身'), I('gaming', '游戏'), I('baking', '烘焙')];
export const C = CATS.length, PER = 5;
export const catColor = (c: number) => `var(--cat-${c})`;
export type Item = { name: I18n; cat: number; pop: number };
const P = (en: string, zh: string, cat: number, pop: number): Item => ({ name: I(en, zh), cat, pop });
/** the products, five per category in their typical order; pop = how often people click it (relative, hidden) */
export const ITEMS: Item[] = [
	P('Camera', '相机', 0, 6), P('Lens', '镜头', 0, 4), P('Tripod', '三脚架', 0, 2), P('Memory card', '存储卡', 0, 10), P('Camera bag', '相机包', 0, 1),
	P('Tent', '帐篷', 1, 5), P('Sleeping bag', '睡袋', 1, 4), P('Camp stove', '野营炉', 1, 2), P('Headlamp', '头灯', 1, 7), P('Water filter', '净水器', 1, 1),
	P('Coffee beans', '咖啡豆', 2, 12), P('Grinder', '磨豆机', 2, 4), P('Kettle', '手冲壶', 2, 2), P('Paper filters', '滤纸', 2, 6), P('Mug', '马克杯', 2, 7),
	P('Running shoes', '跑鞋', 3, 8), P('Sports watch', '运动手表', 3, 4), P('Yoga mat', '瑜伽垫', 3, 3), P('Dumbbells', '哑铃', 3, 1), P('Water bottle', '运动水壶', 3, 10),
	P('Game console', '游戏机', 4, 5), P('Controller', '手柄', 4, 8), P('Headset', '耳机', 4, 4), P('Racing game', '赛车游戏', 4, 2), P('Charging dock', '充电座', 4, 1),
	P('Stand mixer', '厨师机', 5, 2), P('Baking tray', '烤盘', 5, 3), P('Cookbook', '食谱', 5, 5), P('Measuring cups', '量杯', 5, 1), P('Oven mitts', '隔热手套', 5, 1),
	P('Drone', '无人机', 0, 0)
];
export const N_ITEMS = ITEMS.length;
/** the new arrival: in the shop, never in the logs */
export const NEW_ITEM = N_ITEMS - 1;
export const MAX_CLICKS = 12;
/** a session of the logs has 3 to 13 clicks: the first 12 predict the 13th, so all 12 positions are trained */
export const MIN_SESSION = 3, MAX_SESSION = MAX_CLICKS + 1;
export const itemsOf = (c: number) => Array.from({ length: PER }, (_, j) => c * PER + j);

/** The learner's raw input: product ids in the order clicked (at most 12, no repeats). */
export type Clicks = { clicks: number[] };
export function toClicks(raw: unknown): Clicks {
	const a = raw && typeof raw === 'object' ? (raw as Partial<Clicks>).clicks : undefined;
	const ids = Array.isArray(a) ? a.filter((x): x is number => Number.isInteger(x) && x >= 0 && x < N_ITEMS) : [];
	return { clicks: [...new Set(ids)].slice(0, MAX_CLICKS) };
}
const S = (...clicks: number[]): Clicks => ({ clicks });
/** start histories; the first is the page's start input */
export const PRESETS: { id: string; label: I18n; raw: Clicks }[] = [
	{ id: 'trip', label: I('A photo trip', '拍照旅行'), raw: S(0, 1, 5, 6) },
	{ id: 'coffee', label: I('Coffee at home', '在家喝咖啡'), raw: S(10, 11, 12) },
	{ id: 'console', label: I('A new console', '新游戏机'), raw: S(20, 21) },
	{ id: 'back', label: I('Back and forth', '来回切换'), raw: S(15, 10, 16, 11, 17) },
	{ id: 'baker', label: I('Baking, then running', '先烘焙，再跑步'), raw: S(25, 26, 27, 15) }
];
/** inputs every lesson step must survive (the adapter's edge fixtures) */
export const EDGES: { id: string; label: I18n; raw: Clicks }[] = [
	{ id: 'empty', label: I('no clicks yet', '还没有点击'), raw: S() },
	{ id: 'one', label: I('a single click', '只点了一次'), raw: S(8) },
	{ id: 'twelve', label: I('twelve clicks, all over the shop', '十二次点击，逛遍全店'), raw: S(0, 5, 10, 15, 20, 25, 1, 6, 11, 16, 21, 26) },
	{ id: 'drone', label: I('the new drone, then a camera', '先点新到的无人机，再点相机'), raw: S(30, 0) }
];
/** export.py's reference cases: the presets and the non-empty edges (frame 0 of the logs tape carries their ids) */
export const FIXED = [...PRESETS, ...EDGES.filter((e) => e.raw.clicks.length)];

const CAT_W = [1, 0.8, 1.4, 1.1, 0.9, 0.6];
/** the usual second interest of each category */
export const PARTNER = [1, 0, 5, 4, 3, 2];
/** the first click in a category favours its first products (the typical order) */
const START = [4, 2, 1, 0.6, 0.6];
function draw(rng: Rng, w: number[]): number {
	let s = 0;
	for (const x of w) s += x;
	let u = rng.next() * s, last = -1;
	for (let i = 0; i < w.length; i++) {
		if (w[i] <= 0) continue;
		last = i;
		u -= w[i];
		if (u < 0) return i;
	}
	return last;
}
/** Session s of the logs (rng fork 's<s>' of seed 2027, so a session depends only on s): its product ids in click order. */
export function session(s: number): number[] {
	const rng = createRng(2027).fork(`s${s}`);
	const c1 = draw(rng, CAT_W), r2 = rng.next();
	const c2 = r2 < 0.3 ? PARTNER[c1] : r2 < 0.5 ? (c1 + 1 + rng.int(C - 1)) % C : -1;
	const L = MIN_SESSION + rng.int(MAX_SESSION - MIN_SESSION + 1);
	const out: number[] = [], last = CATS.map(() => -1);
	const free = (c: number) => itemsOf(c).filter((i) => !out.includes(i));
	let cur = c1;
	for (let t = 0; t < L; t++) {
		if (t > 0) {
			const r = rng.next();
			if (c2 >= 0 && r < 0.3) cur = cur === c1 ? c2 : c1;
			else if (r >= 0.92) cur = draw(rng, CAT_W);
		}
		if (!free(cur).length) cur = draw(rng, CAT_W.map((w, c) => (free(c).length ? w : 0)));
		const j0 = last[cur];
		const w = itemsOf(cur).map((i, j) => (out.includes(i) ? 0 : ITEMS[i].pop * (j0 < 0 ? START[j] : j === j0 + 1 ? 6 : j > j0 ? 1.5 : 0.6)));
		const j = draw(rng, w);
		out.push(cur * PER + j);
		last[cur] = j;
	}
	return out;
}
/** the logs' split: sessions [0, 8000) train, [8000, 9000) validate, [9000, 10000) test */
export const SPLIT = { train: 8000, val: 9000, all: 10000 };
