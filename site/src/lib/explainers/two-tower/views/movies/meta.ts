// two-tower view 'movies' (kind trace): the catalog and what retrieval does with it. props.mode:
//   catalog    the 36 movies by genre, Orbit Nine outlined, and the 8 numbers the item tower reads of it
//   pick       the learner's choices (genre switches and movie toggles; they change the input)
//   scores     every movie's score for you, best first (hidden while a predict question is open)
//   shortlist  the movies you did not pick, best first, the best K kept (retrieval)
//   rank       the shortlist reordered by the ranker (score and likes in the logs)
//   cost       multiplications per request: two towers against a network that reads you and a movie together
//   twins      Orbit Nine and Nova Station: the same features, the same vector, very different likes
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { MOVIES, TWINS } from '../../world.ts';
import { K, T, costs, ranked, reranked, shortlist } from '../../recs.ts';
import { demoStage } from '../../demo.ts';

const I = (en: string, zh: string): I18n => ({ en, zh });
const name = (i: number) => I(MOVIES[i].title.en, `《${MOVIES[i].title.zh}》`);
const list = (xs: { i: number; s: number }[]) => I(xs.map((x) => `${name(x.i).en} ${fmt3(x.s)}`).join(', '), xs.map((x) => `${name(x.i).zh} ${fmt3(x.s)}`).join('，'));

export function describeMovies(p: Omit<StageProps, 'ctx'>): I18n {
	const mode = String(p.props.mode ?? 'catalog');
	if (mode === 'catalog' || mode === 'pick') return I(`${MOVIES.length} movies in six genres, each with its year and length${mode === 'pick' ? '; press a card to pick it' : ''}.`, `${MOVIES.length} 部电影，分六个类型，各有年份和片长${mode === 'pick' ? '；点卡片可以选中' : ''}。`);
	if (!T(p.rt, 'scores')) return I('Scores: nothing to show yet.', '分数：还没有可显示的内容。');
	if (mode === 'scores') {
		const s = T(p.rt, 'scores')!.data, all = MOVIES.map((_, i) => ({ i, s: s[i] })).sort((a, b) => b.s - a.s || a.i - b.i);
		return I(`Scores of all ${MOVIES.length} movies, best first: ${list(all.slice(0, 5)).en}, …`, `全部 ${MOVIES.length} 部电影的分数，从高到低：${list(all.slice(0, 5)).zh}……`);
	}
	if (mode === 'shortlist') return I(`The best ${K} of the ${ranked(p.rt).length} movies you did not pick: ${list(shortlist(p.rt)).en}.`, `你没选过的 ${ranked(p.rt).length} 部电影里最好的 ${K} 部：${list(shortlist(p.rt)).zh}。`);
	if (mode === 'rank') {
		const r = reranked(p.rt);
		return I(`After the ranker: ${r.map((x) => `${name(x.i).en} ${fmtPct(x.p)}`).join(', ')}.`, `排序之后：${r.map((x) => `${name(x.i).zh} ${fmtPct(x.p)}`).join('，')}。`);
	}
	if (mode === 'cost') {
		const c = costs(p.rt);
		return I(`Per request: two towers ${c.perRequest} multiplications; the item tower for every movie ${c.offline}, done once in advance; a network that reads you and each movie together ${c.joint}.`, `每次请求：双塔 ${c.perRequest} 次乘法；物品塔为每部电影算一遍要 ${c.offline} 次，提前做一次就行；把你和每部电影放在一起读的网络要 ${c.joint} 次。`);
	}
	const s = T(p.rt, 'scores')!.data;
	return I(`${name(TWINS[0]).en} and ${name(TWINS[1]).en}: the same features, the same vector, the same score ${fmt3(s[TWINS[0]])}.`, `${name(TWINS[0]).zh}和${name(TWINS[1]).zh}：特征相同，向量相同，分数也同为 ${fmt3(s[TWINS[0]])}。`);
}

const meta = {
	key: 'two-tower/movies',
	title: { en: 'The movies, scored and shortlisted', zh: '电影：打分与候选' },
	kinds: ['trace'],
	describe: describeMovies,
	demo: () => demoStage({ mode: 'shortlist' }),
	minHeight: 360
} satisfies ViewMeta;

export default meta;
