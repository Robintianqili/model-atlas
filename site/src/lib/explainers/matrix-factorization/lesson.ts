// matrix-factorization: the guided lesson, one idea per step (the brief's nine). Every number is a {placeholder} bound to a
// live frame of 'train', 'cold' or 'nudge', or computed in vars() from your ratings; the sentences that are not live are
// claims.test.ts. A predict step's narration never gives its answer away: the stage shows it after the answer.
import type { At, ExplainerRuntime, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { formatValue } from '../../atlas/lesson/narrate.ts';
import { MOVIES, N_ITEMS, N_OTHERS, N_USERS, OTHERS, PRESETS, YOU, known, ranked, toRaw, vecU, vecV } from './world.ts';
import { closest, factorsAt, num, sameKindNeighbours } from './picture.ts';
import { fit } from './trainer.ts';

export { sameKindNeighbours };
/** how many movies the task asks the learner to rate */
export const RATE_GOAL = 10;
const I = (en: string, zh: string): I18n => ({ en, zh });
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
const title = (i: number): I18n => MOVIES[i] ? { en: MOVIES[i].title.en, zh: `《${MOVIES[i].title.zh}》` } : I('…', '…');
const rated = (rt: ExplainerRuntime) => toRaw(rt.input).ratings.filter((x) => x !== null).length;
/** your error at the end is below half of the start: your two numbers fit your ratings (else they pull different ways) */
export const fitsYou = (start: number, end: number) => end < 0.5 * start;
const fm = (at: At) => factorsAt(at)!;
/** a number inside a sentence built in vars(), printed as a placeholder prints it */
const fmtN = (v: number | undefined) => (typeof v === 'number' ? (formatValue(v, undefined, 'en') ?? '…') : '…');
/** a setting as written in the code (0.05, not 0.0500) */
const setting = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? String(v) : NaN);

const lesson: LessonDef = {
	title: I('Matrix factorization: how a recommender guesses your ratings', '矩阵分解：推荐系统怎样猜出你的评分'),
	summary: I(
		'A table of ratings full of holes, your own ratings, two numbers for every person and movie, a prediction as a dot product, one training step with real numbers, the map organising itself, your recommendations, and the cold-start problem.',
		'满是空格的评分表、你自己的评分、每个人和每部电影的两个数、作为点积的预测、带真实数字的一步训练、自己变得有条理的地图、给你的推荐，以及冷启动问题。'
	),
	steps: [
		{
			id: 'matrix',
			source: 'train',
			at: 'first',
			view: 'matrix-factorization/matrix',
			provenance: 'live',
			title: I('A table full of holes', '一张满是空格的表'),
			narration: {
				en: 'The columns are you and {others|int} made-up people, the rows are {movies|int} movies, and a filled cell is a rating from −2 (“hated it”) to +2 (“loved it”). Only {known|int} of the {cells|int} cells are filled ({share|pct}): each of the others rated just {lo|int} to {hi|int} of the movies. A recommender guesses the empty cells, then suggests the movies with the best guesses.',
				zh: '列是你和 {others|int} 个虚构的人，行是 {movies|int} 部电影；有内容的格子是一个评分，从 −2（“讨厌”）到 +2（“超爱”）。{cells|int} 个格子里只有 {known|int} 个有评分（{share|pct}）：其他人每人只评了其中 {lo|int} 到 {hi|int} 部。推荐系统先猜出空格里的分数，再把猜得最高的电影推荐出来。',
				vars: (rt) => {
					const k = known(rt.input), per = Array.from({ length: N_OTHERS }, (_, u) => OTHERS.filter((x) => x.u === u).length);
					return { others: N_OTHERS, movies: N_ITEMS, known: k.n, cells: N_USERS * N_ITEMS, share: k.n / (N_USERS * N_ITEMS), lo: Math.min(...per), hi: Math.max(...per) };
				}
			},
			look: I('The first, wider column is you. The dot before a title shows the movie’s kind, for you only: training never sees it.', '最前面那一列（更宽的）是你。片名前的圆点表示电影的类型，只是给你看的：训练从来看不到它。'),
			terms: ['rating-matrix']
		},
		{
			id: 'rate',
			source: 'train',
			at: 'first',
			view: { key: 'matrix-factorization/matrix', props: () => ({ edit: true }) },
			provenance: 'live',
			title: I('Rate movies yourself', '亲手打分'),
			narration: {
				en: 'Make the wide column yours: judging by the titles and the kind dots, rate the movies you would or would not enjoy. So far {rated}. Every rating is one more filled cell for training to learn from, and the rest of the lesson runs on your ratings.',
				zh: '把最宽的那一列变成你自己的：看片名和类型圆点，给你觉得会喜欢或不喜欢的电影打分。目前{rated}。你打的每一个分，都给训练多一个可以学习的格子；后面的课程都用你的评分来算。',
				vars: (rt) => ({ rated: plural(rated(rt), { one: 'you have rated {n} movie', many: 'you have rated {n} movies', zh: '你评了 {n} 部' }) })
			},
			look: I('Your column changes as you press the buttons under the table.', '按表格下面的按钮时，你那一列跟着变。'),
			task: {
				prompt: I('Rate at least ten movies with the buttons under the table (press a chosen button again to clear it).', '用表格下面的按钮至少给十部电影打分（再按一次已选的按钮可以清除）。'),
				done: (rt) => rated(rt) >= RATE_GOAL,
				hint: I('Each movie has five buttons, from −2 to +2. The side panel can also load a whole start profile.', '每部电影有五个按钮，从 −2 到 +2。侧栏里也可以一次换上一整份起始口味。'),
				waiting: I('Not yet: rate at least ten movies.', '还没有：至少给十部电影打分。')
			}
		},
		{
			id: 'factors',
			source: 'train',
			at: 'last',
			view: { key: 'matrix-factorization/map', props: () => ({ people: true, kinds: true }) },
			provenance: 'live',
			title: I('Two numbers per person and per movie', '每个人、每部电影各两个数'),
			narration: {
				en: 'Matrix factorization describes each person and each movie by two numbers, {count|int} numbers in all, so each becomes a point on one plane (an embedding). Nobody picks these numbers, the latent factors: training (steps 5 to 7) sets them from the ratings alone. Training never saw the kinds of the movies, yet for {same|int} of the {movies|int} movies the nearest other movie on the map is of the same kind.',
				zh: '矩阵分解用两个数描述每个人和每部电影，一共 {count|int} 个数，于是每个人、每部电影都成了同一个平面上的一个点（一种嵌入）。这些数叫隐因子，不是谁挑的：训练（第 5 到第 7 步）只根据评分来定。训练从没见过电影的类型，可是在 {movies|int} 部电影中，有 {same|int} 部在图上离它最近的另一部电影和它同一类型。',
				vars: (_rt, at) => ({ count: (N_USERS + N_ITEMS) * 2, same: factorsAt(at) ? sameKindNeighbours(fm(at)) : NaN, movies: N_ITEMS })
			},
			look: I('A point sits at (its first number, its second number), as on a graph. Dots of one color sit together, although training never saw the colors; your two numbers are the arrow.', '每个点的位置就是（它的第一个数，第二个数），和坐标图一样。同一颜色的点聚在一起，尽管训练从没拿到过这些颜色；你的两个数画成了箭头。'),
			terms: ['latent-factors', 'embedding']
		},
		{
			id: 'dot',
			source: 'train',
			at: 'last',
			view: { key: 'matrix-factorization/map', props: () => ({ movie: 'closest', dot: true }) },
			provenance: 'live',
			title: I('A prediction is a dot product', '预测就是点积'),
			narration: {
				en: 'To predict your rating of a movie, multiply your first number by the movie’s first number, do the same with the second numbers, and add the two products: that sum is the dot product. Your numbers are ({a}, {b}). The marked movie, {title}, has ({c}, {d}); {which}',
				zh: '要预测你给一部电影打几分，就把你的第一个数和电影的第一个数相乘，第二个数也这样相乘，再把两个乘积相加：这个和就是点积。你的两个数是 ({a}, {b})。标出的电影{title}是 ({c}, {d})；{which}',
				vars: (rt, at): Record<string, TmplVal> => {
					const f = factorsAt(at);
					if (!f) return { a: NaN };
					const c = closest(f, rt.input), [a, b] = vecU(f, YOU), v = vecV(f, c.i);
					return { a: num(a), b: num(b), c: v[0], d: v[1], title: title(c.i), which: pick(c.unrated, I('among the movies you have not rated, its direction from the centre is the closest to your arrow’s.', '在你没评过的电影里，从中心看过去，它的方向和你的箭头最接近。'), I('of all the movies, its direction from the centre is the closest to your arrow’s.', '在所有电影里，从中心看过去，它的方向和你的箭头最接近。')) };
				}
			},
			look: I('Your arrow, the marked movie, and its shadow on your arrow.', '看你的箭头、标出的电影，以及它投在你箭头上的影子。'),
			predict: {
				id: 'matrix-factorization-dot',
				question: I('The marked movie lies roughly in the direction your arrow points. What will the model predict for it?', '标出的电影大致在你的箭头所指的方向上。模型会给它预测什么？'),
				options: [I('Above 0: it predicts you will like it', '大于 0：它预测你会喜欢'), I('Exactly 0', '正好是 0'), I('Below 0: it predicts you will dislike it', '小于 0：它预测你不会喜欢')],
				answer: 0,
				explain: I(
					'Think of the movie as an arrow from the centre to its dot. The dot product of two arrows is the length of one times the length of the other’s shadow on it: positive when they point the same way, negative when they point in opposite directions, and 0 at a right angle. The stage now shows the sum with your numbers.',
					'把电影看成从中心指向它那个点的箭头。两个箭头的点积，等于其中一个的长度乘以另一个投在它上面的影子的长度：方向相同时为正，方向相反时为负，互相垂直时为 0。舞台上现在用你的数算出了这个和。'
				)
			},
			terms: ['dot-product']
		},
		{
			id: 'error',
			source: 'nudge',
			at: { phase: 'error' },
			view: { key: 'matrix-factorization/map', props: () => ({ formula: true }) },
			provenance: 'live',
			title: I('How wrong is one rating?', '一个评分错了多少？'),
			narration: {
				en: 'Back to the start: training begins from random numbers between −{init} and {init}. {whose} {r} for {title}. With the starting numbers the prediction is {pred}, so the error e, prediction minus rating, is {e}. Training squares it, so that misses in both directions count, and calls e² = {loss} the loss of this rating.',
				zh: '回到起点：训练从 −{init} 到 {init} 之间的随机数开始。{whose}{r}，电影是{title}。用起始的数，预测是 {pred}，所以误差 e（预测减评分）是 {e}。训练把误差平方，这样偏高和偏低都算错，并把 e² = {loss} 叫作这个评分的损失。',
				vars: (_rt, at) => ({
					init: setting(fit.defaults.init),
					whose: pick(m(at, 'user') === YOU, I('Take your strongest rating, the one furthest from 0:', '拿你最强烈的评分，也就是离 0 最远的那个：'), I(`You have rated nothing yet, so take the strongest rating of person ${m(at, 'user')} (counting from 0):`, `你还没有评分，所以拿第 ${m(at, 'user')} 个人（从 0 数起）最强烈的评分：`)),
					r: m(at, 'r'), title: title(m(at, 'item')), pred: m(at, 'pred'), e: m(at, 'e'), loss: m(at, 'loss')
				})
			},
			look: I('The formula under the map, filled in with these numbers. The map is zoomed in on this person and this movie.', '看地图下面的公式，里面填的就是这些数。地图放大了，只画出这个人和这部电影。'),
			terms: ['loss']
		},
		{
			id: 'update',
			source: 'nudge',
			at: { phase: 'update' },
			view: { key: 'matrix-factorization/map', props: () => ({ formula: true }) },
			provenance: 'live',
			title: I('One small step downhill', '往下走一小步'),
			narration: {
				en: 'The gradient of a number says how fast the loss grows when that number grows. If {whose} first number u₁ grows by a tiny h, the prediction u₁v₁ + u₂v₂ grows by v₁h and the loss becomes (e + v₁h)² ≈ e² + 2ev₁h (v₁²h² is tinier still), so the gradient is 2ev₁ = 2 × ({e}) × ({v1}) = {g}. Gradient descent subtracts the learning rate {lr} times the gradient from every number, a small step downhill: {u1} − {lr} × ({g}) = {nu1}. The movie’s numbers move too; the prediction goes from {pred} to {predAfter}, closer to {r}.',
				zh: '一个数的梯度，表示这个数变大时损失增长得有多快。如果{whose}第一个数 u₁ 增大一个很小的 h，预测 u₁v₁ + u₂v₂ 就增大 v₁h，损失变成 (e + v₁h)² ≈ e² + 2ev₁h（v₁²h² 更小，可以忽略），所以梯度是 2ev₁ = 2 × ({e}) × ({v1}) = {g}。梯度下降从每个数里减去学习率 {lr} 乘梯度，往下坡走一小步：{u1} − {lr} × ({g}) = {nu1}。电影的两个数也跟着动，预测从 {pred} 变成 {predAfter}，更接近 {r}。',
				vars: (_rt, at) => ({ whose: pick(m(at, 'user') === YOU, I('your', '你的'), I('this person’s', '这个人的')), e: m(at, 'e'), v1: m(at, 'v1'), g: m(at, 'gu1'), lr: setting(m(at, 'lr')), u1: m(at, 'u1'), nu1: m(at, 'nu1'), pred: m(at, 'pred'), predAfter: m(at, 'predAfter'), r: m(at, 'r') })
			},
			look: I('The formula: this gradient and this update. On the map, the pale arrow and dot are before the step, the solid ones after.', '看公式：这个梯度和这次更新。地图上浅色的箭头和点是这一步之前，实色的是之后。'),
			terms: ['gradient', 'gradient-descent', 'learning-rate']
		},
		{
			id: 'watch',
			source: 'train',
			at: 'last',
			view: { key: 'matrix-factorization/map', props: () => ({ people: true, loss: true, kinds: true }) },
			play: { from: 'first', to: 'last', phases: ['init', 'snapshot'], interval: 110 },
			provenance: 'live',
			title: I('Repeat, and watch the map organise', '重复下去，看地图变得有条理'),
			narration: {
				en: 'Training repeats such a step {steps|int} times, each time on one known rating picked at random: this is stochastic gradient descent. The average loss over all {known|int} known ratings (their mean squared error) falls from {l0} to {l1}{yours}. Meanwhile movies that the same people like drift together{arrow}.',
				zh: '训练把这样的一步重复 {steps|int} 次，每次随机挑一个已知评分：这就是随机梯度下降。全部 {known|int} 个已知评分的平均损失（平均平方误差）从 {l0} 降到 {l1}{yours}。与此同时，同一批人喜欢的电影彼此靠拢{arrow}。',
				vars: (_rt, at) => {
					const t = at.tape, first = t && t.length ? t.frame(0).metrics ?? {} : {};
					const mine = Number.isFinite(m(at, 'you'));
					return {
						steps: at.frame?.step ?? NaN, known: m(at, 'n'), l0: first.mse ?? NaN, l1: m(at, 'mse'),
						yours: pick(mine, { en: `, and on your own ratings from ${num(first.you ?? NaN)} to ${num(m(at, 'you'))}`, zh: `；在你自己的评分上，从 ${num(first.you ?? NaN)} 降到 ${num(m(at, 'you'))}` }, I('', '')),
						arrow: !mine
							? I('; your arrow stays where it started, because you rated nothing', '；你的箭头一直停在起点，因为你什么都没评')
							: pick(fitsYou(first.you ?? NaN, m(at, 'you')), I(', and your arrow settles where it fits your ratings', '，你的箭头也停在最符合你评分的位置'), I(', and your arrow ends where it fits your ratings as well as two numbers can: your ratings pull it different ways', '，你的箭头停在两个数所能做到的最贴近你评分的位置：你的评分把它往不同方向拉'))
					};
				}
			},
			look: I('Press Play: the dots move as training goes on, the error curve falls and the count of same-kind neighbours grows.', '按“播放”：随着训练进行，点在移动，误差曲线在下降，同类近邻的数目在增加。'),
			terms: ['sgd']
		},
		{
			id: 'recs',
			source: 'train',
			at: 'last',
			view: { key: 'matrix-factorization/matrix', props: () => ({ fill: true, rank: true }) },
			provenance: 'live',
			title: I('Your recommendations', '给你的推荐'),
			narration: {
				en: 'Now every empty cell gets a prediction, one dot product each. All {cells|int} cells come from {count|int} numbers: the table is rebuilt from two thin tables of two numbers per person and per movie, its “factors”: hence the name matrix factorization. {recs}{beyond}',
				zh: '现在每个空格都得到一个预测，各是一个点积。全部 {cells|int} 个格子都只来自 {count|int} 个数：这张大表由两张窄表（每人两个数、每部电影两个数）重新拼出来，它们就是它的“因子”，矩阵分解因此得名。{recs}{beyond}',
				vars: (rt, at): Record<string, TmplVal> => {
					const f = factorsAt(at);
					if (!f) return { recs: NaN };
					const sizes = { cells: N_USERS * N_ITEMS, count: (N_USERS + N_ITEMS) * 2 };
					const r = ranked(f, rt.input), s = (x: { i: number; p: number }, lang: 'en' | 'zh') => (lang === 'en' ? `${title(x.i).en} (${fmtN(x.p)})` : `${title(x.i).zh}（${fmtN(x.p)}）`);
					if (!r.length) return { ...sizes, recs: I('You rated every movie, so nothing is left to recommend to you; the other people’s holes are filled all the same.', '你把每部电影都评过了，所以没有可以推荐给你的了；其他人的空格照样被填上了。'), beyond: I('', '') };
					const top = r.slice(0, 3), low = r[r.length - 1], over = [top[0], low].find((x) => Math.abs(x.p) > 2);
					return {
						...sizes,
						recs: {
							en: `Your best unrated movies, your recommendations, are ${top.map((x) => s(x, 'en')).join(', ')}; the worst is ${s(low, 'en')}.`,
							zh: `你没评过的电影里预测最高的，也就是给你的推荐，是${top.map((x) => s(x, 'zh')).join('、')}；最低的是${s(low, 'zh')}。`
						},
						beyond: over
							? { en: ` A prediction can pass the scale, as ${fmtN(over.p)} does: a dot product has no limit, so a real service would cap it at ${over.p > 0 ? '+2' : '−2'}.`, zh: `预测可能超出评分范围，比如 ${fmtN(over.p)}：点积没有上限，所以真实的服务会把它截在 ${over.p > 0 ? '+2' : '−2'}。` }
							: I('', '')
					};
				}
			},
			look: I('Your column: the pale numbers are predictions; 1, 2, 3 mark the top recommendations.', '看你那一列：浅色的数是预测；1、2、3 标出推荐的前三名。')
		},
		{
			id: 'cold',
			source: 'cold',
			at: 'last',
			view: { key: 'matrix-factorization/map', props: () => ({ people: true, who: I('new user', '新用户') }) },
			provenance: 'live',
			kindOf: 'limits',
			title: I('A new user nobody knows', '一个谁也不了解的新用户'),
			narration: {
				en: 'Here is the same training for someone who has rated nothing. No known rating involves their column, so no step changes their two numbers: after {steps|int} steps they are still the random start, ({c1}, {c2}), and their top “recommendation”, {top}, is chance. This is the cold-start problem; a new movie that nobody has rated has it too. Real services first ask a newcomer to rate a few things, or show what is popular.',
				zh: '这是同样的训练，换成一个还什么都没评的人。没有任何已知评分涉及这一列，所以没有哪一步会改动这个人的两个数：训练 {steps|int} 步之后，它们仍是随机的起始值 ({c1}, {c2})，排在第一的“推荐”{top}纯属偶然。这就是冷启动问题；一部还没人评过的新电影也有这个问题。真实的服务会先请新用户给几样东西打分，或者先展示热门内容。',
				vars: (rt, at): Record<string, TmplVal> => {
					const f = factorsAt(at);
					return f ? { steps: at.frame?.step ?? NaN, c1: f.U[2 * YOU], c2: f.U[2 * YOU + 1], top: title(ranked(f, PRESETS[3].raw)[0].i) } : { steps: NaN };
				}
			},
			look: I('The new user’s arrow, short and pointing in a random direction, because it never moved.', '看新用户的箭头：又短，方向也是随机的，因为它从来没动过。'),
			terms: ['cold-start']
		}
	],
	// every preset, besides the adapter's edge fixtures (nothing rated, one rating, all twenty, all so-so)
	fixtures: PRESETS.map((p) => ({ id: p.id, label: p.label, raw: p.raw }))
};

export default lesson;
