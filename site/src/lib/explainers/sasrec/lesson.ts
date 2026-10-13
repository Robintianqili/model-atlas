// sasrec: the guided lesson, one idea per step. Every number is a {placeholder}: structure from the spec's weights, live
// values from the trace of the learner's clicks (fwd) or the same model on them reversed, measured ones from facts.json.
// Claims that are not live: claims.test.ts. The predict step's narration never gives its answer away: the stage shows
// the weights only after the answer.
//   1 your clicks (task) → 2 item + position rows → 3 which click matters (predict) → 4 one score worked out → 5 the
//   causal mask and training → 6 mixing the values → 7 a product's score → 8 the top 5 → 9 order matters → 10 limits
import type { ExplainerRuntime, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmtPct, fmtSetting } from '../../atlas/core/format.ts';
import { C, ITEMS, N_ITEMS, toClicks } from './world.ts';
import { K, T, attn, clicksOf, dim, most, newPlace, popTop, positions, ranked, reversed, row } from './seq.ts';

type RT = ExplainerRuntime;
const I = (en: string, zh: string): I18n => ({ en, zh });
const name = (i: number | undefined): I18n => (i !== undefined && ITEMS[i] ? I(ITEMS[i].name.en, `“${ITEMS[i].name.zh}”`) : I('…', '…'));
const lastOf = (rt: RT) => clicksOf(rt).length - 1;
export const TASK_CLICKS = 5, TASK_CATS = 2;
export const taskDone = (raw: unknown) => {
	const c = toClicks(raw).clicks;
	return c.length >= TASK_CLICKS && new Set(c.map((i) => ITEMS[i].cat)).size >= TASK_CATS;
};
/** step 4: the last click's query against the key of the earlier click it reads most (itself when it is alone) */
export const keyOf = (rt: RT) => most(rt, true)?.j ?? lastOf(rt);
/** step 6: the feature of the last row of the mix with the largest size, and its largest part a × v */
export function mixOf(rt: RT) {
	const n = clicksOf(rt).length, d = dim(rt), m = row(T(rt, 'att.mix'), n - 1), a = attn(rt), v = T(rt, 'v');
	if (!m || !a || !v) return null;
	const f = m.reduce((b, x, k) => (Math.abs(x) > Math.abs(m[b]) ? k : b), 0);
	const parts = a.map((w, j) => ({ j, a: w, v: v.data[j * d + f], av: w * v.data[j * d + f] }));
	const big = parts.reduce((b, x) => (Math.abs(x.av) > Math.abs(b.av) ? x : b));
	return { f, y: m[f], ...big };
}
const scoresVars = (rt: RT): Record<string, TmplVal> => {
	const top = ranked(rt)[0];
	return top ? { top: name(top.i), s: top.s, p: top.p, d: dim(rt), n: N_ITEMS } : { top: NaN };
};

const lesson: LessonDef = {
	title: I('SASRec: a recommender that reads your clicks in order', 'SASRec：按顺序读你的点击的推荐系统'),
	summary: I(
		'Your clicks as a sequence, item and position rows, which past click matters (attention), one attention score worked out, the causal mask and how it trains every position, mixing the values, a product’s score against the same item table, the top 5, why order matters, and two limits: a new product and popularity bias.',
		'作为序列的你的点击、商品行和位置行、哪次过去的点击重要（注意力）、算出一个注意力分数、因果掩码以及它怎样让每个位置都参与训练、混合值、用同一张商品表算出商品的分数、前 5 名、为什么顺序重要，以及两个局限：新商品和流行度偏差。'
	),
	steps: [
		{
			id: 'clicks', source: 'fwd', at: { layer: 'item' }, view: { key: 'sasrec/seq', props: () => ({ mode: 'pick' }) }, provenance: 'live',
			title: I('Your clicks, in order', '你的点击，按顺序'),
			narration: {
				en: 'This made-up shop sells {n} products in {c} categories, and each product has an id, from 0 to {maxId}. A sequential recommender reads only the ids you clicked and their order: no names, no prices, no account. So far your history has {now}.',
				zh: '这家虚构的商店卖 {n} 件商品，分 {c} 类，每件商品有一个编号，从 0 到 {maxId}。序列推荐只读你点过的编号和它们的顺序：不读名字、不读价格，也不需要账号。到目前为止，你的历史里有{now}。',
				vars: (rt) => {
					const c = clicksOf(rt);
					return { n: N_ITEMS, c: C, maxId: N_ITEMS - 1, now: c.length ? plural(c.length, { one: '{n} click', many: '{n} clicks', zh: ' {n} 次点击' }) : I('no clicks yet', '还没有点击') };
				}
			},
			look: I('The shop in the stage: press a product to add it to the end of your history.', '看舞台上的商店：点一件商品，它就加到你的历史末尾。'),
			task: {
				prompt: I('Build a history of at least five clicks from at least two categories.', '攒出至少五次点击、至少来自两个类别的历史。'),
				done: (rt) => taskDone(rt.input),
				hint: I('The side panel can also load a start history; then add clicks in the stage.', '侧栏里也可以换一段起始历史，再在舞台上继续点。'),
				waiting: I('Not yet: five clicks from two categories at least.', '还没有：至少五次点击，来自至少两个类别。')
			},
			terms: ['sequential-recommendation']
		},
		{
			id: 'embed', source: 'fwd', at: { layer: 'x' }, view: { key: 'sasrec/seq', props: () => ({ mode: 'embed' }) }, provenance: 'live',
			title: I('Each click becomes a row of numbers', '每次点击变成一行数'),
			narration: {
				en: 'The model keeps an item table, one row of {d} learned numbers for each of the {n} products, and a position table with one row for each of {P} positions. A click becomes its product’s row plus its position’s row, so the same product reads differently first or last. Your last click, {last} at position {t} (counting from 0), starts in feature 0 as {a} + {b} = {x}.',
				zh: '模型有一张商品表，{n} 件商品每件一行，每行 {d} 个学来的数；还有一张位置表，{P} 个位置每个一行。一次点击就是它的商品行加上它的位置行，所以同一件商品排在最前和最后读起来不一样。你最后一次点击{last}在位置 {t}（从 0 数起），它的特征 0 一开始是 {a} + {b} = {x}。',
				vars: (rt) => {
					const t = lastOf(rt), at = (id: string) => row(T(rt, id), t)?.[0] ?? NaN;
					return { d: dim(rt), n: N_ITEMS, P: positions(rt), last: name(clicksOf(rt)[t]), t, a: at('item'), b: at('pos'), x: at('x') };
				}
			},
			look: I('One row per click: the product row, the position row and their sum; the shaded row is your last click.', '每次点击一行：商品行、位置行和它们的和；底色加深的一行是你最后一次点击。'),
			terms: ['embedding', 'positional-encoding']
		},
		{
			id: 'attend', source: 'fwd', at: { layer: 'att.w' }, view: { key: 'sasrec/seq', props: () => ({ mode: 'attend' }) }, provenance: 'live',
			title: I('Which past click matters?', '哪次过去的点击重要？'),
			narration: {
				en: 'To guess your next click, the model reads your history from your last click, {last}. Every click offers a key of {d} numbers, the last click asks with a query of {d} numbers, and a query times a key gives a score. A softmax turns {scores}. Which click will get the largest weight?',
				zh: '为了猜你的下一次点击，模型从你最后一次点击{last}出发去读你的历史。每次点击都提供一个 {d} 个数的键，最后一次点击用一个 {d} 个数的查询来问，查询乘以键就得到一个分数。softmax 把{scores}。哪次点击会得到最大的权重？',
				vars: (rt) => {
					const n = clicksOf(rt).length;
					return { last: name(clicksOf(rt)[lastOf(rt)]), d: dim(rt), scores: n ? pick(n > 1, I(`the ${n} scores into attention weights that add up to 1`, `这 ${n} 个分数变成加起来等于 1 的注意力权重`), I('its one score into a weight of exactly 1', '它唯一的分数变成正好为 1 的权重')) : NaN };
				}
			},
			look: I('Once you answer: one bar per click, its share of the last click’s attention.', '回答之后：每次点击一条，是它分到的最后一次点击的注意力。'),
			predict: {
				id: 'sasrec-attend',
				question: I('Which click gets the largest weight?', '哪次点击得到的权重最大？'),
				options: [I('The most recent one: the last click itself', '最近的那次：最后一次点击自己'), I('The first click of the history', '历史里的第一次点击'), I('Every click about equally', '每次点击差不多一样')],
				answer: 0,
				explain: I(
					'The most recent click, on every history this lesson tries: it says the most about what comes next. Earlier clicks of its own category get more, on average, than clicks of another interest. A history of one click puts all its weight on itself.',
					'最近的那次点击，在这一课试过的每段历史上都是如此：它最能说明接下来会发生什么。同一类别的更早点击，平均分到的比另一种兴趣的点击多。只有一次点击的历史，把全部权重都放在它自己身上。'
				)
			},
			terms: ['attention', 'query-key-value', 'softmax']
		},
		{
			id: 'score', source: 'fwd', at: { layer: 'att.scores' }, select: (rt) => ({ layer: 'att.scores', unit: [0, lastOf(rt), keyOf(rt)] }), view: 'zoom.weighted-sum', provenance: 'live',
			title: I('One attention score, worked out', '算出一个注意力分数'),
			narration: {
				en: 'Each click’s row is first rescaled (layer normalization), then three learned weighted sums make its query, key and value. The query of {last} times {key}, number by number, adds up to {dot|n4}; divided by √{d} = {sq|int} (the same as × {inv}), that gives the score {s|n4}. On the test sessions the most recent click got the largest weight in {fact:attLastIsMax|pct} of the histories.',
				zh: '每次点击的那一行先被重新缩放（层归一化），再由三个学来的加权和变成它的查询、键和值。{last}的查询和{key}逐个相乘再相加，得到 {dot|n4}；除以 √{d} = {sq|int}（也就是乘以 {inv}），分数就是 {s|n4}。在测试会话上，有 {fact:attLastIsMax|pct} 的历史里，最近一次点击得到的权重最大。',
				vars: (rt) => {
					const c = clicksOf(rt), d = dim(rt), j = keyOf(rt), sc = T(rt, 'att.scores'), n = c.length, s = sc ? sc.data[lastOf(rt) * n + j] : NaN;
					return { last: name(c[lastOf(rt)]), key: pick(j === lastOf(rt), I('its own key', '它自己的键'), I(`the key of ${name(c[j]).en}`, `${name(c[j]).zh}的键`)), dot: s * Math.sqrt(d), sq: Math.sqrt(d), inv: fmtSetting(1 / Math.sqrt(d)), s, d };
				}
			},
			look: { en: 'Each row is one query number × one key number; the {d} results add up to {dot|n4}.', zh: '每一行是一个查询数 × 一个键数；{d} 个乘积加起来是 {dot|n4}。', vars: (rt) => { const n = clicksOf(rt).length, sc = T(rt, 'att.scores'); return { d: dim(rt), dot: (sc ? sc.data[lastOf(rt) * n + keyOf(rt)] : NaN) * Math.sqrt(dim(rt)) }; } },
			terms: ['layer-norm', 'query-key-value']
		},
		{
			id: 'mask', source: 'fwd', at: { layer: 'att.w' }, view: { key: 'sasrec/seq', props: () => ({ mode: 'mask' }) }, provenance: 'live',
			title: I('The causal mask: every click guesses the next', '因果掩码：每次点击都猜下一次'),
			narration: {
				en: 'Not only your last click: every click has its own row of weights, and row t (counting from 0) guesses the click at position t + 1. Training used all rows at once: {fact:trainSessions} sessions gave {fact:trainTargets} guesses, each scored with cross-entropy. Row t must not read later clicks, or it would see the answer: the causal mask sets {masked}.',
				zh: '不只是最后一次点击：每次点击都有自己的一行权重，行 t（从 0 数起）猜的是位置 t + 1 上的点击。训练时所有行一起用上：{fact:trainSessions} 个会话给出 {fact:trainTargets} 次猜测，每次都用交叉熵打分。行 t 不能读之后的点击，否则就看到了答案：因果掩码把{masked}。',
				vars: (rt) => {
					const n = clicksOf(rt).length;
					const m = (n * (n - 1)) / 2;
					return {
						masked: !n ? NaN : m === 0 ? I('the scores of later clicks to −∞, so their weights are exactly 0 (with one click there are none)', '之后点击的分数设成 −∞，经过 softmax 后权重正好是 0（只有一次点击，所以一个也没有）')
							: plural(m, { one: 'that {n} score to −∞, so after the softmax its weight is exactly 0', many: 'those {n} scores to −∞, so after the softmax their weights are exactly 0', zh: '这 {n} 个分数设成 −∞，经过 softmax 后权重正好是 0' })
					};
				}
			},
			look: I('Read a row from left to right: the hatched cells are later clicks, masked.', '从左往右读一行：斜线格是之后的点击，被遮住了。'),
			terms: ['causal-mask', 'cross-entropy']
		},
		{
			id: 'mix', source: 'fwd', at: { layer: 'att.mix' }, select: (rt) => ({ layer: 'att.mix', unit: [lastOf(rt), mixOf(rt)?.f ?? 0] }), view: 'zoom.weighted-sum', provenance: 'live',
			title: I('Mix the values', '混合各次点击的值'),
			narration: {
				en: 'Every click also has a value of {d} numbers. Your last click’s row adds up all values, each times its weight: feature {f} (counting from 0) becomes {y|n4}, and its largest part, {a|n4} × {v|n4} = {av|n4}, comes from {src}. The result is added to the row (a residual connection); then a small two-layer network (an MLP) works on each row alone, and a last layer normalization gives the vector that scores the products.',
				zh: '每次点击还有一个 {d} 个数的值。你最后一次点击的那一行把所有值乘上各自的权重再相加：特征 {f}（从 0 数起）变成 {y|n4}，其中最大的一份 {a|n4} × {v|n4} = {av|n4} 来自{src}。结果加回到这一行上（残差连接）；然后一个两层的小网络（MLP，也叫前馈网络）单独加工每一行，最后一次层归一化给出用来给商品打分的向量。',
				vars: (rt): Record<string, TmplVal> => {
					const m = mixOf(rt);
					return m ? { d: dim(rt), f: m.f, y: m.y, a: m.a, v: m.v, av: m.av, src: name(clicksOf(rt)[m.j]) } : { d: dim(rt), f: NaN };
				}
			},
			look: I('Column a holds the weights of step 3 and column v the values; each a·v is one click’s part of the sum.', 'a 列是第 3 步的权重，v 列是值；每个 a·v 是一次点击在这个和里的那一份。'),
			terms: ['residual-connection', 'mlp']
		},
		{
			id: 'scores', source: 'fwd', at: { layer: 'logits' }, select: (rt) => ({ layer: 'logits', unit: [lastOf(rt), ranked(rt)[0]?.i ?? 0] }), view: 'zoom.weighted-sum', provenance: 'live',
			title: I('Score every product with the same table', '用同一张表给每件商品打分'),
			narration: {
				en: 'To score a product, the model multiplies your history’s vector by that product’s row of the item table, the very table your clicks were looked up in, and adds up the {d} results: one dot product per product. {top} scores {s|n4}, the highest of the products you have not clicked. A softmax over all {n} scores turns that into a share of {p|pct}: the model’s guess that it is your next click.',
				zh: '给一件商品打分时，模型把你历史的向量乘以这件商品在商品表里的那一行（正是查你的点击时用的那张表），再把 {d} 个乘积加起来：每件商品一个点积。{top}得到 {s|n4}，是你没点过的商品里最高的。对全部 {n} 个分数做 softmax，它就成了 {p|pct} 的份额：模型猜它就是你的下一次点击。',
				vars: scoresVars
			},
			look: { en: 'Each row multiplies one number of your vector by the number in the same place in {top}’s row.', zh: '每一行把你的向量的一个数，乘以{top}那一行同一位置上的数。', vars: scoresVars },
			terms: ['dot-product', 'softmax']
		},
		{
			id: 'topk', source: 'fwd', at: { layer: 'probs' }, view: { key: 'sasrec/seq', props: () => ({ mode: 'topk' }) }, provenance: 'live',
			title: I('Your top 5', '你的前 5 名'),
			narration: {
				en: 'A shop shows only a few suggestions: the {k} largest shares among the products you have not clicked. For you: {list}. On {fact:testSessions} test sessions with the last click hidden, that click was among the top {k} in {fact:hit5|pct} of them; showing the most clicked products gets {fact:hit5Pop|pct}, and showing what most often follows the last click {fact:hit5Markov|pct}.',
				zh: '商店只展示几条推荐：在你没点过的商品里份额最大的 {k} 件。对你来说是：{list}。在 {fact:testSessions} 个藏起最后一次点击的测试会话上，那次点击有 {fact:hit5|pct} 落在前 {k} 名里；只推点击最多的商品是 {fact:hit5Pop|pct}，推最后一次点击之后最常跟着的商品是 {fact:hit5Markov|pct}。',
				vars: (rt) => {
					const r = ranked(rt).slice(0, K);
					return { k: K, list: r.length ? I(r.map((x) => `${ITEMS[x.i].name.en} ${fmtPct(x.p)}`).join(', '), r.map((x) => `“${ITEMS[x.i].name.zh}” ${fmtPct(x.p)}`).join('、')) : NaN };
				}
			},
			look: { en: 'The shaded rows are your top {k}; each bar is a share of the next click.', zh: '底色加深的几行是你的前 {k} 名；每一条是下一次点击的份额。', vars: () => ({ k: K }) },
			terms: ['hit-rate']
		},
		{
			id: 'order', source: 'fwd', at: { layer: 'probs' }, view: { key: 'sasrec/seq', props: () => ({ mode: 'order' }) }, provenance: 'live',
			title: I('Order matters', '顺序很重要'),
			narration: {
				en: 'Now the same clicks in reverse order. {change} SASRec sees order through the position rows and the causal mask; shuffling every test history drops its hit rate from {fact:hit5|pct} to {fact:hit5Shuffled|pct}. Matrix factorization learns one row per user from their ratings, and the two-tower model reads your picks as a bag: neither sees order.',
				zh: '现在把同样的点击倒过来。{change}SASRec 通过位置行和因果掩码看到顺序；把每段测试历史打乱后，它的命中率从 {fact:hit5|pct} 降到 {fact:hit5Shuffled|pct}。矩阵分解从评分里给每个用户学一行，双塔模型把你的选择当成一袋东西来读：两者都看不到顺序。',
				vars: (rt) => {
					const a = ranked(rt)[0], rv = reversed(rt), b = rv.top[0];
					if (!a || !b) return { change: NaN };
					const pa = fmtPct(a.p), pb = fmtPct(b.p), A = ITEMS[a.i].name, B = ITEMS[b.i].name;
					return {
						change: clicksOf(rt).length < 2 ? I('With one click there is nothing to reverse.', '只有一次点击，没什么可倒过来的。')
							: a.i !== b.i ? I(`The top suggestion changes from ${A.en} (${pa}) to ${B.en} (${pb}).`, `排第一的推荐从“${A.zh}”（${pa}）变成“${B.zh}”（${pb}）。`)
							: I(`The top suggestion stays ${A.en}, at ${pb} instead of ${pa}.`, `排第一的推荐还是“${A.zh}”，份额从 ${pa} 变成 ${pb}。`)
					};
				}
			},
			look: I('Two top-5 lists side by side; the links lead to the two models that ignore order.', '两份前 5 名并排；链接通向两个不看顺序的模型。'),
			terms: ['latent-factors', 'two-tower-model']
		},
		{
			id: 'limits', source: 'fwd', at: { layer: 'probs' }, view: { key: 'sasrec/seq', props: () => ({ mode: 'limits' }) }, provenance: 'live', kindOf: 'limits',
			title: I('Limits: new products and popular ones', '局限：新商品和热门商品'),
			narration: {
				en: 'The drone is new: no session in the logs clicked it, so training only ever used it as a wrong answer and pushed its score down; {drone}, and of {fact:testSessions} test sessions {reach} (cold start). Popularity bias: on the test sessions, the {np} most clicked products take {fact:popShareRec|pct} of all top-{k} slots, though they were only {fact:popShareTarget|pct} of the clicks that actually came next.',
				zh: '无人机是新品：日志里没有哪个会话点过它，所以训练只把它当作错误答案，一直压低它的分数；{drone}；在 {fact:testSessions} 个测试会话里，{reach}（冷启动）。流行度偏差：在测试会话上，点击最多的 {np} 件商品占了所有前 {k} 名推荐位的 {fact:popShareRec|pct}，可在真正接下来的那次点击里，它们只占 {fact:popShareTarget|pct}。',
				vars: (rt) => {
					const p = newPlace(rt), m = rt.num('fact:newItemInTop5');
					return { reach: Number.isNaN(m) ? NaN : m === 0 ? I(`none had it in the top ${K}`, `没有一个把它排进前 ${K} 名`) : I(`${m} had it in the top ${K}`, `有 ${m} 个把它排进前 ${K} 名`), k: K, np: popTop(rt).length || NaN, drone: p ? I(`for you it ranks ${p.place} of ${p.of}`, `对你来说它在 ${p.of} 件里排第 ${p.place}`) : I('you clicked it yourself, but its row was never trained as a click', '你自己点过它，但它的那一行从没作为点击被训练过') };
				}
			},
			look: I('One card for the drone, one for the most clicked products; the two bars are to scale.', '一张卡片讲无人机，一张讲点击最多的商品；两条横条按比例画。'),
			terms: ['cold-start', 'popularity-bias']
		}
	]
	// the fixtures: the adapter's start histories (manifest input.presetsFromAdapter) and its edge fixtures
};

export default lesson;
