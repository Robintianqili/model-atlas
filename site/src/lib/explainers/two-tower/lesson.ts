// two-tower: the guided lesson, one idea per step (the catalog row's ten, with the dot product and the ranker as steps
// of their own). Every number is a {placeholder}: structure from the spec's weights, live values from the trace of the
// learner's choices (fwd) or the batch (inbatch), measured ones from facts.json. Claims that are not live:
// claims.test.ts. The predict step's narration never gives its answer away: the stage shows it after the answer.
//   1 catalog → 2 item tower + table → 3 your choices (task) → 4 user tower → 5 one dot product → 6 all scores (predict)
//   → 7 shortlist → 8 ranker → 9 why it is fast → 10 a new movie → 11 in-batch negatives → 12 limits
import type { At, ExplainerRuntime, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { G, GENRES, MAX_PICKS, MOVIES, N_ITEM_F, N_USER_F, TWINS, toPrefs } from './world.ts';
import { K, T, costs, likesOf, newMovie, rankOne, ranked, reranked, rowOf, shortlist, towerOf } from './recs.ts';

type RT = ExplainerRuntime;
const I = (en: string, zh: string): I18n => ({ en, zh });
const title = (i: number): I18n => (MOVIES[i] ? I(MOVIES[i].title.en, `《${MOVIES[i].title.zh}》`) : I('…', '…'));
const shape = (rt: RT, id: string) => rt.model?.weights[`${id}.weight`]?.shape ?? [NaN, NaN];
/** learned numbers of one tower: two weight matrices and two biases */
const towerParams = (rt: RT, k: string) => [`${k}.fc1`, `${k}.fc2`].reduce((s, id) => s + shape(rt, id)[0] * (shape(rt, id)[1] + 1), 0);
const D = (rt: RT) => shape(rt, 'scores')[1], H = (rt: RT) => shape(rt, 'user.fc1')[0];
const view = (key: string, props: Record<string, unknown>) => ({ key: `two-tower/${key}`, props: () => props });
export const EX = TWINS[0];
export const TASK_GENRES = 1, TASK_PICKS = 3;
export const taskDone = (raw: unknown) => toPrefs(raw).genres.length >= TASK_GENRES && toPrefs(raw).picks.length >= TASK_PICKS;
const score = (rt: RT, i: number) => T(rt, 'scores')?.data[i] ?? NaN;
/** a metric of the step's tape at the frame of a phase */
const metric = (at: At, phase: string, k: string) => {
	const tp = at.tape, i = tp ? tp.find({ phase }) : -1;
	return tp && i >= 0 ? (tp.frame(i).metrics?.[k] ?? NaN) : NaN;
};

const lesson: LessonDef = {
	title: I('Two-tower retrieval: how a recommender finds your movies fast', '双塔召回：推荐系统怎样快速找到你的电影'),
	summary: I(
		'Movies as features, the item tower and its table computed in advance, your choices, the user tower, a score as a dot product, retrieval of a shortlist, a ranker, why it is fast, a new movie without retraining, training with in-batch negatives, and what the towers cannot know.',
		'作为特征的电影、提前算好的物品塔和物品表、你的选择、用户塔、作为点积的分数、召回候选、排序器、它为什么快、无需重新训练的新电影、用批内负样本训练，以及双塔无法知道的东西。'
	),
	steps: [
		{
			id: 'catalog', source: 'fwd', at: { layer: 'item' }, view: view('movies', { mode: 'catalog' }), provenance: 'live',
			title: I('Every movie is a few features', '每部电影就是几个特征'),
			narration: {
				en: 'Picture a streaming service with {n} movies, all made up, in {g} genres. The model never sees a title: it reads {f} numbers per movie, its genre as {g} switches with exactly one on (one-hot), then its year and its length rescaled to about −1 to 1. For {ex}, from {year} and {len} minutes long, they are the numbers below the grid.',
				zh: '设想一个视频网站，有 {n} 部电影，都是虚构的，分 {g} 个类型。模型看不到片名：每部电影它只读 {f} 个数，类型用 {g} 个开关表示，只有一个是开的（独热编码），然后是年份和片长，缩放到大约 −1 到 1。比如{ex}，{year} 年的片子，长 {len} 分钟，它的这些数就在网格下面。',
				vars: () => ({ n: MOVIES.length, g: G, f: N_ITEM_F, ex: title(EX), year: String(MOVIES[EX].year), len: MOVIES[EX].length })
			},
			look: I('The outlined card, and below the grid the numbers the model reads of it.', '看描了边的那张卡片，以及网格下面模型从它读到的数。'),
			terms: ['one-hot']
		},
		{
			id: 'item', source: 'fwd', at: { layer: 'item.fc2' }, view: view('tower', { tower: 'item', movie: EX, table: true }), provenance: 'precomputed',
			title: I('The item tower turns each movie into 8 numbers', '物品塔把每部电影变成 8 个数'),
			narration: {
				en: 'The item tower is a small network with {p} learned numbers: {f} numbers in, a hidden layer of {h} with ReLU (which turns negative numbers into 0), {d} numbers out, the movie’s vector (an embedding). No viewer is involved, so a service runs it once per movie, in advance, and stores the results as the item table: {n} rows of {d} numbers. PyTorch computed this table when the model was exported; the page recomputes {ex}’s row live and gets {same}.',
				zh: '物品塔是一个有 {p} 个学来的数的小网络：输入 {f} 个数，经过 {h} 个数的隐藏层和 ReLU（它把负数变成 0），输出 {d} 个数，也就是电影的向量（一种嵌入）。这里不涉及任何观众，所以服务提前为每部电影跑一次，把结果存成物品表：{n} 行，每行 {d} 个数。这张表是导出模型时由 PyTorch 算好的；页面现场重算{ex}那一行，得到{same}。',
				vars: (rt) => {
					const y = towerOf(rt, 'item', EX)?.y ?? [], row = rowOf(rt, EX);
					const diff = y.length ? Math.max(...row.map((v, k) => Math.abs(v - y[k]))) : NaN;
					return { p: towerParams(rt, 'item'), f: N_ITEM_F, h: H(rt), d: D(rt), n: MOVIES.length, ex: title(EX), same: Number.isNaN(diff) ? NaN : pick(diff === 0, I('exactly the same numbers', '完全相同的数'), I(`the same numbers to within ${fmt3(diff, 2)}`, `相差不到 ${fmt3(diff, 2)} 的数`)) };
				}
			},
			look: I('Top: the tower on one movie. Below: the whole table, one strip of 8 colors per movie; the outlined strip is the one computed above.', '上面：塔处理一部电影。下面：整张表，每部电影一条 8 格的色带；描了边的就是上面算出的那一条。'),
			terms: ['embedding', 'relu']
		},
		{
			id: 'pick', source: 'fwd', at: { layer: 'user' }, view: view('movies', { mode: 'pick' }), provenance: 'live',
			title: I('Tell it what you like', '告诉它你喜欢什么'),
			narration: {
				en: 'Now you are the viewer. Switch on the genres you like and press the movies you liked, up to {max}. So far {now}. The user tower will read only {fu} numbers made from these choices: which genres are on, the genre shares of your picks, their average year and length, and whether you picked any.',
				zh: '现在你就是观众。打开你喜欢的类型，再点你喜欢过的电影，最多 {max} 部。目前{now}。用户塔只读由这些选择算出的 {fu} 个数：哪些类型开着、你所选电影的类型占比、它们的平均年份和片长，以及你有没有选过电影。',
				vars: (rt) => {
					const p = toPrefs(rt.input), g = plural(p.genres.length, { one: '{n} genre', many: '{n} genres', zh: '{n} 个类型' }), m = plural(p.picks.length, { one: '{n} movie', many: '{n} movies', zh: '{n} 部电影' });
					return { max: MAX_PICKS, fu: N_USER_F, now: typeof g === 'object' && typeof m === 'object' ? I(`you like ${g.en} and picked ${m.en}`, `你喜欢 ${g.zh}，选了 ${m.zh}`) : NaN };
				}
			},
			look: I('The genre switches on top, then every movie as a card you can press.', '上面是类型开关，下面每部电影都是一张可以点的卡片。'),
			task: {
				prompt: I('Switch on at least one genre and pick at least three movies you liked.', '至少打开一个类型，并至少选三部你喜欢的电影。'),
				done: (rt) => taskDone(rt.input),
				hint: I('The side panel can also load a whole start profile.', '侧栏里也可以一次换上一整份起始口味。'),
				waiting: I('Not yet: one genre and three movies at least.', '还没有：至少一个类型、三部电影。')
			}
		},
		{
			id: 'user', source: 'fwd', at: { layer: 'user.fc2' }, view: view('tower', { tower: 'user' }), provenance: 'live',
			title: I('The user tower turns you into 8 numbers, live', '用户塔现场把你变成 8 个数'),
			narration: {
				en: 'The user tower has the same shape as the item tower but {p} learned numbers of its own: your {fu} numbers in, {h} hidden, {d} out, your vector. Unlike the item tower it cannot run in advance: it runs on your choices, again each time they change; this time the ReLU turned {zeros} of its {h} hidden numbers into 0. Training made the two towers work together, so your {d} numbers can be compared with any movie’s {d}, place by place.',
				zh: '用户塔和物品塔形状一样，但有自己的 {p} 个学来的数：输入你的 {fu} 个数，隐藏层 {h} 个，输出 {d} 个，就是你的向量。和物品塔不同，它没法提前算：它读你的选择，选择每变一次就得再跑一次；这一次，ReLU 把 {h} 个隐藏数里的 {zeros} 个变成了 0。训练让两座塔配合起来，所以你的 {d} 个数可以和任何一部电影的 {d} 个数逐位对比。',
				vars: (rt) => ({ p: towerParams(rt, 'user'), fu: N_USER_F, h: H(rt), d: D(rt), zeros: Array.from(towerOf(rt, 'user')?.h ?? [NaN]).filter((v) => v === 0).length })
			},
			look: I('Change a choice in the side panel: your vector changes at once; the item table never does.', '在侧栏里改一个选择：你的向量马上变了；物品表从来不变。'),
			terms: ['mlp']
		},
		{
			id: 'dot', source: 'fwd', at: { layer: 'scores' }, select: () => ({ layer: 'scores', unit: [EX] }),
			view: { key: 'zoom.weighted-sum', props: () => ({ names: { x: 'u', w: 'v', p: 'u·v' } }) }, provenance: 'live',
			title: I('A score is a dot product', '分数就是点积'),
			narration: {
				en: 'How well does {ex} suit you? Multiply your {d} numbers u by its {d} numbers v, pair by pair, and add the {d} products: that sum, the dot product, is its score for you, {s}. It is large when the two vectors point the same way. The scoring layer is a weighted sum whose weights are the item table, one movie’s row per output.',
				zh: '{ex}有多适合你？把你的 {d} 个数 u 和它的 {d} 个数 v 一对一对相乘，再把 {d} 个乘积加起来：这个和就是点积，也就是它对你的分数，{s}。两个向量方向越一致，它就越大。打分的那一层是一个加权和，它的权重就是物品表，每个输出对应一部电影的那一行。',
				vars: (rt) => ({ ex: title(EX), d: D(rt), s: score(rt, EX) })
			},
			look: I('The table of products u·v: orange ones push the score up, blue ones down.', '看乘积 u·v 的表：橙色的把分数往上推，蓝色的往下拉。'),
			terms: ['dot-product']
		},
		{
			id: 'scores', source: 'fwd', at: { layer: 'scores' }, view: view('movies', { mode: 'scores' }), provenance: 'live',
			title: I('Every movie gets a score', '每部电影都得到一个分数'),
			narration: {
				en: 'The same dot product for all {n} movies is a single step, your vector times the whole table: {n} × {d} = {mul} multiplications, and every movie has its score. Before you look: which movies come out on top for you?',
				zh: '对全部 {n} 部电影做同样的点积，就是用你的向量乘一次整张表，{n} × {d} = {mul} 次乘法，每部电影就都有了分数。先别看：对你来说，哪些电影会排在最前面？',
				vars: (rt) => ({ n: MOVIES.length, d: D(rt), mul: MOVIES.length * D(rt) })
			},
			look: I('The list, best first: the dot is the genre, and rows in a dashed outline are movies you picked.', '列表从高到低：圆点是类型，带虚线框的行是你选过的电影。'),
			predict: {
				id: 'two-tower-top',
				question: I('Which movies will get your highest scores?', '哪些电影会得到你最高的分数？'),
				options: [I('Movies like the ones you chose', '和你选的相似的电影'), I('The most liked movies, whatever you chose', '最受欢迎的电影，不管你选了什么'), I('A mix of every genre, for variety', '各种类型混在一起，图个多样')],
				answer: 0,
				explain: I(
					'Movies like the ones you chose. Training pulled each viewer’s vector toward the vectors of the movies they went on to like, so your vector points where the vectors of movies like your choices point. With nothing chosen, your vector is the one for “anyone”, and the list starts with a movie that many viewers liked.',
					'和你选的相似的电影。训练把每位观众的向量拉向他们后来喜欢的电影的向量，所以你的向量指向和你的选择相似的电影的向量所指的方向。什么都没选时，你的向量就是“随便哪个人”的向量，列表开头是一部很多观众都喜欢的电影。'
				)
			}
		},
		{
			id: 'shortlist', source: 'fwd', at: { layer: 'scores' }, view: view('movies', { mode: 'shortlist' }), provenance: 'live',
			title: I('Retrieval keeps the best few', '召回：只留最好的几部'),
			narration: {
				en: 'A recommender does not show you all {n}. {drop} by score and keeps the best {k}: the shortlist, or candidates. This first stage is called retrieval. Your shortlist runs from {first} ({s1}) down to {last} ({sk}); {most}.',
				zh: '推荐系统不会把 {n} 部全给你看。{drop}按分数排序，留下最好的 {k} 部：这就是候选名单。这第一阶段叫召回。你的候选从{first}（{s1}）排到{last}（{sk}）；{most}。',
				vars: (rt): Record<string, TmplVal> => {
					const s = shortlist(rt), r = ranked(rt), cnt = GENRES.map((_, g) => s.filter((x) => MOVIES[x.i].genre === g).length), g = cnt.indexOf(Math.max(...cnt));
					if (!s.length) return { n: NaN };
					const np = MOVIES.length - r.length;
					return { n: MOVIES.length, drop: pick(np > 0, I(`It drops the ${np} you picked, sorts the other ${r.length}`, `它去掉你选过的 ${np} 部，把剩下的 ${r.length} 部`), I(`You picked nothing, so it sorts all ${r.length}`, `你什么都没选，所以它把全部 ${r.length} 部`)), k: K, first: title(s[0].i), s1: s[0].s, last: title(s[s.length - 1].i), sk: s[s.length - 1].s, most: I(`${cnt[g]} of the ${K} are ${GENRES[g].en}`, `${K} 部里有 ${cnt[g]} 部是${GENRES[g].zh}片`) };
				}
			},
			look: I('Highlighted rows are the shortlist; the line marks where it ends.', '高亮的行是候选名单；那条线标出它在哪里结束。'),
			terms: ['candidate-retrieval']
		},
		{
			id: 'rank', source: 'fwd', at: { layer: 'scores' }, view: view('movies', { mode: 'rank' }), provenance: 'live',
			title: I('A ranker reorders the shortlist', '排序器重排候选'),
			narration: {
				en: 'The second stage, ranking, looks harder at the {k} shortlisted movies only. Here the ranker is tiny: chance = σ({ws} × score + {wp} × popularity {b}). Popularity measures the movie’s likes in the logs (on a log scale); σ squeezes any number into a chance between 0 and 1. {moved} On the test viewers it gains little over the scores alone: {before|n4} → {after|n4} liked movies in the first three, on average.',
				zh: '第二阶段叫排序，它只仔细看候选里的 {k} 部电影。这里的排序器很小：概率 = σ({ws} × 分数 + {wp} × 热度 {b})。热度衡量这部电影在日志里的点赞次数（取对数）；σ 把任何数压到 0 和 1 之间，变成概率。{moved}在测试观众上，它比只看分数强不了多少：前三名里观众喜欢的电影平均从 {before|n4} 部变成 {after|n4} 部。',
				vars: (rt) => {
					const r = reranked(rt), rk = (rt.spec?.ranker ?? {}) as { w?: number[]; b?: number }, b = rk.b ?? NaN;
					const mv = [...r].map((x, j) => ({ ...x, d: x.k - j })).sort((a, z) => z.d - a.d), up = mv[0], dn = mv[mv.length - 1];
					return {
						k: K, ws: rk.w?.[0] ?? NaN, wp: rk.w?.[1] ?? NaN, b: b < 0 ? `− ${(-b).toPrecision(3)}` : `+ ${b.toPrecision(3)}`, after: rt.num('fact:hits3After'), before: rt.num('fact:hits3Before'),
						// the biggest move up and the biggest move down (a reorder always has both)
						moved: !up ? NaN : up.d > 0 ? I(`For you, ${title(up.i).en} moves up from place ${up.k + 1} to ${up.k + 1 - up.d}, and ${title(dn.i).en} down from ${dn.k + 1} to ${dn.k + 1 - dn.d}.`, `对你来说，${title(up.i).zh}从第 ${up.k + 1} 名升到第 ${up.k + 1 - up.d} 名，${title(dn.i).zh}从第 ${dn.k + 1} 名降到第 ${dn.k + 1 - dn.d} 名。`) : I('For you, the order does not change.', '对你来说，顺序没有变。')
					};
				}
			},
			look: I('The “was” column is the place by score alone; likes come from the training logs.', '“原来”一列是只按分数时的名次；点赞数来自训练日志。'),
			terms: ['ranking-stage']
		},
		{
			id: 'fast', source: 'fwd', at: { layer: 'scores' }, view: view('movies', { mode: 'cost' }), provenance: 'live',
			title: I('Why two towers make retrieval fast', '为什么双塔让召回很快'),
			narration: {
				en: 'For you the page ran the user tower once ({user} multiplications) and one dot product per movie ({n} × {d}): {per} in all. The item tower’s {offline} multiplications were done in advance, once for everybody. A network of the same size that read your {fu} numbers and a movie’s {f} together might judge each pair better, but it costs {one} per movie, on every request: {joint}. With a million movies that is {bigPer} against {bigJoint}.',
				zh: '为了你，页面跑了一次用户塔（{user} 次乘法），再给每部电影做一个点积（{n} × {d}）：一共 {per} 次。物品塔的 {offline} 次是提前做好的，为所有人只做一次。换成一个同样大小、把你的 {fu} 个数和一部电影的 {f} 个数放在一起读的网络，也许能把每一对判断得更准，但每次请求每部电影都要 {one} 次：一共 {joint} 次。有一百万部电影时，就是 {bigPer} 次对 {bigJoint} 次。',
				vars: (rt) => {
					const c = costs(rt), b = costs(rt, 1e6);
					return { user: c.user, n: c.N, d: c.D, per: c.perRequest, offline: c.offline, fu: N_USER_F, f: N_ITEM_F, one: c.jointOne, joint: c.joint, bigPer: b.perRequest, bigJoint: b.joint };
				}
			},
			look: I('The middle row is paid once, before anyone asks.', '中间那一行的计算只做一次，在任何人发出请求之前就做好了。'),
			terms: ['two-tower-model']
		},
		{
			id: 'new', source: 'fwd', at: { layer: 'item.fc2' }, view: view('tower', { tower: 'item', edit: true }), provenance: 'live',
			title: I('A new movie needs no retraining', '新电影不用重新训练'),
			narration: {
				en: 'Add a movie the logs have never seen: set its genre, year and length in the editor. The item tower turns its features into a vector at once, and its score for you is {s}, place {place} of {of}. Nothing was retrained. Matrix factorization learns each movie’s numbers from ratings, so a movie nobody has rated has no useful numbers yet (cold start); the item tower needs only the features.',
				zh: '加一部日志里从没出现过的电影：在编辑器里设定它的类型、年份和片长。物品塔马上把它的特征变成一个向量，它对你的分数是 {s}，在 {of} 部里排第 {place}。没有重新训练任何东西。矩阵分解从评分里学出每部电影的数，所以一部还没人评过的电影没有有用的数（冷启动）；物品塔只需要特征。',
				vars: (rt): Record<string, TmplVal> => {
					const m = newMovie(rt);
					return m ? { s: m.s, place: m.place, of: m.of } : { s: NaN };
				}
			},
			look: I('Change the genre or a slider: the vector, the score and the place change at once.', '换个类型或拖动滑块：向量、分数和名次马上跟着变。'),
			terms: ['cold-start']
		},
		{
			id: 'train', source: 'inbatch', at: { phase: 'grad' }, view: view('batch', { show: 'P' }), provenance: 'live',
			title: I('Training: in-batch negatives in one picture', '训练：一张图看懂批内负样本'),
			narration: {
				en: 'Training read {pairs} (viewer, liked movie) pairs from the logs, {batch} at a time. Here are {b}: each viewer is scored against every movie of the batch; their own movie is the right answer, and the other viewers’ movies serve as wrong answers for free: the in-batch negatives. A softmax turns each row into shares, and the loss averages −log(share of the right movie): {loss} here, {chance} for a blind guess that gives all {b} the same share.',
				zh: '训练从日志里读了 {pairs} 对（观众，喜欢的电影），每次 {batch} 对。这里是 {b} 对：每位观众和这一批里的每部电影都打一次分；自己的那部是正确答案，其他观众的电影顺手当作错误答案，不用另外去找：这就是批内负样本。softmax 把每一行变成份额，损失是 −log(正确那部的份额) 的平均：这里是 {loss}；如果瞎猜、给 {b} 部一样的份额，是 {chance}。',
				vars: (rt, at) => ({ pairs: rt.num('fact:trainPairs'), batch: rt.num('fact:batch'), b: metric(at, 'scores', 'B'), loss: metric(at, 'softmax', 'loss'), chance: metric(at, 'softmax', 'chance') })
			},
			look: I('The outlined cells are the right answers. Press “Gradient of the loss”: negative on the diagonal, so gradient descent raises the right scores; positive elsewhere, so it lowers the wrong ones.', '描了边的格子是正确答案。点“损失的梯度”：对角线上是负的，所以梯度下降会提高正确的分数；其他地方是正的，所以会压低错误的分数。'),
			terms: ['in-batch-negatives', 'softmax', 'cross-entropy']
		},
		{
			id: 'limits', source: 'fwd', at: { layer: 'scores' }, view: view('movies', { mode: 'twins' }), provenance: 'live', kindOf: 'limits',
			title: I('Limits: the towers know only the features', '局限：双塔只知道特征'),
			narration: {
				en: '{a} and {b} share genre, year and length, so the item tower gives them one vector, and every viewer gives them one score: {s} for you. Yet the logs liked {a} {la} times and {b} only {lb}: something no feature records made one far better. The towers cannot learn what the features do not say. Only the ranker, which reads the likes, separates them ({pa|pct} against {pb|pct}), and a brand-new movie has no likes for it to read.',
				zh: '{a}和{b}的类型、年份、片长都一样，所以物品塔给它们同一个向量，每位观众给它们的分数也一样：对你是 {s}。可是日志里{a}被点赞 {la} 次，{b}只有 {lb} 次：某种没有特征记录的东西让前者好得多。特征没说的，双塔学不到。只有读点赞数的排序器能把它们分开（{pa|pct} 对 {pb|pct}），而一部全新的电影还没有点赞可读。',
				vars: (rt) => {
					const [a, b] = TWINS, s = score(rt, a);
					return { a: title(a), b: title(b), s, la: likesOf(rt, a), lb: likesOf(rt, b), pa: rankOne(rt, { i: a, s }).p, pb: rankOne(rt, { i: b, s: score(rt, b) }).p };
				}
			},
			look: I('Two cards, two identical strips of numbers, two very different like counts.', '两张卡片，两条一模一样的数字带，两个相差很大的点赞数。')
		}
	]
	// the fixtures: the adapter's start profiles (manifest input.presetsFromAdapter) and its edge fixtures
};

export default lesson;
