// two-tower: the package manifest (design §4.9). A two-tower retrieval model on synthetic movies and viewers:
//   fwd      the exported model (pipeline/two-tower/export.py) on the learner's choices: the item tower on their new
//            movie, the user tower on their genres and picks, and the scores of all 36 movies against the item table
//            (the item tower's output for every movie, computed once at export and shipped as a weight)
//   logs     the training logs, made by the same TS code that made train.py's data (world.ts; 24 viewers here; no
//            lesson step plays it: `atlas run two-tower logs --save` writes the data)
//   inbatch  six (viewer, liked movie) pairs of the logs through the exported towers: the 6×6 scores, the row softmax,
//            the loss and its gradient (kit autograd; reference.py twin)
// Measured numbers (recall, the ranker's effect, the twins' likes) are in facts.json (pipeline/two-tower/train.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { PRESETS } from './world.ts';
import { batchOf } from './recs.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'two-tower',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/two-tower', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	// the start profiles come from the adapter (presets(), no file): the page, lint and the claim tests all get them
	input: { adapter: 'prefs', start: PRESETS[0].raw, presetsFromAdapter: true },
	output: { view: 'two-tower/movies', layer: 'scores', props: { mode: 'rank' } },
	sources: {
		fwd: { kind: 'trace' },
		logs: { kind: 'process', process: 'logs', run: 'main', reference: false },
		inbatch: { kind: 'process', process: 'inbatch', run: 'main', input: (rt) => batchOf(rt) }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			I('Compute a movie’s score as the dot product of your vector and the movie’s vector, and say which movies come out on top.', '把你的向量和一部电影的向量做点积，算出它的分数，并说出哪些电影会排在最前面。'),
			I('Explain why the item tower can run once per movie in advance, and count what that saves on each request.', '解释为什么物品塔可以提前为每部电影跑一次，并算出每次请求因此省下多少计算。'),
			I('Tell retrieval (a quick shortlist from the whole catalog) from ranking (a closer look at the shortlist only).', '分清召回（从整个片库里快速挑出候选）和排序（只仔细看候选）。'),
			I('Show that a new movie gets a vector and a score from its features alone, without retraining, unlike in matrix factorization.', '说明一部新电影只凭特征就能得到向量和分数，不用重新训练，这一点和矩阵分解不同。'),
			I('Read an in-batch training table: which scores the loss pushes up and which down; and name what the towers cannot learn.', '读懂一张批内训练表：损失把哪些分数往上推、哪些往下压；并说出双塔学不到什么。')
		],
		prequestion: {
			id: 'two-tower-pre',
			question: I('A site with a million movies must find the best ones for you in a split second. How can it?', '一个有一百万部电影的网站，要在一瞬间为你找出最好的几部。它怎么做到？'),
			options: [I('Each movie’s numbers are worked out in advance, so for you it needs one quick multiply-and-add per movie', '每部电影的数都提前算好，所以对你只需每部电影做一次简单的乘加'), I('It asks a big network about every movie, on a very fast computer', '它用一台很快的计算机，让一个大网络逐部判断'), I('It only ever looks at the most popular movies', '它只看最热门的电影')],
			answer: 0,
			explain: I('That is the two-tower idea: one network turns every movie into a vector ahead of time, another turns you into a vector when you arrive, and a movie’s score is their dot product. Only a short list then gets a closer look.', '这就是双塔的思路：一个网络提前把每部电影变成向量，另一个在你到来时把你变成向量，电影的分数就是两者的点积。之后只有一份短名单会被仔细查看。')
		},
		tryThis: [
			{
				id: 'scary', title: I('History without genres', '只有观看历史，没有类型'), setup: { input: PRESETS[3].raw },
				// claims.test.ts checks the look on this profile
				predict: I('This profile switched on no genre at all and only picked four horror movies. Will horror still lead its scores?', '这份口味一个类型也没打开，只选了四部恐怖片。恐怖片还会排在它的分数最前面吗？'),
				look: I('Yes: the two horror movies it did not pick lead its shortlist. The user tower reads the genre shares of your picks too, so a history speaks even when no genre is switched on.', '会：它没选的那两部恐怖片排在候选最前面。用户塔也读你所选电影的类型占比，所以即使没打开任何类型，观看历史也在起作用。'),
				goto: { step: 'shortlist' }
			}
		],
		quiz: [
			{
				q: I('Your vector is (1, 2, 0) and a movie’s vector is (3, −1, 5). What is its score?', '你的向量是 (1, 2, 0)，一部电影的向量是 (3, −1, 5)。它的分数是多少？'),
				options: [I('1', '1'), I('8', '8'), I('−2', '−2')],
				answer: 0,
				explain: I('Multiply pair by pair and add: 1 × 3 + 2 × (−1) + 0 × 5 = 3 − 2 + 0 = 1.', '一对一对相乘再相加：1 × 3 + 2 × (−1) + 0 × 5 = 3 − 2 + 0 = 1。')
			}
		],
		misconception: {
			claim: I('A recommender runs a big network on you and every movie whenever you open the page.', '每次你打开页面，推荐系统都会把你和每部电影一起送进一个大网络。'),
			truth: I('Retrieval is one dot product per movie with movie vectors computed in advance; only the short list gets a closer look.', '召回只是每部电影一个点积，电影的向量是提前算好的；只有短名单才会被仔细查看。'),
			step: 'fast'
		},
		sections: {
			titles: { output: I('Your shortlist after the ranker', '排序之后的候选') },
			intros: { training: I('Trained in PyTorch on the synthetic logs with in-batch negatives (batches of 64 pairs, Adam; the weights of the best of 40 epochs on 500 validation viewers), then the ranker was fitted on those validation viewers. The page runs the exported towers; the numbers below were measured on 500 other, test viewers.', '用 PyTorch 在合成日志上以批内负样本训练（每批 64 对，Adam；取 40 轮里在 500 位验证观众上最好的那一轮的权重），之后在这些验证观众上拟合排序器。页面运行导出的两座塔；下面的数是在另外 500 位测试观众上测出来的。') }
		},
		metrics: {
			test_recall8: { label: I('Test: share of a viewer’s liked movies in their shortlist of 8', '测试：观众喜欢的电影落在其 8 部候选里的比例'), fmt: 'pct' },
			test_recall8_pop: { label: I('The same for “most liked first”', '“最热门优先”的同一比例'), fmt: 'pct' },
			test_recall8_random: { label: I('The same for a random pick', '随机挑选的同一比例'), fmt: 'pct' }
		},
		lessonNote: I('Twelve steps: movies as features, the item tower and its table, your choices, the user tower, one dot product, all the scores, the shortlist, the ranker, why it is fast, a new movie, training with in-batch negatives, and the limits.', '十二步：作为特征的电影、物品塔和物品表、你的选择、用户塔、一个点积、全部分数、候选名单、排序器、为什么快、一部新电影、用批内负样本训练，以及局限。')
	},
	glossary: [
		{ id: 'two-tower-model', term: I('two-tower model', '双塔模型'), def: I('A recommender made of two networks: one turns a user into a vector, the other turns an item into a vector, and the score is their dot product.', '由两个网络组成的推荐模型：一个把用户变成向量，另一个把物品变成向量，分数是两者的点积。'), related: ['embedding', 'dot-product'] },
		{ id: 'candidate-retrieval', term: I('retrieval (candidate generation)', '召回（候选生成）'), def: I('The first stage of a recommender: from the whole catalog, quickly pick a short list of candidates, for example the items with the highest dot products.', '推荐系统的第一阶段：从整个物品库里快速挑出一份短的候选名单，比如点积最高的那些物品。'), related: ['two-tower-model', 'ranking-stage'] },
		{ id: 'ranking-stage', term: I('ranking stage', '排序阶段'), def: I('The second stage of a recommender: a model that looks harder at the short list only and puts it in its final order.', '推荐系统的第二阶段：一个只仔细看短名单、并给出最终顺序的模型。'), related: ['candidate-retrieval'] },
		{ id: 'in-batch-negatives', term: I('in-batch negatives', '批内负样本'), def: I('A training trick: in a batch of (user, item) pairs, each user’s own item is the right answer and the other users’ items serve as wrong ones, so no extra negatives have to be drawn.', '一种训练技巧：在一批（用户，物品）对里，每个用户自己的物品是正确答案，其他用户的物品当作错误答案，所以不用另外抽负样本。'), related: ['softmax', 'cross-entropy', 'batch'] }
	],
	catalog: {
		params: I('672 learned numbers in two towers, plus a 36 × 8 item table computed from them', '两座塔共 672 个学来的数，外加由它们算出的 36 × 8 物品表'),
		size: I('About 16 KB: spec, weights (with the item table) and facts', '约 16 KB：规格、权重（含物品表）和事实数据'),
		// the site's row for an exported model (tech-zh.ts has it): the towers' forward pass, checked against PyTorch
		runtime: I('Own JS forward pass (checked against PyTorch)', '自研 JS 前向计算（已与 PyTorch 核对）'),
		// the planned entry said "a dot product decides what you see first"; on the page the dot product picks the
		// shortlist and the ranker then reorders it
		hook: I('One tower for you, one for the movies; a dot product picks your shortlist from the whole catalog.', '一座塔给你，一座塔给电影；点积从整个片库里为你挑出候选。'),
		shows: I(
			'Movies as features, the item tower and its precomputed table, your choices through the user tower, a score as a dot product, retrieval of a shortlist, a tiny ranking stage, why the split is fast, a new movie without retraining, in-batch negatives, and what the features cannot tell.',
			'作为特征的电影、物品塔和提前算好的物品表、经过用户塔的你的选择、作为点积的分数、召回候选、一个小小的排序阶段、这种拆分为什么快、无需重新训练的新电影、批内负样本，以及特征说不出的东西。'
		)
	},
	// No dataset, model or code from anyone else: the movies, viewers and logs are synthetic (world.ts). The model and its
	// training follow published descriptions (ideas only).
	credits: [
		{
			name: 'Yi, Yang, Hong, Cheng, Heldt, Kumthekar, Zhao, Wei & Chi, “Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations” (RecSys 2019)',
			url: 'https://doi.org/10.1145/3298689.3346996',
			license: 'Article; ideas only (no text, figures, code or data used)',
			usedFor: 'the two-tower retrieval model trained with in-batch negatives (softmax over the items of a batch), at toy size',
			attribution: 'Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei and Ed Chi (Google)'
		},
		{
			name: 'Covington, Adams & Sargin, “Deep Neural Networks for YouTube Recommendations” (RecSys 2016)',
			url: 'https://doi.org/10.1145/2959100.2959190',
			license: 'Article; ideas only (no text, figures, code or data used)',
			usedFor: 'the two stages: candidate retrieval by dot products over the whole catalog, then ranking of the short list',
			attribution: 'Paul Covington, Jay Adams and Emre Sargin (Google)'
		}
	]
};

export default manifest;
