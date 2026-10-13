// sasrec: the package manifest (design §4.9). A tiny SASRec on a made-up shop:
//   fwd   the exported model (pipeline/sasrec/export.py) on the learner's clicks: item + position rows, one causal
//         attention block, the score of every product against the same item table, the shares of the next click
//   logs  the training sessions, made by the same TS code that made train.py's data (world.ts; 24 sessions here; no
//         lesson step plays it: `atlas run sasrec logs --opt '{"n":10000}' --save` writes the data)
// Measured numbers (hit rates, the baselines, popularity bias, the new arrival) are in facts.json (train.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { PRESETS } from './world.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'sasrec',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/sasrec', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	// the start histories come from the adapter (presets(), no file): the page, lint and the claim tests all get them
	input: { adapter: 'clicks', start: PRESETS[0].raw, presetsFromAdapter: true },
	output: { view: 'sasrec/seq', layer: 'probs', props: { mode: 'topk' }, row: 'last' },
	sources: {
		fwd: { kind: 'trace' },
		logs: { kind: 'process', process: 'logs', run: 'main', reference: false }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			I('Turn a click history into rows of numbers: a product’s row plus a position’s row.', '把一段点击历史变成一行行的数：商品的那一行加上位置的那一行。'),
			I('Read the attention weights of your last click and say which earlier clicks it relies on.', '读懂你最后一次点击的注意力权重，说出它依靠哪些更早的点击。'),
			I('Explain why the causal mask lets one session teach the model at every position at once.', '解释为什么因果掩码让一个会话在每个位置上同时训练模型。'),
			I('Compute a product’s score as a dot product with the same item table, and pick the top 5 you have not clicked.', '用同一张商品表做点积算出商品的分数，并挑出你还没点过的前 5 件。'),
			I('Show with numbers that order matters here, unlike in matrix factorization or a two-tower model, and name two limits: a new product and popular products.', '用数字说明这里顺序很重要，这和矩阵分解、双塔模型不同；并说出两个局限：新商品和热门商品。')
		],
		prequestion: {
			id: 'sasrec-pre',
			question: I('You clicked a tent, then a sleeping bag. What should a shop show you next?', '你先点了帐篷，又点了睡袋。商店接下来该给你看什么？'),
			options: [I('What people usually click after those two, in that order', '人们按这个顺序点完这两样后，通常接着点的东西'), I('The shop’s best sellers', '店里卖得最好的东西'), I('More tents', '更多帐篷')],
			answer: 0,
			explain: I('A sequential recommender reads your clicks as a sequence and guesses the next one, as a language model guesses the next word. Best sellers ignore you; more tents ignore that you already have one.', '序列推荐把你的点击当成一个序列来读，猜下一次点击，就像语言模型猜下一个词。畅销榜不看你是谁；再推帐篷则忽略了你已经有一顶了。')
		},
		tryThis: [
			{
				id: 'late', title: I('The last click weighs most', '最后一次点击分量最重'), setup: { input: PRESETS[4].raw },
				// claims.test.ts checks the look on this history
				predict: I('This history has three baking clicks, then running shoes. Will its top 5 be mostly baking or mostly fitness?', '这段历史先是三次烘焙，再是一双跑鞋。它的前 5 名主要是烘焙用品还是健身用品？'),
				look: I('Mostly fitness, led by the sports watch: the most recent click weighs most. Step 9 reverses the history, and then baking products lead.', '主要是健身用品，排第一的是运动手表：最近一次点击分量最重。第 9 步把历史倒过来，烘焙用品就排到了前面。'),
				goto: { step: 'topk' }
			}
		],
		quiz: [
			{
				q: I('A history has 4 clicks. How many attention weights does the causal mask set to exactly 0?', '一段历史有 4 次点击。因果掩码把多少个注意力权重设成正好为 0？'),
				options: [I('6', '6'), I('16', '16'), I('4', '4')],
				answer: 0,
				explain: I('The table has 4 × 4 = 16 weights; row t (counting from 0) may read the t + 1 clicks up to itself, so 1 + 2 + 3 + 4 = 10 are allowed and 16 − 10 = 6 are masked.', '这张表有 4 × 4 = 16 个权重；行 t（从 0 数起）能读到它自己为止的 t + 1 次点击，所以 1 + 2 + 3 + 4 = 10 个被允许，16 − 10 = 6 个被遮住。')
			}
		],
		misconception: {
			claim: I('A recommender needs to know who you are: your account, your profile, your past ratings.', '推荐系统必须知道你是谁：你的账号、资料和以前的评分。'),
			truth: I('This sequential recommender has no table of users at all: your clicks in this visit are the whole input, so it works for a visitor it has never seen.', '这个序列推荐模型根本没有用户表：你这次访问的点击就是全部输入，所以它对从没见过的访客也能用。'),
			step: 'clicks'
		},
		sections: {
			intros: { training: I('Trained in PyTorch on 8,000 synthetic sessions: at every position, the next click (a softmax over all products, Adam, 60 epochs; the weights of the epoch with the best hit rate on 1,000 validation sessions). The page runs the exported model; the numbers below were measured on 1,000 other, test sessions, with each one’s last click hidden.', '用 PyTorch 在 8,000 个合成会话上训练：在每个位置预测下一次点击（对所有商品做 softmax，Adam，60 轮；取在 1,000 个验证会话上命中率最高的那一轮的权重）。页面运行导出的模型；下面的数是在另外 1,000 个测试会话上测的，每个会话的最后一次点击都被藏起来。') }
		},
		metrics: {
			test_hit5: { label: I('Test: the hidden next click is in the top 5', '测试：被藏起来的下一次点击落在前 5 名里'), fmt: 'pct' },
			test_hit5_markov: { label: I('The same for “what usually follows the last click”', '“最后一次点击之后通常点什么”的同一比例'), fmt: 'pct' },
			test_hit5_pop: { label: I('The same for “most clicked first”', '“点击最多的优先”的同一比例'), fmt: 'pct' },
			test_hit5_shuffled: { label: I('SASRec with each history shuffled', '把每段历史打乱顺序后的 SASRec'), fmt: 'pct' }
		},
		lessonNote: I('Ten steps: your clicks, their rows, which click matters (attention), one score worked out, the causal mask and training, mixing the values, scoring every product, the top 5, why order matters, and the limits.', '十步：你的点击、它们的行、哪次点击重要（注意力）、算出一个分数、因果掩码与训练、混合值、给每件商品打分、前 5 名、为什么顺序重要，以及局限。')
	},
	glossary: [
		{ id: 'sequential-recommendation', term: I('sequential recommendation', '序列推荐'), def: I('Recommending the next item from the order of a user’s recent actions (clicks, views, purchases), the way a language model predicts the next word.', '根据用户最近行为（点击、浏览、购买）的顺序推荐下一个物品，就像语言模型预测下一个词。'), related: ['attention', 'causal-mask', 'next-token-prediction'] },
		{ id: 'hit-rate', term: I('hit rate @k', '命中率 @k'), def: I('The share of test cases whose true next item is among the model’s k best suggestions.', '测试样例中，真实的下一个物品落在模型前 k 个推荐里的比例。'), related: ['sequential-recommendation'] },
		{ id: 'popularity-bias', term: I('popularity bias', '流行度偏差'), def: I('A recommender’s tendency to suggest already popular items more often than people actually go on to choose them, so the rest get even less attention.', '推荐系统倾向于推荐本来就热门的物品，次数多于人们实际会选的，于是其余物品得到的关注更少。'), related: ['cold-start'] }
	],
	catalog: {
		params: I('2,144 learned numbers: a 31 × 16 item table, 12 position rows and one attention block', '2,144 个学来的数：31 × 16 的商品表、12 行位置表和一个注意力块'),
		size: I('About 25 KB: spec, weights and facts', '约 25 KB：规格、权重和事实数据'),
		// the site's row for an exported model (tech-zh.ts has it): the forward pass, checked against PyTorch
		runtime: I('Own JS forward pass (checked against PyTorch)', '自研 JS 前向计算（已与 PyTorch 核对）'),
		hook: I('Your click history as a sequence, read with causal attention.', '把你的点击历史当作序列，用因果注意力来读。'),
		shows: I(
			'Your clicks as item and position rows, which past click matters (attention weights with real numbers), the causal mask and how it trains every position, the next-product scores against the same item table, the top 5, why order matters, and two limits: a new product and popularity bias.',
			'作为商品行和位置行的你的点击、哪次过去的点击重要（带真实数值的注意力权重）、因果掩码以及它怎样让每个位置都参与训练、用同一张商品表算出的下一件商品的分数、前 5 名、为什么顺序重要，以及两个局限：新商品和流行度偏差。'
		),
		prereqs: ['two-tower', 'matrix-factorization']
	},
	// No dataset, model or code from anyone else: the shop and its sessions are synthetic (world.ts). The model follows
	// the published description (ideas only).
	credits: [
		{
			name: 'Kang & McAuley, “Self-Attentive Sequential Recommendation” (ICDM 2018)',
			url: 'https://doi.org/10.1109/ICDM.2018.00035',
			license: 'Article; ideas only (no text, figures, code or data used)',
			usedFor: 'the SASRec model (item and position embeddings, causal self-attention blocks, scores against the shared item embeddings), at toy size',
			attribution: 'Wang-Cheng Kang and Julian McAuley (UC San Diego)'
		}
	]
};

export default manifest;
