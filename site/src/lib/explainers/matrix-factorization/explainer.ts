// matrix-factorization: the package manifest (design §4.9). Kinds B + C, all live in your browser, no download:
//   train   trainer 'fit' (trainer.ts): stochastic gradient descent learns two numbers per person and per movie from every
//           known rating, the learner's included; checked against a PyTorch float64 twin (pipeline/.../twin.py)
//   cold    the same run with the learner's ratings left out: a new user who rated nothing (the limits step)
//   nudge   process 'nudge' (process.ts): one step of that training on one rating, slowed down; checked frame by frame
//           against an independent Python run (pipeline/.../reference.py)
// The data (world.ts) is synthetic: 24 made-up people with hidden tastes and 20 made-up movies of four kinds.
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { PRESETS, toRaw } from './world.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'matrix-factorization',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	input: { adapter: 'ratings', start: PRESETS[0].raw },
	sources: {
		train: { kind: 'train', trainer: 'fit', detail: 'coarse', opt: { seed: 1 } },
		cold: { kind: 'train', trainer: 'fit', detail: 'coarse', opt: { seed: 1, cold: true } },
		nudge: { kind: 'process', process: 'nudge', run: 'main', seed: 1 }
	},
	page: {
		goals: [
			I('Compute a predicted rating as the dot product of a person’s two numbers and a movie’s two numbers, and say what a positive or a negative result means.', '把一个人的两个数和一部电影的两个数做点积，算出预测评分，并说出结果为正或为负意味着什么。'),
			I('Work out one gradient-descent step for one known rating with its numbers: the error, the gradient, the new numbers.', '针对一个已知评分，带着数字走完一步梯度下降：误差、梯度、新的数。'),
			I('Explain why movies of the same kind end up close together on the map although training never saw the kinds.', '解释为什么训练从没见过电影类型，同类电影在图上却会聚到一起。'),
			I('Explain why a new user who has rated nothing gets no useful recommendations (cold start).', '解释为什么一个还没评过分的新用户得不到有用的推荐（冷启动）。')
		],
		prequestion: {
			id: 'matrix-factorization-pre',
			question: I('A site suggests movies you have never rated. To guess your rating of one, what must it be told about the movie?', '一个网站给你推荐你从没评过分的电影。要猜你会给其中一部打几分，它必须事先知道这部电影的什么？'),
			options: [I('Its kind (comedy, crime, …), entered by hand', '它的类型（喜剧、犯罪……），由人工录入'), I('Nothing: the ratings that people gave are enough', '什么都不用：大家打过的分就够了'), I('Its title and plot', '它的片名和剧情')],
			answer: 1,
			explain: I('Matrix factorization reads only the table of ratings. From who liked what, it learns two numbers per movie, and movies of one kind end up with similar numbers anyway.', '矩阵分解只读评分表。它从“谁喜欢什么”里为每部电影学出两个数，同一类型的电影自然会得到相近的数。')
		},
		tryThis: [
			{
				id: 'dark',
				title: I('A taste for crime', '偏爱犯罪片'),
				setup: { input: toRaw(PRESETS[1].raw) },
				// claims.test.ts checks every statement of the look on this profile
				predict: I('This profile likes three crime films and hates a comedy. Which kind will fill its top recommendations?', '这份口味喜欢三部犯罪片，讨厌一部喜剧。它的推荐榜前几名会是哪一类？'),
				look: I(
					'“Your recommendations” above: the top three are sci-fi, not crime. Three of the five crime films are rated already, and only unrated movies are recommended. On the map below, the arrow points between the crime dots and the sci-fi dots, nearer the sci-fi ones (the profile also gave a sci-fi film +1). The model never sees the word “crime”, only ratings.',
					'看上面的“给你的推荐”：前三名都是科幻片，不是犯罪片。五部犯罪片里已经评了三部，而推荐只从没评过的电影里挑。在下面的图上，箭头指在犯罪片的点和科幻片的点之间，更靠近科幻片（这份口味还给一部科幻片打了 +1）。模型从没见过“犯罪”这个词，只见过评分。'
				),
				goto: { step: 'recs' }
			}
		],
		quiz: [
			{
				q: I('Your two numbers are (1, 2) and a movie’s are (3, −1). What rating does the model predict?', '你的两个数是 (1, 2)，一部电影的两个数是 (3, −1)。模型预测的评分是多少？'),
				options: [I('1', '1'), I('5', '5'), I('−2', '−2')],
				answer: 0,
				explain: I('Multiply the matching numbers and add: 1 × 3 + 2 × (−1) = 3 − 2 = 1.', '对应的数相乘再相加：1 × 3 + 2 × (−1) = 3 − 2 = 1。')
			}
		],
		misconception: {
			claim: I('The recommender knows which movies are comedies.', '推荐系统知道哪些电影是喜剧。'),
			truth: I('It never sees a kind, a title or a plot, only the ratings. Movies of one kind still land close together, because the same people like them.', '它从没见过类型、片名或剧情，只见过评分。同类电影还是会落到一起，因为喜欢它们的是同一批人。'),
			step: 'factors'
		},
		sections: { titles: { output: I('Your recommendations', '给你的推荐'), training: I('Training: the map and its error', '训练：地图和误差') } },
		lessonNote: I('Nine steps: the rating table with its holes, your own ratings, two numbers per person and per movie, the dot product, one training step with its numbers, training to the end, your recommendations, and a new user nobody knows.', '九步：带空格的评分表、你自己的评分、每个人和每部电影的两个数、点积、带数字的一步训练、训练到底、给你的推荐，以及一个谁也不了解的新用户。')
	},
	glossary: [
		{ id: 'rating-matrix', term: I('rating matrix', '评分矩阵'), def: I('A table with one row per person and one column per item; a cell holds that person’s rating of that item, and most cells are empty.', '每个人一行、每个物品一列的表；格子里是这个人给这个物品的评分，大多数格子是空的。') },
		{ id: 'latent-factors', term: I('latent factors', '隐因子'), def: I('The few learned numbers that describe each person and each item in matrix factorization; nobody names them, training finds them.', '矩阵分解中描述每个人、每个物品的少数几个学出来的数；没有人给它们命名，是训练找出来的。'), related: ['rating-matrix'] },
		{ id: 'dot-product', term: I('dot product', '点积'), def: I('Multiply two lists of numbers entry by entry and add the products: (a, b)·(c, d) = a·c + b·d.', '把两列数逐项相乘，再把乘积加起来：(a, b)·(c, d) = a·c + b·d。') },
		{ id: 'cold-start', term: I('cold start', '冷启动'), def: I('A recommender knows nothing about a new user or a new item until it has some ratings for it.', '在得到一些评分之前，推荐系统对新用户或新物品一无所知。'), related: ['rating-matrix'] }
	],
	catalog: {
		runtime: I('Own autograd SGD in a worker (checked against PyTorch)', '在 Worker 线程中运行的自研自动求导 SGD（已与 PyTorch 核对）'),
		params: I('(24 people + you + 20 movies) × 2 = 90 numbers', '（24 个人 + 你 + 20 部电影）× 2 = 90 个数'),
		size: I('No files: synthetic ratings made in your browser', '没有文件：合成评分在你的浏览器里生成'),
		// the page as it is (the planned entry promised "your 2-D user vector moving among item factors, the dot product and the loss")
		shows: I(
			'The rating matrix with its holes, your own ratings, two numbers per person and per movie on a map, a prediction as a dot product, one gradient-descent step worked out with its numbers, the map organising itself as training runs, your top recommendations, and cold start.',
			'带空格的评分矩阵、你自己的评分、地图上每个人和每部电影的两个数、作为点积的预测、带着数字算出的一步梯度下降、训练中自己变得有条理的地图、给你的前三名推荐，以及冷启动。'
		)
	},
	// No dataset, model or code from anyone else: the ratings are synthetic (world.ts). The method and the update rule
	// follow the published description of matrix factorization for recommenders (ideas only).
	credits: [
		{
			name: 'Koren, Bell & Volinsky, “Matrix Factorization Techniques for Recommender Systems” (IEEE Computer, 2009)',
			url: 'https://doi.org/10.1109/MC.2009.263',
			license: 'Article; ideas only (no text, figures, code or data used)',
			usedFor: 'the model (a rating as the dot product of a user vector and an item vector) and its stochastic-gradient-descent training',
			attribution: 'Yehuda Koren, Robert Bell and Chris Volinsky'
		}
	]
};

export default manifest;
