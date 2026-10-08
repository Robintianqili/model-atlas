// classic-ml: the manifest. Three classic algorithms on one canvas of the learner's points (the kit's points2d input,
// up to three classes): k nearest neighbours, a decision tree (CART, Gini) and k-means. Kind C only: every stage is a
// process from process.ts, run live in the browser on dataOf(input) and checked frame by frame against an independent
// Python run (pipeline/classic-ml/reference.py). No data, model or code from anyone else.
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { dataOf, I, RINGS, START } from './ml.ts';

const on = (process: string, o: { run?: 'main' | 'worker'; seed?: number; opt?: Record<string, unknown> } = {}) => ({ kind: 'process' as const, process, run: o.run ?? ('main' as const), seed: o.seed ?? 1, opt: o.opt, input: (rt: { input: unknown }) => dataOf(rt.input) });

const manifest: ExplainerManifest = {
	slug: 'classic-ml',
	type: 'model',
	kinds: ['process'],
	quality: 'reviewed',
	input: { adapter: 'points2d', start: START, options: { classes: 3 } },
	sources: {
		knn: on('knn'),
		map: on('knnMap', { run: 'worker' }),
		tree: on('tree', { run: 'worker' }),
		km: on('kmeans', { seed: 3 }),
		rings: { ...on('kmeans', { seed: 3, opt: { sweep: 6 } }), input: () => dataOf(RINGS) }
	},
	page: {
		goals: [
			I('Classify a new point with k nearest neighbours by hand: its distances, the k nearest, the vote; and say how k changes the decision map.', '亲手用 k 近邻给一个新点分类：算距离、找最近的 k 个、投票；并说出 k 怎样改变决策图。'),
			I('Compute the Gini impurity of a set of points and pick the best split of a decision tree from the weighted impurity of its two sides.', '算出一组点的基尼不纯度，并用两侧的加权不纯度挑出决策树最好的切分。'),
			I('Find the depth where a tree’s accuracy on held-out points peaks, and explain why a deeper tree can do worse on new points.', '找出让树在留出点上准确率最高的深度，并解释为什么更深的树在新点上可能更差。'),
			I('Run k-means by hand for one round (assign, then update), and name two things it cannot do.', '亲手做一轮 k-means（先分配，再更新），并说出它做不到的两件事。')
		],
		prequestion: {
			id: 'classic-ml-pre',
			question: I('A new point lands on your canvas. Without any training at all, what is a sensible way to guess its class?', '一个新点落在你的画布上。完全不训练，怎样猜它的类别才合理？'),
			options: [I('Look at the points closest to it and take the most common class', '看离它最近的那几个点，取最常见的类别'), I('Take the most common class on the whole canvas', '取整张画布上最常见的类别'), I('You cannot guess without training a model', '不训练模型就没法猜')],
			answer: 0,
			explain: I('That is k nearest neighbours: its “model” is the stored points themselves. The whole-canvas majority ignores where the new point is.', '这就是 k 近邻：它的“模型”就是存下来的点本身。整张画布的多数类完全不管新点在哪里。')
		},
		tryThis: [
			{ id: 'rings', title: I('A disc inside a ring', '圆环里的一团点'), setup: { input: { ...RINGS, k: 5, depth: 8, kc: 2 } }, predict: I('Can a tree, made only of straight cuts parallel to the axes, carve out a round disc?', '一棵只会沿坐标轴方向直切的树，能切出一个圆盘吗？'), look: I('Look at the boxes in the depth step: a staircase that follows the circle.', '看“深度”那一步里的方框：一道沿着圆走的阶梯。'), goto: { step: 'tree-depth' } },
			{ id: 'xor', title: I('XOR quadrants', '异或象限'), setup: { input: { preset: { name: 'xor', n: 200, seed: 1 }, k: 5, depth: 8, kc: 4 } }, predict: I('With four groups in four corners, will k-means with k = 4 put one centroid in each corner?', '四团点在四个角上，k = 4 的 k-means 会在每个角放一个质心吗？'), look: I('Look at where the centroids end up, and at the total squared distance.', '看质心最后停在哪里，以及总平方距离。'), goto: { step: 'km-repeat' } }
		],
		quiz: [
			{ q: I('Two classes on one side of a split hold 30 and 10 points. What is their Gini impurity?', '切分某一侧有两类，分别 30 个点和 10 个点。它的基尼不纯度是多少？'), options: [I('0.375', '0.375'), I('0.25', '0.25'), I('0.5', '0.5')], answer: 0, explain: I('p = 0.75 and 0.25, so G = 1 − 0.75² − 0.25² = 1 − 0.5625 − 0.0625 = 0.375.', 'p = 0.75 和 0.25，所以 G = 1 − 0.75² − 0.25² = 1 − 0.5625 − 0.0625 = 0.375。') },
			{ q: I('Which of the three algorithms never looks at the class colours?', '三种算法里，哪一种从来不看类别颜色？'), options: [I('k nearest neighbours', 'k 近邻'), I('The decision tree', '决策树'), I('k-means', 'k-means')], answer: 2, explain: I('k-means is unsupervised: it groups points by distance alone. kNN votes with the classes and the tree measures their impurity.', 'k-means 是无监督的：它只按距离分组。kNN 用类别投票，决策树用类别算不纯度。') }
		],
		misconception: { claim: I('A deeper tree is always a better tree.', '树越深越好。'), truth: I('A deeper tree fits the training points ever better, but past some depth it carves boxes around single noisy points, and its accuracy on new points falls.', '更深的树对训练点拟合得越来越好，但过了某个深度，它开始给个别带噪声的点单独切方框，在新点上的准确率反而下降。'), step: 'tree-depth' },
		lessonNote: I('Twelve steps in three short chapters, all computed live on your points: k nearest neighbours, a decision tree, k-means.', '十二步，分三个短章节，全部在你的点上实时计算：k 近邻、决策树、k-means。')
	},
	glossary: [
		{ id: 'knn', term: I('k nearest neighbours (kNN)', 'k 近邻（kNN）'), def: I('A classifier that stores the training points and answers with the most common class among the k points nearest to a new point.', '一种分类器：把训练点存起来，对新点取离它最近的 k 个点中最常见的类别作为回答。'), related: ['decision-boundary'] },
		{ id: 'decision-boundary', term: I('decision boundary, decision map', '决策边界、决策图'), def: I('The map of a classifier’s answer at every place of the input space; the boundary is where the answer changes.', '分类器在输入空间每一处给出的回答画成的图；边界就是回答发生变化的地方。'), related: ['classifier'] },
		{ id: 'decision-tree', term: I('decision tree', '决策树'), def: I('A model that asks yes/no questions about one feature at a time (“is x ≤ 0.3?”); each answer leads to the next question, and the last box gives the class.', '一次只就一个特征问一个是非题（“x ≤ 0.3 吗？”）的模型；每个回答通向下一个问题，最后落到的方框给出类别。'), related: ['gini'] },
		{ id: 'gini', term: I('Gini impurity', '基尼不纯度'), def: I('1 − Σ p², where p is the share of each class in a set of points: 0 when one class only, larger when the classes are mixed.', '1 − Σ p²，其中 p 是每个类别在一组点中的占比：只有一类时为 0，类别越混越大。'), related: ['decision-tree'] },
		{ id: 'clustering', term: I('clustering', '聚类'), def: I('Grouping points by similarity without any labels (unsupervised learning).', '在没有任何标签的情况下按相似程度给点分组（无监督学习）。'), related: ['k-means'] },
		{ id: 'k-means', term: I('k-means', 'k-means（k 均值）'), def: I('A clustering method that repeats two steps: every point joins its nearest centroid, then every centroid moves to the mean of its points.', '一种聚类方法，反复做两步：每个点归到最近的质心，然后每个质心移到它那组点的平均位置。'), related: ['centroid', 'clustering'] },
		{ id: 'centroid', term: I('centroid', '质心'), def: I('The centre of a group of points: the mean of their x values and the mean of their y values.', '一组点的中心：它们 x 的平均值和 y 的平均值。'), related: ['k-means'] }
	],
	catalog: {
		hook: I('Draw your own dataset and watch k nearest neighbours, a decision tree and k-means work on it, step by step.', '画出你自己的数据集，一步步看 k 近邻、决策树和 k-means 怎样处理它。'),
		shows: I('kNN (distances, the vote, decision maps as k changes), a CART decision tree (Gini impurity, the best split, the tree growing box by box, depth versus overfitting) and k-means (assign, update, repeat; its limits) on one shared “draw your own dataset” canvas, every number computed live.', '在同一张“画你自己的数据集”画布上：kNN（距离、投票、随 k 变化的决策图）、CART 决策树（基尼不纯度、最好的切分、树逐个方框地生长、深度与过拟合）和 k-means（分配、更新、重复，以及它的局限），每个数都实时算出。'),
		runtime: I('Own TypeScript, live in the browser (no neural network)', '自研 TypeScript，在浏览器中实时运行（无神经网络）'),
		params: I('none learned by gradient: kNN stores the points, the tree its splits, k-means its centroids', '没有用梯度学的参数：kNN 存下所有点，决策树存它的切分，k-means 存它的质心')
	},
	credits: [{ name: 'R2D3, A visual introduction to machine learning (2015)', url: 'http://www.r2d3.us/visual-intro-to-machine-learning-part-1/', license: 'Ideas only (no text, figures or code used)', usedFor: 'the idea of showing each tree split on the data next to the growing tree', attribution: 'Stephanie Yee and Tony Chu (R2D3)' }]
};

export default manifest;
