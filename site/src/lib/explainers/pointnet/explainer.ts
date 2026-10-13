// pointnet: the package manifest. A tiny PointNet (pipeline/pointnet) on the learner's own points:
//   fwd   the exported model traced on the points (centred and scaled by the local adapter, shapes.ts normalize)
// The lesson also runs the same model (rt.model, cloud.ts variants) on the shuffled table, on the critical rows only and
// with one stray point. Measured numbers (test accuracy, removal and stray-point effects) are in facts.json (stats.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';

const manifest: ExplainerManifest = {
	slug: 'pointnet',
	type: 'model',
	kinds: ['trace'],
	quality: 'reviewed',
	assets: { dir: 'models/pointnet', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'points2d', presetsFromAdapter: true, options: { classes: 1 } },
	output: { view: 'kit.probbars' },
	sources: { fwd: { kind: 'trace' } },
	overview: { layout: 'stack' },
	page: {
		goals: [
			{ en: 'Say how a set of loose points becomes a table of x and y, and why the page centres and scales it first.', zh: '说出一组散点如何变成一张 x、y 数表，以及页面为什么先把它居中、缩放。' },
			{ en: 'Explain why the same small network, run on every point alone, followed by a max, gives an answer that does not depend on the order of the points.', zh: '解释为什么先让同一个小网络单独处理每个点、再取最大值，得到的答案就与点的顺序无关。' },
			{ en: 'Find the critical points of your shape and show that they alone give exactly the same answer.', zh: '找出你的形状的关键点，并说明只用它们就能得到完全相同的答案。' },
			{ en: 'Say why PointNet shrugs off missing points but can be thrown by one stray point.', zh: '说出为什么 PointNet 不怕缺点，却可能被一个离群点带偏。' }
		],
		prequestion: {
			id: 'pointnet-pre',
			question: { en: 'You draw a star as loose points. Which part of the drawing will decide the answer most?', zh: '你用散点画了一个五角星。画里的哪一部分对答案影响最大？' },
			options: [
				{ en: 'Every point counts equally', zh: '每个点都同样重要' },
				{ en: 'The points you drew first', zh: '你最先画的那些点' },
				{ en: 'A few extreme points, such as the tips', zh: '少数几个极端的点，比如尖角' }
			],
			answer: 2,
			explain: {
				en: 'After the max, each feature remembers only the one point where it is largest. Those few winners, the critical points, often sit at tips and corners; the order of drawing plays no part.',
				zh: '取最大值之后，每个特征只记住它最大的那一个点。这些少数赢家，也就是关键点，常常落在尖角和拐角上；画的先后顺序不起作用。'
			}
		},
		tryThis: [
			{
				id: 'square',
				title: { en: 'A square', zh: '一个正方形' },
				setup: { preset: 'square' },
				predict: { en: 'Where will the critical points of a square sit?', zh: '正方形的关键点会落在哪里？' },
				look: { en: 'Count how many of the large dots sit near the four corners.', zh: '数一数大圆点里有多少个靠近四个角。' },
				goto: { step: 'critical' }
			},
			{
				id: 'ring',
				title: { en: 'Erase the inner circle', zh: '擦掉内圈' },
				setup: { preset: 'ring' },
				predict: { en: 'A ring is two circles. Erase the inner one in the input: what will the network answer?', zh: '圆环是两个圆。在输入里擦掉内圈：网络会怎么回答？' },
				look: { en: 'The answer bars, before and after you erase.', zh: '擦掉前后的答案条。' },
				goto: { step: 'output' }
			}
		],
		quiz: [
			{
				q: { en: 'You shuffle the rows of the table of points. What happens to the network’s answer?', zh: '你打乱点表的行顺序。网络的答案会怎样？' },
				options: [
					{ en: 'It stays exactly the same', zh: '完全不变' },
					{ en: 'It changes a little', zh: '变一点点' },
					{ en: 'It changes completely', zh: '完全改变' }
				],
				answer: 0,
				explain: { en: 'Every point goes through the same network alone, and a max ignores order.', zh: '每个点都单独经过同一个网络，而最大值不管顺序。' }
			},
			{
				q: { en: 'What does the classifier at the end read?', zh: '最后的分类器读的是什么？' },
				options: [
					{ en: 'All the points, one after another', zh: '所有的点，一个接一个' },
					{ en: 'One row: the largest value of each feature', zh: '一行：每个特征的最大值' },
					{ en: 'Only the first point', zh: '只有第一个点' }
				],
				answer: 1,
				explain: { en: 'The max squeezes the table into one row, and only that row reaches the classifier.', zh: '最大值把整张表压成一行，只有这一行进入分类器。' }
			}
		],
		misconception: {
			claim: { en: 'A network needs the points in a fixed order, like the pixels of an image.', zh: '网络需要点按固定顺序排好，就像图像的像素一样。' },
			truth: { en: 'PointNet runs one shared network on each point and then takes a max, so any order gives the same answer.', zh: 'PointNet 对每个点用同一个共享网络，再取最大值，所以任何顺序都给出相同的答案。' },
			step: 'order'
		},
		lessonNote: {
			en: 'Ten steps: your points become a table, the order question, the shared network on each point, the max and its critical points, a task, the classifier, the answer, and one stray point.',
			zh: '十步：你的点变成数表，顺序之问，每个点上的共享网络，最大值和它的关键点，一个任务，分类器，答案，以及一个离群点。'
		}
	},
	glossary: [
		{ id: 'point-cloud', term: { en: 'point cloud', zh: '点云' }, def: { en: 'A set of points given only by their coordinates, in no particular order, such as the output of a 3-D scanner.', zh: '只由坐标给出、没有特定顺序的一组点，比如三维扫描仪的输出。' } },
		{ id: 'shared-mlp', term: { en: 'shared MLP', zh: '共享 MLP' }, def: { en: 'One small network applied to every item of a set on its own, with the same weights for every item.', zh: '对集合中的每一项单独使用的同一个小网络，所有项用同一组权重。' }, related: ['mlp'] },
		{ id: 'permutation-invariance', term: { en: 'permutation invariance', zh: '置换不变性' }, def: { en: 'An output that stays the same however the items of the input are reordered.', zh: '无论输入中的各项怎样重新排序，输出都保持不变。' } },
		{ id: 'critical-point', term: { en: 'critical points', zh: '关键点' }, def: { en: 'In PointNet, the points that give the largest value of at least one feature; they alone fix the result of the max.', zh: '在 PointNet 中，至少在一个特征上取得最大值的点；只靠它们就决定了取最大值的结果。' }, related: ['pooling'] }
	],
	credits: [
		{
			name: 'PointNet (Qi, Su, Mo and Guibas, CVPR 2017)',
			url: 'https://arxiv.org/abs/1612.00593',
			license: 'idea only (no code or data used)',
			usedFor: 'the architecture (shared per-point MLP, max pooling, classifier) and the idea of critical points',
			attribution: 'C. R. Qi, H. Su, K. Mo, L. J. Guibas, “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation”'
		}
	],
	catalog: {
		params: { en: '2,373', zh: '2,373' },
		size: { en: '9.3 KB of weights', zh: '权重 9.3 KB' }
	}
};

export default manifest;
