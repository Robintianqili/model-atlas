// mlp-playground: the manifest. A small MLP learns to sort the learner's 2-D points into two classes, live (kind B); one
// point is traced through the live weights (kind A on B). Every source has seed 1: the same network starts the same.
//   run      process 'play' = trainer 'fit': 2 -> [4, 4] -> 2, tanh, Adam lr 0.03, all points every step, 300 steps
//   sgd1     the same network, one step of plain gradient descent (lr 0.5) in fine detail
//   spiral2/4/8   the spiral with 2, 4 or 8 neurons per hidden layer;  lrs  process 'lrPair' on it: lr 0.03 and 3
//   lr0003, lr03, lr3   run at learning rates 0.003, 0.3 and 3 (the Training section's presets)
//   train    'fit' as a train source: the twin's parity run, the page's Training section and the weights pt traces
//   pt       a kind-A trace of the first point through train's weights (zoom.weighted-sum)
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { presetRaw } from '../../atlas/adapters/points2d/adapter.ts';

/** The spiral of the limits step: 300 points on two arms (atlas/data/points2d.ts, seed 1). */
export const SPIRAL = presetRaw('spiral');
const play = (opt: Record<string, unknown> = {}, process = 'play') => ({ kind: 'process' as const, process, run: 'worker' as const, seed: 1, opt });
const L = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'mlp-playground',
	type: 'model',
	kinds: ['train', 'trace'],
	quality: 'reviewed',
	input: { adapter: 'points2d', start: presetRaw('moons') },
	sources: {
		run: play(),
		sgd1: play({ optimizer: 'sgd', lr: 0.5, steps: 1, fine: true }),
		spiral2: play({ data: SPIRAL, hidden: [2, 2] }),
		spiral4: play({ data: SPIRAL, hidden: [4, 4] }),
		spiral8: play({ data: SPIRAL, hidden: [8, 8] }),
		lrs: play({ data: SPIRAL, hidden: [8, 8], lr: 0.03, lrHi: 3 }, 'lrPair'),
		// the learning-rate presets of the Training section, on the learner's points (run is 0.03)
		lr0003: play({ lr: 0.003 }),
		lr03: play({ lr: 0.3 }),
		lr3: play({ lr: 3 }),
		train: { kind: 'train', trainer: 'fit', detail: 'coarse', opt: { seed: 1 } },
		pt: { kind: 'live-model', of: 'train' }
	},
	page: {
		goals: [
			L(
				'Work out what one neuron computes for a point (a weighted sum of x and y plus a bias, squashed by tanh), and explain why over the whole plane that is a soft straight line.',
				'算出一个神经元对一个点做了什么（x 和 y 的加权和加上偏置，再用 tanh 压缩），并解释为什么在整个平面上它是一条“软”直线。'
			),
			L('Explain how a second layer combines straight lines into a curved border.', '解释第二层如何把几条直线组合成一条弯曲的边界。'),
			L('Follow one gradient-descent update with its numbers: loss, gradient, learning rate, new weight.', '带着具体数字走完一次梯度下降更新：损失、梯度、学习率、新权重。'),
			L('Predict when a network is too small, or its learning rate too large, to learn a shape.', '预测网络在什么时候太小、或者学习率太大，以至于学不会一个形状。')
		],
		prequestion: {
			id: 'mlp-playground-pre',
			question: L(
				'Each neuron of the first layer can only draw a straight border. Can such a network separate a disc of points from a ring of points around it?',
				'第一层的每个神经元只能画出一条直的边界。这样的网络能把一团点和围着它的一圈点分开吗？'
			),
			options: [L('No: straight lines only make straight borders', '不能：直线只能组成直的边界'), L('Yes: a second layer can combine the lines into a curved border', '能：第二层可以把这些直线组合成弯曲的边界')],
			answer: 1,
			explain: L(
				'One neuron alone gives a straight (soft) border, but the next layer adds several of them up and squashes the sum again, and that bends the border. Without a hidden layer the network really is stuck with one straight line.',
				'单个神经元只给出一条直的（软）边界，但下一层把好几条加起来、再压缩一次，边界就弯了。如果没有隐藏层，网络确实只能用一条直线。'
			)
		},
		tryThis: [
			{
				id: 'ring',
				title: L('A disc inside a ring', '圆环里的一团点'),
				setup: { input: presetRaw('circles') },
				predict: L('Can four lines in the first layer surround the inner disc?', '第一层的四条线能把中间那团点围起来吗？'),
				look: L('Look at the output square: the border closes into a loop made of the four soft lines.', '看输出方块：边界合成了一个圈，它由四条软直线拼成。'),
				goto: { step: 'combine' }
			}
		],
		quiz: [
			{
				q: L('How many weights and biases does the 2 → 4 → 4 → 2 network have?', '2 → 4 → 4 → 2 这个网络一共有多少个权重和偏置？'),
				options: [L('22', '22'), L('42', '42'), L('16', '16')],
				answer: 1,
				explain: L('Layer 1: 2×4 weights + 4 biases; layer 2: 4×4 + 4; output: 4×2 + 2. That is 12 + 20 + 10 = 42.', '第 1 层：2×4 个权重 + 4 个偏置；第 2 层：4×4 + 4；输出层：4×2 + 2。合计 12 + 20 + 10 = 42。')
			}
		],
		misconception: {
			claim: L('A bigger learning rate always learns faster.', '学习率越大，学得越快。'),
			truth: L(
				'Past some size the steps overshoot: with learning rate 3 the loss jumps far above where it started and stays far above what a small learning rate reaches.',
				'学习率大过某个程度，步子就会迈过头：学习率为 3 时，损失跳到比起点高得多的地方，之后一直远远高于小学习率能达到的水平。'
			),
			step: 'high-lr'
		},
		lessonNote: L(
			'Eleven steps: a neuron, a layer and the whole network on your points, one training step slowed down, then training to the end, a network that is too small and a learning rate that is too big.',
			'十一步：在你的点上看一个神经元、一层、整个网络；放慢看一步训练；然后训练到底；最后看一个太小的网络和一个太大的学习率。'
		)
	},
	catalog: { runtime: L('Own autograd core in a worker', '自研自动微分内核，在 Web Worker 中运行'), params: L('2 → 4 → 4 → 2 by default (42 parameters)', '默认 2 → 4 → 4 → 2（42 个参数）') },
	// layout ideas only (neuron heatmaps in columns, weights as colored edges); no code was copied
	credits: [
		{
			name: 'TensorFlow Playground',
			url: 'https://github.com/tensorflow/playground',
			license: 'Apache-2.0',
			usedFor: 'Layout ideas for the neuron heatmaps and the network diagram; reimplemented on the Model Atlas autograd core',
			attribution: 'Daniel Smilkov and Shan Carter, Google (TensorFlow Playground)'
		}
	]
};

export default manifest;
