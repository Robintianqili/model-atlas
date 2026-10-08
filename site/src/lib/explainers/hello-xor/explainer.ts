// hello-xor: the package manifest (design §4.9). A lab-only (devOnly) package that covers all three kinds with one
// tiny network, a 2-2-1 tanh MLP on XOR:
//   A  fwd     the weights trained in PyTorch (pipeline/hello-xor/train.py) and exported with export.py
//   B  train   the same network trained live in the browser (trainer.ts); stuck = a start that gets stuck
//   C  step1   one gradient-descent step, slowed down into phases (process.ts)
//   A on B     live = the live-trained weights traced on the learner's point
// Open it in the Lesson Lab: npm run dev -- --port 5214, then /dev/atlas/lab/hello-xor/ (never routed under /m/).
import type { ExplainerManifest } from '../../atlas/types/index.ts';

const manifest: ExplainerManifest = {
	slug: 'hello-xor',
	type: 'model',
	kinds: ['trace', 'train', 'process'],
	quality: 'draft',
	devOnly: true,
	assets: { dir: 'models/hello-xor', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'xor-point', presets: 'presets.json', start: [1, 0] },
	output: { view: 'layer.tensor' },
	sources: {
		fwd: { kind: 'trace' },
		train: { kind: 'train', trainer: 'xor', detail: 'coarse', opt: { seed: 5 } },
		stuck: { kind: 'train', trainer: 'xor', detail: 'coarse', opt: { seed: 1 } },
		step1: { kind: 'process', process: 'sgdStep', run: 'main', seed: 1 },
		live: { kind: 'live-model', of: 'train' }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			{ en: 'Follow one point through a 2-2-1 network and say what each of its 9 weights and biases does.', zh: '跟着一个点走过 2-2-1 网络，说出它的 9 个权重和偏置各自的作用。' },
			{ en: 'Do one gradient-descent step by hand: loss, gradient, update.', zh: '亲手做一步梯度下降：损失、梯度、更新。' },
			{ en: 'Explain why a network this small can get stuck, and one way out.', zh: '解释这么小的网络为什么会卡住，以及一种解决办法。' }
		],
		prequestion: {
			id: 'hello-xor-pre',
			question: { en: 'Can one weighted sum (a single straight line) separate the XOR points?', zh: '一个加权和（一条直线）能把 XOR 的点分开吗？' },
			options: [
				{ en: 'Yes, with the right weights', zh: '能，只要权重合适' },
				{ en: 'No, it needs a hidden layer', zh: '不能，需要一个隐藏层' }
			],
			answer: 1,
			explain: {
				en: 'XOR is true at (0, 1) and (1, 0) but false at (0, 0) and (1, 1): no single line puts the two true points on one side. Two hidden units draw two lines, and the output combines them.',
				zh: 'XOR 在 (0, 1) 和 (1, 0) 为真，在 (0, 0) 和 (1, 1) 为假：没有一条直线能把两个真点放在同一边。两个隐藏单元画两条线，输出再把它们组合起来。'
			}
		},
		tryThis: [
			{
				id: 'between',
				title: { en: 'A point between the corners', zh: '角落之间的点' },
				setup: { input: [0.5, 0.5] },
				predict: { en: 'The network only ever saw the four corners. What will it say for (0.5, 0.5)?', zh: '网络只见过四个角。它对 (0.5, 0.5) 会怎么说？' },
				look: { en: 'Look at the output score: the network still answers, but nothing forced that answer to mean anything.', zh: '看输出分数：网络照样给出回答，但没有任何东西保证这个回答有意义。' },
				goto: { layer: 'fc2' }
			}
		],
		quiz: [
			{
				q: { en: 'How many weights and biases does the 2-2-1 network have?', zh: '这个 2-2-1 网络一共有多少个权重和偏置？' },
				options: [
					{ en: '6', zh: '6' },
					{ en: '9', zh: '9' },
					{ en: '4', zh: '4' }
				],
				answer: 1,
				explain: { en: '2×2 + 2 in the hidden layer, 2 + 1 in the output layer: 9.', zh: '隐藏层 2×2 + 2 个，输出层 2 + 1 个：一共 9 个。' }
			}
		],
		misconception: {
			claim: { en: 'Training always finds the answer if you run it long enough.', zh: '只要训练得够久，总能找到答案。' },
			truth: { en: 'A tiny network can get stuck in a bad spot from some starting weights; more steps do not help there.', zh: '从某些起始权重出发，小网络会卡在不好的位置；再多的步数也没用。' },
			step: 'stuck'
		},
		lessonNote: { en: 'Nine steps: the trained network on your point, one slowed-down training step, then live training and its limits.', zh: '九步：训练好的网络如何处理你的点，一步放慢的训练，然后是实时训练和它的局限。' }
	},
	// No third-party data, model or code: the XOR truth table and a network written here.
	credits: []
};

export default manifest;
