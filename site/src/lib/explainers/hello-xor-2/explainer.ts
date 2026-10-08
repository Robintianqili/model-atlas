// hello-xor-2: the package manifest. A lab-only (devOnly) variant of hello-xor: the same 2-2-1 network on XOR, but
// with a SIGMOID output (a probability) trained with binary cross-entropy. All three kinds:
//   A  fwd     the weights trained in PyTorch (pipeline/hello-xor-2/train.py) and exported with export.py
//   B  train   the same network trained live in the browser (trainer.ts); stuck = a start that gets stuck
//   C  step1   one gradient-descent step, slowed down into phases (process.ts)
//   A on B     live = the live-trained weights traced on the learner's point
// Open it in the Lesson Lab: npm run dev -- --port 5221, then /dev/atlas/lab/hello-xor-2/ (never routed under /m/).
import type { ExplainerManifest } from '../../atlas/types/index.ts';

const manifest: ExplainerManifest = {
	slug: 'hello-xor-2',
	type: 'model',
	kinds: ['trace', 'train', 'process'],
	quality: 'draft',
	devOnly: true, // lab only (/dev/atlas/lab/hello-xor-2/): exempt from the catalog rules, never routed under /m/
	assets: { dir: 'models/hello-xor-2', auto: ['spec.json', 'weights.bin', 'facts.json'] }, // written by pipeline/hello-xor-2/export.py
	input: { adapter: 'xor-point', presets: 'presets.json', start: [1, 0] },
	output: { view: 'layer.tensor' },
	sources: {
		fwd: { kind: 'trace' },
		train: { kind: 'train', trainer: 'xorSigmoid', detail: 'coarse', opt: { seed: 1 } },
		stuck: { kind: 'train', trainer: 'xorSigmoid', detail: 'coarse', opt: { seed: 2 } },
		step1: { kind: 'process', process: 'sigmoidStep', run: 'main', seed: 1 },
		live: { kind: 'live-model', of: 'train' }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			{
				en: 'Follow one point through a 2-2-1 network with a sigmoid output and read its answer as a probability.',
				zh: '跟着一个点走过带 sigmoid 输出的 2-2-1 网络，把它的回答读成一个概率。'
			},
			{
				en: 'Do one gradient-descent step by hand with binary cross-entropy: loss, gradient, update.',
				zh: '用二元交叉熵亲手做一步梯度下降：损失、梯度、更新。'
			},
			{ en: 'Explain why a network this small can get stuck, and one way out.', zh: '解释这么小的网络为什么会卡住，以及一种解决办法。' }
		],
		prequestion: {
			id: 'hello-xor-2-pre',
			question: {
				en: 'The last layer gives one score, any number. What does a sigmoid after it change?',
				zh: '最后一层给出一个分数，可以是任意数。在它后面加一个 sigmoid 会改变什么？'
			},
			options: [
				{ en: 'It squashes the score into 0 to 1, so it reads as a probability', zh: '它把分数压到 0 到 1 之间，于是可以读成概率' },
				{ en: 'It makes the network able to learn XOR', zh: '它让网络能够学会 XOR' }
			],
			answer: 0,
			explain: {
				en: 'The sigmoid is a fixed squashing curve on the last number; it keeps the order of scores. What lets the network learn XOR is the hidden layer, as in hello-xor.',
				zh: 'sigmoid 只是作用在最后那个数上的一条固定的压缩曲线，它不改变分数的大小顺序。让网络能学会 XOR 的是隐藏层，和 hello-xor 一样。'
			}
		},
		misconception: {
			claim: { en: 'Training always finds the answer if you run it long enough.', zh: '只要训练得够久，总能找到答案。' },
			truth: {
				en: 'A tiny network can get stuck in a bad spot from some starting weights; more steps do not help there.',
				zh: '从某些起始权重出发，小网络会卡在不好的位置；再多的步数也没用。'
			},
			step: 'stuck'
		},
		lessonNote: {
			en: 'The trained network on your point, read as a probability; one slowed-down training step with cross-entropy; then live training and its limits.',
			zh: '训练好的网络如何处理你的点并给出概率；放慢的一步交叉熵训练；然后是实时训练和它的局限。'
		}
	},
	// No third-party data, model or code: the XOR truth table and a network written here (adapter copied from hello-xor).
	credits: []
};

export default manifest;
