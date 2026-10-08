// flow-2d: the manifest. 2-D flow matching, trained live in the package worker on the learner's points (kind B), with
// Euler sampling as part of the run (kind C) and one pair traced through the network's weights (kind A on live weights).
// Every train source runs process 'train:flow' (process.ts: trainer 'fit', then Euler sampling) with seed 1, so all
// start from the same weights and draw the same first batch:
//   flow        Adam 1e-3, batch 256, 1,500 steps, a snapshot every 100; then 400 noise points, Euler K = 50 and K = 4
//   sgd1        one plain gradient-descent step (lr 0.5) in fine detail: the batch, the loss, the gradients, the update
//   t0          no training (steps 0): the starting weights, for the live-model source `job`
//   job         a kind-A trace of training pair 0's (x, y, t) through t0's weights (zoom.weighted-sum)
//   eulerCheck  20 Adam steps + Euler on the first 200 points: the run pipeline/flow-2d/reference.py repeats in PyTorch
import type { ExplainerManifest, ExplainerRuntime } from '../../atlas/types/index.ts';
import { presetOf } from './adapters/points2d/adapter.ts';
import { dataOf } from './flow.ts';

const L = (en: string, zh: string) => ({ en, zh });
const train = (opt: Record<string, unknown> = {}, detail: 'coarse' | 'fine' = 'coarse') => ({ kind: 'train' as const, trainer: 'flow', detail, opt: { seed: 1, ...opt } });

const manifest: ExplainerManifest = {
	slug: 'flow-2d',
	type: 'model',
	kinds: ['train', 'process', 'trace'],
	quality: 'reviewed',
	input: { adapter: 'points2d', presets: 'presets.json', start: presetOf('smiley'), options: { classes: 1 } },
	sources: {
		flow: { ...train(), precomputed: 'tapes/flow' },
		sgd1: train({ optimizer: 'sgd', lr: 0.5, steps: 1 }, 'fine'),
		t0: train({ steps: 0, sample: false }),
		job: { kind: 'live-model', of: 't0' },
		eulerCheck: {
			kind: 'process',
			process: 'eulerCheck',
			input: (rt: ExplainerRuntime) => ({ xy: Array.from(dataOf(rt.input).xy.slice(0, 400)) }),
			opt: { steps: 20, snapshotEvery: 10 },
			seed: 1,
			recordDraws: true
		}
	},
	page: {
		goals: [
			L(
				'Build one training pair by hand: a noise point x₀, a data point x₁, a time t and the point x_t on the straight line between them.',
				'亲手搭出一个训练对：噪声点 x₀、数据点 x₁、时间 t，以及它们连线上的点 x_t。'
			),
			L('Say what the network must output for (x, y, t), and work out the squared-error loss of one pair.', '说出网络对 (x, y, t) 要输出什么，并算出一个训练对的平方误差损失。'),
			L('Follow Euler’s method from noise to new points, and explain why a few big steps pull the points toward the middle.', '沿着欧拉法从噪声走到新的点，并解释为什么几大步会把点拉向中间。')
		],
		prequestion: {
			id: 'flow-2d-pre',
			question: L('A network will turn random noise into new points shaped like your drawing. What does it learn?', '一个网络要把随机噪声变成形状像你的画的新点。它学的是什么？'),
			options: [L('Where each of your points is, so it can copy them', '你的每个点在哪里，好把它们复制出来'), L('Which way to push a point, at every place and time', '在每个位置、每个时刻，该把点往哪边推')],
			answer: 1,
			explain: L(
				'It learns a velocity: an arrow for every place and time. Following the arrows carries noise onto the shape; the network keeps only its weights, not your points.',
				'它学的是速度：每个位置、每个时刻都有一个箭头。顺着箭头走，噪声就被带到形状上；网络只保存自己的权重，不保存你的点。'
			)
		},
		tryThis: [
			{
				id: 'ring',
				title: L('A ring', '一个圆环'),
				setup: { preset: 'ring' },
				predict: L('Noise fills a disc. Will the 400 points spread evenly around the ring, or bunch up?', '噪声铺成一个圆盘。这 400 个点会沿圆环均匀散开，还是挤成一团？'),
				look: L('Look at the pink points at t = 1: they spread around the whole ring. Noise from outside moves in; noise from the middle moves out.', '看 t = 1 时的粉色点：它们散布在整个圆环上。外面的噪声往里走，中间的噪声往外走。'),
				goto: { step: 'euler' }
			}
		],
		quiz: [
			{
				q: L('Which three numbers does the network read for one point?', '对一个点，网络读入哪三个数？'),
				options: [L('Its x, its y and the time t', '它的 x、y 和时间 t'), L('The noise x₀, the data point x₁ and t', '噪声 x₀、数据点 x₁ 和 t'), L('x, y and the number of its pair', 'x、y 和它所在配对的编号')],
				answer: 0,
				explain: L('Only where the point is and when. It never sees x₀ or x₁, so where lines cross it has to learn the average arrow.', '只有点在哪里、在什么时刻。它看不到 x₀ 和 x₁，所以在直线交叉的地方，它只能学到平均的箭头。')
			}
		],
		misconception: {
			claim: L('The model stores your points and copies them.', '模型把你的点存起来，再复制出来。'),
			truth: L('It keeps only the network’s weights, not your points. Its samples are new points that land near the shape, not copies of the data.', '它只保存网络的权重，不保存你的点。它采出的是落在形状附近的新点，而不是数据的复制品。'),
			step: 'euler'
		},
		lessonNote: L(
			'Twelve steps: training pairs built from your points and noise, the untrained network, one training step, training to the end, then Euler sampling and what four big steps do.',
			'十二步：用你的点和噪声搭训练对，看未训练的网络，放慢看一步训练，训练到底，然后用欧拉法采样，再看四大步会怎样。'
		),
		sections: {
			titles: { training: L('Training on your points', '在你的点上训练') },
			intros: { training: L('The loss of every training step of the lesson’s run (one batch of 256 pairs each).', '本课那次训练每一步的损失（每步一批 256 对）。') }
		}
	},
	glossary: [
		{ id: 'generative-model', term: L('generative model', '生成模型'), def: L('A model that learns to produce new examples that look like its training data, instead of labelling them.', '学会造出与训练数据相似的新样本的模型，而不是给样本贴标签。') },
		{ id: 'gaussian-noise', term: L('standard normal noise', '标准正态噪声'), def: L('Random numbers from a bell curve centred on 0 with spread (standard deviation) 1; about 95% of them fall between −2 and 2.', '取自以 0 为中心、宽度（标准差）为 1 的钟形曲线的随机数；大约 95% 落在 −2 和 2 之间。') },
		{ id: 'velocity-field', term: L('velocity field', '速度场'), def: L('An arrow at every place (and time) that says which way, and how fast, a point there moves.', '每个位置（和时刻）都有一个箭头，表示那里的点往哪个方向、以多快的速度移动。') },
		{ id: 'mse', term: L('mean squared error', '均方误差'), def: L('The average of the squared differences between predictions and targets.', '预测值与目标值之差的平方的平均。'), related: ['loss'] },
		{ id: 'euler-method', term: L('Euler’s method', '欧拉法'), def: L('Following a velocity in small steps: new position = position + step size × velocity.', '沿着速度一小步一小步地走：新位置 = 位置 + 步长 × 速度。'), related: ['velocity-field'] }
	],
	catalog: {
		shows: L('Training pairs on straight lines, the velocity field on a time slider, Euler sampling paths, and 4 steps against 50.', '直线上的训练配对、带时间滑块的速度场、欧拉采样的路径，以及 4 步与 50 步的对比。')
	},
	// ideas only: no text, figures or code of these papers is used
	credits: [
		{
			name: 'Lipman et al., Flow Matching for Generative Modeling (ICLR 2023)',
			url: 'https://arxiv.org/abs/2210.02747',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'the training objective: a network regresses the velocity of paths from noise to data',
			attribution: 'Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le'
		},
		{
			name: 'Liu et al., Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (ICLR 2023)',
			url: 'https://arxiv.org/abs/2209.03003',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'straight paths x_t = (1 − t)·x0 + t·x1 with target x1 − x0, and few-step Euler sampling',
			attribution: 'Xingchao Liu, Chengyue Gong, Qiang Liu'
		}
	]
};

export default manifest;
