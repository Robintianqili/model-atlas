// regularization: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef).
// hook (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { dec, metricsOf, setting } from './data.ts';
import { CNN, shiftPixels } from './cnn.ts';

const L = (en: string, zh: string) => ({ en, zh });

export const hook: I18n = {
	en: 'A network can sort every one of its training points correctly and still be wrong on new ones: it has learned the points, noise and all, instead of the pattern behind them. Here a network trains in your browser on two noisy moons. Watch the moment it starts to overfit, then stop it four ways: a smaller network, a penalty on big weights, dropout and early stopping, plus more or augmented data.',
	zh: '一个网络可以把它的训练点全部分对，却在新的点上出错：它连同噪声一起记住了这些点，而没有学到点背后的规律。这里，一个神经网络在你的浏览器里学习两个带噪声的月牙。先看它从哪一刻开始过拟合，再用几种办法阻止它：更小的网络、给大权重加惩罚、dropout 和早停，以及更多或增强过的数据。'
};

export const toy: StepDef = {
	id: 'toy',
	title: L('Your knobs, your network', '你的旋钮，你的网络'),
	source: 'run',
	at: 'last',
	view: 'regularization/field',
	provenance: 'live',
	narration: {
		en: 'Your knobs give {nTrain} training points with noise {noise} and a network with {width} units per layer, λ = {lam} and p = {p}. After {n} updates its training loss is {train} and its test loss {test}. Change a knob and it trains again from the start.',
		zh: '你的旋钮给出 {nTrain} 个噪声为 {noise} 的训练点，以及一个每层 {width} 个单元、λ = {lam}、p = {p} 的网络。更新 {n} 次之后，它的训练损失是 {train}，测试损失是 {test}。改动任意一个旋钮，它都会从头重新训练。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			return { nTrain: m.nTrain, noise: setting(m.noise), width: m.width, lam: setting(m.l2), p: setting(m.dropout), n: m.step, train: dec(m.trainLoss), test: m.testLoss };
		}
	},
	look: L('The border, the rings with a second ring (test points it gets wrong) and the two losses in the line of numbers.', '分界线、外面多一个圈的测试点（被分错的），以及那行数字里的两个损失。')
};

export const real: StepDef = {
	id: 'real',
	title: L('A real model: how TinyCNN was trained', '真实的模型：TinyCNN 是怎么训练的'),
	source: 'cnn',
	at: 'last',
	// curve1d: an epoch axis and a marked point (the kept epoch), which loss-curve has not
	view: {
		key: 'curve1d',
		props: (_rt, at) => ({
			series: [{ y: { shape: [CNN.epochs, 2], data: Float64Array.from(CNN.valAcc.flatMap((v, i) => [i + 1, v])) }, label: L('validation accuracy', '验证准确率'), kind: 'model' }],
			xLabel: L('epoch (counting from 1)', '轮次（从 1 数起）'),
			yLabel: L('validation accuracy', '验证准确率'),
			yRange: [0.7, 1],
			highlight: metricsOf(at).bestEpoch - 1,
			legend: true
		})
	},
	provenance: 'precomputed',
	terms: ['epoch', 'dataset-split'],
	narration: {
		en: 'TinyCNN, the drawing classifier of this site, used three of these tools. For {epochs} epochs it saw every drawing shifted by up to {shift} pixels, resized and turned by up to {rot} degrees, many also drawn thicker or thinner, with weight decay {wd} on its weights. Like early stopping, the site keeps the weights from epoch {best}, where its accuracy on held-out validation drawings peaked at {val|pct}; on its test drawings it scores {test|pct}.',
		zh: 'TinyCNN 是本站的手绘图分类器，它用了其中三种办法。在 {epochs} 轮训练里，它看到的每张图都被平移最多 {shift} 个像素、随机缩放、旋转最多 {rot} 度，不少图的笔画还被加粗或变细；它的权重还带有 {wd} 的权重衰减。和早停一样，网站保留的是第 {best} 轮的权重，那时它在留出的验证图上准确率最高，达到 {val|pct}；在测试图上，它的准确率是 {test|pct}。',
		vars: (_rt, at) => ({ epochs: CNN.epochs, shift: shiftPixels, rot: CNN.rotDeg, wd: setting(CNN.weightDecay), best: metricsOf(at).bestEpoch, val: CNN.valAcc[metricsOf(at).bestEpoch - 1], test: CNN.testAcc })
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: L('Five fixes, one starting point', '五种办法，同一个起点'),
	source: 'methods',
	at: 'last',
	view: {
		key: 'loss-curve',
		props: () => ({
			metrics: ['testLoss', 'testLossL2', 'testLossDrop', 'testLossAug', 'testLossSmall'],
			labels: {
				testLoss: L('plain', '原样'),
				testLossL2: L('L2 penalty', 'L2 惩罚'),
				testLossDrop: L('dropout', 'dropout'),
				testLossAug: L('nudged points', '挪动的点'),
				testLossSmall: L('small network', '小网络')
			}
		})
	},
	provenance: 'live',
	narration: {
		en: 'The same {nTrain} training points, the same first weights, {n} updates; every line is a test loss. At the end: plain {plain}, L2 penalty {l2}, dropout {drop}, nudged points {aug}, small network {small}. Each fix ends lower than the plain run; early stopping would keep the plain run\'s own lowest point, {best}.',
		zh: '同样的 {nTrain} 个训练点、同样的初始权重、{n} 次更新；每条线都是测试损失。最后：原样 {plain}，L2 惩罚 {l2}，dropout {drop}，挪动的点 {aug}，小网络 {small}。每种办法最后都比原样的训练低；早停则会保留原样训练自己的最低点 {best}。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			return { nTrain: m.nTrain, n: m.step, plain: m.testLoss, l2: m.testLossL2, drop: m.testLossDrop, aug: m.testLossAug, small: m.testLossSmall, best: m.best };
		}
	}
};

export const pitfalls = [
	{
		claim: L('As long as the training loss falls, more training helps.', '只要训练损失还在下降，多训练就有好处。'),
		truth: L(
			'Not on new points. Here the test loss turns back up long before the training loss stops falling; early stopping keeps the weights from its lowest point.',
			'在新的点上不是这样。这里测试损失早在训练损失停止下降之前就开始回升；早停会保留它最低点时的权重。'
		),
		step: 'early-stop'
	},
	{
		claim: L('Dropout stays on when the network is used.', '使用网络时，dropout 也还开着。'),
		truth: L(
			'It acts only during training. When the network is tested or used, every unit is on, and the scaling by 1/(1 − p) during training keeps the totals the same on average.',
			'它只在训练时起作用。测试或使用网络时，所有单元都开着；训练时乘以 1/(1 − p) 的缩放保证了总和平均不变。'
		),
		step: 'dropout'
	},
	{
		claim: L('Pick λ, p and the stopping point by the test loss.', '用测试损失来挑选 λ、p 和停止的时间点。'),
		truth: L(
			'Settings picked by the test loss are tuned to the test points, so the test loss no longer measures new data fairly. In practice a third set, the validation set, picks them; this page uses the test points for both only to stay small.',
			'按测试损失挑出来的设置是针对测试点调好的，测试损失就不再能公平地衡量新数据了。实际中由第三组数据，也就是验证集，来挑选这些设置；本页把测试点同时用于这两件事，只是为了保持简单。'
		),
		step: 'early-stop'
	},
	{
		claim: L('More regularization is always safer.', '正则化越强越保险。'),
		truth: L(
			'Too much and the network underfits: with the largest λ on the knob, the training loss and the test loss both end high.',
			'太强了网络就会欠拟合：把 λ 调到旋钮上的最大值，训练损失和测试损失最后都停在高处。'
		),
		step: 'too-much'
	},
	{
		claim: L('Weight decay and an L2 penalty in the loss are always the same thing.', '权重衰减和在损失里加 L2 惩罚，永远是一回事。'),
		truth: L(
			"With plain gradient descent they are: the penalty's slope 2λw shrinks every weight by the same share at every update. Adam rescales every gradient, the penalty's too, so there the two differ; AdamW, which TinyCNN used, shrinks the weights directly instead.",
			'用普通的梯度下降时是一回事：惩罚项的斜率 2λw 在每次更新时把每个权重按同样的比例缩小。Adam 会重新缩放每个梯度，惩罚项的梯度也不例外，所以用 Adam 时两者就不一样了；TinyCNN 用的 AdamW 则直接缩小权重。'
		),
		step: 'l2'
	}
];

export const further = [
	{ label: L('Dive into Deep Learning: weight decay (free online book)', '《动手学深度学习》：权重衰减（免费在线书）'), href: 'https://d2l.ai/chapter_linear-regression/weight-decay.html' },
	{ label: L('Dive into Deep Learning: dropout', '《动手学深度学习》：暂退法（dropout）'), href: 'https://d2l.ai/chapter_multilayer-perceptrons/dropout.html' },
	{ label: L('Loshchilov and Hutter (2017), Decoupled weight decay regularization (AdamW)', 'Loshchilov 和 Hutter（2017），解耦权重衰减正则化（AdamW）'), href: 'https://arxiv.org/abs/1711.05101' },
	{ label: L('Srivastava et al. (2014), Dropout: a simple way to prevent neural networks from overfitting', 'Srivastava 等（2014），Dropout：防止神经网络过拟合的简单方法'), href: 'https://jmlr.org/papers/v15/srivastava14a.html' },
	{ label: L('TensorFlow Playground: noise, train ratio and regularization sliders', 'TensorFlow Playground：噪声、训练比例和正则化滑块'), href: 'https://playground.tensorflow.org' },
	{ label: L('MLU-Explain: double descent', 'MLU-Explain：双下降'), href: 'https://mlu-explain.github.io/double-descent/' }
];
