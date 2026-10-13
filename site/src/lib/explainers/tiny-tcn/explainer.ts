// tiny-tcn: the manifest. A temporal convolutional network (TCN: causal convolutions with dilations 1, 2, 4, 8 and
// residual blocks) learns the learner's own series live in the browser (kind B), and our own ARIMA(p, 1, q)-lite is
// fitted on the same series as a baseline (kind C, with a Python reference). Every source has seed 1.
//   arima     process 'arima': the series and its changes, ARIMA's fit, its forecast (Python reference)
//   init      trainer 'tcn' with no steps: the starting weights (steps 3 to 7 look inside the network with them)
//   train     trainer 'tcn', 200 steps of Adam on every position at once (twin: pipeline/tiny-tcn/twin.py)
//   forecast  process 'forecast' after 'train': the trained TCN and ARIMA forecast the held-out part
//   long      trainer 'tcn' on a fixed square wave of period 40 (the limits step; the same for every learner)
import type { ExplainerManifest, TrainState } from '../../atlas/types/index.ts';
import { seriesPresetRaw } from '../../atlas/adapters/series/adapter.ts';
import { dataOf, longSquare } from './data.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'tiny-tcn',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/tiny-tcn', auto: [] },
	input: { adapter: 'series', start: seriesPresetRaw('seasonal'), options: { mode: 'curve', presets: ['seasonal', 'sine', 'square', 'trend', 'randomWalk'] } },
	sources: {
		arima: { kind: 'process', process: 'arima', run: 'main', input: (rt) => Array.from(dataOf(rt.input).y) },
		init: { kind: 'train', trainer: 'tcn', detail: 'coarse', opt: { seed: 1, steps: 0 } },
		train: { kind: 'train', trainer: 'tcn', detail: 'coarse', opt: { seed: 1 } },
		forecast: {
			kind: 'process',
			process: 'forecast',
			after: 'train',
			reference: false,
			input: (rt) => {
				const t = rt.sources.train.tape!;
				return { y: Array.from(dataOf(rt.input).y), params: (t.state(t.length - 1) as TrainState).params };
			}
		},
		long: { kind: 'train', trainer: 'tcn', detail: 'coarse', runOnEmpty: true, opt: { seed: 1, data: longSquare() } }
	},
	page: {
		goals: [
			I('Read a causal convolution: each output mixes the current change and one earlier change, never a later one.', '读懂因果卷积：每个输出只混合当前的变化量和一个更早的变化量，从不读后面的。'),
			I('Count a receptive field: with kernel 2 and dilations 1, 2, 4, 8, one output sees exactly 16 changes.', '数出感受野：卷积核为 2、空洞率为 1、2、4、8 时，一个输出恰好看到 16 个变化量。'),
			I('Follow one forecast through the tree of nodes, the residual blocks and the head, with real numbers.', '用真实的数，沿着节点树、残差块和输出头，跟踪一个预测的来历。'),
			I('Compare the TCN with an ARIMA baseline on held-out values, and name a pattern no 16-change reach can catch.', '在留出的值上把 TCN 和 ARIMA 基准比一比，并说出一种 16 个变化量的视野抓不住的规律。')
		],
		prequestion: {
			id: 'tiny-tcn-pre',
			question: I(
				'A network has four layers. Each layer mixes two inputs: the current one and one a few steps back. Can one of its outputs depend on something 15 steps back?',
				'一个网络有四层。每层混合两个输入：当前的一个，和几步之前的一个。它的一个输出能依赖 15 步之前的东西吗？'
			),
			options: [I('No: four layers of two inputs reach at most 4 steps back', '不能：四层、每层两个输入，最多只能看到 4 步之前'), I('Yes, if the layers skip further and further back', '能，只要各层往回跳得越来越远')],
			answer: 1,
			explain: I(
				'Yes. If the layers skip 1, 2, 4 and 8 steps back, every layer doubles the reach: 1 + 1 + 2 + 4 + 8 = 16 values, the current one and 15 before it. That is a dilated convolution, the core of a TCN.',
				'能。如果各层分别往回跳 1、2、4、8 步，每加一层视野就翻一倍：1 + 1 + 2 + 4 + 8 = 16 个值，即当前的一个加上之前的 15 个。这就是空洞卷积，TCN 的核心。'
			)
		},
		tryThis: [
			{
				id: 'square',
				title: I('A square wave', '方波'),
				setup: { input: seriesPresetRaw('square') },
				predict: I('The wave jumps every 8 steps. Which forecaster gets the jumps on time: the TCN, which reads 16 changes, or ARIMA, which reads at most 5?', '这个波每 8 步跳一次。谁能准时预测跳变：读 16 个变化量的 TCN，还是最多读 5 个的 ARIMA？'),
				look: I('Look at the step that compares the two forecasts: only one of them can see the last jump 8 steps back.', '看比较两种预测的那一步：只有其中一个能看到 8 步之前的上一次跳变。'),
				goto: { step: 'compare' }
			},
			{
				id: 'sine',
				title: I('A sine wave', '正弦波'),
				setup: { input: seriesPresetRaw('sine') },
				predict: I('Each change of a sine wave is a fixed mix of the two before it. Can the 441-parameter TCN beat a model with 3 coefficients there?', '正弦波的每个变化量都是前两个变化量的固定组合。在这里，有 441 个参数的 TCN 能胜过只有 3 个系数的模型吗？'),
				look: I('Look at the ARIMA step: it finds the exact rule, so its forecast lies on the truth.', '看 ARIMA 那一步：它找到了精确的规则，所以它的预测正好落在真实值上。'),
				goto: { step: 'arima' }
			}
		],
		quiz: [
			{
				q: I('A TCN has kernel 2 and dilations 1, 2, 4, 8, 16. How many inputs does one output see?', '一个 TCN 的卷积核为 2，空洞率为 1、2、4、8、16。一个输出能看到多少个输入？'),
				options: [I('5', '5'), I('16', '16'), I('32', '32')],
				answer: 2,
				explain: I('1 + 1 + 2 + 4 + 8 + 16 = 32: one more layer doubles the reach of the four-layer network on this page.', '1 + 1 + 2 + 4 + 8 + 16 = 32：多一层，就把本页四层网络的视野翻一倍。')
			},
			{
				q: I('Why must the convolutions of a forecaster be causal?', '为什么预测模型里的卷积必须是因果的？'),
				options: [I('To make training faster', '为了让训练更快'), I('So that no output reads a value from its own future, which is unknown when you forecast', '这样就没有哪个输出会读到它自己“未来”的值，而预测时未来是未知的'), I('Because a kernel of 2 cannot be centred', '因为大小为 2 的卷积核没法居中')],
				answer: 1,
				explain: I('A forecast may only use the past. Padding only at the start keeps every output from reading later values, also during training, where all positions are computed at once.', '预测只能用过去。只在序列开头补零，就能让每个输出都读不到更晚的值，在一次算出所有位置的训练里也是如此。')
			}
		],
		misconception: {
			claim: I('A deep network always beats a classic statistical model.', '深度网络总是胜过经典的统计模型。'),
			truth: I(
				'On a sine wave or a straight trend, ARIMA finds the exact rule with a handful of coefficients and forecasts the held-out part with no visible miss; the TCN only gets close.',
				'在正弦波或直线趋势上，ARIMA 用几个系数就找到了精确的规则，对留出部分的预测几乎没有误差；TCN 只是接近。'
			),
			step: 'compare'
		},
		sections: {
			intros: {
				training: I('Train the TCN on your series once more: every step reads the whole training part in one pass.', '在你的序列上再训练一次 TCN：每一步都一次读完整个训练部分。')
			}
		},
		lessonNote: I(
			'Eleven steps on your own series: its changes, a causal convolution, dilations and the receptive field, the tree of one forecast with real numbers, residual blocks, the head, training on every position at once, ARIMA-lite, the comparison on held-out values, and a pattern longer than the reach.',
			'在你自己的序列上走十一步：变化量、因果卷积、空洞率与感受野、一个预测的节点树（真实的数）、残差块、输出头、在所有位置上同时训练、ARIMA 简化版、在留出值上的比较，以及一种比视野更长的规律。'
		)
	},
	glossary: [
		{ id: 'tcn', term: I('TCN (temporal convolutional network)', 'TCN（时间卷积网络）'), def: I('A network for sequences built from causal, dilated convolutions, usually in residual blocks; it computes every position in one pass.', '由因果的空洞卷积（通常组成残差块）搭成的序列网络；一次就能算出所有位置。'), related: ['convolution', 'receptive-field'] },
		{ id: 'causal-convolution', term: I('causal convolution', '因果卷积'), def: I('A convolution whose output at time t reads only inputs at t and before: zeros are padded only at the start.', '在时刻 t 的输出只读 t 及之前输入的卷积：只在序列开头补零。'), related: ['convolution', 'padding'] },
		{ id: 'dilation', term: I('dilation', '空洞率（膨胀率）'), def: I('The gap between the inputs a kernel reads: with dilation d, a kernel of 2 reads t − d and t.', '卷积核读取的输入之间的间隔：空洞率为 d 时，大小为 2 的卷积核读 t − d 和 t。'), related: ['kernel', 'receptive-field'] },
		{ id: 'differencing', term: I('differencing', '差分'), def: I('Replacing a series by its changes, y_t − y_{t−1}; a trend becomes a constant. The d of ARIMA counts how often.', '把序列换成它的变化量 y_t − y_{t−1}；趋势就变成了常数。ARIMA 里的 d 就是差分的次数。'), related: ['time-series'] },
		{ id: 'arima', term: I('ARIMA(p, d, q)', 'ARIMA(p, d, q)'), def: I('A classic linear forecaster: difference d times, then predict each change from the last p changes (AR) and the last q forecast errors (MA).', '经典的线性预测模型：先差分 d 次，再用最近 p 个变化量（AR）和最近 q 个预测误差（MA）去预测每个变化量。'), related: ['differencing', 'baseline'] },
		{ id: 'aic', term: I('AIC (Akaike information criterion)', 'AIC（赤池信息量准则）'), def: I('A score for choosing a model: how badly it fits, plus 2 for every coefficient. The smallest score wins.', '用来挑模型的分数：拟合得有多差，再给每个系数加 2。分数最小的胜出。'), related: ['arima'] }
	],
	// The catalog row (promote prints it): this page has the TCN and ARIMA only, no Transformer or PatchTST part.
	catalog: {
		hook: I(
			'Four causal convolutions that skip 1, 2, 4 and 8 steps back learn your series in the browser, against a classic ARIMA.',
			'四层分别往回跳 1、2、4、8 步的因果卷积，在浏览器里学习你的序列，并和经典的 ARIMA 比一比。'
		),
		shows: I(
			'Causal convolutions, dilations and the 16-change receptive field, the tree of nodes behind one forecast with real numbers, residual blocks and the head, training on every position at once, an ARIMA(p, 1, q) baseline chosen by AIC, a held-out comparison that each model wins on some series, and a square wave longer than the reach.',
			'因果卷积、空洞率与 16 个变化量的感受野、一个预测背后带真实数值的节点树、残差块与输出头、在所有位置上同时训练、按 AIC 选阶的 ARIMA(p, 1, q) 基准、在留出值上的比较（两个模型各有胜场），以及一个比视野更长的方波。'
		),
		runtime: I('Own autograd core in a worker (checked against PyTorch); own ARIMA (checked against a Python run)', '在 Worker 线程中运行的自研自动求导核心（已与 PyTorch 核对）；自研 ARIMA（已与 Python 运行核对）'),
		params: I('TCN with kernel 2, dilations 1, 2, 4, 8 and 8 features per layer + a 1×1 head (441 parameters)', 'TCN：卷积核为 2，空洞率 1、2、4、8，每层 8 个特征，加一个 1×1 输出头（441 个参数）'),
		size: I('No files: the network trains in your browser', '没有文件：网络在你的浏览器里训练')
	},
	// Synthetic series only (atlas/data/series.ts); no data, model or code from anyone else. Ideas only, no text, figures
	// or code: the TCN design (Bai et al.), the dilated causal stack (WaveNet), the two-stage ARIMA fit (Hannan &
	// Rissanen) and the order limits and evaluation on held-out values (Hyndman & Athanasopoulos).
	credits: [
		{
			name: 'Bai, Kolter & Koltun, An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling (2018)',
			url: 'https://arxiv.org/abs/1803.01271',
			license: 'Paper; ideas only (no text, figures or code used)',
			usedFor: 'the TCN: causal dilated convolutions in residual blocks',
			attribution: 'Shaojie Bai, J. Zico Kolter and Vladlen Koltun'
		},
		{
			name: 'van den Oord et al., WaveNet: A Generative Model for Raw Audio (2016)',
			url: 'https://arxiv.org/abs/1609.03499',
			license: 'Paper; ideas only (no text, figures or code used)',
			usedFor: 'dilations that double layer by layer, and the receptive field they give',
			attribution: 'Aäron van den Oord and co-authors'
		},
		{
			name: 'Hannan & Rissanen, Recursive estimation of mixed autoregressive-moving average order (Biometrika, 1982)',
			url: 'https://doi.org/10.1093/biomet/69.1.81',
			license: 'Paper; ideas only (no text, figures or code used)',
			usedFor: 'ARIMA-lite: the two-stage least-squares fit of the AR and MA coefficients',
			attribution: 'E. J. Hannan and J. Rissanen'
		},
		{
			name: 'Hyndman & Athanasopoulos, Forecasting: Principles and Practice (3rd ed., 2021)',
			url: 'https://otexts.com/fpp3/',
			license: 'Book, free online; ideas only (no text, figures or code used)',
			usedFor: 'ARIMA with differencing, order choice by AIC (p up to 5, as auto.arima), checking forecasts on held-out values',
			attribution: 'Rob J. Hyndman and George Athanasopoulos'
		}
	]
};

export default manifest;
