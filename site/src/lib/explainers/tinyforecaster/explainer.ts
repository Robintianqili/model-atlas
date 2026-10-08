// tinyforecaster: the manifest (design §4.9). An LSTM forecaster learns the learner's own series live in the browser
// (kind B), the series becomes training windows and the forecasts are checked step by step (kind C), and one window is
// traced through the network with the kit's shared lstm op (kind A on the live weights). Every source has seed 1. The
// process sources get their series as plain numbers, so the Python reference (pipeline/tinyforecaster/reference.py)
// reads exactly the values the TS side uses.
//   windows   process 'windows': the series, then every training window and its target (Python reference)
//   init      trainer 'lstm' with no steps: the starting weights;  net0  a trace of the last training window through them
//   train     trainer 'lstm', 200 steps of Adam (twin: pipeline/tinyforecaster/twin.py)
//   forecast  process 'forecast': the same training, then the rollout over the held-out part and the baseline
//   walks     process 'walks': the same training on three random walks (fixed data, the same for every learner)
//   walksSaved  the limits step: the 'walks' run recorded ahead of time (npm run atlas -- run tinyforecaster walks --precompute;
//             claim-tested to equal a fresh run), so step 10 needs no training in the browser and no rerun after a new input
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { seriesPresetRaw } from '../../atlas/adapters/series/adapter.ts';
import { dataOf, WALK_SEEDS } from './data.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'tinyforecaster',
	type: 'model',
	kinds: ['train', 'process', 'trace'],
	quality: 'reviewed',
	assets: { dir: 'models/tinyforecaster', auto: [] },
	input: { adapter: 'series', presets: 'presets.json', start: seriesPresetRaw('seasonal'), options: { mode: 'curve' } },
	sources: {
		windows: { kind: 'process', process: 'windows', run: 'main', input: (rt) => Array.from(dataOf(rt.input).y) },
		train: { kind: 'train', trainer: 'lstm', detail: 'coarse', opt: { seed: 1 } },
		init: { kind: 'train', trainer: 'lstm', detail: 'coarse', opt: { seed: 1, steps: 0 } },
		net0: { kind: 'live-model', of: 'init' },
		forecast: { kind: 'process', process: 'forecast', run: 'worker', seed: 1, input: (rt) => Array.from(dataOf(rt.input).y) },
		walks: { kind: 'process', process: 'walks', run: 'worker', seed: 1, input: () => WALK_SEEDS.map((seed) => Array.from(dataOf(seriesPresetRaw('randomWalk', { seed })).y)) },
		walksSaved: { kind: 'tape', file: 'tapes/walks' }
	},
	page: {
		goals: [
			I('Turn a series into training examples: windows of 24 values, each with the value after it as its target.', '把一条序列变成训练样例：24 个值组成的窗口，各配上紧随其后的值作为目标。'),
			I('Follow one forecast through the network: the LSTM’s gates at the last step, then last value + predicted change.', '跟着一个预测走完整个网络：LSTM 最后一步的各个门，再到“最后一个值 + 预测的变化量”。'),
			I('Explain why forecasting many steps ahead feeds the model its own forecasts, and why its errors can grow.', '解释为什么向前预测很多步时，模型读的是自己的预测，以及误差为什么会越积越大。'),
			I('Judge a forecaster against a simple baseline on values it never saw, and name a series no model can forecast.', '在模型从未见过的值上，拿它和一个简单基准比较；并说出一种任何模型都预测不了的序列。')
		],
		prequestion: {
			id: 'tinyforecaster-pre',
			question: I(
				'In a random walk, each value is the previous one plus a fresh random step. Can a neural network learn to forecast the next value better than “the same as now”?',
				'随机游走中，每个值都是前一个值加上一个新的随机步。神经网络能学会比“和现在一样”更准地预测下一个值吗？'
			),
			options: [I('Yes, with enough data and training', '能，只要数据和训练足够'), I('No: the next step is random, so “the same as now” is already the best guess', '不能：下一步是随机的，“和现在一样”已经是最好的猜测')],
			answer: 1,
			explain: I(
				'No. Each step is new randomness, so nothing in the past predicts it. A network can fit the past, noise included, but on new values it does no better than repeating the last one, except by luck. The last step of the lesson shows it on three random walks.',
				'不能。每一步都是新的随机数，过去的任何东西都预测不了它。网络能把过去（连同噪声）拟合得很好，但在新的值上，它并不比“重复最后一个值”更好，除非碰上运气。这节课的最后一步会在三条随机游走上展示这一点。'
			)
		},
		tryThis: [
			{
				id: 'square',
				title: I('A square wave', '方波'),
				setup: { input: seriesPresetRaw('square') },
				predict: I('The wave jumps between two levels every 8 steps. Will the forecast of the 24 held-out values keep the jumps on time?', '这个波每 8 步在两个高度之间跳一次。对 24 个留出值的预测能准时跳吗？'),
				look: I('Look at the step that forecasts the held-out part: the first forecasts follow the wave closely, but once a jump goes wrong, every later forecast is built on the mistake.', '看预测留出部分的那一步：开头的预测紧跟着波形，可一旦有一次跳错了，后面的每个预测都建立在这个错误之上。'),
				goto: { step: 'rollout' }
			},
			{
				id: 'trend',
				title: I('A straight trend', '直线趋势'),
				setup: { input: seriesPresetRaw('trend') },
				predict: I('Can the forecast continue a straight line above every value it trained on?', '预测能把直线延续到比所有训练值都高的地方吗？'),
				look: I('Look at the baseline step: the network predicts the change, so the line goes on rising.', '看“基准”那一步：网络预测的是变化量，所以直线会继续往上走。'),
				goto: { step: 'baseline' }
			}
		],
		quiz: [
			{
				q: I('A series has 96 values and the last 24 are held out. How many training windows of 24 values (each with the next value as target) fit before the held-out part?', '一条序列有 96 个值，最后 24 个留出。在留出部分之前，能放下多少个 24 值的训练窗口（每个以下一个值为目标）？'),
				options: [I('48', '48'), I('72', '72'), I('3', '3')],
				answer: 0,
				explain: I('The training part has 72 values. A window starts at positions 0 to 47 so that its target (position 24 to 71) is still in the training part: 48 windows.', '训练部分有 72 个值。窗口从第 0 到第 47 个位置开始，这样它的目标（第 24 到第 71 个位置）仍在训练部分里：一共 48 个窗口。')
			},
			{
				q: I('What does the seasonal-naive baseline forecast for the next value?', '季节性朴素基准对下一个值给出什么预测？'),
				options: [I('The average of the whole series', '整条序列的平均值'), I('The value one season (m steps) earlier', '一个季节（m 步）之前的值'), I('A line through the last two values', '过最后两个值的直线')],
				answer: 1,
				explain: I('It copies the value from m steps back, with m the lag that worked best on the training part. With m = 1 it is “the same as now”.', '它照抄 m 步之前的值，m 取在训练部分上效果最好的间隔。m = 1 时就是“和现在一样”。')
			}
		],
		misconception: {
			claim: I('A model that fits the past well will forecast well.', '把过去拟合得好的模型，预测也一定好。'),
			truth: I(
				'On a random walk the network fits the training part better than repeating the last value, and still forecasts new values worse: it learned the noise.',
				'在随机游走上，网络对训练部分的拟合比“重复最后一个值”更好，可预测新的值时反而更差：它学到的是噪声。'
			),
			step: 'walk'
		},
		sections: {
			intros: {
				training: I('Train the forecaster on your series once more: the line is the loss of each step’s random windows.', '在你的序列上再训练一次预测模型：这条线是每一步那几个随机窗口的损失。')
			}
		},
		lessonNote: I(
			'Ten steps on your own series: windows, the LSTM’s gates, one forecast worked out (the head, then last value + change), the loss, training live, forecasting the held-out part, a baseline to beat, and a series no model can forecast.',
			'在你自己的序列上走十步：窗口、LSTM 的门、算出一个预测（先是输出头，再是“最后一个值 + 变化量”）、损失、实时训练、预测留出的部分、要打败的基准，以及一种任何模型都预测不了的序列。'
		)
	},
	glossary: [
		{ id: 'time-series', term: I('time series', '时间序列'), def: I('Numbers measured one after another in time, such as a temperature every hour.', '按时间先后测得的一串数，比如每小时的气温。') },
		{ id: 'forecast-window', term: I('window', '窗口'), def: I('The last few values a forecaster reads to predict the next one. Sliding it one step gives the next training example.', '预测模型为了预测下一个值而读取的最近几个值。把它滑动一步，就得到下一个训练样例。'), related: ['time-series'] },
		{ id: 'lstm', term: I('LSTM (long short-term memory)', 'LSTM（长短期记忆网络）'), def: I('A recurrent network that reads a sequence one step at a time and keeps a memory, the cell state c, which gates update.', '一种循环网络：一步一步读序列，并保留一份由各个门更新的记忆，即细胞状态 c。') },
		{ id: 'gate', term: I('gate', '门'), def: I('A number between 0 and 1 that an LSTM multiplies a signal by: near 0 blocks it, near 1 lets it through.', 'LSTM 用来乘一个信号的数，介于 0 和 1 之间：接近 0 就挡住它，接近 1 就让它通过。'), related: ['lstm'] },
		{ id: 'mse', term: I('mean squared error', '均方误差'), def: I('The average of the squared misses (forecast − truth)²; squaring makes big misses count much more.', '各次误差平方 (预测 − 真实值)² 的平均；平方让大的误差影响大得多。'), related: ['loss'] },
		{ id: 'rollout', term: I('rollout', '滚动预测'), def: I('Forecasting many steps ahead by appending each forecast to the history and reading it back as input.', '把每个预测接到历史后面、再当作输入读回去，以此向前预测很多步。'), related: ['autoregressive'] },
		{ id: 'baseline', term: I('baseline', '基准'), def: I('A simple rule a model is compared with. A model is only worth using if it beats it on new data.', '拿来和模型比较的简单规则。模型只有在新数据上胜过它，才值得用。') },
		{ id: 'random-walk', term: I('random walk', '随机游走'), def: I('A series in which each value is the previous one plus a new random step.', '每个值都等于前一个值加上一个新的随机步的序列。'), related: ['time-series'] }
	],
	catalog: { runtime: I('Own autograd core in a worker', '自研自动微分内核，在 Web Worker 中运行'), params: I('LSTM with 16 memory numbers + a linear head (1,233 parameters)', '16 个记忆数的 LSTM + 线性输出头（1,233 个参数）') },
	// Synthetic series only (atlas/data/series.ts); no data, model or code from anyone else. The evaluation on held-out
	// values and the seasonal-naive baseline follow a textbook (ideas only, no text, figures or code).
	credits: [
		{
			name: 'Hyndman & Athanasopoulos, Forecasting: Principles and Practice (3rd ed., 2021)',
			url: 'https://otexts.com/fpp3/',
			license: 'Book, free online; ideas only (no text, figures or code used)',
			usedFor: 'the seasonal-naive baseline and checking forecasts on held-out values',
			attribution: 'Rob J. Hyndman and George Athanasopoulos'
		}
	]
};

export default manifest;
