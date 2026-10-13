// gaussian-process: the manifest. Gaussian process regression with an RBF kernel on points the learner clicks on a 1-D
// plot. Kind C only: every stage is the process 'gp' (process.ts), run live in the browser on dataOf(input) and checked
// frame by frame against an independent numpy run (pipeline/gaussian-process/reference.py).
//   gp      the learner's points and settings (every step but the last)
//   short   the fixed step data with a short length-scale, and long with a long one (the kernel-choice step)
// No data, model or code from anyone else: the points are the learner's or presets made in gp.ts.
import type { ExplainerManifest, ExplainerRuntime } from '../../atlas/types/index.ts';
import { dataOf, I, LONG, SHORT, START, STEP } from './gp.ts';

const on = (input: (rt: ExplainerRuntime) => unknown) => ({ kind: 'process' as const, process: 'gp', seed: 1, input });

const manifest: ExplainerManifest = {
	slug: 'gaussian-process',
	type: 'model',
	kinds: ['process'],
	quality: 'reviewed',
	input: { adapter: 'gp-points', start: START },
	sources: {
		gp: on((rt) => dataOf(rt.input)),
		short: on(() => dataOf(STEP, { ls: SHORT })),
		long: on(() => dataOf(STEP, { ls: LONG }))
	},
	page: {
		goals: [
			I('Say what the RBF kernel computes for two points and how the length-scale ℓ sets how far one point’s influence reaches.', '说出 RBF 核对两个点算的是什么，以及长度尺度 ℓ 如何决定一个点的影响能伸多远。'),
			I('Work out the posterior mean and band after one point with their numbers, and say why the band narrows at the points and widens between them.', '带着数字算出加入一个点后的后验均值和误差带，并说出为什么误差带在点附近变窄、在点之间又变宽。'),
			I('Choose a length-scale with the marginal likelihood, and say what too short and too long length-scales do.', '用边缘似然挑选长度尺度，并说出长度尺度太短或太长会怎样。'),
			I('Name two limits: the cost grows with the cube of the number of points, and one kernel assumes the same smoothness everywhere.', '说出两个局限：计算量随点数的立方增长；一个核函数假定处处同样平滑。')
		],
		prequestion: {
			id: 'gaussian-process-pre',
			question: I('You know a function’s value at five places. How sure can you be about its value halfway between two of them?', '你知道一个函数在五个位置上的值。在其中两个位置的正中间，你对它的值能有多大把握？'),
			options: [I('Exactly as sure as at the known places', '和已知位置一样有把握'), I('Less sure, and the less the farther the known places are apart', '把握更小，两个已知位置离得越远，把握越小'), I('Not at all: it could be anything', '完全没把握：什么值都有可能')],
			answer: 1,
			explain: I('Between known values a smooth function cannot do just anything, but it is not pinned down either. A Gaussian process turns this into numbers: a band that is narrow at the data and widens with the distance to them.', '在已知值之间，平滑的函数不能随便乱走，但也没有被完全确定。高斯过程把这种直觉变成数字：误差带在数据处很窄，离数据越远越宽。')
		},
		tryThis: [
			{ id: 'wiggle', title: I('A fast wiggle', '快速起伏'), setup: { input: { preset: 'wiggle' } }, predict: I('Will a long or a short length-scale explain these points best?', '这些点用长的还是短的长度尺度解释得最好？'), look: I('In the marginal-likelihood step, find the highest point of the log p(y) curve.', '在边缘似然那一步，找 log p(y) 曲线的最高点。'), goto: { step: 'evidence' } },
			{ id: 'smooth', title: I('A gentle curve', '平缓的曲线'), setup: { input: { preset: 'smooth', ls: 0.25 } }, predict: I('With ℓ = 0.25 on a gentle curve, what does the band do between the points?', '在平缓的曲线上用 ℓ = 0.25，点与点之间的误差带会怎样？'), look: I('Look at the band between neighbouring points, then try the slider.', '看相邻两点之间的误差带，再试试滑块。'), goto: { step: 'tune' } }
		],
		quiz: [
			{ q: I('With σf² = 1, how strongly are two points one length-scale apart linked: what is k(x, x + ℓ)?', 'σf² = 1 时，相距一个长度尺度的两个点关联有多强：k(x, x + ℓ) 是多少？'), options: [I('about 0.61', '约 0.61'), I('about 0.37', '约 0.37'), I('0', '0')], answer: 0, explain: I('k = σf² exp(−ℓ²/(2ℓ²)) = exp(−0.5) ≈ 0.61. At two length-scales it is exp(−2) ≈ 0.14, at three already exp(−4.5) ≈ 0.011.', 'k = σf² exp(−ℓ²/(2ℓ²)) = exp(−0.5) ≈ 0.61。相距两个长度尺度时是 exp(−2) ≈ 0.14，三个长度尺度时只剩 exp(−4.5) ≈ 0.011。') }
		],
		misconception: { claim: I('The band shows where the data are noisy.', '误差带表示数据在哪里噪声大。'), truth: I('Here the noise is the same everywhere. The band shows how unsure the model is about the function: narrow near the points, wide in gaps and beyond the data.', '这里的噪声处处相同。误差带表示模型对函数本身有多没把握：点附近窄，空白处和数据之外宽。'), step: 'gap' },
		lessonNote: I('Eleven steps on your own points: the prior, the kernel, one point worked out with its numbers, the band between points, the posterior point by point, the kernel matrix, sample functions, the marginal likelihood, choosing a length-scale, and two limits.', '十一步，都在你自己的点上：先验、核函数、带数字算一个点、点之间的误差带、逐点更新的后验、核矩阵、样本函数、边缘似然、挑选长度尺度，以及两个局限。')
	},
	glossary: [
		{ id: 'gaussian-process', term: I('Gaussian process', '高斯过程'), def: I('A probability distribution over functions: the values of the function at any set of inputs are jointly normal, with a mean (here 0) and a covariance given by a kernel.', '函数上的概率分布：函数在任意一组输入处的值服从联合正态分布，均值（这里为 0）和协方差由核函数给出。'), related: ['rbf-kernel', 'gp-posterior'] },
		{ id: 'rbf-kernel', term: I('RBF kernel and length-scale', 'RBF 核与长度尺度'), def: I('k(x, x′) = σf² exp(−(x − x′)² / (2ℓ²)): the covariance of the function’s values at x and x′. It is σf² for the same place and falls towards 0 as the distance grows past a few length-scales ℓ.', 'k(x, x′) = σf² exp(−(x − x′)² / (2ℓ²))：函数在 x 和 x′ 处取值的协方差。同一位置时等于 σf²，距离超过几个长度尺度 ℓ 后趋近于 0。') },
		{ id: 'gp-posterior', term: I('posterior mean and band', '后验均值与误差带'), def: I('After seeing the data, the function’s value at x is still normal: its mean is the best guess, and mean ± 2 standard deviations is a band that holds the true value about 95 times in 100 if the model is right.', '看到数据之后，函数在 x 处的值仍服从正态分布：均值是最佳猜测，均值 ± 2 个标准差构成误差带；若模型正确，大约 100 次里有 95 次包含真实值。'), related: ['gaussian-process'] },
		{ id: 'marginal-likelihood', term: I('marginal likelihood (evidence)', '边缘似然（证据）'), def: I('The probability of the observed y under the model, with every possible function averaged out. Its logarithm is a data-fit term plus a complexity penalty; the setting with the highest value explains the data best without overfitting.', '在模型下观测到这些 y 的概率，所有可能的函数都已被平均掉。它的对数等于数据拟合项加上复杂度惩罚；数值最高的设置在不过拟合的前提下最能解释数据。') },
		{ id: 'cholesky', term: I('Cholesky factorization', 'Cholesky 分解'), def: I('Writing a symmetric positive-definite matrix as L Lᵀ with L lower-triangular; for n points it solves the GP’s linear system in about n³/3 multiplications and subtractions.', '把一个对称正定矩阵写成 L Lᵀ，其中 L 是下三角矩阵；对 n 个点，它求解高斯过程的线性方程组大约需要 n³/3 次乘法和减法。') }
	],
	catalog: {
		hook: I('Click points and watch the uncertainty band shrink around them.', '点击添加数据点，看误差带在这些点周围收窄。'),
		shows: I('The prior and posterior band updating point by point on your own clicks, the RBF kernel and the kernel matrix with its numbers, functions drawn from the prior and the posterior, the marginal likelihood for choosing the length-scale, and the limits: cubic cost and one fixed smoothness.', '在你自己点出的数据上逐点更新的先验与后验误差带、带数字的 RBF 核与核矩阵、从先验和后验中抽取的函数、用来挑选长度尺度的边缘似然，以及局限：立方级的计算量和固定的平滑程度。'),
		runtime: I('Own TypeScript, live in the browser (checked against a numpy run)', '自研 TypeScript，在浏览器中实时运行（已与 numpy 运行核对）'),
		params: I('none learned: the settings ℓ, σf² and the noise σn², and one weight per data point', '没有学习的参数：设定 ℓ、σf² 和噪声 σn²，以及每个数据点一个权重'),
		size: I('no files: the points are yours or presets made in your browser', '没有文件：数据点由你点出，或是在你的浏览器里生成的预设')
	},
	credits: [
		{
			name: 'C. E. Rasmussen and C. K. I. Williams, “Gaussian Processes for Machine Learning” (MIT Press, 2006), chapters 2 and 5',
			url: 'https://gaussianprocess.org/gpml/',
			license: 'Book; ideas only (no text, figures, code or data used)',
			usedFor: 'the posterior equations, the Cholesky-based prediction (algorithm 2.1) and the log marginal likelihood',
			attribution: 'Carl Edward Rasmussen and Christopher K. I. Williams'
		}
	]
};

export default manifest;
