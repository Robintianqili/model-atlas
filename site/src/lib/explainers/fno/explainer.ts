// fno: the package manifest. A tiny Fourier neural operator (pipeline/fno, PyTorch) trained offline on pairs made in TS
// (heat.ts: a starting curve and the numerical solver's curve 50 time steps later), traced live on the learner's curve:
//   fwd    the exported model: lift -> two Fourier layers (FFT, keep the lowest modes, complex weights, inverse FFT,
//          plus a pointwise path) -> one height per point; the slider's `keep` is a second input
//   solve  kind C: the numerical solver on the learner's curve, one frame per time step (checked frame by frame against
//          pipeline/fno/reference.py)
//   (process.ts data made the training pairs train.py read; no page source uses it)
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { I } from './heat.ts';
import { valuesOf } from './adapters/series/adapter.ts';

const manifest: ExplainerManifest = {
	slug: 'fno',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/fno', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'series', presetsFromAdapter: true, options: { mode: 'curve', yRange: [-1.5, 1.5] } },
	output: { view: 'fno/field', layer: 'proj' },
	sources: {
		fwd: { kind: 'trace' },
		solve: { kind: 'process', process: 'solve', seed: 1, input: (rt) => { const u = valuesOf(rt.input); return u.length ? Array.from(u) : null; } }
	},
	page: {
		goals: [
			I('Say what a numerical solver does in each of its 50 small steps, and how a neural operator reaches the same answer in one pass.', '说出数值求解器在 50 个小步的每一步里做什么，以及神经算子怎样一次就得到同样的答案。'),
			I('Name the parts of a Fourier layer in order (the FFT, keeping the lowest modes, the learned complex weights, the inverse FFT, the pointwise path, the activation) and say what each does to a curve.', '按顺序说出傅里叶层的各个部分（FFT、只保留最低的模式、学来的复数权重、逆 FFT、逐点通路、激活函数），以及每一部分对曲线做了什么。'),
			I('Explain why dropping the high modes costs so little for heat, and find how many modes this curve needs.', '解释为什么对热传导来说丢掉高模式几乎没有代价，并找出这条曲线需要多少个模式。'),
			I('Say what the operator cannot do: curves taller than its training data, and detail finer than its 64 points.', '说出这个算子做不到什么：比训练数据更高的曲线，以及比 64 个点更细的细节。')
		],
		prequestion: {
			id: 'fno-pre',
			question: I('Heat spreads from hot to cold. Write a temperature curve as a sum of waves. Which waves fade fastest?', '热量从热处流向冷处。把一条温度曲线写成许多波的和。哪些波消失得最快？'),
			options: [I('The short, fast wiggles', '短而快的小抖动'), I('The long, slow waves', '长而慢的大波'), I('All waves fade at the same rate', '所有波消失得一样快')],
			answer: 0,
			explain: I('A short wiggle has hot and cold points close together, so heat evens it out quickly; a long wave has to move heat a long way. That is why a model that keeps only the slow waves can still predict heat well.', '短的小抖动里冷热点挨得很近，热量很快就把它抹平；长波则要把热量搬很远。所以一个只保留慢波的模型，照样能把热传导预测好。')
		},
		tryThis: [
			{ id: 'square', title: I('Sharp edges', '锐利的边'), setup: { preset: 'square' }, predict: I('A square pulse has sharp corners, made of many high modes. Will the network, which keeps only 16 modes, still match the solver?', '方波脉冲有锐利的拐角，由许多高模式组成。只保留 16 个模式的网络，还能和求解器对上吗？'), look: I('Compare the prediction with the solver in the stage: heat rounds the corners off anyway, so the dropped modes were about to vanish.', '在舞台上比较预测和求解器：热量本来就会把拐角磨圆，被丢掉的模式本来就要消失。'), goto: { step: 'predict' } },
			{ id: 'waves', title: I('A fast wiggle', '一个快速抖动'), setup: { preset: 'waves' }, predict: I('This curve is a slow wave plus a fast wiggle of 12 bumps. What is left after the solver’s 50 steps?', '这条曲线是一个慢波加一个有 12 个鼓包的快速抖动。求解器算完 50 步后还剩下什么？'), look: I('The wiggle is gone and the slow wave has only shrunk a little.', '抖动没了，慢波只缩小了一点。'), goto: { step: 'solver' } }
		],
		quiz: [
			{
				q: I('You lower the number of kept modes from 16 to 8 on a smooth curve. What happens to the prediction?', '对一条平滑曲线，你把保留的模式数从 16 降到 8。预测会怎样？'),
				options: [I('Almost nothing: modes 8 to 15 had nearly faded in the solver’s answer anyway', '几乎不变：模式 8 到 15 在求解器的答案里本来就快消失了'), I('It becomes noisy', '变得很嘈杂'), I('It shifts sideways', '整体向旁边移动')],
				answer: 0,
				explain: I('Heat shrinks mode k by a factor that falls very fast with k, so the high modes carry almost nothing at the end. Dropping them changes little.', '热量让模式 k 缩小的倍数随 k 急剧下降，所以高模式到最后几乎什么都不剩。丢掉它们几乎没有影响。')
			},
			{
				q: I('Why is the FFT a good first step for this problem?', '为什么对这个问题，FFT 是很好的第一步？'),
				options: [I('Heat acts on each wave separately, so in frequency space one weight per mode does most of the work', '热量对每个波分别起作用，所以在频域里每个模式一个权重就能完成大部分工作'), I('It makes the curve shorter, so the network is faster', '它让曲线变短，网络就更快'), I('It removes the noise from the data', '它去掉了数据里的噪声')],
				answer: 0,
				explain: I('The heat equation multiplies every Fourier mode by its own factor. A layer that multiplies modes by learned weights matches that structure, so it learns from few examples.', '热传导方程把每个傅里叶模式乘以它自己的因子。一个把模式乘以学来权重的层正好符合这种结构，所以只用少量例子就能学会。')
			}
		],
		misconception: {
			claim: I('A neural operator solves the equation exactly, at any resolution and for any starting curve.', '神经算子能在任何分辨率、对任何起始曲线精确地解方程。'),
			truth: I('It learned from a solver’s examples on 64 points. It is only as right as that solver, and it drifts away on curves unlike its training data.', '它是从一个求解器在 64 个点上的例子学来的。它最多和那个求解器一样准，遇到和训练数据不像的曲线还会跑偏。'),
			step: 'limits'
		},
		metrics: {
			train_mse: { label: I('Mean squared error, training pairs', '训练样本对的平均平方误差'), fmt: 'n3' },
			val_mse: { label: I('Mean squared error, validation pairs', '验证样本对的平均平方误差'), fmt: 'n3' },
			test_rel: { label: I('Relative error on the test pairs', '测试样本对上的相对误差'), fmt: 'n3' }
		},
		lessonNote: I('Eleven steps: your curve; the solver spreads heat in 50 small steps; then your curve goes through the network: lifted into channels, turned into waves by the FFT, cut to the lowest modes, multiplied by learned weights, turned back, added to a pointwise path; then you compare it with the solver, find how many modes your curve needs, and see the limits.', '十一步：先是你的曲线；求解器用 50 个小步让热量扩散；然后你的曲线穿过网络：先升到多个通道，再由 FFT 变成波，截到最低的几个模式，乘以学来的权重，变回曲线，再加上逐点通路；接着你把它和求解器比较，找出你的曲线需要多少个模式，最后看局限。')
	},
	glossary: [
		{ id: 'heat-equation', term: I('heat equation', '热传导方程'), def: I('The rule for how temperature spreads: at every moment, each point moves toward the average of its neighbours, so bumps flatten and the total heat stays the same.', '描述温度如何扩散的规律：每一刻，每个点都向邻居的平均值靠拢，所以鼓包会变平，总热量保持不变。'), related: ['numerical-solver'] },
		{ id: 'numerical-solver', term: I('numerical solver', '数值求解器'), def: I('A program that follows an equation in many small time steps, each computed from the one before. The heat solver on the Fourier neural operator page moves every point a little toward its two neighbours, 50 times.', '把方程拆成许多小时间步、每一步都从上一步算出来的程序。傅里叶神经算子页面上的热传导求解器就是这样：每个点向两边邻居靠拢一点，重复 50 次。') },
		{ id: 'fourier-mode', term: I('Fourier mode', '傅里叶模式'), def: I('One of the waves a curve is made of. On a ring, mode k goes k times around; mode 0 is the average height. The FFT gives each mode’s two numbers (cosine and sine part); the inverse FFT adds the waves back up.', '组成一条曲线的波之一。在环上，模式 k 绕环 k 圈；模式 0 就是平均高度。FFT 给出每个模式的两个数（余弦部分和正弦部分）；逆 FFT 再把这些波加回去。'), related: ['fft'] },
		{ id: 'fft', term: I('FFT (fast Fourier transform)', 'FFT（快速傅里叶变换）'), def: I('A fast way to rewrite a list of values as a sum of waves, and back again (inverse FFT).', '把一串值快速改写成许多波之和的方法，也能反过来变回去（逆 FFT）。'), related: ['fourier-mode'] },
		{ id: 'neural-operator', term: I('neural operator', '神经算子'), def: I('A network that maps a whole function to a whole function (for example, a starting temperature curve to the curve some time later) instead of numbers to a label.', '把一整个函数映射成另一整个函数的网络（例如把起始温度曲线映射成一段时间后的曲线），而不是把一些数映射成一个标签。'), related: ['fourier-layer'] },
		{ id: 'fourier-layer', term: I('Fourier layer', '傅里叶层'), def: I('The block of a Fourier neural operator: FFT, keep the lowest modes, multiply them by learned complex weights, inverse FFT, add a pointwise linear path, then an activation.', '傅里叶神经算子的基本模块：FFT，只保留最低的模式，乘以学来的复数权重，逆 FFT，加上逐点线性通路，再接激活函数。'), related: ['fourier-mode', 'neural-operator'] }
	],
	credits: [
		{
			name: 'Fourier Neural Operator (Li et al., 2021)',
			url: 'https://arxiv.org/abs/2010.08895',
			license: 'Paper; ideas only (no text, figures or code used)',
			usedFor: 'The architecture (lift, Fourier layers with a spectral and a pointwise path, projection) and the idea of keeping only the lowest modes; no code or data',
			attribution: 'Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar. Fourier Neural Operator for Parametric Partial Differential Equations. ICLR 2021'
		},
		{
			name: 'An Interactive Introduction to Fourier Transforms (Jez Swanson)',
			url: 'https://github.com/Jezzamonn/fourier',
			license: 'MIT; ideas only (no code copied)',
			usedFor: 'Ideas only (no code): drawing a curve and watching it rebuilt from a few waves as the number of kept waves changes',
			attribution: 'Jez Swanson (Jezzamonn), MIT License'
		}
	],
	catalog: {
		hook: I('Draw a temperature curve; a neural operator predicts how heat spreads by filtering it in frequency space.', '画一条温度曲线；神经算子在频域里对它滤波，预测热量如何扩散。'),
		shows: I('The numerical solver step by step, the Fourier layer (FFT, the kept modes, the learned complex weights, the inverse FFT, the pointwise path), the prediction against the solver, a slider for the number of kept modes, and the learned filter next to the solver’s.', '逐步运行的数值求解器，傅里叶层（FFT、保留的模式、学来的复数权重、逆 FFT、逐点通路），预测与求解器的对比，保留模式数的滑块，以及学到的滤波器与求解器的对比。'),
		runtime: I('Own JS forward pass (checked against PyTorch); the heat solver in a worker', '自研 JS 前向计算（已与 PyTorch 核对）；热传导求解器在 Worker 线程中运行'),
		params: I('1,077', '1,077'),
		size: I('4.3 KB of weights', '权重 4.3 KB')
	}
};

export default manifest;
