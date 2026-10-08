// ddpm-unet: the package manifest (design §4.9). A tiny DDPM: a small U-Net noise predictor trained offline on three
// Quick, Draw! classes (pipeline/ddpm-unet), traced live on the learner's noised drawing, and sampling run live:
//   fwd      the exported model on (drawing x₀, noise ε, step t): x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε, then the U-Net's ε̂
//   noise    the drawing noised at every step t = 0 … T (process.ts forward)
//   chain    DDPM sampling, all T steps, from the learner's seed (process.ts sample; live, in the package worker)
//   few      the same seed with 10 bigger steps
//   gallery  seeds 0 … 11, their pictures at t = 100, 80, …, 0 (process.ts gallery; only for `atlas run --precompute`)
//   seeds    that precomputed gallery (tapes/gallery): full chains for 12 seeds without 1,200 network runs on the page
// Measured numbers (training, the judge's shares, the 10-step comparison) are in facts.json (export.py, stats.py).
import type { ExplainerManifest, ExplainerRuntime } from '../../atlas/types/index.ts';
import { FEW, modelInput } from './ddpm.ts';

const L = (en: string, zh: string) => ({ en, zh });
const noiseIn = (rt: ExplainerRuntime) => (rt.pre ? { x0: rt.pre.tensors.input.data, eps: rt.pre.tensors.eps.data } : null);

const manifest: ExplainerManifest = {
	slug: 'ddpm-unet',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/ddpm-unet', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'drawing28', presets: 'presets.json' },
	output: { view: 'ddpm-unet/pics', props: { mode: 'predict' } },
	sources: {
		fwd: { kind: 'trace' },
		noise: { kind: 'process', process: 'forward', input: noiseIn },
		chain: { kind: 'process', process: 'sample', input: modelInput },
		few: { kind: 'process', process: 'sample', input: modelInput, opt: { stride: FEW } },
		gallery: { kind: 'process', process: 'gallery', input: (rt) => { const m = modelInput(rt); return m && { ...m, seed: 0 }; }, reference: false },
		seeds: { kind: 'tape', file: 'tapes/gallery' }
	},
	overview: { layout: 'stack' },
	page: {
		goals: [
			L('Noise a drawing to any step t with x_t = √ᾱ_t·(2x₀ − 1) + √(1 − ᾱ_t)·ε, and say what ᾱ_t does as t grows.', '用 x_t = √ᾱ_t·(2x₀ − 1) + √(1 − ᾱ_t)·ε 把一幅画加噪到任意第 t 步，并说出 t 变大时 ᾱ_t 怎样变化。'),
			L('Say what the U-Net is trained to output for a noisy drawing, and how its guess is scored.', '说出 U-Net 对一幅加噪的画要输出什么，以及它的猜测怎样打分。'),
			L('Work out one denoising step, from x_t to the slightly cleaner x_(t−1), for one pixel with the real numbers.', '用真实的数算出一个像素的一步去噪：从 x_t 到稍微干净一点的 x_(t−1)。'),
			L('Explain why making one drawing takes many network runs, and what the seed and the number of steps change.', '解释为什么画一幅画要运行网络很多次，以及种子和步数会改变什么。')
		],
		prequestion: {
			id: 'ddpm-unet-pre',
			question: L('A diffusion model turns pure random noise into a new drawing. What does its network learn to do?', '扩散模型把纯随机噪声变成一幅新画。它的网络学会做什么？'),
			options: [L('Draw the whole picture from noise in one go', '一下子从噪声画出整幅画'), L('Guess the noise in a noisy picture, so it can be removed bit by bit', '猜出一幅加噪的画里的噪声，好一点点去掉它'), L('Find the closest training drawing and copy it', '找到最接近的训练画并复制它')],
			answer: 1,
			explain: L(
				'It only ever guesses the noise. Sampling subtracts part of the guess, adds a little fresh noise, and repeats, so a drawing emerges over many small steps; no training drawing is stored.',
				'它只做一件事：猜噪声。采样时减去猜测的一部分，再加一点新的噪声，反复进行，画就在许多小步里慢慢出现；它不存任何训练画。'
			)
		},
		quiz: [
			{
				q: L('What does the network get as input?', '网络的输入是什么？'),
				options: [L('The noisy picture x_t and the step t', '加噪的画 x_t 和步数 t'), L('The clean drawing and the noise', '干净的画和噪声'), L('Only the seed', '只有种子')],
				answer: 0,
				explain: L('x_t and t, nothing else: the step tells it how much noise to expect.', '只有 x_t 和 t：步数告诉它该预期多少噪声。')
			}
		],
		misconception: {
			claim: L('The model stores its training drawings and blends them.', '模型把训练画存起来，再把它们混在一起。'),
			truth: L('It keeps only the U-Net’s weights. Every drawing is made from fresh noise by many small denoising steps.', '它只保存 U-Net 的权重。每幅画都是从新的噪声出发，经过许多小的去噪步骤做出来的。'),
			step: 'seeds'
		},
		metrics: { val_noise_mse: { label: L('Noise guess error (mean squared) on the validation drawings', '验证集画上猜噪声的误差（均方）'), fmt: 'n4' } },
		lessonNote: L(
			'Eleven steps: bury your drawing in noise, see the network guess the noise, work out one denoising step, then watch drawings appear from noise: other seeds, fewer steps, the U-Net inside, and the limits.',
			'十一步：把你的画埋进噪声，看网络猜噪声，算出一步去噪，再看画从噪声里出现：换种子、减步数、看 U-Net 内部，最后看局限。'
		)
	},
	glossary: [
		{ id: 'diffusion-model', term: L('diffusion model', '扩散模型'), def: L('A generative model that learns to undo a gradual noising: it makes data by starting from noise and removing it step by step.', '学会撤销逐步加噪的生成模型：它从噪声出发，一步步去掉噪声来造出数据。'), related: ['generative-model'] },
		{ id: 'forward-process', term: L('forward process', '前向过程'), def: L('Adding a little noise at each of T steps until only noise is left; x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε (x₀ scaled to −1 … 1) jumps to step t at once.', '在 T 步里每步加一点噪声，直到只剩噪声；x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε（x₀ 先缩放到 −1 … 1）一下子跳到第 t 步。'), related: ['diffusion-model'] },
		{ id: 'noise-schedule', term: L('noise schedule', '噪声调度'), def: L('How much noise each step adds (β_t); ᾱ_t, the product of 1 − β over steps 1 to t, says how much of the data is left.', '每一步加多少噪声（β_t）；ᾱ_t 是第 1 到第 t 步的 1 − β 之积，表示数据还剩多少。'), related: ['forward-process'] },
		{ id: 'denoising-step', term: L('denoising step', '去噪步'), def: L('One step of sampling: subtract part of the predicted noise from x_t, rescale, and add a little fresh noise to get x_{t−1}.', '采样的一步：从 x_t 中减去预测噪声的一部分，重新缩放，再加一点新噪声，得到 x_{t−1}。'), related: ['diffusion-model'] },
		{ id: 'u-net', term: L('U-Net', 'U-Net'), def: L('A convolutional network shaped like a U: it shrinks the picture step by step, grows it back, and copies maps across with skip connections.', '形状像 U 的卷积网络：它一步步把图缩小，再放大回来，并用跳跃连接把图横着复制过去。'), related: ['skip-connection', 'convolution'] },
		{ id: 'skip-connection', term: L('skip connection', '跳跃连接'), def: L('A path that copies a layer’s maps past the layers in between and joins them to a later layer, here by concatenating channels.', '把某一层的图越过中间的层复制过去、并接到后面某一层的通路；这里是把通道拼接起来。'), related: ['u-net', 'concatenation'] },
		{ id: 'timestep-embedding', term: L('timestep embedding', '时间步嵌入'), def: L('The step number t turned into a list of sine and cosine waves, so the network knows how much noise to expect.', '把步数 t 变成一串正弦和余弦波，让网络知道该预期多少噪声。'), related: ['positional-encoding'] },
		{ id: 'upsampling', term: L('upsampling', '上采样'), def: L('Making a map bigger; nearest-neighbour upsampling copies each number into a 2×2 block.', '把图放大；最近邻上采样把每个数复制成一个 2×2 的块。'), related: ['pooling'] }
	],
	catalog: {
		hook: L('Bury your drawing in noise, then watch a small U-Net turn fresh noise into a new drawing, step by step.', '把你的画埋进噪声，再看一个小 U-Net 把新的噪声一步步变成一幅新画。'),
		shows: L(
			'The forward noising with the real ᾱ schedule, the U-Net’s noise guess on your drawing, one denoising step with real numbers, 100-step chains run live, 12 seeds, 10 steps against 100, and the U-Net’s skip connections.',
			'按真实 ᾱ 噪声调度的前向加噪、U-Net 对你的画的噪声猜测、代入真实数值的一步去噪、在浏览器里实时运行的 100 步链条、12 个种子、10 步与 100 步的对比，以及 U-Net 的跳跃连接。'
		),
		runtime: L('Own JS forward pass (checked against PyTorch); sampling in a worker', '自研 JS 前向计算（已与 PyTorch 核对）；采样在 Worker 线程中运行'),
		params: L('60,001', '60,001'),
		size: L('240 KB of weights', '权重 240 KB')
	},
	credits: [
		{
			name: 'Quick, Draw! dataset',
			url: 'https://github.com/googlecreativelab/quickdraw-dataset',
			license: 'CC BY 4.0',
			usedFor: 'DDPM training drawings (star, umbrella, house; TinyCNN’s split) and the preset drawings',
			attribution: 'Quick, Draw! dataset by Google Creative Lab'
		},
		{
			name: 'Ho, Jain and Abbeel, Denoising Diffusion Probabilistic Models (NeurIPS 2020)',
			url: 'https://arxiv.org/abs/2006.11239',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'the forward process, the noise-prediction loss, the sampling formula with σ_t² = β_t, the linear β schedule, sinusoidal timestep embeddings and the EMA of weights',
			attribution: 'Jonathan Ho, Ajay Jain, Pieter Abbeel'
		},
		{
			name: 'Nichol and Dhariwal, Improved Denoising Diffusion Probabilistic Models (ICML 2021)',
			url: 'https://arxiv.org/abs/2102.09672',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'scaling the linear schedule to fewer steps (1000 / T) and sampling on a subset of the steps (respacing)',
			attribution: 'Alex Nichol, Prafulla Dhariwal'
		},
		{
			name: 'Ronneberger, Fischer and Brox, U-Net (MICCAI 2015)',
			url: 'https://arxiv.org/abs/1505.04597',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'the U shape: down by pooling, up by upsampling, skip connections by concatenation',
			attribution: 'Olaf Ronneberger, Philipp Fischer, Thomas Brox'
		}
	]
};

export default manifest;
