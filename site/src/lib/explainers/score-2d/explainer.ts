// score-2d: the manifest. Score matching on the learner's points: a network learns the score (the direction in which the
// density of the points grows fastest) by denoising score matching (kind B, in the package worker), then Langevin chains
// follow it (kind C). Sources (seed 1 everywhere, so the train sources start from the same weights and every Langevin
// source from the same uniform starts):
//   train     trainer 'fit': Adam 3e-3, batch 256, 1,000 steps, a snapshot every 50
//   sgd1      one plain gradient-descent step (lr 1) in fine detail: the batch, the loss, the gradients, the update
//   walk      Langevin at one noise level σ = 0.16, 100 steps, α = 0.0032 (process 'langevin', after training)
//   plain     Langevin at the smallest level σ = 0.04 only, 240 steps, α = 2e-4
//   annealed  annealed Langevin: σ = 1.28, 0.64, …, 0.04, 40 steps each, α = 2e-4 × (σ / 0.04)²
//   still     annealed without the noise term (gradient ascent on log p at each level)
//   limits    the learned score against the true one at σ = 0.04 where no noisy copy lands (process 'limits', no reference)
import type { ExplainerManifest, ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { presetRaw } from '../../atlas/adapters/points2d/adapter.ts';
import { dataOf, SHOW, SIGMAS, SMIN } from './score.ts';

const L = (en: string, zh: string) => ({ en, zh });
const train = (opt: Record<string, unknown> = {}, detail: 'coarse' | 'fine' = 'coarse') => ({ kind: 'train' as const, trainer: 'fit', detail, opt: { seed: 1, ...opt } });
/** the trained weights (the train source's last frame) and the points, for a Langevin source */
export function trained(rt: ExplainerRuntime) {
	const t = rt.sources.train?.tape;
	const params = t?.length ? (t.state(t.length - 1) as { params?: Record<string, Tensor> }).params : undefined;
	return params ? { params, xy: Array.from(dataOf(rt.input).xy) } : null;
}
const lang = (opt: Record<string, unknown>) => ({ kind: 'process' as const, process: 'langevin', after: 'train', seed: 1, input: trained, opt });
export const ALPHA = 2e-4;

const manifest: ExplainerManifest = {
	slug: 'score-2d',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	input: { adapter: 'points2d', start: presetRaw('smiley'), options: { classes: 1, presets: ['smiley', 'ring', 'moons', 'gaussians'] } },
	sources: {
		train: train(),
		sgd1: train({ optimizer: 'sgd', lr: 1, steps: 1 }, 'fine'),
		walk: lang({ sigmas: [SHOW], T: 100, alpha: ALPHA }),
		plain: lang({ sigmas: [SMIN], T: 240, alpha: ALPHA }),
		annealed: lang({ sigmas: SIGMAS, T: 40, alpha: ALPHA }),
		still: lang({ sigmas: SIGMAS, T: 40, alpha: ALPHA, noise: 0 }),
		limits: { kind: 'process', process: 'limits', after: 'train', seed: 1, input: trained, opt: { sigma: SMIN }, reference: false }
	},
	page: {
		goals: [
			L('Say what the score of a set of points is, and which way its arrow points at a place you pick.', '说出一组点的分数是什么，以及在你选的位置它的箭头指向哪里。'),
			L('Build one denoising example by hand (a point, a noise level σ, noise ε) and say what the network must output for it.', '亲手搭出一个去噪样本（一个点、一个噪声水平 σ、噪声 ε），并说出网络对它要输出什么。'),
			L('Work out one Langevin step, and explain why the sampler needs its noise and why it starts at a large σ.', '算出朗之万法的一步，并解释采样器为什么需要噪声、为什么从大的 σ 开始。')
		],
		prequestion: {
			id: 'score-2d-pre',
			question: L('A network will learn to turn random points into points shaped like yours. What will it learn?', '一个网络要学会把随机的点变成形状像你的点的新点。它学的是什么？'),
			options: [L('A list of where your points are', '一张记录你的点在哪里的清单'), L('An arrow at every place, pointing toward where points are dense', '每个位置一个箭头，指向点密集的地方')],
			answer: 1,
			explain: L(
				'It learns the score: at every place, an arrow pointing the way the density of your points rises fastest. A sampler then walks along the arrows, with a little noise, from anywhere to your shape.',
				'它学的是分数：在每个位置，一个指向你的点的密度上升最快方向的箭头。然后采样器顺着箭头走，再加一点噪声，从任何地方走到你的形状上。'
			)
		},
		lessonNote: L(
			'Ten steps: the density of your points and its score, how a network learns the score from noisy copies, one training step, Langevin sampling with and without noise, annealing from large to small noise, and where the learned score fails.',
			'十步：你的点的密度和它的分数，网络怎样从加了噪声的副本学到分数，一步训练，有噪声和没噪声的朗之万采样，从大噪声到小噪声的退火，以及学到的分数在哪里失灵。'
		),
		// every batch draws new noise, so the loss jumps from step to step: the Training section also draws the 50-step mean
		// that step 6 narrates (K2 page.trainingCurve)
		trainingCurve: { smooth: 50 }
	},
	glossary: [
		{ id: 'density', term: L('density', '密度'), def: L('How crowded the points are near a place: high where many points lie close together, near 0 far from all of them.', '某个位置附近点的拥挤程度：许多点挨在一起的地方高，远离所有点的地方接近 0。') },
		{ id: 'score-function', term: L('score', '分数'), def: L('The gradient of the log density, ∇ log p(x): at every place, an arrow pointing the way the density rises fastest.', '对数密度的梯度 ∇ log p(x)：在每个位置，一个指向密度上升最快方向的箭头。'), related: ['density', 'gradient'] },
		{ id: 'denoising-score-matching', term: L('denoising score matching', '去噪分数匹配'), def: L('Learning the score from samples: add noise of size σ to a data point and train a network to point back, (x − x̃) / σ².', '从样本学分数：给数据点加上大小为 σ 的噪声，训练网络指回去，即 (x − x̃) / σ²。'), related: ['score-function'] },
		{ id: 'langevin-dynamics', term: L('Langevin dynamics', '朗之万动力学'), def: L('A sampler that repeats x ← x + (α/2)·score + √α·z with fresh noise z: a climb toward dense places plus a random jiggle.', '一种采样方法，反复执行 x ← x + (α/2)·分数 + √α·z（z 每次重新抽）：朝密集处爬一点，再随机抖一下。'), related: ['score-function'] },
		{ id: 'annealing', term: L('annealing', '退火'), def: L('Sampling with a large noise level first and lowering it step by step, so the samples settle from a rough shape onto the fine one.', '先用大的噪声水平采样，再一步步降低，让样本从粗略的形状慢慢落到精细的形状上。'), related: ['langevin-dynamics'] }
	],
	catalog: {
		// the count is a claim test (claims.test.ts); the runtime text reuses the tech-zh row of the other own-core pages
		params: L('3 → 64 → 64 → 2 (4,546 parameters)', '3 → 64 → 64 → 2（4,546 个参数）'),
		size: L('No files: the network trains in your browser', '没有文件：网络在你的浏览器里训练'),
		runtime: L('Own autograd core in a worker', '在 Worker 线程中运行的自研自动求导核心'),
		shows: L('The density and score of your points, denoising score matching with real numbers, the learned score field at six noise levels, Langevin chains with step-size and noise sliders, and annealing from large to small noise.', '你的点的密度和分数、带真实数值的去噪分数匹配、六个噪声水平上学到的分数场、带步长和噪声滑块的朗之万链，以及从大噪声到小噪声的退火。')
	},
	// ideas only: no text, figures or code of these papers is used
	credits: [
		{
			name: 'Song and Ermon, Generative Modeling by Estimating Gradients of the Data Distribution (NeurIPS 2019)',
			url: 'https://arxiv.org/abs/1907.05600',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'a noise-conditional score network trained at many noise levels, and annealed Langevin sampling with step α·σ²/σ_L²',
			attribution: 'Yang Song, Stefano Ermon'
		},
		{
			name: 'Vincent, A Connection Between Score Matching and Denoising Autoencoders (Neural Computation 2011)',
			url: 'https://doi.org/10.1162/NECO_a_00142',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'denoising score matching: the target (x − x̃) / σ² for a noisy point x̃',
			attribution: 'Pascal Vincent'
		},
		{
			name: 'Ho, Jain and Abbeel, Denoising Diffusion Probabilistic Models (NeurIPS 2020)',
			url: 'https://arxiv.org/abs/2006.11239',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'the link drawn in the lesson: a network that predicts the noise ε is a score network, s = −ε̂ / σ',
			attribution: 'Jonathan Ho, Ajay Jain, Pieter Abbeel'
		}
	]
};

export default manifest;
