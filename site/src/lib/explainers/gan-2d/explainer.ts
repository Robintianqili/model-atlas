// gan-2d: the manifest. A generator G and a discriminator D (two small MLPs, trainer.ts) are trained against each other
// in your browser on the points you brush (kind B); the lesson slows one D step and one G step down (fine detail) and
// shows mode collapse on eight blobs. Every source has seed 1, so they all start from the same weights.
//   train   the page's GAN on your points: Adam, D at lr 0.01, G at lr 0.002, 3000 steps (D and G take turns)
//   slow    the same run, first two steps only, in fine detail (forward, loss, backward, update)
//   same    eight blobs, G as fast as D (lrG 0.01): mode collapse;  slower: eight blobs at the page's settings
// runOnEmpty (K1): every source also runs while you have no points. same and slower train on fixed data (opt.data); on
// train and slow every gradient is then 0 (trainer.ts), so nothing moves, steps 2-4 still show the noise, G and D, and the
// narration says "— (no points)" where a number needs your points (without it every step would read "…").
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { EIGHT_CENTRES, presetOf } from './data.ts';

const L = (en: string, zh: string) => ({ en, zh });
const EIGHT = { data: presetOf('eight'), centres: EIGHT_CENTRES };

const manifest: ExplainerManifest = {
	slug: 'gan-2d',
	type: 'model',
	kinds: ['train'],
	quality: 'reviewed',
	assets: { dir: 'models/gan-2d', auto: [] },
	input: { adapter: 'points2d', presets: 'presets.json', start: presetOf('ring'), options: { classes: 1 } },
	sources: {
		train: { kind: 'train', trainer: 'gan', detail: 'coarse', runOnEmpty: true, opt: { seed: 1 } },
		slow: { kind: 'train', trainer: 'gan', detail: 'fine', runOnEmpty: true, opt: { seed: 1, steps: 2 } },
		same: { kind: 'train', trainer: 'gan', detail: 'coarse', runOnEmpty: true, opt: { seed: 1, lrG: 0.01, ...EIGHT } },
		slower: { kind: 'train', trainer: 'gan', detail: 'coarse', runOnEmpty: true, opt: { seed: 1, ...EIGHT } }
	},
	page: {
		goals: [
			L('Say what the generator and the discriminator each take in and put out, and trace one noise point to the point the generator makes from it.', '说出生成器和判别器各自的输入和输出，并跟着一个噪声点，看生成器把它变成平面上的哪个点。'),
			L('Read the discriminator’s heat map and compute its loss on one batch from the scores it gives real and generated points.', '读懂判别器的热力图，并根据它给真实点和生成点的分数，算出它在一批数据上的损失。'),
			L('Predict which way one generator step moves the generated points, and explain why the generator never needs to see the real points.', '预测生成器的一步训练会把生成的点往哪里推，并解释为什么生成器从来不需要看到真实的点。'),
			L('Recognize mode collapse and explain why a GAN’s two losses do not fall to zero.', '认出模式坍缩，并解释为什么 GAN 的两个损失都不会降到零。')
		],
		prequestion: {
			id: 'gan-2d-pre',
			question: L('A GAN learns to make new points that look like your points. Does its generator ever look at your points?', 'GAN 学着造出和你的点相像的新点。它的生成器会直接看你的点吗？'),
			options: [L('Yes: it copies them, with a little noise', '会：它把它们抄下来，再加一点噪声'), L('No: it only hears how a second network, the discriminator, judges its points', '不会：它只听另一个网络（判别器）怎么评判它造的点')],
			answer: 1,
			explain: L(
				'The generator never sees the real points. The discriminator sees both, learns to tell them apart, and the generator learns only from the discriminator’s gradient: which way to move its points so that they look more real.',
				'生成器从来看不到真实的点。判别器两种点都看，学着把它们分开；生成器只从判别器的梯度里学习：把点往哪个方向移，才显得更像真的。'
			)
		},
		tryThis: [
			{
				id: 'two',
				title: L('Two blobs', '两团点'),
				setup: { input: presetOf('two') },
				predict: L('Will the generator split its points between the two blobs, or stretch one sheet across the gap?', '生成器会把点分到两团里，还是把一张“布”拉过两团之间的空隙？'),
				look: L('Watch the bent grid (the generator’s sheet) and the points it leaves in the gap between the blobs.', '看那张弯曲的网格（生成器的“布”），以及它留在两团之间空隙里的点。'),
				goto: { step: 'watch' }
			}
		],
		quiz: [
			{
				q: L('The discriminator gives every point, real or generated, the score 0.5. What is its loss?', '判别器给每个点（无论真假）都打 0.5 分。它的损失是多少？'),
				options: [L('0', '0'), L('2 ln 2 ≈ 1.39', '2 ln 2 ≈ 1.39'), L('0.5', '0.5')],
				answer: 1,
				explain: L('−ln 0.5 for the real points plus −ln(1 − 0.5) for the generated ones: ln 2 + ln 2 ≈ 1.39.', '真实点贡献 −ln 0.5，生成点贡献 −ln(1 − 0.5)：ln 2 + ln 2 ≈ 1.39。')
			}
		],
		misconception: {
			claim: L('When training goes well, both losses fall towards 0.', '训练顺利时，两个损失都会降向 0。'),
			truth: L('The two networks work against each other: when the generator gets good, the discriminator can no longer tell real from fake, so its loss stays near 2 ln 2 ≈ 1.39 instead of falling to 0.', '两个网络彼此对抗：生成器变好时，判别器就分不清真假，所以它的损失停在 2 ln 2 ≈ 1.39 附近，而不是降到 0。'),
			step: 'game'
		},
		// the auto Training section would draw one loss curve that mixes D's and G's steps; the lesson's 'game' step shows both
		sections: { hide: ['training'] },
		lessonNote: L(
			'Nine steps: the real points, the noise, the generator, the discriminator, one step of each slowed down, the whole game, how it differs from flow matching, and mode collapse.',
			'九步：真实的点、噪声、生成器、判别器，各放慢看一步训练，然后看整场博弈、它和流匹配有什么不同，以及模式坍缩。'
		)
	},
	glossary: [
		{ id: 'gan', term: L('GAN (generative adversarial network)', 'GAN（生成对抗网络）'), def: L('Two networks trained against each other: a generator makes samples, a discriminator tells them from real data.', '两个互相对抗着训练的网络：生成器造样本，判别器把它们和真实数据分开。') },
		{ id: 'generator', term: L('generator', '生成器'), def: L('The network that turns random noise into samples (here: points on a plane).', '把随机噪声变成样本（这里是平面上的点）的网络。'), related: ['gan'] },
		{ id: 'discriminator', term: L('discriminator', '判别器'), def: L('The network that scores a sample: its estimated probability that the sample is real.', '给样本打分的网络：它估计样本是真实数据的概率。'), related: ['gan'] },
		{ id: 'noise', term: L('noise (latent input)', '噪声（潜在输入）'), def: L('The random numbers a generator starts from; a new draw gives a new sample.', '生成器的出发点：一组随机数，每抽一次就得到一个新样本。'), related: ['generator'] },
		{ id: 'mode-collapse', term: L('mode collapse', '模式坍缩'), def: L('A GAN failure: the generator covers only some of the data’s clusters (modes) and ignores the rest.', 'GAN 的一种失败：生成器只覆盖数据里的一部分团（模式），忽略其余的。'), related: ['gan'] }
	],
	// runtime: the catalog's existing string (MLP playground, TinyForecaster); GAN Lab is credited below (ideas only)
	catalog: {
		shows: L('D’s heat map with its ½ line, G’s sheet (the bent noise grid), gradient arrows, one D step and one G step with their numbers, both loss curves, and mode collapse on eight blobs.', '判别器的热力图和 ½ 线、生成器的“布”（弯曲的噪声网格）、梯度箭头、带数字的一步判别器训练和一步生成器训练、两条损失曲线，以及八团点上的模式坍缩。'),
		runtime: L('Own autograd core in a worker', '在 Worker 线程中运行的自研自动求导核心'),
		size: L('No weights; 0.3 KB of presets', '无权重文件；预设 0.3 KB'),
		params: L('G 2 → 16 → 16 → 2 and D 2 → 16 → 16 → 1 (691 parameters)', '生成器 2 → 16 → 16 → 2，判别器 2 → 16 → 16 → 1（共 691 个参数）')
	},
	// ideas only (the D heat map, the generator's bent grid, gradient arrows on generated points, slow-motion D and G
	// steps); no code was copied
	credits: [
		{
			name: 'GAN Lab',
			url: 'https://github.com/poloclub/ganlab',
			license: 'Apache-2.0',
			usedFor: 'Visualization ideas (discriminator heat map, generator manifold, gradient arrows, slow-motion steps); reimplemented on the Model Atlas autograd core',
			attribution: 'Minsuk Kahng, Nikhil Thorat, Duen Horng (Polo) Chau, Fernanda Viégas and Martin Wattenberg (GAN Lab, Georgia Tech and Google)'
		}
	]
};

export default manifest;
