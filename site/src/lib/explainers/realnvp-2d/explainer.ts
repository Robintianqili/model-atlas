// realnvp-2d: the manifest. A RealNVP normalizing flow on the learner's points: eight affine coupling layers trained by
// maximum likelihood live in the package worker (kind B), then the trained flow is run step by step (kind C): your
// points through the layers to noise, with the exact log-likelihood, and noise back through the layers to samples.
// Sources (seed 1 everywhere, so every train source starts from the same weights):
//   train       trainer 'fit': Adam 3e-3, batch 128, 1,200 steps, a snapshot every 50
//   sgd1        one plain gradient-descent step (lr 0.1) in fine detail: the batch, the loss, the gradients, the update
//   flow        your points data -> noise, one frame per layer, then log p on a grid (process 'flow', after training)
//   sample      500 noise draws pulled back through the layers (process 'sample', after training)
//   rings       the same trainer on the 'circles' points (two rings), whatever you painted (the limits step)
//   ringSample  500 samples of the two-rings flow
import type { ExplainerManifest, ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { presetRaw } from '../../atlas/adapters/points2d/adapter.ts';
import { dataOf } from './nvp.ts';

const L = (en: string, zh: string) => ({ en, zh });
/** the two rings of the limits step */
export const RINGS = presetRaw('circles');
/** the final weights of a train source and the points it trained on, for a process source */
export function trainedOf(rt: ExplainerRuntime, source: string, raw: unknown) {
	const t = rt.sources[source]?.tape;
	const params = t?.length ? (t.state(t.length - 1) as { params?: Record<string, Tensor> }).params : undefined;
	return params ? { params, xy: Array.from(dataOf(raw).xy) } : null;
}
export const trained = (rt: ExplainerRuntime) => trainedOf(rt, 'train', rt.input);
const fit = (opt: Record<string, unknown> = {}, detail: 'coarse' | 'fine' = 'coarse') => ({ kind: 'train' as const, trainer: 'fit', detail, opt: { seed: 1, ...opt } });

const manifest: ExplainerManifest = {
	slug: 'realnvp-2d',
	type: 'model',
	kinds: ['train', 'process'],
	quality: 'reviewed',
	input: { adapter: 'points2d', start: presetRaw('smiley'), options: { classes: 1, presets: ['smiley', 'ring', 'moons', 'gaussians'] } },
	sources: {
		train: fit(),
		sgd1: fit({ optimizer: 'sgd', lr: 0.1, steps: 1 }, 'fine'),
		flow: { kind: 'process', process: 'flow', after: 'train', seed: 1, input: trained },
		sample: { kind: 'process', process: 'sample', after: 'train', seed: 1, input: trained },
		rings: { ...fit({ data: RINGS }), runOnEmpty: true },
		ringSample: { kind: 'process', process: 'sample', after: 'rings', seed: 1, input: (rt) => trainedOf(rt, 'rings', RINGS) }
	},
	page: {
		goals: [
			L('Work out one affine coupling layer by hand (y′ = y·e^s + t) and undo it without inverting the network inside it.', '亲手算出一个仿射耦合层（y′ = y·e^s + t），并在不对里面的网络求逆的情况下把它撤销。'),
			L('Compute the exact log-likelihood of a point as log N(z) + Σs, and say why the stretches s enter it.', '把一个点精确的对数似然算成 log N(z) + Σs，并说出各层的拉伸 s 为什么要算进去。'),
			L('Sample by running the layers backwards, and explain why one blob of noise cannot cleanly become two separate rings.', '把各层倒着跑来采样，并解释为什么一团噪声没法干净地变成两个分开的圆环。')
		],
		prequestion: {
			id: 'realnvp-2d-pre',
			question: L('A model will turn a round cloud of noise into points shaped like yours, and back. What must every one of its layers be?', '一个模型要把一团圆圆的噪声变成形状像你的点的新点，还要能变回去。它的每一层必须是什么样的？'),
			options: [L('As large as possible', '越大越好'), L('Invertible: every output comes from exactly one input', '可逆的：每个输出都恰好来自一个输入'), L('Random, so the samples differ', '随机的，这样样本才各不相同')],
			answer: 1,
			explain: L(
				'Invertible. Then the same layers carry your points to the noise (and give the exact likelihood on the way) and carry fresh noise back to new points. The randomness comes only from the noise you start from.',
				'可逆的。这样同样的层既能把你的点送到噪声那边（顺路给出精确的似然），也能把新的噪声送回来，变成新的点。随机性只来自出发时的噪声。'
			)
		},
		lessonNote: L(
			'Eleven steps: your points and the noise, one coupling layer and its inverse, eight layers that bend your shape into the noise, the log-determinant and the exact log-likelihood, one training step and the whole training, sampling backwards, the two-rings limit, and how this compares with flow matching and score matching.',
			'十一步：你的点和噪声，一个耦合层和它的逆，把你的形状弯成噪声的八层，对数行列式和精确的对数似然，一步训练和整个训练，倒着采样，两个圆环的局限，以及它和流匹配、分数匹配的比较。'
		),
		metrics: { nllAll: { label: L('Mean −log p over all your points', '你所有的点上 −log p 的平均'), fmt: 'n3' } },
		trainingCurve: { metrics: ['loss', 'nllAll'], labels: { loss: L('−log p of the batch', '这一批的 −log p'), nllAll: L('mean −log p over all your points', '所有点上 −log p 的平均') } }
	},
	glossary: [
		{ id: 'normalizing-flow', term: L('normalizing flow', '归一化流'), def: L('A generative model made of invertible layers that carry data points to plain noise and back; it gives the exact likelihood of any point and samples by running the layers backwards.', '由可逆层组成的生成模型，它把数据点送到普通的噪声，也能送回来；它能给出任何一个点精确的似然，并把各层倒着跑来采样。'), related: ['generative-model', 'coupling-layer'] },
		{ id: 'coupling-layer', term: L('coupling layer', '耦合层'), def: L('An invertible layer that keeps part of its input as it is and lets a small network of that part stretch and shift the rest: y′ = y·e^s + t. Undoing it is (y′ − t)·e^−s.', '一种可逆层：保留输入的一部分不动，用这部分经过一个小网络算出的量去拉伸和平移其余部分：y′ = y·e^s + t。撤销它就是 (y′ − t)·e^−s。'), related: ['normalizing-flow'] },
		{ id: 'log-determinant', term: L('log-determinant', '对数行列式'), def: L('How much a layer scales a small area around a point, on a log scale. For a coupling layer it is just s; the density must shrink by the same factor, so it enters the log-likelihood.', '一层把某点周围的小面积放大了多少倍，取对数。对耦合层来说它就是 s；密度必须缩小同样的倍数，所以它要算进对数似然。'), related: ['density', 'log-likelihood'] },
		{ id: 'log-likelihood', term: L('log-likelihood', '对数似然'), def: L('log p(x): how probable a model says a point is, on a log scale. Higher means more likely; a flow computes it exactly.', 'log p(x)：模型认为一个点有多大可能，取对数。越高越可能；流模型能精确地算出它。'), related: ['density', 'maximum-likelihood'] },
		{ id: 'maximum-likelihood', term: L('maximum likelihood', '最大似然'), def: L('Training a model by making the data as probable as possible: minimise the mean of −log p over the data points.', '通过让数据出现的概率尽可能大来训练模型：把数据点上 −log p 的平均降到最小。'), related: ['log-likelihood', 'loss'] }
	],
	catalog: {
		// the count is a claim test (claims.test.ts); the runtime text reuses the tech-zh row of the other own-core pages
		params: L('8 coupling layers, each 1 → 24 → 24 → 2 (5,584 parameters)', '8 个耦合层，每层 1 → 24 → 24 → 2（5,584 个参数）'),
		size: L('No files: the network trains in your browser', '没有文件：网络在你的浏览器里训练'),
		runtime: L('Own autograd core in a worker', '在 Worker 线程中运行的自研自动求导核心'),
		shows: L('One affine coupling layer worked out and undone, eight layers warping a grid over your points into noise, the log-determinant and the exact log-likelihood, maximum-likelihood training, sampling by inverting, and the two-rings limit.', '算出并撤销一个仿射耦合层，八层把覆盖你的点的网格扭成噪声，对数行列式和精确的对数似然，最大似然训练，求逆采样，以及两个圆环的局限。')
	},
	// ideas only: no text, figures or code of these papers is used
	credits: [
		{
			name: 'Dinh, Sohl-Dickstein and Bengio, Density estimation using Real NVP (ICLR 2017)',
			url: 'https://arxiv.org/abs/1605.08803',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'affine coupling layers y′ = y·e^s + t with alternating masks, the log-determinant Σs, the exact log-likelihood and sampling by inverting',
			attribution: 'Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio'
		},
		{
			name: 'Dinh, Krueger and Bengio, NICE: Non-linear Independent Components Estimation (ICLR 2015 workshop)',
			url: 'https://arxiv.org/abs/1410.8516',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'coupling layers that keep one part of the input and are inverted without inverting their network',
			attribution: 'Laurent Dinh, David Krueger, Yoshua Bengio'
		}
	]
};

export default manifest;
