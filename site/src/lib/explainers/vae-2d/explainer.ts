// vae-2d: the package manifest (design §4.9). A small variational autoencoder (VAE) with a 2-D latent space, trained
// offline on five Quick, Draw! classes (pipeline/vae-2d), traced live on the learner's drawing:
//   fwd   the exported model on the drawing and on the noise ε (two inputs; the local drawing28 adapter adds ε)
// The map of test drawings (latent.json) and the measured numbers (facts.json) come from pipeline/vae-2d/stats.py.
import type { ExplainerManifest } from '../../atlas/types/index.ts';

const manifest: ExplainerManifest = {
	slug: 'vae-2d',
	type: 'model',
	kinds: ['trace'],
	quality: 'reviewed',
	assets: { dir: 'models/vae-2d', auto: ['spec.json', 'weights.bin', 'facts.json', 'latent.json'] },
	input: { adapter: 'drawing28', presets: 'presets.json' },
	output: { view: 'vae-2d/pair', layer: 'out' },
	sources: { fwd: { kind: 'trace' } },
	page: {
		goals: [
			{ en: 'Say what the encoder outputs for a drawing: a centre μ and a spread σ on a map with only 2 numbers per point.', zh: '说出编码器对一幅画输出什么：在每个点只有 2 个数的地图上，一个中心 μ 和一个散布 σ。' },
			{ en: 'Work out one sample z = μ + σ·ε with real numbers, and say why the randomness is kept in ε.', zh: '用真实的数算出一次采样 z = μ + σ·ε，并说出为什么把随机性放在 ε 里。' },
			{ en: 'Name the two parts of the training loss (redrawing error and KL) and what each one pushes the model to do.', zh: '说出训练损失的两部分（重画误差和 KL），以及它们各自推动模型做什么。' },
			{ en: 'Read the latent map: find a border between two kinds of drawing, and say why a 2-D VAE redraws blurry, typical drawings.', zh: '读懂隐空间地图：找到两类画之间的边界，并说出为什么二维 VAE 重画出的是模糊、典型的画。' }
		],
		prequestion: {
			id: 'vae-2d-pre',
			question: {
				en: 'This model must squeeze your whole drawing, 784 numbers, into just 2 numbers and then draw it again from those 2. What will the redrawing look like?',
				zh: '这个模型要把你的整幅画，784 个数，挤成仅仅 2 个数，再从这 2 个数把它画出来。重画出来的会是什么样？'
			},
			options: [
				{ en: 'An exact copy of my drawing', zh: '和我的画一模一样' },
				{ en: 'A blurry, typical drawing of the same kind', zh: '一幅模糊的、同类的典型画' },
				{ en: 'Random noise', zh: '随机的噪点' }
			],
			answer: 1,
			explain: {
				en: 'Two numbers can say roughly what and where on the map, not every stroke. The decoder draws what drawings near that place on the map look like on average, so the result is a soft, typical version.',
				zh: '两个数只能大致说明“是什么、在地图上哪里”，记不下每一笔。解码器画出的是地图上那附近的画平均起来的样子，所以结果是一幅柔和的、典型的画。'
			}
		},
		tryThis: [
			{
				id: 'house',
				title: { en: 'A house', zh: '一座房子' },
				setup: { preset: 'heldout-house' },
				predict: { en: 'The VAE never saw a house. What will it redraw?', zh: '这个 VAE 从没见过房子。它会重画出什么？' },
				look: { en: 'It lands between kinds it knows, and the redrawing is a faint, blurry mix of them: the map has no place for a house.', zh: '它落在它认识的几类之间，重画出来的是它们淡淡的、模糊的混合：地图上没有房子的位置。' },
				goto: { step: 'limits' }
			},
			{
				id: 'dot',
				title: { en: 'One dot', zh: '一个点' },
				setup: { input: { strokes: [{ points: [{ x: 0.5, y: 0.5 }] }] } },
				predict: { en: 'A single dot is nothing like the training drawings. Where will it land on the map?', zh: '一个点和训练用的画完全不像。它会落在地图上的哪里？' },
				look: { en: 'It still gets a place among the five kinds and is redrawn as one of them, but its σ is several times the usual one (step 3): the encoder is much less sure.', zh: '它照样在五类之间得到一个位置，并被重画成其中一类；但它的 σ 是平常的好几倍（第 3 步）：编码器没那么确定了。' },
				goto: { step: 'map' }
			}
		],
		quiz: [
			{
				q: { en: 'How many numbers does the encoder pass on to the decoder?', zh: '编码器传给解码器多少个数？' },
				options: [
					{ en: '2 (one point z)', zh: '2 个（一个点 z）' },
					{ en: '4 (μ and log σ²)', zh: '4 个（μ 和 log σ²）' },
					{ en: '784', zh: '784 个' }
				],
				answer: 0,
				explain: { en: 'The encoder computes μ and log σ² (4 numbers), but only the sample z = μ + σ·ε, 2 numbers, reaches the decoder.', zh: '编码器算出 μ 和 log σ²（4 个数），但传到解码器的只有采样 z = μ + σ·ε，2 个数。' }
			},
			{
				q: { en: 'What does the KL part of the loss push the encoder to do?', zh: '损失里的 KL 部分推动编码器做什么？' },
				options: [
					{ en: 'Keep every cloud near the centre of the map and not too small', zh: '让每团云靠近地图中心、又不要太小' },
					{ en: 'Make the redrawing sharper', zh: '让重画更清晰' },
					{ en: 'Put each kind in its own corner', zh: '把每一类放到各自的角落' }
				],
				answer: 0,
				explain: { en: 'KL compares the cloud N(μ, σ²) with the standard cloud N(0, 1): it grows when μ is far from 0 and when σ is far from 1.', zh: 'KL 把云 N(μ, σ²) 和标准云 N(0, 1) 比较：μ 离 0 越远、σ 离 1 越远，它就越大。' }
			}
		],
		misconception: {
			claim: { en: 'A VAE stores each training drawing and looks up the closest one.', zh: 'VAE 把每幅训练画存起来，再查出最接近的一幅。' },
			truth: { en: 'It stores no drawings: the decoder computes a drawing from any point z, also from places where no training drawing landed.', zh: '它不存任何画：解码器从任何一个点 z 算出一幅画，哪怕那个位置上没有落下任何训练画。' },
			step: 'drag'
		},
		// K1: names for the Training section (spec.training.metrics; test_loss and test_n are named by the kit)
		metrics: {
			test_bce: { label: { en: 'Redrawing error (BCE) on the test set, z sampled', zh: '测试集上的重画误差（BCE），z 为采样' }, fmt: 'n4' },
			test_bce_at_mu: { label: { en: 'Redrawing error (BCE) on the test set, z = μ', zh: '测试集上的重画误差（BCE），z = μ' }, fmt: 'n4' },
			test_kl: { label: { en: 'KL on the test set', zh: '测试集上的 KL' }, fmt: 'n4' }
		},
		lessonNote: {
			en: 'Ten steps: your drawing is squeezed into a point with a spread, a sample is drawn and decoded, the loss is computed with your numbers, then you explore the map: drag, cross a border, walk between two drawings, and see the limits.',
			zh: '十步：你的画被挤成一个带散布的点，采样后再解码，用你的数算出损失；然后探索地图：拖动、跨过边界、在两幅画之间走一走，最后看局限。'
		}
	},
	glossary: [
		{ id: 'latent-space', term: { en: 'latent space', zh: '隐空间' }, def: { en: 'The space of the code numbers a model squeezes its input into; here a plane, 2 numbers per point.', zh: '模型把输入压缩成的编码数所在的空间；这里是一个平面，每个点 2 个数。' } },
		{ id: 'encoder-decoder', term: { en: 'encoder and decoder', zh: '编码器与解码器' }, def: { en: 'The encoder turns an input into a few code numbers; the decoder turns code numbers back into an input-like output.', zh: '编码器把输入变成少数几个编码数；解码器再把编码数变回和输入同类的输出。' } },
		{ id: 'reparameterization', term: { en: 'reparameterization trick', zh: '重参数化技巧' }, def: { en: 'Writing a random sample as z = μ + σ·ε with ε drawn from N(0, 1), so z depends smoothly on μ and σ and training can adjust them.', zh: '把随机采样写成 z = μ + σ·ε，其中 ε 从 N(0, 1) 中抽取，这样 z 平滑地依赖 μ 和 σ，训练就能调整它们。' } },
		{ id: 'kl-divergence', term: { en: 'KL divergence', zh: 'KL 散度' }, def: { en: 'A measure of how different one probability distribution is from another; 0 when they are the same.', zh: '衡量一个概率分布和另一个有多不同的量；两者相同时为 0。' } },
		{ id: 'bce', term: { en: 'binary cross-entropy', zh: '二元交叉熵' }, def: { en: 'A loss for numbers between 0 and 1: small when the predicted value is close to the target, large when it is confidently wrong.', zh: '用于 0 到 1 之间数值的损失：预测值接近目标时小，很自信却错了时大。' } }
	],
	credits: [
		{
			name: 'Quick, Draw! dataset',
			url: 'https://github.com/googlecreativelab/quickdraw-dataset',
			license: 'CC BY 4.0',
			usedFor: 'VAE training and test drawings (star, ladder, umbrella, eyeglasses, t-shirt; TinyCNN’s split), the latent map and the preset drawings',
			attribution: 'Quick, Draw! dataset by Google Creative Lab'
		},
		{
			name: 'VAE Explainer',
			url: 'https://github.com/xnought/vae-explainer',
			license: 'MIT',
			usedFor: 'Ideas only (no code): the reparameterization trick as its own step, and dragging a point through the latent space',
			attribution: 'VAE Explainer (github.com/xnought/vae-explainer), MIT License'
		}
	],
	catalog: {
		hook: { en: 'Draw a star, a ladder, an umbrella, glasses or a T-shirt, then drag a point through the 2-D map it lives on.', zh: '画一颗星、一架梯子、一把伞、一副眼镜或一件 T 恤，然后在它所在的二维地图上拖动一个点。' },
		runtime: { en: 'Own JS forward pass (checked against PyTorch)', zh: '自研 JS 前向计算（已与 PyTorch 核对）' },
		params: { en: '435,988', zh: '435,988' },
		size: { en: '1.7 MB of weights', zh: '权重 1.7 MB' }
	}
};

export default manifest;
