// vq-vae: the package manifest. A tiny VQ-VAE trained offline on five Quick, Draw! classes (pipeline/vq-vae), traced
// live on the learner's drawing:
//   fwd   the exported model on the drawing (the kit's drawing28 adapter): 16 patches -> z_e -> the nearest of 32 codes
//         -> the decoder -> the redrawing. Swaps, code pictures and the gradient are computed from the same weights (vq.ts).
// The measured numbers (facts.json) and the test-set code usage (usage.json) come from pipeline/vq-vae/stats.py.
import type { ExplainerManifest } from '../../atlas/types/index.ts';

const manifest: ExplainerManifest = {
	slug: 'vq-vae',
	type: 'model',
	kinds: ['trace'],
	quality: 'reviewed',
	assets: { dir: 'models/vq-vae', auto: ['spec.json', 'weights.bin', 'facts.json', 'usage.json'] },
	input: { adapter: 'drawing28', presets: 'presets.json' },
	output: { view: 'vq-vae/grid', layer: 'out' },
	sources: { fwd: { kind: 'trace' } },
	page: {
		goals: [
			{ en: 'Say what the encoder gives for each 7×7 patch of a drawing: a short list of numbers z_e.', zh: '说出编码器对画里每个 7×7 图块给出什么：一小串数 z_e。' },
			{ en: 'Find the code a patch snaps to by working out its squared distance to the codebook entries.', zh: '算出一个图块到码本条目的距离平方，找到它吸附到的码字。' },
			{ en: 'Read a drawing as 16 code indices, and predict what swapping one index changes in the redrawing.', zh: '把一幅画读成 16 个码字编号，并预测换掉一个编号会让重画改变什么。' },
			{ en: 'Explain how training gets a gradient past the nearest-code lookup (the straight-through estimator), and name two limits: rough redrawings and codes that are rarely or never used.', zh: '解释训练怎样让梯度越过“查最近码字”这一步（直通估计器），并说出两个局限：重画粗糙，以及很少用到或从不使用的码字。' }
		],
		prequestion: {
			id: 'vq-vae-pre',
			question: {
				en: 'This model stores your whole drawing as 16 small whole numbers, each picked from a list of 32. What will its redrawing look like?',
				zh: '这个模型把你的整幅画存成 16 个小整数，每个都从一张 32 项的清单里挑。它重画出来会是什么样？'
			},
			options: [
				{ en: 'An exact copy of my drawing', zh: '和我的画一模一样' },
				{ en: 'My drawing rebuilt from stock pieces: the strokes roughly in place, but smudged', zh: '用现成的小块拼回我的画：笔画大致在原位，但糊了' },
				{ en: 'A typical drawing of the same kind, not mine', zh: '一幅同类的典型画，而不是我的' }
			],
			answer: 1,
			explain: {
				en: 'Each of the 16 patches keeps only which of 32 stock pieces it is closest to. That keeps where the strokes are, unlike a 2-number VAE, but not their exact shape, so the redrawing is your drawing rebuilt from approximate pieces.',
				zh: '16 个图块各自只记住它最接近 32 种现成小块里的哪一种。这样能保住笔画在哪里（这点和只有 2 个数的 VAE 不同），却保不住笔画的确切形状，所以重画是用近似的小块把你的画拼回来。'
			}
		},
		tryThis: [
			{
				id: 'house',
				title: { en: 'A house', zh: '一座房子' },
				setup: { preset: 'heldout-house' },
				predict: { en: 'The model never saw a house. Will it still redraw one?', zh: '这个模型从没见过房子。它还能重画出房子吗？' },
				look: { en: 'Mostly yes: its codes are pieces of strokes, not whole kinds, so a new kind is rebuilt from the same pieces.', zh: '大体可以：它的码字是笔画的小片段，而不是整类的画，所以新的类别也能用同样的小块拼出来。' },
				goto: { step: 'decode' }
			},
			{
				id: 'dot',
				title: { en: 'One dot', zh: '一个点' },
				setup: { input: { strokes: [{ points: [{ x: 0.5, y: 0.5 }] }] } },
				predict: { en: 'One dot fills only part of one patch. How many different codes will the 16 patches use?', zh: '一个点只占一个图块的一部分。16 个图块会用到几种不同的码字？' },
				look: { en: 'The empty patches all share one code; only the patches the dot touches take another.', zh: '空的图块全都共用一个码字；只有被这个点碰到的图块才取别的码字。' },
				goto: { step: 'indices' }
			}
		],
		quiz: [
			{
				q: { en: 'What does the decoder receive from the encoder?', zh: '解码器从编码器那里拿到什么？' },
				options: [
					{ en: 'The 8 numbers z_e of every patch', zh: '每个图块的 8 个数 z_e' },
					{ en: 'One codebook row per patch, chosen by the patch’s index', zh: '每个图块一行码本，按图块的编号取出' },
					{ en: 'The 784 pixels', zh: '784 个像素' }
				],
				answer: 1,
				explain: { en: 'z_e is only used to pick the nearest code; the decoder sees the codebook rows of the 16 indices, z_q.', zh: 'z_e 只用来挑最近的码字；解码器看到的是 16 个编号对应的码本行，即 z_q。' }
			},
			{
				q: { en: 'Why does training need the straight-through estimator?', zh: '训练为什么需要直通估计器？' },
				options: [
					{ en: 'Picking the nearest code has no useful slope, so no gradient would reach the encoder', zh: '挑最近码字这一步没有可用的斜率，梯度到不了编码器' },
					{ en: 'To make the codebook larger', zh: '为了让码本变大' },
					{ en: 'To add noise like a VAE', zh: '为了像 VAE 那样加噪声' }
				],
				answer: 0,
				explain: { en: 'A small change of z_e usually keeps the same code, so the lookup’s slope is 0. Training copies the gradient at z_q to z_e instead.', zh: 'z_e 稍微变一点，通常还是同一个码字，所以这一步的斜率是 0。训练改为把 z_q 处的梯度直接复制给 z_e。' }
			}
		],
		misconception: {
			claim: { en: 'Each code stands for a whole kind of drawing, like “star” or “ladder”.', zh: '每个码字代表一整类画，比如“星星”或“梯子”。' },
			truth: { en: 'A code stands for a small piece of stroke in one 7×7 patch; drawings of different kinds reuse the same pieces in different places.', zh: '一个码字只代表一个 7×7 图块里的一小段笔画；不同类别的画在不同的位置重复使用同样的小块。' },
			step: 'codebook'
		},
		metrics: {
			test_bce_px: { label: { en: 'Redrawing error (BCE per pixel) on the test set', zh: '测试集上的重画误差（每像素 BCE）' }, fmt: 'n4' },
			test_commit: { label: { en: 'Squared distance of z_e to its code, per number, on the test set', zh: '测试集上 z_e 到其码字的距离平方（每个数）' }, fmt: 'n4' },
			codes_used: { label: { en: 'Codes used on the test set', zh: '测试集上用到的码字数' }, fmt: 'int' }
		},
		lessonNote: {
			en: 'Ten steps: your drawing is cut into 16 patches, each becomes 8 numbers and snaps to the nearest of 32 codes; the decoder redraws it from those 16 indices. Then you swap a code, see how training gets past the lookup, how the codes are used, and the limits.',
			zh: '十步：你的画被切成 16 个图块，每块变成 8 个数，再吸附到 32 个码字里最近的那个；解码器从这 16 个编号把画重画出来。然后你换一个码字，看训练怎样越过这一步查找、码字用得怎样，最后看局限。'
		}
	},
	glossary: [
		{ id: 'vector-quantization', term: { en: 'vector quantization', zh: '向量量化' }, def: { en: 'Replacing a list of numbers by the nearest entry of a fixed, learned list (the codebook), so it can be stored as one index.', zh: '把一串数换成一张固定的、学来的清单（码本）里离它最近的那一项，于是只需存一个编号。' }, related: ['codebook'] },
		{ id: 'codebook', term: { en: 'codebook', zh: '码本' }, def: { en: 'The learned table of code vectors a VQ-VAE snaps to; each row is one code, named by its index.', zh: 'VQ-VAE 吸附用的、学来的码向量表；每一行是一个码字，用它的编号称呼。' }, related: ['vector-quantization'] },
		{ id: 'straight-through', term: { en: 'straight-through estimator', zh: '直通估计器' }, def: { en: 'A training trick for a step with no useful gradient (such as picking the nearest code): the backward pass copies the gradient of its output to its input unchanged.', zh: '用于没有可用梯度的步骤（比如挑最近码字）的训练技巧：反向传播时把输出处的梯度原样复制给输入。' }, related: ['gradient', 'backpropagation'] },
		{ id: 'commitment-loss', term: { en: 'commitment loss', zh: '承诺损失' }, def: { en: 'The part of a VQ-VAE’s loss that pulls the encoder’s output z_e towards the code it snapped to, so z_e does not drift away from the codebook.', zh: 'VQ-VAE 损失中把编码器输出 z_e 拉向它所吸附码字的那一部分，让 z_e 不会远离码本。' }, related: ['loss'] },
		{ id: 'dead-code', term: { en: 'dead code', zh: '死码字' }, def: { en: 'A codebook entry that no input ever snaps to: it takes up space but is never used.', zh: '没有任何输入会吸附到的码本条目：占着位置却从不被使用。' }, related: ['codebook'] }
	],
	credits: [
		{
			name: 'Quick, Draw! dataset',
			url: 'https://github.com/googlecreativelab/quickdraw-dataset',
			license: 'CC BY 4.0',
			usedFor: 'VQ-VAE training and test drawings (star, ladder, umbrella, eyeglasses, t-shirt; TinyCNN’s split), the code usage and the preset drawings',
			attribution: 'Quick, Draw! dataset by Google Creative Lab'
		},
		{
			name: 'van den Oord, Vinyals and Kavukcuoglu, Neural Discrete Representation Learning (NeurIPS 2017)',
			url: 'https://arxiv.org/abs/1711.00937',
			license: 'Paper; idea only (no text, figures or code used)',
			usedFor: 'the VQ-VAE: nearest-code quantization, the straight-through gradient, the commitment loss and the moving-average codebook update',
			attribution: 'Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu'
		}
	],
	catalog: {
		hook: { en: 'Draw something and watch it become 16 code numbers, then swap one code and see what it redraws.', zh: '画点什么，看它变成 16 个码字编号；再换掉一个码字，看它重画出什么。' },
		shows: { en: 'Nearest-code lookup with real distances, the codebook and its usage, code swaps, and how the straight-through estimator gets the gradient past the snap.', zh: '带真实距离的最近码字查找、码本及其使用情况、码字替换，以及直通估计器怎样让梯度越过“吸附”。' },
		runtime: { en: 'Own JS forward pass (checked against PyTorch)', zh: '自研 JS 前向计算（已与 PyTorch 核对）' },
		params: { en: '15,993', zh: '15,993' },
		size: { en: '62 KB of weights', zh: '权重 62 KB' }
	}
};

export default manifest;
