// adversarial-tinycnn: the package manifest (design §4.9). Adversarial examples on TinyCNN, the shipped model (not
// retrained): FGSM and PGD on the learner's own drawing. Kinds A + C:
//   fwd     TinyCNN (static/models/tinycnn, read through the kit's legacy.ts) traced on the drawing
//   attack  the attack, event by event (process.ts): the loss, the gradient for every input pixel (the kit's autograd
//           core, net.ts), its sign, one FGSM step, PGD's small clipped steps and a scan over ε. Its twin,
//           pipeline/adversarial-tinycnn/reference.py (PyTorch autograd, float64), gives the same frames (atlas parity).
// ε is part of the learner's raw input ({ ..., eps }; the stage's slider sets it), so the process, the narration and the
// task read one value. Measured numbers over TinyCNN's test drawings: facts.json (pipeline/adversarial-tinycnn/stats.py).
import type { ExplainerManifest, ExplainerRuntime } from '../../atlas/types/index.ts';
import { epsOf, type AttackIn } from './process.ts';
import { N, type Weights } from './net.ts';

const I = (en: string, zh: string) => ({ en, zh });

/** the attack's input: the 784 numbers TinyCNN sees, ε, and TinyCNN's weights (plain arrays: the worker gets a copy) */
export function attackInput(rt: ExplainerRuntime): AttackIn {
	const x = rt.pre?.tensors.input?.data ?? new Float32Array(N);
	return { x, eps: epsOf(rt.input), w: (rt.sources.fwd?.model?.weights as Weights | undefined) ?? null };
}

const manifest: ExplainerManifest = {
	slug: 'adversarial-tinycnn',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/adversarial-tinycnn', auto: ['facts.json'] },
	input: { adapter: 'drawing28', presets: 'presets.json' },
	output: { view: 'kit.probbars' },
	sources: {
		fwd: { kind: 'trace', assets: 'models/tinycnn' },
		attack: { kind: 'process', process: 'attack', input: attackInput }
	},
	page: {
		goals: [
			I('Say what the input gradient is: one number per pixel, how fast the loss for the model’s answer grows when that pixel gets a little more ink.', '说出输入梯度是什么：每个像素一个数，表示这个像素的墨稍微多一点时，模型答案的损失增长得有多快。'),
			I('Build an FGSM example by hand: keep only the sign of the gradient, multiply it by ε, add it to the drawing and clip to the range 0 to 1.', '亲手做出一个 FGSM 样本：只留下梯度的符号，乘以 ε，加到画上，再裁剪到 0 到 1 的范围。'),
			I('Explain with the measured numbers why PGD, several small clipped steps, flips more drawings than one FGSM step of the same ε.', '用实测的数字说明：为什么 PGD（几个裁剪过的小步）在同样的 ε 下比一步 FGSM 翻转更多的画。'),
			I('Find the smallest ε that flips your drawing, and name what limits the result: a white-box attack on a tiny model of 28×28 drawings.', '找到能让你的画翻转的最小 ε，并说出这个结果的局限：这是对一个处理 28×28 小画的小模型的白盒攻击。')
		],
		prequestion: {
			id: 'adversarial-tinycnn-pre',
			question: I(
				'TinyCNN gets almost all of its test drawings right. How much would you have to change a drawing to make it give a different answer?',
				'TinyCNN 几乎能答对所有测试画。要让它给出别的答案，你得把一幅画改动多少？'
			),
			options: [
				I('A lot: redraw it until it looks like another class', '很多：重画到它像另一类为止'),
				I('Very little, if every pixel moves in just the right direction', '很少，只要每个像素都朝恰好的方向动一点'),
				I('No change will do: a trained model is sure of its answer', '怎么改都没用：训练好的模型对答案很有把握')
			],
			answer: 1,
			explain: I(
				'On most test drawings, moving every pixel by at most a tenth of the way from paper to ink is enough, if each pixel moves the way the gradient says. You will find out how much your own drawing needs.',
				'对大多数测试画来说，只要每个像素都按梯度指的方向走，每个像素最多改动从纸到墨的十分之一就够了。你会亲手测出自己的画需要改多少。'
			)
		},
		tryThis: [
			{
				id: 'sure',
				title: I('A drawing it is very sure of', '一幅它很有把握的画'),
				setup: { preset: 'sun' },
				predict: I('TinyCNN is almost completely sure of this sun, and its gradient is tiny. Can one FGSM step flip it? Can PGD?', 'TinyCNN 对这个太阳几乎完全有把握，它的梯度也极小。一步 FGSM 能让它翻转吗？PGD 呢？'),
				look: I('One FGSM step never flips it anywhere on the slider; PGD’s small steps, each with a fresh gradient, do.', '一步 FGSM 在整条滑块上都翻转不了它；PGD 的小步每步都重新算梯度，就能做到。'),
				goto: { step: 'pgd' }
			},
			{
				id: 'blank',
				title: I('An empty pad', '一块空白画板'),
				setup: { input: { strokes: [] } },
				predict: I('With no ink at all, TinyCNN still gives an answer. Which pixels will the gradient pick?', '一点墨都没有，TinyCNN 照样会给出答案。梯度会挑中哪些像素？'),
				look: I('Even empty paper has a gradient: most pixels can still be pushed one way or the other.', '空白的纸也有梯度：大多数像素仍然可以被推向某个方向。'),
				goto: { step: 'grad' }
			}
		],
		quiz: [
			{
				q: I('Why does FGSM keep only the sign of the gradient?', 'FGSM 为什么只保留梯度的符号？'),
				options: [I('So every pixel moves by the full ε, the most the budget allows', '让每个像素都走满 ε，也就是预算允许的最大一步'), I('Because the gradient is always +1 or −1', '因为梯度总是 +1 或 −1'), I('To make the drawing look the same', '为了让画看起来不变')],
				answer: 0,
				explain: I('With a budget of ε per pixel, the step that raises the loss most (to a first approximation) moves every pixel by ε in the direction of its gradient’s sign.', '每个像素的预算是 ε 时，（一阶近似下）让损失涨得最多的一步，就是让每个像素都朝它梯度的符号方向走 ε。')
			},
			{
				q: I('PGD takes many small steps. What keeps its change small?', 'PGD 走很多小步。是什么让它的改动保持很小？'),
				options: [I('After every step each pixel is clipped back to within ε of your drawing, and to between 0 and 1', '每一步之后，把每个像素裁剪回离你的画不超过 ε 的范围，并限制在 0 到 1 之间'), I('The steps get smaller and smaller', '步子越走越小'), I('It stops after one step', '它走一步就停')],
				answer: 0,
				explain: I('The projection (the clipping) is what the P in PGD stands for: the change never leaves the box of size ε around your drawing.', '投影（也就是裁剪）就是 PGD 里的 P：改动永远不会超出你的画周围 ±ε 的范围。')
			}
		],
		misconception: {
			claim: I('If a change is too small to notice, it cannot change a model’s answer.', '小到注意不到的改动，不可能改变模型的答案。'),
			truth: I('A change picked by the gradient moves hundreds of pixels a little, each in the direction that raises the loss; the network adds those small pushes up, and the answer flips.', '按梯度挑出的改动让几百个像素各动一点，每个都朝着让损失变大的方向；网络把这些小推动加起来，答案就翻转了。'),
			step: 'flip'
		},
		lessonNote: I(
			'Ten steps: your drawing and its class, the loss, the gradient for every pixel, its sign, one FGSM step, the flip, PGD’s small steps, how big ε must be, a defence, and the limits.',
			'十步：你的画和它的类别、损失、每个像素的梯度、它的符号、一步 FGSM、翻转、PGD 的小步、ε 要多大、一种防御，以及局限。'
		)
	},
	glossary: [
		{ id: 'adversarial-example', term: I('adversarial example', '对抗样本'), def: I('An input changed on purpose, often by very little, so that a model gives a wrong answer.', '被故意改动（常常只改一点点）、让模型给出错误答案的输入。'), related: ['fgsm', 'pgd'] },
		{ id: 'perturbation', term: I('perturbation', '扰动'), def: I('The change x′ − x added to an input; an attack keeps it small, for example at most ε per pixel.', '加到输入上的改动 x′ − x；攻击会让它保持很小，比如每个像素最多改 ε。') },
		{ id: 'fgsm', term: I('FGSM (fast gradient sign method)', 'FGSM（快速梯度符号法）'), def: I('One step x′ = x + ε·sign(∇ₓL): every pixel moves by ε in the direction that raises the loss.', '一步 x′ = x + ε·sign(∇ₓL)：每个像素都朝让损失变大的方向移动 ε。'), related: ['gradient'] },
		{ id: 'pgd', term: I('PGD (projected gradient descent)', 'PGD（投影梯度下降）'), def: I('Many small sign steps on the loss, each followed by clipping back into the allowed box around the input.', '在损失上走很多个符号小步，每步之后都裁剪回输入周围允许的范围里。'), related: ['fgsm'] },
		{ id: 'adversarial-training', term: I('adversarial training', '对抗训练'), def: I('Training on adversarial examples with their correct labels, so the model learns to resist them.', '用带正确标签的对抗样本来训练，让模型学会抵抗它们。') },
		{ id: 'white-box-attack', term: I('white-box attack', '白盒攻击'), def: I('An attack that knows the model’s weights and can compute its exact gradient.', '知道模型权重、能算出它精确梯度的攻击。') }
	],
	credits: [
		{
			name: 'Quick, Draw! dataset',
			url: 'https://github.com/googlecreativelab/quickdraw-dataset',
			license: 'CC BY 4.0',
			usedFor: 'The preset drawings (TinyCNN’s test drawings) and the measured attack rates over TinyCNN’s 15,000 test drawings',
			attribution: 'Quick, Draw! dataset by Google Creative Lab'
		},
		{
			name: 'Goodfellow, Shlens & Szegedy, Explaining and Harnessing Adversarial Examples (ICLR 2015)',
			url: 'https://arxiv.org/abs/1412.6572',
			license: 'arXiv paper; ideas only (no text, figures or code used)',
			usedFor: 'The fast gradient sign method (FGSM) and adversarial training',
			attribution: 'Ian J. Goodfellow, Jonathon Shlens and Christian Szegedy'
		},
		{
			name: 'Madry et al., Towards Deep Learning Models Resistant to Adversarial Attacks (ICLR 2018)',
			url: 'https://arxiv.org/abs/1706.06083',
			license: 'arXiv paper; ideas only (no text, figures or code used)',
			usedFor: 'Projected gradient descent (PGD) inside an ε box, and adversarial training with it',
			attribution: 'Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras and Adrian Vladu'
		}
	],
	catalog: {
		hook: I('Nudge every pixel of your drawing a little along the gradient, and flip TinyCNN’s answer.', '让你的画的每个像素沿着梯度挪一点点，就能翻转 TinyCNN 的答案。'),
		shows: I('The gradient of the loss for every pixel and its sign, an ε slider with x + ε·sign = x′, the change amplified, the class shares before and after, PGD’s steps on the step player, the smallest ε that flips your drawing, and measured rates on the test drawings with the transfer to TinyViT.', '损失对每个像素的梯度和它的符号、带 ε 滑块的 x + ε·sign = x′、放大显示的改动、改动前后的类别份额、逐步播放的 PGD、能让你的画翻转的最小 ε，以及在测试画上实测的比例和对 TinyViT 的转移。'),
		runtime: I('Own JS (input gradient in the autograd core, checked against PyTorch)', '自研 JS（由自动求导核心计算输入梯度，已与 PyTorch 核对）'),
		params: I('6,722 (TinyCNN’s)', '6,722（TinyCNN 的）'),
		size: I('27 KB of weights (TinyCNN’s, shared)', '权重 27 KB（与 TinyCNN 共用）')
	}
};

export default manifest;
