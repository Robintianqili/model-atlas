// contrastive-learning: the concept page around the lesson (K2 concept packages; the page's order is goals, hook, lesson,
// toy, real, compare, pitfalls, checkpoint, links). toy, real and compare are StepDefs on the lesson's own runtime and
// sources, shown as free-standing stages; the toy section shows the batch widget (manifest.input) above the toy.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { fm, st, whenOf } from './lesson.ts';

export const hook: I18n = {
	en: 'Nobody told CLIP what a dog is. It learned from a huge number of pictures, each with the text someone wrote next to it, and from one rule: give a picture and its own text similar numbers, and the other texts in the batch different ones. Here you build such a batch from your own drawings and captions and watch the same loss at work on TinyCLIP.',
	zh: '没有人告诉过 CLIP 狗是什么。它从海量图片中学习，每张都配着别人写在旁边的文字，靠的只是一条规则：让一张图和它自己的文字得到相近的数，和同一批里其他文字得到不同的数。在这里，你用自己的画和描述搭出这样一批，看同样的损失如何在 TinyCLIP 上起作用。'
};

/** The temperature, live: the batch's loss over τ, with the learner's τ marked. */
export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Your batch, its loss and the temperature', zh: '你的这一批、它的损失和温度' },
	source: 'nce',
	at: { phase: 'softmax' },
	view: { key: 'contrastive-learning/matrix', props: () => ({ mode: 'tau' }) },
	provenance: 'live',
	narration: {
		en: 'Your {n} pairs score a loss of {loss} at τ = {tau}; a blind guess would score {chance}. Add or remove pairs, rewrite a caption or drag τ in the widget, and the curve follows.',
		zh: '你的 {n} 对在 τ = {tau} 时损失为 {loss}；瞎猜的损失是 {chance}。在输入工具里增删图文对、改写描述或拖动 τ，曲线都会跟着变。',
		vars: (_rt, at) => ({ n: fm(at, 'scores', 'N'), loss: fm(at, 'softmax', 'loss'), tau: st(fm(at, 'scores', 'tau')), chance: fm(at, 'softmax', 'chance') })
	}
};

/** The training run, checkpoint by checkpoint, on the learner's batch. */
export const real: StepDef = {
	id: 'real',
	title: { en: 'A real training run, on your batch', zh: '一次真实的训练，作用在你的这一批上' },
	source: 'progress',
	at: 'last',
	view: { key: 'contrastive-learning/matrix', props: () => ({ mode: 'progress' }) },
	play: { from: 'first', to: 'last', interval: 900, unit: { en: 'checkpoint', zh: '检查点 {n}' } },
	provenance: { mode: 'precomputed', note: { en: 'Weights saved during a re-run of TinyCLIP’s training (CPU, seed 0); your batch runs through them live.', zh: '权重保存自 TinyCLIP 的一次重新训练（CPU，种子 0）；你的这一批在浏览器里实时通过它们。' } },
	narration: {
		en: 'With the weights saved {when}, the diagonal of your batch averages {diag} and the other cells {off}. Play from the first checkpoint to watch the table turn into a diagonal.',
		zh: '用保存于{when}的权重，你这一批的对角线平均为 {diag}，其余格子为 {off}。从第一个检查点开始播放，看这张表如何变成一条对角线。',
		vars: (_rt, at) => {
			const m = (k: string) => Number(at.frame?.metrics?.[k] ?? NaN);
			return { when: whenOf(m('update'), m('last')), diag: m('diag'), off: m('off') };
		}
	}
};

/** SimCLR's pairs: two distorted views of every drawing, no captions. */
export const compare: StepDef = {
	id: 'compare',
	title: { en: 'CLIP’s pairs and SimCLR’s pairs', zh: 'CLIP 的正样本对和 SimCLR 的正样本对' },
	source: 'simclr',
	at: { phase: 'loss' },
	view: { key: 'contrastive-learning/matrix', props: () => ({ mode: 'simclr' }) },
	provenance: { mode: 'live', note: { en: 'TinyCLIP’s image tower stands in for a SimCLR encoder: it was trained with captions, not with this loss.', zh: '这里用 TinyCLIP 的图像塔代替 SimCLR 的编码器：它是用描述训练的，不是用这个损失训练的。' } },
	narration: {
		en: 'CLIP pairs a drawing with its caption; SimCLR pairs two distorted views of the same drawing. Your {n} drawings make {m} views, and the loss over them at τ = {tau} is {loss}, against {chance} for a blind guess.',
		zh: 'CLIP 把一幅画和它的描述配成一对；SimCLR 把同一幅画的两个变形视图配成一对。你的 {n} 幅画变成 {m} 个视图，在 τ = {tau} 时它们的损失是 {loss}，而瞎猜是 {chance}。',
		vars: (_rt, at) => ({ n: fm(at, 'views', 'N'), m: fm(at, 'views', 'M'), tau: st(fm(at, 'loss', 'tau')), loss: fm(at, 'loss', 'loss'), chance: fm(at, 'loss', 'chance') })
	}
};

/** Misconceptions after manifest.page.misconception; `step` links the lesson step that shows it. */
export const pitfalls: { claim: I18n; truth: I18n; step?: string }[] = [
	{
		claim: { en: 'A lower temperature always gives a lower loss.', zh: '温度越低，损失总是越低。' },
		truth: { en: 'Only while every item already ranks its own partner first. A low τ makes the softmax very sure of itself, so a single wrong ranking costs a lot; a very high τ turns every row into a blind guess, log N.', zh: '只有在每一项都已经把自己的搭档排第一时才成立。低 τ 让 softmax 非常笃定，一次排错就代价很大；τ 很高时，每一行都变成瞎猜，损失是 log N。' },
		step: 'drag'
	},
	{
		claim: { en: 'Every other item in the batch is a true negative.', zh: '同一批里的其他每一项都是真正的负样本。' },
		truth: { en: 'Another caption in the batch may describe the drawing just as well. The loss still pushes it away: a false negative.', zh: '同一批里的另一句描述可能同样符合这幅画。损失还是会把它推开：这就是伪负样本。' },
		step: 'limits'
	},
	{
		claim: { en: 'A loss of 1 means the same in any batch.', zh: '损失为 1 在任何批里含义都一样。' },
		truth: { en: 'A blind guess scores log N, so the same loss is good in a large batch and poor in a small one. Compare losses only at the same batch size.', zh: '瞎猜的损失是 log N，所以同样的损失在大批里算好，在小批里算差。只在批大小相同时比较损失。' },
		step: 'hot'
	}
];

/** Further reading under the links (absolute URLs, or site paths starting with '/'). */
export const further: { label: I18n; href: string }[] = [
	{ label: { en: 'TinyCLIP on this site: the two towers, layer by layer', zh: '本站的 TinyCLIP：两座塔，逐层讲解' }, href: '/m/tinyclip/' },
	{ label: { en: 'CLIP: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)', zh: 'CLIP：从自然语言监督中学习可迁移的视觉模型（Radford 等，2021）' }, href: 'https://arxiv.org/abs/2103.00020' },
	{ label: { en: 'SimCLR: A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., 2020)', zh: 'SimCLR：一个简单的视觉表示对比学习框架（Chen 等，2020）' }, href: 'https://arxiv.org/abs/2002.05709' },
	{ label: { en: 'Contrastive Predictive Coding, where the InfoNCE loss comes from (van den Oord et al., 2018)', zh: '对比预测编码：InfoNCE 损失的出处（van den Oord 等，2018）' }, href: 'https://arxiv.org/abs/1807.03748' },
	{ label: { en: 'Contrastive Representation Learning, an overview (Lilian Weng, 2021)', zh: '对比表示学习综述（Lilian Weng，2021）' }, href: 'https://lilianweng.github.io/posts/2021-05-31-contrastive/' }
];
