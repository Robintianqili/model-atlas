// self-supervised-objectives: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts
// ConceptDef). hook (text); toy, real and compare are StepDefs on the lesson's own runtime and sources.
//   toy      every objective as a mask pattern on the learner's text and the drawing, one at a time (illustrative, with
//            the masked-LM row from the live 80/10/10 draw)
//   real     TinyBERT and TinyGPT on a token the learner clicks (live)
//   compare  all objectives side by side
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { bertGuess, gap, gptGuess, nextProbs, say } from './text.ts';
import { GRID } from './patterns.ts';
import { fmt3, fmtPct } from '../../atlas/core/format.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';

/** TinyGPT's causal loss on the whole text, as a sentence (none for a single token: nothing to grade) */
function lossLine(rt: Parameters<typeof gap>[0]): I18n {
	const N = nextProbs(rt);
	return pick(N.targets > 0, { en: ` Over your whole text, TinyGPT’s causal loss is ${fmt3(N.loss)} nats.`, zh: `在你的整段文字上，TinyGPT 的因果损失是 ${fmt3(N.loss)} 奈特。` }, { en: '', zh: '' });
}

export const hook: I18n = {
	en: 'Most models on this site learned from text or pictures that nobody labelled. The trick is a puzzle made from the data itself: hide part of it and predict the hidden part. Which part is hidden, and what must come back, decides what the model becomes good at. Here every objective works on the same sentence, yours, and on the same drawing.',
	zh: '这个网站上的大多数模型，都是从没人标注过的文字或图片里学会的。诀窍是用数据本身出一道谜题：藏起一部分，再预测被藏的部分。藏起哪一部分、要求还原什么，决定了模型最终擅长什么。这里每一种目标都作用在同一句话（你的句子）和同一幅简笔画上。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Pick an objective: what does it hide, what must come back?', zh: '选一种目标：它藏起什么，要还原什么？' },
	source: 'hide',
	at: 'last',
	view: { key: 'self-supervised-objectives/patterns', props: () => ({ pick: true }) },
	provenance: { mode: 'illustrative', note: { en: 'Drawings, except causal and masked language modelling: their rows use your real tokens and the live 80/10/10 draw.', zh: '示意图；因果语言建模和掩码语言建模除外：它们那两行用的是你真实的词元和实时的 80/10/10 抽签。' } },
	narration: {
		en: 'Your text is {n}; the drawing is {np} patches. Pick an objective with the buttons and read its pattern: what the model sees, what is hidden from it, and what it must predict. Type another sentence or click a token in the input: every pattern follows it.',
		zh: '你的文字是 {n}；简笔画是 {np} 个图块。用按钮选一种目标，读它的图案：模型看到什么、什么对它藏着、它要预测什么。换一句话，或者在输入框里点一个词元：每个图案都会跟着变。',
		vars: (rt) => ({ n: plural(gap(rt).n, { one: '{n} token', many: '{n} tokens', zh: '{n} 个词元' }), np: GRID.n * GRID.n })
	},
	look: { en: 'Outlined chips and patches are the target.', zh: '加框的词元和图块就是目标。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'Two live models, one hidden token', zh: '两个实时模型，一个被藏的词元' },
	source: 'bert',
	at: { layer: 'probs' },
	view: { key: 'self-supervised-objectives/guesses', props: () => ({ mode: 'mlm' }) },
	provenance: 'live',
	narration: {
		en: 'Click any token to hide it. TinyBERT, trained to fill gaps, reads both sides and gives “{hidden}” {pB}; TinyGPT, trained on the next token, reads only the left side and {pG}.{loss}',
		zh: '点任意一个词元把它藏起来。被训练来填空的 TinyBERT 读两侧，给“{hidden}”{pB}；被训练来预测下一个词元的 TinyGPT 只读左边，{pG}。{loss}',
		vars: (rt) => ({ hidden: say(gap(rt).hidden), pB: { v: bertGuess(rt).pHidden, fmt: 'pct' }, pG: pick(gap(rt).k >= 1, { en: `gives it ${fmtPct(gptGuess(rt).pHidden)}`, zh: `给它 ${fmtPct(gptGuess(rt).pHidden)}` }, { en: 'cannot guess it: nothing comes before the first token', zh: '猜不了它：第一个词元前面什么都没有' }), loss: lossLine(rt) })
	},
	look: { en: 'The accent bar in each set is your hidden token.', zh: '每组里强调色的条就是你藏起的词元。' }
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Every objective on your sentence and the drawing', zh: '你的句子和简笔画上的全部目标' },
	source: 'hide',
	at: 'last',
	view: { key: 'self-supervised-objectives/patterns', props: () => ({ brief: false }) },
	provenance: { mode: 'illustrative', note: { en: 'Drawings, except the first two rows (TinyGPT and TinyBERT were trained with them; the masked row is the live 80/10/10 draw).', zh: '示意图，前两行除外（TinyGPT 和 TinyBERT 就是用它们训练的；掩码那一行是实时的 80/10/10 抽签）。' } },
	narration: {
		en: 'Side by side, the objectives differ in two choices: what the model may see, and what it must produce. Causal modelling grades every token after the first, {clm} of your {n}; masked modelling grades only the tokens it hid; next-sentence prediction gives one label per pair; contrastive learning asks for no hidden part back at all.',
		zh: '放在一起看，这些目标的区别在于两个选择：模型能看到什么，以及它要产出什么。因果建模给第一个之后的每个词元打分，也就是你 {n}中的 {clm} 个；掩码建模只给它藏起的词元打分；下一句预测每对句子只有一个标签；对比学习根本不要求找回被藏的部分。',
		vars: (rt) => ({ n: plural(gap(rt).n, { one: '{n} token', many: '{n} tokens', zh: '{n} 个词元' }), clm: Math.max(gap(rt).n - 1, 0) })
	}
};

export const pitfalls = [
	{
		claim: { en: 'In masked language modelling every chosen token becomes [MASK].', zh: '在掩码语言建模里，每个被选中的词元都变成 [MASK]。' },
		truth: {
			en: 'Only 80% do; 10% become a random token and 10% stay as they are. Real text never contains [MASK], so the model must also check the tokens it can see.',
			zh: '只有 80% 会变成 [MASK]；10% 变成随机词元，10% 保持不变。真实的文字里从来没有 [MASK]，所以模型也得检查它看得见的词元。'
		},
		step: 'eighty'
	},
	{
		claim: { en: 'Hiding more is always better.', zh: '藏得越多越好。' },
		truth: {
			en: 'It depends on the data. BERT hides 15% of the tokens, MAE 75% of the image patches: every word carries a lot, while a small hole in an image is easy to fill from its neighbours.',
			zh: '这取决于数据。BERT 藏起 15% 的词元，MAE 藏起 75% 的图块：每个词都承载很多信息，而图上的小洞很容易靠周围补上。'
		},
		step: 'patches'
	},
	{
		claim: { en: 'Next-sentence prediction is what makes BERT work.', zh: '下一句预测是 BERT 起作用的关键。' },
		truth: {
			en: 'The RoBERTa paper reports that removing it matched or slightly improved results on later tasks; masked language modelling does the heavy lifting.',
			zh: 'RoBERTa 论文报告说，去掉它之后，在后续任务上的结果持平甚至略好；真正起主要作用的是掩码语言建模。'
		},
		step: 'nsp'
	}
];

export const further = [
	{ label: { en: 'BERT: masked language modelling and next-sentence prediction (Devlin et al., 2019)', zh: 'BERT：掩码语言建模与下一句预测（Devlin 等，2019）' }, href: 'https://arxiv.org/abs/1810.04805' },
	{ label: { en: 'RoBERTa: what happens without next-sentence prediction (Liu et al., 2019)', zh: 'RoBERTa：去掉下一句预测会怎样（Liu 等，2019）' }, href: 'https://arxiv.org/abs/1907.11692' },
	{ label: { en: 'T5: span corruption and many other objectives compared (Raffel et al., 2020)', zh: 'T5：片段破坏，以及许多目标的对比（Raffel 等，2020）' }, href: 'https://arxiv.org/abs/1910.10683' },
	{ label: { en: 'Masked Autoencoders Are Scalable Vision Learners (He et al., 2022)', zh: '掩码自编码器是可扩展的视觉学习者（He 等，2022）' }, href: 'https://arxiv.org/abs/2111.06377' },
	{ label: { en: 'Self-supervised representation learning, an overview (Lilian Weng)', zh: '自监督表示学习综述（Lilian Weng）' }, href: 'https://lilianweng.github.io/posts/2019-11-10-self-supervised/' }
];
