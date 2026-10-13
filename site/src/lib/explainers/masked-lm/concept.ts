// masked-lm: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef). hook
// (text); toy, real and compare are StepDefs on the lesson's own runtime and sources.
//   toy      click tokens to hide them; TinyBERT's guesses at every hidden position (live)
//   real     the 80/10/10 draw on a batch, and the loss when your row is sent to TinyBERT (live)
//   compare  TinyBERT and TinyGPT on the same token: which context each reads (live)
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { chosenLine, mlmLoss, say, text } from './text.ts';

export const hook: I18n = {
	en: 'BERT and most text encoders after it learned from text that nobody labelled. The trick is a fill-in-the-blank puzzle made from the text itself: hide some tokens, ask for them back, and grade only those. Here the puzzle runs on your own sentence, with a real TinyBERT filling the gaps and a real TinyGPT, trained on the same stories to predict the next token, beside it.',
	zh: 'BERT 以及它之后的大多数文本编码器，都是从没人标注过的文字里学会的。诀窍是用文字本身出一道填空题：藏起一些词元，让模型把它们找回来，并且只批改这些词元。这里的谜题就出在你自己的句子上：一个真实的 TinyBERT 来填空，旁边是用同样的故事训练、负责预测下一个词元的真实 TinyGPT。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Click any token to hide it', zh: '点任意词元把它藏起来' },
	source: 'bert',
	at: { layer: 'probs' },
	view: 'masked-lm/fill',
	provenance: 'live',
	narration: {
		en: 'Click tokens in the row to hide them or give them back. TinyBERT fills every gap at once. {chosen}',
		zh: '点这一行里的词元，把它们藏起来或还原。TinyBERT 一次填好所有空缺。{chosen}',
		vars: (rt) => ({ chosen: chosenLine(rt) })
	},
	look: { en: 'The accent bar in each panel is the right token.', zh: '每个小面板里强调色的条就是正确词元。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'A training draw, graded', zh: '一次训练抽签，以及它的批改' },
	source: 'batch',
	at: 'last',
	view: { key: 'masked-lm/loss', props: () => ({ draw: true }) },
	provenance: 'live',
	narration: {
		en: 'BERT’s masking rule drew row 0 of a batch from your text. Send it to TinyBERT: the table grades only the chosen rows, whatever they became. Now the loss over {k} graded rows is {loss}.',
		zh: 'BERT 的掩码规则从你的文字里抽出了批次的第 0 行。把它交给 TinyBERT：表格只批改被选中的行，不管它们变成了什么。现在 {k} 个被批改行的损失是 {loss}。',
		vars: (rt) => {
			const L = mlmLoss(rt);
			return { k: L.graded.length, loss: L.graded.length ? { en: `${fmt3(L.loss)} nats`, zh: `${fmt3(L.loss)} 奈特` } : { en: 'not defined: nothing is chosen', zh: '无定义：没有选中任何词元' } };
		}
	},
	look: { en: 'Shaded rows are graded.', zh: '底色加深的行会被批改。' }
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Masked and causal, on the same token', zh: '掩码与因果，同一个词元' },
	source: 'bert',
	at: { layer: 'probs' },
	view: 'masked-lm/context',
	provenance: 'live',
	narration: {
		en: '{who} TinyBERT reads both sides of a gap; TinyGPT, trained to predict the next token, reads only the tokens before it.',
		zh: '{who}TinyBERT 读空缺的两侧；被训练来预测下一个词元的 TinyGPT 只读它前面的词元。',
		vars: (rt) => {
			const x = text(rt);
			return { who: x.focus >= 0 ? { en: `The token compared is “${say(x.c[x.focus])}”.`, zh: `比较的词元是“${say(x.c[x.focus])}”。` } : { en: 'Hide a token in the input to compare the two models on it.', zh: '在输入框里藏起一个词元，就能在它上面比较两个模型。' } };
		}
	},
	look: { en: 'Underlined chips are what each model reads.', zh: '带下划线的词元是每个模型读到的内容。' }
};

export const pitfalls = [
	{
		claim: { en: 'Every chosen token becomes [MASK].', zh: '每个被选中的词元都会变成 [MASK]。' },
		truth: {
			en: 'Only 80% do; 10% become a random token and 10% stay as they are, and all of them are graded. Real text never contains [MASK], so the model must also check the tokens it can see.',
			zh: '只有 80% 会变成 [MASK]；10% 变成随机词元，10% 保持不变，而且它们都要被批改。真实的文字里从来没有 [MASK]，所以模型也得检查它看得见的词元。'
		},
		step: 'eighty'
	},
	{
		claim: { en: 'TinyBERT’s loss is lower than TinyGPT’s, so it is the better language model.', zh: 'TinyBERT 的损失比 TinyGPT 低，所以它是更好的语言模型。' },
		truth: {
			en: 'The two losses grade different puzzles. With only the left side, TinyGPT’s puzzle, TinyBERT guesses worse than TinyGPT does.',
			zh: '这两个损失批改的是不同的谜题。只给左边的内容（也就是 TinyGPT 的谜题）时，TinyBERT 猜得比 TinyGPT 差。'
		},
		step: 'limits'
	},
	{
		claim: { en: 'The hidden tokens are guessed together, so their guesses fit each other.', zh: '被藏的词元是一起猜的，所以这些猜测彼此吻合。' },
		truth: {
			en: 'Each gap is filled on its own, in one pass, without seeing the model’s other guesses; had it seen them, its guesses would change.',
			zh: '每个空缺都是在同一次计算里单独填的，看不到模型的其他猜测；如果看得到，它的猜测会变。'
		},
		step: 'crowd'
	}
];

export const further = [
	{ label: { en: 'On this site: next-token prediction, the causal objective TinyGPT was trained with', zh: '本站：预测下一个词元，也就是 TinyGPT 所用的因果目标' }, href: '/train/next-token-prediction/' },
	{ label: { en: 'On this site: self-supervised objectives, masked language modelling among many others', zh: '本站：自监督目标，掩码语言建模和其他许多目标' }, href: '/train/self-supervised-objectives/' },
	{ label: { en: 'BERT: masked language modelling and the 80/10/10 rule (Devlin et al., 2019)', zh: 'BERT：掩码语言建模与 80/10/10 规则（Devlin 等，2019）' }, href: 'https://arxiv.org/abs/1810.04805' },
	{ label: { en: 'ELECTRA: grading every token instead of the chosen 15% (Clark et al., 2020)', zh: 'ELECTRA：批改每个词元，而不只是选中的 15%（Clark 等，2020）' }, href: 'https://arxiv.org/abs/2003.10555' },
	{ label: { en: 'RoBERTa: masking drawn anew every time a text is seen (Liu et al., 2019)', zh: 'RoBERTa：每次读到一段文字都重新抽取掩码（Liu 等，2019）' }, href: 'https://arxiv.org/abs/1907.11692' }
];
