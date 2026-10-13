// self-supervised-objectives: the guided lesson (eleven steps; the concept page's primary path). One sentence (the
// learner's) and one drawing. Live: TinyGPT ('gpt'), TinyBERT ('bert') and the 80/10/10 draw ('hide'); illustrative: the
// objectives no model of this site was trained with (patterns.ts, objectives.ts). Every number is a placeholder: live
// values in vars(), measured ones from facts.json, settings from the papers' constants in patterns.ts.
//   puzzle -> next-token -> every (all positions) -> gap (predict) -> click (task) -> eighty (80/10/10, play) -> nsp ->
//   spans -> patches -> views -> limits
import type { At, I18n, LessonDef } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmt3, fmtPct } from '../../atlas/core/format.ts';
import { bertGuess, gap, gptGuess, nextProbs, say, type RT } from './text.ts';
import { BART, GRID, MAE, T5, bartSpans, cropViews, maePatches, t5Spans } from './patterns.ts';

const tokens = (n: number) => plural(n, { one: '{n} token', many: '{n} tokens', zh: '{n} 个词元' });
const pct = (v: number) => `${Math.round(v * 100)}%`;
/** a measured share from facts.json as a whole percentage ("15%") */
const pctFact = (rt: RT, key: string) => {
	const v = rt.num(`fact:${key}`);
	return Number.isFinite(v) ? pct(v) : NaN;
};
/** TinyGPT's guess for the hidden token as a sentence (or why there is none) */
function gptSays(rt: RT): I18n {
	const g = gap(rt);
	const q = gptGuess(rt);
	if (q.top < 0) return { en: 'Nothing comes before the first token, so TinyGPT cannot guess it at all.', zh: '第一个词元前面什么都没有，所以 TinyGPT 根本猜不了它。' };
	const w = say(g.hidden);
	return pick(
		q.rank === 1,
		{ en: `TinyGPT ranks “${w}” first, at ${fmtPct(q.pHidden)}.`, zh: `TinyGPT 把“${w}”排在第一，概率 ${fmtPct(q.pHidden)}。` },
		{ en: `TinyGPT gives “${w}” ${fmtPct(q.pHidden)} (rank ${q.rank}); its top guess is “${say(q.top)}”, ${fmtPct(q.pTop)}.`, zh: `TinyGPT 给“${w}”的概率是 ${fmtPct(q.pHidden)}（第 ${q.rank} 名）；它的第一猜测是“${say(q.top)}”，${fmtPct(q.pTop)}。` }
	);
}
const doneOf = (at: At) => at.frame?.metrics ?? {};

const lesson: LessonDef = {
	title: { en: 'Self-supervised objectives, step by step', zh: '自监督目标，分步课程' },
	summary: {
		en: 'One sentence and one drawing, many objectives: what each hides and what it predicts. TinyGPT and TinyBERT run live on your text (next token, a token you click, the 80/10/10 rule); next-sentence prediction, spans, masked patches and contrastive views are labelled drawings.',
		zh: '一句话、一幅简笔画，多种目标：每种目标藏起什么、预测什么。TinyGPT 和 TinyBERT 在你的文字上实时运行（下一个词元、你点中的词元、80/10/10 规则）；下一句预测、片段、被遮的图块和对比视图是标明“示意”的图。'
	},
	steps: [
		{
			id: 'puzzle',
			title: { en: 'A text is its own answer key', zh: '文字本身就是答案' },
			source: 'bert',
			at: { layer: 'probs' },
			view: { key: 'self-supervised-objectives/chips', props: () => ({ mode: 'plain' }) },
			provenance: 'live',
			terms: ['self-supervised', 'token', 'tokenizer'],
			narration: {
				en: 'Nobody labelled your text, yet it can grade a model: hide part of it and ask for the hidden part back. The hidden part is the right answer, so no person has to write one. That is self-supervised learning. TinyGPT’s tokenizer cuts your text into {n}; every objective on this page is one rule for what to hide and what to predict.',
				zh: '没有人给你的文字做过标注，但它可以给模型打分：藏起一部分，再让模型把它找回来。藏起来的部分就是正确答案，不需要任何人去写。这就是自监督学习。TinyGPT 的分词器把你的文字切成 {n}；本页的每一种目标，都是一条“藏什么、预测什么”的规则。',
				vars: (rt) => ({ n: tokens(gap(rt).n) })
			},
			look: { en: 'Each chip is one token, numbered from 0.', zh: '每个小块是一个词元，从 0 开始编号。' }
		},
		{
			id: 'next-token',
			title: { en: 'Causal language modelling: hide the future', zh: '因果语言建模：把后面的藏起来' },
			source: 'gpt',
			at: { layer: 'probs' },
			view: { key: 'self-supervised-objectives/guesses', props: () => ({ mode: 'clm' }) },
			provenance: 'live',
			terms: ['next-token-prediction', 'causal-mask'],
			narration: {
				en: 'TinyGPT was trained with the causal objective: read the tokens so far, predict the next one. A causal mask hides every later token from it. Here the token to predict is “{hidden}”. {gpt}',
				zh: 'TinyGPT 是用因果目标训练的：读已有的词元，预测下一个。因果掩码把后面的每个词元都对它藏起来。这里要预测的词元是“{hidden}”。{gpt}',
				vars: (rt) => ({ hidden: say(gap(rt).hidden), gpt: gptSays(rt) })
			},
			look: { en: 'Underlined chips: what TinyGPT reads. Outlined: the answer. Dashed: later tokens, hidden from it.', zh: '带下划线的词元：TinyGPT 读到的内容。加框的：答案。虚线框的：后面的词元，对它藏着。' }
		},
		{
			id: 'every',
			title: { en: 'One text, an answer at every position', zh: '一段文字，每个位置都有答案' },
			source: 'gpt',
			at: { layer: 'probs' },
			view: 'self-supervised-objectives/every',
			provenance: 'live',
			terms: ['loss', 'cross-entropy'],
			narration: {
				en: 'The causal puzzle is asked at every position at once: token 1 from token 0, token 2 from tokens 0 and 1, and so on. One pass over your text gives {answers}. The loss is the mean of −log p over the answers, where p is the probability TinyGPT gave the token that really came next. {here}',
				zh: '因果谜题在每个位置同时出题：用第 0 个词元猜第 1 个，用第 0、1 个猜第 2 个，依此类推。对你的文字算一遍就得到 {answers}。损失是这些答案的 −log p 的平均值，其中 p 是 TinyGPT 给真正出现的下一个词元的概率。{here}',
				vars: (rt) => {
					const N = nextProbs(rt);
					const val = fmt3(rt.num('fact:gptValLoss'));
					const uni = fmt3(rt.num('fact:gptUniformLoss'));
					return {
						answers: plural(N.targets, { one: 'one answer', many: '{n} answers', zh: '{n} 个答案' }),
						here: pick(
							N.targets > 0,
							{ en: `On your text it is ${fmt3(N.loss)} nats; on unseen stories TinyGPT averages ${val}, and blind guessing would score ${uni}.`, zh: `在你的文字上是 ${fmt3(N.loss)} 奈特；在没见过的故事上，TinyGPT 平均是 ${val}，盲猜则是 ${uni}。` },
							{ en: `On unseen stories TinyGPT averages ${val} nats; blind guessing would score ${uni}.`, zh: `在没见过的故事上，TinyGPT 平均是 ${val} 奈特；盲猜则是 ${uni}。` }
						)
					};
				}
			},
			look: { en: 'Each bar is TinyGPT’s probability for the token that really came next; the accent bar is the least expected one.', zh: '每根条是 TinyGPT 给真正的下一个词元的概率；强调色的条是最出乎它意料的那个。' }
		},
		{
			id: 'gap',
			title: { en: 'Masked language modelling: hide a token, read both sides', zh: '掩码语言建模：藏起一个词元，读两侧' },
			source: 'bert',
			at: { layer: 'probs' },
			view: { key: 'self-supervised-objectives/guesses', props: () => ({ mode: 'mlm' }) },
			provenance: 'live',
			terms: ['masked-language-model', 'special-token'],
			narration: {
				en: 'BERT’s objective hides a token in the middle and lets the model read both sides of the gap. TinyBERT was trained this way: it reads [CLS], your text with “{hidden}” replaced by [MASK], and [SEP]. {left} Answer first, then compare the two sets of bars.',
				zh: 'BERT 的目标是藏起中间的一个词元，让模型读空缺的两侧。TinyBERT 就是这样训练的：它读 [CLS]、把“{hidden}”换成 [MASK] 的你的文字，以及 [SEP]。{left}先回答，再比较两组条形图。',
				vars: (rt) => ({
					hidden: say(gap(rt).hidden),
					left: pick(
						gap(rt).k >= 1,
						{ en: 'Next to it, TinyGPT guesses the same token from the left side only.', zh: '旁边的 TinyGPT 只用左边的内容去猜同一个词元。' },
						{ en: 'TinyGPT could use only the left side, and nothing comes before the first token.', zh: 'TinyGPT 只能用左边的内容，而第一个词元前面什么都没有。' }
					)
				})
			},
			predict: {
				id: 'self-supervised-objectives-gap',
				question: { en: 'Over thousands of hidden tokens in unseen stories, which model guesses the hidden token right more often?', zh: '在没见过的故事里藏起成千上万个词元，哪个模型更常猜对？' },
				options: [
					{ en: 'TinyGPT, which reads only the tokens before the gap', zh: 'TinyGPT，它只读空缺前面的词元' },
					{ en: 'TinyBERT, which reads both sides of the gap', zh: 'TinyBERT，它读空缺的两侧' },
					{ en: 'Both about equally often', zh: '两者差不多' }
				],
				answer: 1,
				explain: {
					en: 'The words after a gap narrow it down a lot, and TinyBERT was trained to use them; the next step gives the measured shares. On one sentence either model can win, so check the bars for your token.',
					zh: '空缺后面的词能把范围缩小很多，而 TinyBERT 正是被训练来利用它们的；下一步给出实测的比例。在单独一句话上，两个模型都可能胜出，所以看看你这个词元的条形图。'
				}
			},
			look: { en: 'Two sets of bars for the same hidden token, drawn in the accent colour.', zh: '两组条形图对应同一个被藏的词元，它用强调色画出。' }
		},
		{
			id: 'click',
			title: { en: 'Click a token to mask it', zh: '点一个词元把它遮住' },
			source: 'bert',
			at: { layer: 'probs' },
			view: { key: 'self-supervised-objectives/guesses', props: () => ({ mode: 'mlm' }) },
			provenance: 'live',
			narration: {
				en: 'For “{hidden}”, TinyBERT gives {pB} (rank {rB}). {gpt} Over {fact:cmpN|int} hidden tokens in unseen stories, TinyBERT’s top guess is right {fact:accBoth|pct} of the time and TinyGPT’s {fact:accGpt|pct}: the words after a gap help.',
				zh: '对“{hidden}”，TinyBERT 给出 {pB}（第 {rB} 名）。{gpt}在没见过的故事里藏起 {fact:cmpN|int} 个词元，TinyBERT 的第一猜测有 {fact:accBoth|pct} 是对的，TinyGPT 是 {fact:accGpt|pct}。可见空缺后面的词有帮助。',
				vars: (rt) => {
					const b = bertGuess(rt);
					return { hidden: say(gap(rt).hidden), pB: { v: b.pHidden, fmt: 'pct' }, rB: { v: b.rank, fmt: 'int' }, gpt: gptSays(rt) };
				}
			},
			task: {
				prompt: { en: 'Click another token in the row to hide it instead, then compare the two models’ shares for it.', zh: '点这一行里的另一个词元，改为藏起它，再比较两个模型给它的份额。' },
				done: (rt) => gap(rt).picked,
				waiting: { en: 'Not yet: click one of the tokens.', zh: '还没有：点其中一个词元。' },
				doneText: { en: 'Done: both models re-ran on the token you hid.', zh: '完成：两个模型都对你藏起的词元重新算了一遍。' }
			},
			linger: 12000,
			look: { en: 'Every number in this step follows the token you hide.', zh: '这一步的每个数都跟着你藏起的词元变化。' }
		},
		{
			id: 'eighty',
			title: { en: 'How masked training picks its puzzles: the 80/10/10 rule', zh: '掩码训练怎样出题：80/10/10 规则' },
			source: 'hide',
			at: 'last',
			play: { from: 'first', to: 'last', interval: 450, unit: { en: 'token', zh: '词元' } },
			view: { key: 'self-supervised-objectives/chips', props: () => ({ mode: 'train' }) },
			provenance: 'live',
			terms: ['masked-language-model', 'pre-training'],
			narration: {
				en: 'In training the gaps are drawn by chance. Each token is chosen with chance {rate}; a chosen one becomes [MASK] {pm} of the time, a random token {pr}, and stays as it is {pk}, so no visible token can be trusted blindly. This draw chose {hid} of your {ofN}: one pass of the masked objective grades {hidAns}, the causal one {clm}.',
				zh: '训练时，空缺是随机抽出来的。每个词元被选中的机会是 {rate}；被选中的词元有 {pm} 变成 [MASK]，{pr} 变成随机词元，{pk} 保持不变，所以看得见的词元也不能盲目相信。这一次抽签选中了你 {ofN}中的 {hid} 个：掩码目标算一遍只批改 {hidAns}，因果目标批改 {clm} 个。',
				vars: (rt, at) => {
					const n = gap(rt).n;
					const hid = doneOf(at).hidden ?? NaN;
					return { rate: pctFact(rt, 'maskRate'), pm: pctFact(rt, 'maskShare'), pr: pctFact(rt, 'randomShare'), pk: pctFact(rt, 'keepShare'), hid, hidAns: plural(hid, { one: '{n} answer', many: '{n} answers', zh: '{n} 个答案' }), ofN: tokens(n), clm: Math.max(n - 1, 0) };
				}
			},
			look: { en: 'Play to watch each token draw a number u between 0 and 1: u below {u} chooses it, then a second number v decides what it becomes.', zh: '点播放，看每个词元依次抽一个 0 到 1 之间的数 u：u 小于 {u} 时它被选中，再由第二个数 v 决定它变成什么。', vars: (rt) => ({ u: String(rt.num('fact:maskRate')) }) }
		},
		{
			id: 'nsp',
			title: { en: 'Next-sentence prediction: a puzzle about pairs', zh: '下一句预测：关于句子对的谜题' },
			source: 'hide',
			at: 'last',
			view: { key: 'self-supervised-objectives/patterns', props: () => ({ show: ['nsp'] }) },
			provenance: { mode: 'illustrative', note: { en: 'A drawing: no model on this site was trained with next-sentence prediction. The pairs are made from your text.', zh: '示意图：本站没有模型用下一句预测训练过。句子对由你的文字组成。' } },
			terms: ['next-sentence-prediction'],
			narration: {
				en: 'The original BERT had a second puzzle: given sentences A and B, did B really come right after A? Half the training pairs are real (IsNext), half take B from another text (NotNext), and the model’s output at [CLS] gives one of the two labels. {pairs}',
				zh: '最初的 BERT 还有第二种谜题：给出句子 A 和 B，B 真的紧跟在 A 后面吗？训练用的句子对一半是真的（IsNext），一半的 B 取自另一段文字（NotNext），模型在 [CLS] 处的输出给出两个标签之一。{pairs}',
				vars: (rt) =>
					({
						pairs: pick(
							gap(rt).split >= 0,
							{ en: 'Your text has more than one sentence, so it splits into a real pair, A then B; the second pair borrows B from another example.', zh: '你的文字不止一句，所以能分成一对真实的句子：先 A 后 B；第二对的 B 借自另一个例句。' },
							{ en: 'Your text is one sentence, so it can only be part A, paired here with a sentence from another example.', zh: '你的文字只有一句，所以只能当 A，这里和另一个例句配成一对。' }
						)
					})
			},
			look: { en: 'The label at the end is the whole target: one answer for a pair of sentences.', zh: '末尾的标签就是全部目标：一对句子只有一个答案。' }
		},
		{
			id: 'spans',
			title: { en: 'Hide whole spans: T5 and BART', zh: '藏起整段：T5 和 BART' },
			source: 'hide',
			at: 'last',
			view: { key: 'self-supervised-objectives/patterns', props: () => ({ show: ['span', 'denoise'] }) },
			provenance: { mode: 'illustrative', note: { en: 'A drawing: no model on this site was trained this way. The spans are drawn on your text with the settings of the T5 and BART papers.', zh: '示意图：本站没有模型这样训练过。片段按 T5 和 BART 论文的设定画在你的文字上。' } },
			terms: ['span-corruption'],
			narration: {
				en: 'Span corruption (T5) hides runs of tokens, {t5rate} of the text with runs of {t5mean} on average, each run replaced by one sentinel such as <X>; the target is only the hidden runs. Text infilling (BART) hides {bartRate}, leaves one [MASK] per run, so even the length of each run is hidden, and asks for the whole text back. {mine}',
				zh: '片段破坏（T5）藏起连续的几段词元：约占文字的 {t5rate}，每段平均 {t5mean} 个，每段换成一个哨兵词元（例如 <X>）；目标只有被藏的那几段。文本填充（BART）藏起 {bartRate}，每段只留一个 [MASK]，连每段有多长也藏起来，并要求把整段文字写回来。{mine}',
				vars: (rt) => {
					const n = gap(rt).n;
					const t = t5Spans(n);
					const b = bartSpans(n);
					return {
						t5rate: pct(T5.rate),
						t5mean: T5.meanSpan,
						bartRate: pct(BART.rate),
						mine: pick(
							n >= 2,
							{ en: `On your text: T5’s target has ${t.outLen} tokens, BART’s all ${b.outLen}.`, zh: `在你的文字上：T5 的目标有 ${t.outLen} 个词元，BART 的是全部 ${b.outLen} 个。` },
							{ en: 'One token is too short to hide a span from.', zh: '只有一个词元，太短，藏不了片段。' }
						)
					};
				}
			},
			look: { en: 'Compare the two targets: T5 writes only what was hidden; BART rewrites everything.', zh: '比较两个目标：T5 只写被藏的部分；BART 把全部重写一遍。' }
		},
		{
			id: 'patches',
			title: { en: 'The same rule on an image: hide most patches', zh: '同样的规则用在图上：藏起大部分图块' },
			source: 'hide',
			at: 'last',
			view: { key: 'self-supervised-objectives/patterns', props: () => ({ show: ['mae', 'jepa'] }) },
			provenance: { mode: 'illustrative', note: { en: 'A drawing: no model on this site was trained this way. The drawing is from Quick, Draw!; the patch grid is TinyViT’s.', zh: '示意图：本站没有模型这样训练过。简笔画来自 Quick, Draw!，图块划分与 TinyViT 相同。' } },
			terms: ['image-patch', 'masked-autoencoder'],
			narration: {
				en: 'An image can be read as patches the way text is read as tokens: TinyViT cuts a {side} × {side} drawing into {np} patches of {ps} × {ps} pixels. Masked autoencoding (MAE) hides {k} of the {np}, {rate} of them, and must paint their pixels back from the few it sees; the MAE paper hides so many because a small hole in an image is easy to fill from its neighbours. JEPA hides a block of neighbouring patches instead and predicts numbers that summarize it, not its pixels.',
				zh: '图片可以像文字切成词元那样切成图块来读：TinyViT 把 {side} × {side} 的简笔画切成 {np} 个 {ps} × {ps} 像素的图块。掩码自编码（MAE）藏起 {np} 个中的 {k} 个，也就是 {rate}，要根据看得见的几块把它们的像素画回来；MAE 论文藏这么多，是因为图上的小洞很容易靠周围补上。JEPA 则藏起一整块相邻的图块，预测的是概括这块的数值，而不是它的像素。',
				vars: () => ({ side: GRID.side, np: GRID.n * GRID.n, ps: GRID.patch, k: maePatches().k, rate: pct(MAE.rate) })
			},
			look: { en: 'Hatched patches are hidden: scattered ones for MAE, one block for JEPA. In MAE’s target, the outlined patches are the ones to paint back.', zh: '打斜线的图块被藏起：MAE 藏的是分散的图块，JEPA 藏的是一整块。在 MAE 的目标里，加框的图块就是要画回来的。' }
		},
		{
			id: 'views',
			title: { en: 'Contrastive learning: nothing to fill in, match two views', zh: '对比学习：不用填空，配对两个视图' },
			source: 'hide',
			at: 'last',
			view: { key: 'self-supervised-objectives/patterns', props: () => ({ show: ['contrastive'] }) },
			provenance: { mode: 'illustrative', note: { en: 'A drawing: no model on this site was trained on these crops. TinyCLIP learns contrastively from pictures and captions.', zh: '示意图：本站没有模型在这些裁剪上训练过。TinyCLIP 用图片和文字说明做对比学习。' } },
			terms: ['contrastive-learning', 'data-augmentation'],
			narration: {
				en: 'Contrastive learning asks for no hidden part back. It makes two altered views of one image, here two crops of {crop} × {crop} pixels with the second one mirrored, and trains the model to give the pair matching numbers while views of other images get different ones. The target is the match itself, not any pixel.',
				zh: '对比学习不要求找回任何被藏的部分。它把一张图做成两个变形的视图，这里是两块 {crop} × {crop} 像素的裁剪，第二块还左右翻转，然后训练模型给这一对算出相配的数值，而别的图片的视图得到不同的数值。目标就是“配对”本身，而不是任何像素。',
				vars: () => ({ crop: cropViews()[0].size })
			},
			look: { en: 'Both views show parts of the same house; neither is complete.', zh: '两个视图都是同一座房子的一部分；哪个都不完整。' }
		},
		{
			id: 'limits',
			title: { en: 'Each objective teaches one skill', zh: '每种目标只教一种本领' },
			source: 'bert',
			at: { layer: 'probs' },
			view: { key: 'self-supervised-objectives/guesses', props: () => ({ mode: 'mlm' }) },
			provenance: 'live',
			kindOf: 'limits',
			terms: ['autoregressive'],
			narration: {
				en: 'What a model can do is what its puzzle trained. TinyBERT fills gaps with text on both sides; when the text stops right after the gap it is right only {fact:accLeft|pct} of the time, below TinyGPT’s {fact:accGpt|pct}, so it writes poorly. TinyGPT was trained for exactly that, but it never uses the words after a gap: {after}',
				zh: '模型会做什么，取决于它的谜题训练了什么。TinyBERT 擅长填两侧都有文字的空；如果文字在空缺之后就结束，它只能猜对 {fact:accLeft|pct}，比 TinyGPT 的 {fact:accGpt|pct} 还低，所以它写不好文章。TinyGPT 正是为此而训练的，但它从不利用空缺后面的词：{after}',
				vars: (rt) => {
					const g = gap(rt);
					const later = g.n - g.k - 1;
					const w = say(g.hidden);
					const rest = { en: later === 1 ? 'the token after it' : `the ${later} tokens after it`, zh: `后面的 ${later} 个词元` };
					const after: I18n =
						later <= 0
							? { en: 'your hidden token is the last one, so here TinyBERT has no words after the gap to read either.', zh: '你藏的是最后一个词元，所以这里 TinyBERT 在空缺后面也没有词可读。' }
							: g.k === 0
								? { en: `for “${w}”, the first token, it has nothing to read at all, while TinyBERT reads ${rest.en}.`, zh: `对第一个词元“${w}”，它根本没有东西可读，而 TinyBERT 读了${rest.zh}。` }
								: { en: `for “${w}” it ignores ${rest.en}, which TinyBERT reads.`, zh: `对“${w}”，它忽略了${rest.zh}，而 TinyBERT 读了${later === 1 ? '它' : '它们'}。` };
					return { after };
				}
			},
			look: { en: 'Click the last token of your text: then TinyBERT, too, has nothing after the gap to read.', zh: '点你文字的最后一个词元：这时 TinyBERT 也没有空缺后面的词可读了。' }
		}
	],
	fixtures: [
		{ id: 'first', label: { en: 'the first token hidden', zh: '藏起第一个词元' }, raw: { text: 'Tom was sad because he lost his ball.', ids: [735, 282, 493, 711, 285, 435, 350, 341, 477, 46], mask: 0, picked: true } },
		{ id: 'last', label: { en: 'two sentences, the last word hidden', zh: '两句话，藏起最后一个词' }, raw: { text: 'Lily went to the park to play with her friends. She had a lot of fun.', ids: [664, 427, 265, 262, 522, 265, 324, 328, 335, 410, 46, 338, 363, 258, 645, 369, 443, 46], mask: 16, picked: true } },
		{ id: 'colour', label: { en: 'a colour hidden', zh: '藏起一个颜色' }, raw: { text: 'She opened the box and found a red ball inside.', ids: [83, 257, 906, 262, 615, 266, 504, 258, 673, 477, 743, 46], mask: 8, picked: true } },
		{ id: 'three', label: { en: 'three sentences', zh: '三句话' }, raw: { text: 'Ben has a red ball. He likes to throw it. His dog runs after it.', ids: [66, 300, 294, 115, 258, 673, 477, 46, 315, 514, 115, 265, 309, 114, 317, 308, 46, 804, 463, 731, 115, 893, 308, 46] } }
	]
};

export default lesson;
