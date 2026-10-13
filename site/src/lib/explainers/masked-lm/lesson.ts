// masked-lm: the guided lesson (ten steps; the concept page's primary path) on the learner's own text. Live: TinyBERT
// ('bert') on the text with the tokens the learner chose, TinyGPT ('gpt') on the same text, and the 80/10/10 draw on a
// small batch ('batch'). Every number is a placeholder: live values in vars(), measured ones from facts.json.
//   hook -> click (task) -> loss (predict) -> copy -> eighty (play) -> draw (task) -> context -> graded -> limits ->
//   crowd (breakIt, task)
import type { At, I18n, LessonDef } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmt3, fmtPct } from '../../atlas/core/format.ts';
import { bertAt, chosenLine, clmLoss, gptAt, mlmLoss, say, text, type RT } from './text.ts';

const tokens = (n: number) => plural(n, { one: '{n} token', many: '{n} tokens', zh: '{n} 个词元' });
/** a measured share from facts.json as a whole percentage ("15%") */
const pctFact = (rt: RT, key: string) => {
	const v = rt.num(`fact:${key}`);
	return Number.isFinite(v) ? `${Math.round(v * 100)}%` : NaN;
};
const done = (at: At) => at.frame?.metrics ?? {};
const NONE: I18n = { en: '', zh: '' };

/** How TinyBERT did at the focus position, as a sentence. */
function focusLine(rt: RT): I18n {
	const x = text(rt);
	if (x.focus < 0) return NONE;
	const g = bertAt(rt, x.focus);
	if (!g.p) return NONE;
	const w = say(x.c[x.focus]);
	return pick(
		g.rank === 1,
		{ en: `For “${w}” it is right, at ${fmtPct(g.pAnswer)}.`, zh: `对“${w}”，它猜对了，概率 ${fmtPct(g.pAnswer)}。` },
		{ en: `For “${w}” it gives ${fmtPct(g.pAnswer)} (rank ${g.rank}) and prefers “${say(g.top)}”.`, zh: `对“${w}”，它给出 ${fmtPct(g.pAnswer)}（第 ${g.rank} 名），更看好“${say(g.top)}”。` }
	);
}

/** The loss on the learner's text and what the visible positions would add, as sentences. */
function lossLines(rt: RT): I18n {
	const L = mlmLoss(rt);
	if (!L.rows.length) return NONE;
	const loss: I18n = L.graded.length
		? { en: `On your text the loss is ${fmt3(L.loss)} nats, the mean of ${L.graded.length} graded row${L.graded.length === 1 ? '' : 's'}.`, zh: `在你的文字上，损失是 ${fmt3(L.loss)} 奈特，也就是 ${L.graded.length} 个被批改行的平均值。` }
		: { en: 'No token is chosen, so nothing is graded and there is no loss.', zh: '没有选中任何词元，所以没有东西被批改，也没有损失。' };
	const seen: I18n = L.seen.length
		? { en: ` At the ${L.seen.length} visible position${L.seen.length === 1 ? '' : 's'} TinyBERT gives the token it can see ${fmtPct(L.seenP)} on average and ranks it first at ${L.seenRight}: there the answer is in the input.`, zh: `在 ${L.seen.length} 个看得见的位置上，TinyBERT 给它看到的词元的平均概率是 ${fmtPct(L.seenP)}，其中 ${L.seenRight} 个排第一：那里的答案就在输入里。` }
		: NONE;
	return { en: loss.en + seen.en, zh: (loss.zh ?? '') + (seen.zh ?? '') };
}

/** After the draw was sent: the counts by fate and the loss; before: what to do. */
function drawLine(rt: RT): I18n {
	const x = text(rt);
	const L = mlmLoss(rt);
	if (!x.n) return NONE;
	if (!x.fromDraw) return { en: 'Send your row of the draw to TinyBERT to see it graded.', zh: '把抽签里你那一行交给 TinyBERT，看它怎样被批改。' };
	const kp = fmtPct(L.kept.reduce((s, r) => s + r.p, 0) / L.kept.length);
	const kept = L.kept.length === 1 ? { en: ` At the kept token it gives the right token ${kp}.`, zh: `在保留不变的词元上，它给正确词元的概率是 ${kp}。` } : L.kept.length ? { en: ` At the kept tokens it gives the right token ${kp} on average.`, zh: `在保留不变的词元上，它给正确词元的平均概率是 ${kp}。` } : NONE;
	return {
		en: `TinyBERT now reads your text as training would show it: ${L.mask.length} [MASK], ${L.random.length} random, ${L.kept.length} kept, and the loss over these ${L.graded.length} rows is ${fmt3(L.loss)} nats.${kept.en}`,
		zh: `TinyBERT 现在读到的就是训练时你的文字的样子：${L.mask.length} 个 [MASK]、${L.random.length} 个随机词元、${L.kept.length} 个保留，这 ${L.graded.length} 行的损失是 ${fmt3(L.loss)} 奈特。${kept.zh}`
	};
}

/** TinyBERT and TinyGPT on the focus token, as a sentence. */
function bothLine(rt: RT): I18n {
	const x = text(rt);
	if (x.focus < 0) return NONE;
	const b = bertAt(rt, x.focus);
	const g = gptAt(rt, x.focus);
	if (!b.p) return NONE;
	return pick(
		x.focus >= 1 && !!g.p,
		{ en: `TinyBERT gives “${say(x.c[x.focus])}” ${fmtPct(b.pAnswer)}, TinyGPT ${fmtPct(g.pAnswer)}.`, zh: `TinyBERT 给“${say(x.c[x.focus])}”的概率是 ${fmtPct(b.pAnswer)}，TinyGPT 是 ${fmtPct(g.pAnswer)}。` },
		{ en: `TinyBERT gives “${say(x.c[x.focus])}” ${fmtPct(b.pAnswer)}; TinyGPT cannot guess the first token at all.`, zh: `TinyBERT 给“${say(x.c[x.focus])}”的概率是 ${fmtPct(b.pAnswer)}；TinyGPT 根本猜不了第一个词元。` }
	);
}

const lesson: LessonDef = {
	title: { en: 'Masked language modelling, step by step', zh: '掩码语言建模，分步课程' },
	summary: {
		en: 'Hide tokens of your own sentence and read TinyBERT’s live guesses, compute the loss on the hidden positions only, watch the 80/10/10 rule draw a batch and send your row to TinyBERT, and set it beside TinyGPT on the same sentence.',
		zh: '在你自己的句子里藏起词元，读 TinyBERT 的实时猜测；只在被藏的位置上算损失；看 80/10/10 规则在一个批次上抽签，并把你那一行交给 TinyBERT；再在同一句话上把它和 TinyGPT 放在一起比较。'
	},
	steps: [
		{
			id: 'hook',
			title: { en: 'Hide a token, ask for it back', zh: '藏起一个词元，再把它要回来' },
			source: 'bert',
			at: { layer: 'probs' },
			view: { key: 'masked-lm/fill', props: () => ({ bars: false }) },
			provenance: 'live',
			terms: ['masked-language-model', 'self-supervised', 'pre-training', 'special-token', 'token'],
			narration: {
				en: 'Masked language modelling is BERT’s pre-training puzzle: hide some tokens of a text and ask the model for them back. The hidden tokens are the right answers, so nobody has to label anything. Your text is {n}; TinyBERT reads it between [CLS] and [SEP], with {hid} hidden as [MASK].',
				zh: '掩码语言建模是 BERT 的预训练谜题：把一段文字里的一些词元藏起来，再让模型把它们找回来。被藏的词元就是正确答案，所以不需要任何人标注。你的文字是 {n}；TinyBERT 读它时前面加 [CLS]、后面加 [SEP]，其中 {hid}被换成了 [MASK]。',
				vars: (rt) => {
					const x = text(rt);
					return { n: tokens(x.n), hid: x.masked.length > 0 ? tokens(x.masked.length) : { en: 'no token', zh: '没有词元' } };
				}
			},
			look: { en: 'Dashed chips are special tokens: [CLS], [SEP] and [MASK]. Underlined chips are the chosen ones.', zh: '虚线框的是特殊词元：[CLS]、[SEP] 和 [MASK]。带下划线的是被选中的词元。' }
		},
		{
			id: 'click',
			title: { en: 'Click tokens to hide them', zh: '点词元，把它们藏起来' },
			source: 'bert',
			at: { layer: 'probs' },
			view: 'masked-lm/fill',
			provenance: 'live',
			terms: ['vocabulary', 'softmax', 'bidirectional-context'],
			narration: {
				en: 'Every hidden position gets its own guess: TinyBERT scores each token of its vocabulary there, reading the whole text around the gap, and softmax turns the scores into shares. {chosen} {focus}',
				zh: '每个被藏的位置都有自己的猜测：TinyBERT 读空缺周围的整段文字，给词表里的每个词元打分，再用 softmax 把分数变成份额。{chosen}{focus}',
				vars: (rt) => ({ chosen: chosenLine(rt), focus: focusLine(rt) })
			},
			look: { en: 'Each panel is one hidden position; the bar in the accent colour is the right token, when it is among the top three.', zh: '每个小面板对应一个被藏的位置；强调色的条是正确词元（如果它排在前三）。' },
			task: {
				prompt: { en: 'Hide at least two tokens in all: click a word you could guess from the rest, and one you could not.', zh: '一共至少藏起两个词元：点一个你能从上下文猜出来的词，再点一个猜不出来的。' },
				done: (rt) => text(rt).picked && text(rt).graded.length >= 2,
				waiting: { en: 'Not yet: click tokens in the row until at least two are hidden.', zh: '还没有：在这一行里点词元，直到至少藏起两个。' },
				doneText: { en: 'Done: TinyBERT re-ran on your gaps.', zh: '完成：TinyBERT 对你的空缺重新算了一遍。' }
			},
			linger: 12000
		},
		{
			id: 'loss',
			title: { en: 'Which positions does the loss grade?', zh: '损失批改哪些位置？' },
			source: 'bert',
			at: { layer: 'probs' },
			view: 'masked-lm/loss',
			provenance: 'live',
			terms: ['loss', 'cross-entropy', 'loss-mask'],
			narration: {
				en: 'Training needs one number that says how wrong the guesses are. At a graded position it is −log p, where p is TinyBERT’s share for the right token: 0 when p is 1, larger the smaller p is (cross-entropy; with the natural log its unit is the nat). The loss of the text is the mean over its graded positions. Your text has {n}, {k} of them chosen.',
				zh: '训练需要一个数来说明猜测错得有多厉害。在一个被批改的位置上，它是 −log p，其中 p 是 TinyBERT 给正确词元的份额：p 为 1 时它是 0，p 越小它越大（交叉熵；用自然对数时，单位是奈特）。整段文字的损失是所有被批改位置的平均值。你的文字有 {n}，其中选中了 {k} 个。',
				vars: (rt) => ({ n: tokens(text(rt).n), k: text(rt).graded.length })
			},
			predict: {
				id: 'masked-lm-graded',
				question: { en: 'Which positions count in the masked-language-model loss?', zh: '掩码语言模型的损失算哪些位置？' },
				options: [
					{ en: 'Every position of the text', zh: '文字的每一个位置' },
					{ en: 'Only the chosen positions', zh: '只有被选中的位置' },
					{ en: 'Only the positions that were not chosen', zh: '只有没被选中的位置' }
				],
				answer: 1,
				explain: {
					en: 'Only the chosen positions count; PyTorch skips the others with the label −100. At a position that was not chosen the right token is in the input, so grading it would reward copying. The table shows both kinds of rows.',
					zh: '只算被选中的位置；PyTorch 用标签 −100 跳过其余位置。没被选中的位置上，正确词元就在输入里，批改它只会奖励照抄。表格里两种行都有。'
				}
			},
			look: { en: 'Shaded rows are graded; their bars are −log p, all on one scale.', zh: '底色加深的行会被批改；它们的条表示 −log p，用同一个刻度。' }
		},
		{
			id: 'copy',
			title: { en: 'Why the visible tokens are left out', zh: '为什么看得见的词元不算' },
			source: 'bert',
			at: { layer: 'probs' },
			view: 'masked-lm/loss',
			provenance: 'live',
			terms: ['loss-mask'],
			narration: {
				en: '{lines} Grading the visible rows would mostly reward copying, so they add nothing to the loss. On unseen stories, TinyBERT’s loss over the chosen tokens averages {fact:valLoss} nats.',
				zh: '{lines}批改看得见的行，多半只是在奖励照抄，所以它们不计入损失。在没见过的故事上，TinyBERT 在被选中词元上的平均损失是 {fact:valLoss} 奈特。',
				vars: (rt) => ({ lines: lossLines(rt) })
			},
			look: { en: 'Compare the p column of the shaded rows with the others.', zh: '比较底色加深的行和其他行的 p 列。' }
		},
		{
			id: 'eighty',
			title: { en: 'How training chooses: 15%, then 80/10/10', zh: '训练怎样选：先选 15%，再按 80/10/10 处理' },
			source: 'batch',
			at: 'last',
			play: { from: 'first', to: 'last', interval: 140, unit: { en: 'token', zh: '词元' } },
			view: 'masked-lm/batch',
			provenance: 'live',
			terms: ['masking-rate', 'batch'],
			narration: {
				en: 'In training nobody clicks. Each token is chosen with chance {rate}; a chosen token becomes [MASK] {pm} of the time, a random token {pr}, and stays as it is {pk}, and every chosen token is graded. This batch has {rows} rows and {N} tokens: {chosen} chosen, {masked} [MASK], {random} random, {kept} kept. Chance sets these counts; on a full training batch they settle near the rule.',
				zh: '训练时没人去点。每个词元以 {rate} 的机会被选中；被选中的词元有 {pm} 变成 [MASK]，{pr} 变成随机词元，{pk} 保持不变，而且每个被选中的词元都要被批改。这个批次有 {rows} 行、{N} 个词元：选中 {chosen} 个，其中 {masked} 个 [MASK]、{random} 个随机、{kept} 个保留。这些数由运气决定；在完整的训练批次上，它们会接近规则的比例。',
				vars: (rt, at) => {
					const m = done(at);
					return { rate: pctFact(rt, 'maskRate'), pm: pctFact(rt, 'maskShare'), pr: pctFact(rt, 'randomShare'), pk: pctFact(rt, 'keepShare'), rows: m.rows ?? NaN, N: m.n ?? NaN, chosen: m.chosen ?? NaN, masked: m.masked ?? NaN, random: m.random ?? NaN, kept: m.kept ?? NaN };
				}
			},
			look: { en: 'Play to watch each token draw a number u: below {u} chooses it, then a second number v decides what it becomes. The last line counts the same rule on a full training batch.', zh: '点播放，看每个词元抽一个数 u：小于 {u} 就被选中，再由第二个数 v 决定它变成什么。最后一行是同样的规则在一个完整训练批次上的计数。', vars: (rt) => ({ u: String(rt.num('fact:maskRate')) }) }
		},
		{
			id: 'draw',
			title: { en: 'Send the draw to TinyBERT', zh: '把抽签结果交给 TinyBERT' },
			source: 'batch',
			at: 'last',
			view: { key: 'masked-lm/loss', props: () => ({ draw: true }) },
			provenance: 'live',
			terms: ['masking-rate', 'fine-tuning'],
			narration: {
				en: 'Real text never contains [MASK]: if every chosen token were [MASK], TinyBERT could trust every token it sees, and later tasks, which feed it plain text, would find it unprepared. Random and kept tokens stop that: they are graded too, so it must check each visible token against its neighbours. {now}',
				zh: '真实的文字里从来没有 [MASK]：如果每个被选中的词元都变成 [MASK]，TinyBERT 就可以相信它看到的每个词元；而后续任务给它的是普通文字，它会措手不及。随机词元和保留的词元阻止了这一点：它们也要被批改，所以模型必须对照邻居检查每个看得见的词元。{now}',
				vars: (rt) => ({ now: drawLine(rt) })
			},
			task: {
				prompt: { en: 'Press the button in the stage to send your row of the training draw to TinyBERT.', zh: '点舞台里的按钮，把训练抽签中你那一行交给 TinyBERT。' },
				done: (rt) => text(rt).fromDraw,
				waiting: { en: 'Not yet: press “Send your row of the training draw to TinyBERT”.', zh: '还没有：点“把训练抽签里你那一行交给 TinyBERT”。' },
				doneText: { en: 'Done: the table now grades your text as training would.', zh: '完成：表格现在按训练的方式批改你的文字。' }
			},
			linger: 12000,
			look: { en: 'In the table, the arrow after a graded token says what TinyBERT read there.', zh: '在表格里，被批改词元后面的箭头说明 TinyBERT 在那里读到了什么。' }
		},
		{
			id: 'context',
			title: { en: 'The same token, two objectives', zh: '同一个词元，两种目标' },
			source: 'bert',
			at: { layer: 'probs' },
			view: 'masked-lm/context',
			provenance: 'live',
			terms: ['next-token-prediction', 'causal-mask', 'bidirectional-context'],
			narration: {
				en: 'TinyGPT was trained on the same stories with the causal objective: predict each token from the tokens before it. {reads} {both}',
				zh: 'TinyGPT 用同样的故事训练，但目标是因果的：用前面的词元预测每一个词元。{reads}{both}',
				vars: (rt) => {
					const x = text(rt);
					const k = x.focus;
					const reads: I18n =
						k < 0
							? { en: 'No token is chosen now: hide one in the input to compare the two models on it.', zh: '现在没有选中任何词元：在输入框里藏起一个，就能在它上面比较两个模型。' }
							: k === 0
								? { en: `For token 0 (counting from 0), “${say(x.c[0])}”, TinyBERT reads the ${x.n - 1} token${x.n === 2 ? '' : 's'} after it, and TinyGPT has nothing to read.`, zh: `对第 0 个词元（从 0 数起）“${say(x.c[0])}”，TinyBERT 读它后面的 ${x.n - 1} 个词元；TinyGPT 在它前面什么也读不到。` }
								: { en: `For token ${k} (counting from 0), “${say(x.c[k])}”, TinyBERT reads ${k} token${k === 1 ? '' : 's'} before it and ${x.n - k - 1} after it; TinyGPT reads only the ${k} before it.`, zh: `对第 ${k} 个词元（从 0 数起）“${say(x.c[k])}”，TinyBERT 读它前面的 ${k} 个词元和后面的 ${x.n - k - 1} 个；TinyGPT 只读前面的 ${k} 个。` };
					return { reads, both: bothLine(rt) };
				}
			},
			look: { en: 'Underlined chips are what each model reads; the outlined chip is the token to guess.', zh: '带下划线的词元是每个模型读到的内容；加框的是要猜的词元。' }
		},
		{
			id: 'graded',
			title: { en: 'Answers per pass: every token, or the chosen few', zh: '每算一遍的答案：每个词元，还是被选中的几个' },
			source: 'bert',
			at: { layer: 'probs' },
			view: { key: 'masked-lm/loss', props: () => ({ gpt: true }) },
			provenance: 'live',
			terms: ['next-token-prediction', 'loss-mask'],
			narration: {
				en: 'One pass of TinyGPT over your text grades {clmN}, every token after the first; one pass of TinyBERT grades only the {k} chosen. At the usual {rate}, a masked model gets about one answer per {per} tokens it reads, so it needs more text for the same number of answers. In return, every answer is made with both sides in view.',
				zh: 'TinyGPT 在你的文字上算一遍，批改 {clmN}，也就是第一个之后的每个词元；TinyBERT 算一遍只批改选中的 {k} 个。按通常的 {rate}，掩码模型大约每读 {per} 个词元才得到一个答案，所以要得到同样多的答案，它需要更多文字。作为交换，它的每个答案都是看着两侧做出的。',
				vars: (rt) => ({ clmN: plural(clmLoss(rt).rows.length, { one: '{n} position', many: '{n} positions', zh: '{n} 个位置' }), k: text(rt).graded.length, rate: pctFact(rt, 'maskRate'), per: Math.round(1 / rt.num('fact:maskRate')) })
			},
			look: { en: 'The last column has a bar on nearly every row; the TinyBERT column only on the shaded ones.', zh: '最后一列几乎每行都有条；TinyBERT 那一列只有底色加深的行才有。' }
		},
		{
			id: 'limits',
			title: { en: 'What masking does not teach: writing', zh: '掩码教不会的事：写作' },
			source: 'bert',
			at: { layer: 'probs' },
			view: 'masked-lm/context',
			provenance: 'live',
			kindOf: 'limits',
			terms: ['autoregressive'],
			narration: {
				en: 'Each objective trains the skill it grades. On {fact:cmpN|int} tokens of unseen stories, hidden one at a time, TinyBERT’s top guess is right {fact:accBoth|pct} of the time with both sides, but only {fact:accLeft|pct} when the text stops right after the gap, below TinyGPT’s {fact:accGpt|pct} on the same gaps. So it predicts the next token, the skill writing needs, worse than TinyGPT, and its lower loss, {fact:valLoss} against {fact:gptValLoss} nats, comes from an easier puzzle, not better writing.',
				zh: '每种目标只训练它所批改的那种本领。在没见过的故事里每次只藏一个词元，共藏 {fact:cmpN|int} 次：TinyBERT 读两侧时第一猜测对了 {fact:accBoth|pct}，但文字在空缺后就结束时只有 {fact:accLeft|pct}，比 TinyGPT 在同样空缺上的 {fact:accGpt|pct} 还低。所以它预测下一个词元（写作靠的正是这个本领）不如 TinyGPT；它更低的损失（{fact:valLoss} 对 {fact:gptValLoss} 奈特）来自更容易的谜题，而不是更好的写作能力。'
			},
			look: { en: 'Hide the last token of your text in the input: then only [SEP] follows the gap, as in the {fact:accLeft|pct} setting.', zh: '在输入框里藏起你文字的最后一个词元：这时空缺后面只剩 [SEP]，正是 {fact:accLeft|pct} 那种情形。' }
		},
		{
			id: 'crowd',
			title: { en: 'Hide too much, and the guesses fall apart', zh: '藏得太多，猜测就散了' },
			source: 'bert',
			at: { layer: 'probs' },
			view: { key: 'masked-lm/fill', props: () => ({ crowd: true }) },
			provenance: 'live',
			kindOf: 'breakIt',
			terms: ['masking-rate'],
			narration: {
				en: 'Each gap is filled on its own: TinyBERT never sees its other guesses, so with many gaps it has little left to read and nothing ties the guesses together. {chosen} On unseen stories, with {rate} chosen, it is right at {fact:valAccMasked|pct} of the [MASK] positions.',
				zh: '每个空缺都是单独填的：TinyBERT 看不到自己的其他猜测，所以空缺一多，它能读的就少了，各个猜测之间也没有任何联系。{chosen}在没见过的故事上，选中 {rate} 时，它在 [MASK] 位置上猜对 {fact:valAccMasked|pct}。',
				vars: (rt) => ({ chosen: chosenLine(rt, true), rate: pctFact(rt, 'maskRate') })
			},
			task: {
				prompt: { en: 'Press “Hide two tokens of every three” and read what is left of the guesses.', zh: '点“每三个词元藏起两个”，看看还剩下哪些猜测是对的。' },
				done: (rt) => {
					const x = text(rt);
					return x.picked && x.n > 1 && x.graded.length * 2 >= x.n;
				},
				waiting: { en: 'Not yet: hide at least half of the tokens.', zh: '还没有：至少藏起一半的词元。' }
			},
			linger: 12000,
			look: { en: 'Count the panels with the accent bar on top. With its neighbours hidden too, a gap is usually filled right less often than when it is hidden alone.', zh: '数一数强调色的条排在最上面的小面板。邻居也被藏起来时，一个空缺被填对的机会通常比单独藏它时更小。' }
		}
	],
	fixtures: [
		{ id: 'first', label: { en: 'the first token hidden', zh: '藏起第一个词元' }, raw: { text: 'Tom was sad because he lost his ball.', ids: [735, 282, 493, 711, 285, 435, 350, 341, 477, 46], fate: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], picked: true, focus: 0 } },
		{ id: 'none', label: { en: 'every token given back', zh: '所有词元都还原了' }, raw: { text: 'Tom was sad because he lost his ball.', ids: [735, 282, 493, 711, 285, 435, 350, 341, 477, 46], fate: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], picked: true } },
		{ id: 'drawn', label: { en: 'a training draw: [MASK], random and kept', zh: '一次训练抽签：[MASK]、随机和保留' }, raw: { text: 'Tom was sad because he lost his ball.', ids: [735, 282, 493, 711, 285, 435, 350, 341, 477, 46], fate: [0, 0, 1, 0, 0, 2, 0, 0, 3, 0], swap: [-1, -1, -1, -1, -1, 600, -1, -1, -1, -1], picked: true, fromDraw: true, focus: 2 } },
		{ id: 'crowd', label: { en: 'two tokens of every three hidden', zh: '每三个词元藏起两个' }, raw: { text: 'She opened the box and found a red ball inside.', ids: [83, 257, 906, 262, 615, 266, 504, 258, 673, 477, 743, 46], fate: [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], picked: true, focus: 0 } },
		{ id: 'three', label: { en: 'three sentences', zh: '三句话' }, raw: { text: 'Ben has a red ball. He likes to throw it. His dog runs after it.', ids: [66, 300, 294, 115, 258, 673, 477, 46, 315, 514, 115, 265, 309, 114, 317, 308, 46, 804, 463, 731, 115, 893, 308, 46] } }
	]
};

export default lesson;
