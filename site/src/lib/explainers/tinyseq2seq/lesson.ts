// tinyseq2seq: the guided lesson (11 steps). Every number is a {placeholder}: structure from the spec (dims()), live
// values from the trace of the learner's date (source 'fwd') or the 'decode' process, measured ones from facts.json.
// Claims that are not computed live are tested in claims.test.ts.
//   1 characters → 2 the encoder reads → 3 the last state is a bottleneck → 4 the decoder's first step → 5 scores
//   (predict) → 6 softmax → 7 the context vector → 8 the next character → 9 repeat: the alignment matrix →
//   10 your date in another format (task) → 11 formats it never saw (limits)
import type { I18n, LessonDef } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmtPct } from '../../atlas/core/format.ts';
import { FORMATS, readDate, TGT } from './data.ts';
import { chr, fx, look, mixParts, restOf, stepAt, sym, verdict, type RT } from './look.ts';
import { textOf } from './adapters/date-text/adapter.ts';
import { TRIES } from './views/align/meta.ts';

/** Structural numbers from the spec (never typed into the text). */
export function dims(rt: RT) {
	const L = (id: string) => rt.spec?.layers.find((l) => l.id === id);
	const steps = Number(L('dec')?.params?.steps ?? NaN);
	return { E: Number(L('emb')?.outShape[1] ?? NaN), H: Number(L('enc')?.params?.hidden ?? NaN), steps, chars: steps - 1, nsym: TGT.length, nf: FORMATS.length };
}
/** the decoder step that writes the month's last digit (YYYY-MM-DD: index 6) */
export const MONTH_ROW = 6;
const q = (c: string) => `“${chr(c)}”`;
/** the training format the learner's text is written in, by name */
export function fmtName(rt: RT): I18n {
	const f = FORMATS.find((x) => x.id === look(rt).fmt);
	return f ? { en: `in the format “${f.label.en}”`, zh: `用的是“${f.label.zh}”格式` } : { en: 'in none of the training formats', zh: '不属于任何一种训练格式' };
}

const lesson: LessonDef = {
	title: { en: 'A tiny seq2seq model: convert a date and watch the decoder look back', zh: '小型 seq2seq：转换日期，看解码器如何回看输入' },
	summary: {
		en: 'An encoder reads your date one character at a time; a decoder writes it as 2021-03-05, and at every step attention scores your characters, turns the scores into weights and mixes the encoder states into a context vector. Watch the alignment matrix fill in, then test formats it never saw.',
		zh: '编码器一次读你日期的一个字符；解码器把它写成 2021-03-05 的样子，每一步注意力都给你的字符打分，把分数变成权重，再把编码器状态混合成上下文向量。看对齐矩阵一行行填满，再测试它没见过的格式。'
	},
	steps: [
		{
			id: 'chars', provenance: 'live',
			view: { key: 'tinyseq2seq/encoder', props: () => ({ mode: 'chars' }) },
			title: { en: 'Your date becomes characters', zh: '你的日期变成字符' },
			narration: {
				en: 'The model reads your date as {n}: each one gets an id, its place in the model’s alphabet (a character it does not know becomes ?). Each id picks a row of {E} learned numbers from a table, the character’s vector (an embedding). The columns below are these vectors.',
				zh: '模型把你的日期读成 {n}：每个字符得到一个 id，也就是它在模型字母表里的位置（不认识的字符变成 ?）。每个 id 从一张表里取出一行 {E} 个学来的数，也就是这个字符的向量（嵌入）。下面的每一列就是这些向量。',
				vars: (rt) => ({ ...dims(rt), n: plural(look(rt).T, { one: '{n} character', many: '{n} characters', zh: '{n} 个字符' }) })
			},
			look: { en: 'The number under each character is its id. Hover or tap a cell to read its value.', zh: '每个字符下面的数是它的 id。把鼠标悬停在格子上（或点按）可以读出数值。' }, terms: ['embedding']
		},
		{
			id: 'read', provenance: 'live',
			view: { key: 'tinyseq2seq/encoder', props: () => ({ mode: 'read' }) },
			play: 'view',
			title: { en: 'The encoder reads left to right', zh: '编码器从左往右读' },
			narration: {
				en: 'The model has two halves: an encoder that reads your date and a decoder that writes the answer. The encoder is a recurrent network called a GRU (gated recurrent unit): it keeps a state of {H} numbers and updates it after every character, with the same weights each time. Gates, numbers between 0 and 1, decide how much of each number is kept and how much is replaced by something new from the character. Column j is the state after reading characters 0 to j.',
				zh: '模型分两半：编码器读你的日期，解码器写出答案。编码器是一种叫 GRU（门控循环单元）的循环网络：它保存一个由 {H} 个数组成的状态，每读一个字符就更新一次，每次用的都是同一组权重。门是介于 0 和 1 之间的数，决定每个数保留多少旧值、又有多少换成根据这个字符算出的新值。第 j 列是读完第 0 到第 j 个字符之后的状态。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'The columns appear one by one as the encoder reads; the outlined one belongs to the character just read. Pause, or step with the arrows.', zh: '编码器每读一个字符就多出一列；框出的那一列属于刚读的字符。可以暂停，或用箭头一步步看。' }, linger: 12000, terms: ['encoder-decoder', 'recurrent-network', 'hidden-state']
		},
		{
			id: 'bottleneck', provenance: 'live',
			view: { key: 'tinyseq2seq/encoder', props: () => ({ mode: 'last' }) },
			title: { en: 'The last state is a bottleneck', zh: '最后一个状态是瓶颈' },
			narration: {
				en: 'Only the last column, after {last}, has read the whole date. Without attention (steps 5 to 7: the decoder looks back at every column), an encoder-decoder hands just these {H} numbers to the decoder, however long the input: a bottleneck. For short dates that still works: a same-size model without attention also got {fact:noAttnExact|pct} of the {fact:testN|int} test dates right. Squeezed to {fact:h8|int} numbers, it got only {fact:h8NoAttnExact|pct}; with attention, a model just as small got {fact:h8AttnExact|pct}.',
				zh: '只有最后一列，也就是读完{last}之后的状态，读过了整个日期。没有注意力（第 5 到第 7 步：解码器回看每一列）时，编码器–解码器只把这 {H} 个数交给解码器，不管输入多长：这就是瓶颈。对短短的日期这还够用：同样大小、没有注意力的模型，在 {fact:testN|int} 个测试日期上也答对了 {fact:noAttnExact|pct}。把状态压到 {fact:h8|int} 个数，它只答对 {fact:h8NoAttnExact|pct}；而同样小、带注意力的模型答对了 {fact:h8AttnExact|pct}。',
				vars: (rt) => ({ ...dims(rt), last: q(look(rt).chars.at(-1) ?? '') })
			},
			look: { en: 'The outlined column is all the decoder would get without attention. The table was measured offline on the test dates.', zh: '框出的这一列，就是没有注意力时解码器能拿到的全部。表格是在测试日期上离线测得的。' }, terms: ['encoder-decoder', 'attention']
		},
		{
			id: 'start', provenance: 'live',
			view: { key: 'tinyseq2seq/attend', props: () => ({ mode: 'start', t: 0 }) },
			title: { en: 'The decoder’s first step', zh: '解码器的第一步' },
			narration: {
				en: 'The decoder writes the answer one symbol at a time: {chars} characters, then an end symbol ■. It is a second GRU with its own weights. It starts from the encoder’s last state and reads a start symbol ▶ in place of a previous output. Its first job is the year’s first digit.',
				zh: '解码器一次写一个符号：先写 {chars} 个字符，再写一个结束符号 ■。它是另一个 GRU，有自己的一组权重。它从编码器的最后一个状态出发，读入一个起始符号 ▶ 来代替“上一个输出”。它的第一项任务是写出年份的第一位数字。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'These {H} numbers are the decoder’s state before step 0 (counting from 0).', zh: '这 {H} 个数是解码器在第 0 步（从 0 数起）之前的状态。', vars: (rt) => dims(rt) }, terms: ['encoder-decoder', 'hidden-state']
		},
		{
			id: 'scores', provenance: 'live',
			view: { key: 'tinyseq2seq/attend', props: () => ({ mode: 'scores', t: 0 }) },
			title: { en: 'A score for every character', zh: '给每个字符打分' },
			narration: {
				en: 'Attention lets the decoder look back at every encoder state, not only the last one. For each character j it computes a score: its own state becomes a query (what it looks for), the encoder state becomes a key (what is there), and the score is v · tanh(query + key): tanh squeezes each of the {H} sums into −1 to 1, and the learned weights v turn them into one weighted sum. Here the {n} scores run from {low|n2} to {high|n2}.',
				zh: '注意力让解码器回看每一个编码器状态，而不只是最后一个。对每个字符 j，它算一个分数：解码器自己的状态变成查询（它要找什么），编码器状态变成键（那里有什么），分数就是 v · tanh(查询 + 键)：tanh 把 {H} 个和分别压到 −1 到 1 之间，再用学来的权重 v 把它们合成一个加权和。这里 {n} 个分数从 {low|n2} 到 {high|n2}。',
				vars: (rt, _, e = Array.from(stepAt(rt, 0).e ?? [])) => ({ ...dims(rt), n: e.length, low: e.length ? Math.min(...e) : NaN, high: e.length ? Math.max(...e) : NaN })
			},
			predict: {
				id: 'tinyseq2seq-scores',
				question: { en: 'To write the year’s first digit, which characters will get the highest scores?', zh: '为了写出年份的第一位数字，哪些字符会得到最高的分数？' },
				options: [{ en: 'The letters of the month', zh: '月份的字母' }, { en: 'The digits of the year', zh: '年份的数字' }, { en: 'The day', zh: '日' }, { en: 'All characters about the same', zh: '所有字符都差不多' }],
				answer: 1,
				explain: { en: 'The year’s digits. Look closely: the top score is usually on the year’s second digit, not the first. The encoder reads left to right, so its state at the second digit has already seen the first one, and the pair tells it a year starts here.', zh: '年份的数字。仔细看：最高分通常落在年份的第二位数字上，而不是第一位。编码器从左往右读，所以它在第二位数字上的状态已经看过第一位数字，这两位一起告诉它：年份从这里开始。' }
			},
			look: { en: 'The strips work out the score of the circled character: add up the numbers of the last strip and you get its score (the line under them).', zh: '这几条算出了圈出字符的分数：把最后一条里的数加起来，就是它的分数（见下面那一行）。' }, terms: ['attention', 'query-key-value']
		},
		{
			id: 'softmax', provenance: 'live',
			view: { key: 'tinyseq2seq/attend', props: () => ({ mode: 'weights', t: 0 }) },
			title: { en: 'Scores become weights', zh: '分数变成权重' },
			narration: {
				en: 'Softmax turns the scores into weights between 0 and 1 that add up to 1: e (about {eu|n4}) to the power (score − largest score), divided by the sum of these over all characters. {c} at position {j} (counting from 0) gets {w|pct}{year}: {few}. On {fact:yearSecondN|int} of the {fact:testN|int} test dates, this step put the most weight on the year’s second digit.',
				zh: 'Softmax 把分数变成介于 0 和 1 之间、加起来等于 1 的权重：先求 e（约 {eu|n4}）的（分数 − 最大分数）次方，再除以所有字符这个值的和。位置 {j}（从 0 数起）上的{c}得到 {w|pct}{year}：{few}。在 {fact:testN|int} 个测试日期中，有 {fact:yearSecondN|int} 个在这一步把最多的权重放在年份的第二位数字上。',
				vars: (rt) => {
					const S = stepAt(rt, 0);
					const top3 = Array.from(S.a ?? []).sort((x, y) => y - x).slice(0, 3).reduce((a, b) => a + b, 0);
					const few = pick(top3 > 0.9, { en: 'nearly all the weight sits on a few characters, and that is what “looking” means here', zh: '几乎所有权重都落在少数几个字符上，这里的“看”就是这个意思' }, { en: 'the weight is spread out, a sign that the model is unsure where to look', zh: '权重很分散，说明模型拿不准该看哪里' });
					return { eu: Math.E, j: S.j, c: q(S.chars[S.j] ?? ''), w: S.a?.[S.j] ?? NaN, few, year: Number.isFinite(S.yearShare) ? { en: `; the year’s four digits together get ${fmtPct(S.yearShare)}`, zh: `；年份的四位数字合起来得到 ${fmtPct(S.yearShare)}` } : { en: '', zh: '' } };
				}
			},
			look: { en: 'Each bar is one character’s weight, in %.', zh: '每根条是一个字符的权重，单位 %。' }, terms: ['softmax']
		},
		{
			id: 'context', provenance: 'live',
			view: { key: 'tinyseq2seq/attend', props: () => ({ mode: 'context', t: 0 }) },
			title: { en: 'The context vector', zh: '上下文向量' },
			narration: {
				en: 'Each encoder state is multiplied by its weight, and the results are added up, number by number: the context vector, {H} numbers that say what this step read. With {w|pct} on {c}, {mostly}. Take its number {f} (counting from 0; the outlined row): {c0} gives {w0} × {h0} = {wh}, the other characters add {rest}, so it is {cf} ({c0} alone has {h0}).',
				zh: '每个编码器状态乘以自己的权重，再逐个数相加，就得到上下文向量：{H} 个数，表示这一步读到了什么。{c}占了 {w|pct}，{mostly}。以它的第 {f} 个数（从 0 数起；框出的那一行）为例：{c0}贡献 {w0} × {h0} = {wh}，其他字符加上 {rest}，所以它是 {cf}（{c0}单独的这个数是 {h0}）。',
				vars: (rt) => {
					const S = stepAt(rt, 0);
					const M = mixParts(rt, 0);
					const p = M?.parts[0];
					const s = (v: number | undefined) => (v === undefined || !Number.isFinite(v) ? NaN : fx(v));
					return { ...dims(rt), w: S.a?.[S.j] ?? NaN, mostly: pick((S.a?.[S.j] ?? 0) >= 0.5, { en: 'it is mostly that character’s state', zh: '所以它基本就是这个字符的状态' }, { en: 'it mixes the states of several characters', zh: '它混合了好几个字符的状态' }), c: q(S.chars[S.j] ?? ''), f: M?.f ?? NaN, w0: s(p?.w), h0: s(p?.h), wh: s(p?.wh), c0: q(S.chars[p?.j ?? -1] ?? ''), rest: s(M && p ? restOf(M.cf, p.wh) : undefined), cf: s(M?.cf) };
				}
			},
			look: { en: 'Faded columns had little weight; the column on the right is the context vector. The line under the picture works out the outlined number.', zh: '颜色淡的列权重小；最右边那一列就是上下文向量。图下面那一行算出了框出的那个数。' }, terms: ['context-vector']
		},
		{
			id: 'next', provenance: 'live',
			view: { key: 'tinyseq2seq/attend', props: () => ({ mode: 'next', t: 0 }) },
			title: { en: 'Write the next character', zh: '写下一个字符' },
			narration: {
				en: 'The decoder’s GRU now updates its state from the symbol it read (▶) and the context vector. From the new state and the context, a last weighted sum gives one score for each of the {nsym} symbols (the digits, the dash, the end symbol and the start symbol, which it learned never to write), and softmax turns them into chances. {y} gets {p|pct}, the runner-up {r} {pr|pct}. Greedy decoding writes the most likely symbol and feeds it back as the next step’s input.',
				zh: '解码器的 GRU 根据读入的符号（▶）和上下文向量更新自己的状态。再用新状态和上下文向量做最后一次加权和，给 {nsym} 个符号各一个分数（数字、短横线、结束符号，还有它学会了从不写的起始符号），softmax 把它们变成概率。{y}得到 {p|pct}，第二名{r}得到 {pr|pct}。贪心解码写下概率最大的符号，并把它作为下一步的输入送回去。',
				vars: (rt, _, S = stepAt(rt, 0)) => ({ ...dims(rt), y: q(sym(S.y)), p: S.p?.[S.y] ?? NaN, r: q(sym(S.runnerUp)), pr: S.p?.[S.runnerUp] ?? NaN })
			},
			look: { en: 'The bar marked as chosen is the symbol written at this step.', zh: '标为“选中”的那根条，就是这一步写下的符号。' }, terms: ['greedy-decoding', 'logit']
		},
		{
			id: 'align', provenance: 'live',
			source: 'decode',
			at: 'last',
			play: 'view',
			view: { key: 'tinyseq2seq/align', props: () => ({ play: true }) },
			title: { en: 'Repeat: the alignment matrix', zh: '重复：对齐矩阵' },
			narration: {
				en: 'The decoder repeats steps 5 to 8 for every output symbol, {n} in all, each time looking back at your characters. Each row of this alignment matrix is one step’s weights. {month}It wrote “{wrote}”{verdict}.',
				zh: '解码器为每个输出符号重复第 5 到第 8 步，一共 {n} 次，每次都回看你的字符。这个对齐矩阵的每一行就是某一步的权重。{month}它写下“{wrote}”{verdict}。',
				vars: (rt) => {
					const L = look(rt);
					const S = stepAt(rt, MONTH_ROW);
					const m = S.a && /^\d{4}-\d{2}-\d{2}$/.test(L.written) ? { en: `To write the month’s last digit, ${q(sym(S.y))}, it looked hardest at ${q(S.chars[S.j])} at position ${S.j} (${fmtPct(S.a[S.j])}): by there the encoder has read enough of the month to tell which one it is. `, zh: `为了写出月份的最后一位${q(sym(S.y))}，它看得最多的是位置 ${S.j} 上的${q(S.chars[S.j])}（${fmtPct(S.a[S.j])}）：读到那里，编码器已经读到足够多的月份信息，能分清是哪个月。` } : { en: 'Its answer is not even shaped like a date. ', zh: '它的答案连日期的样子都不像。' };
					return { n: L.end >= 0 ? L.end + 1 : L.ids.length, month: m, wrote: L.written, verdict: verdict(rt) };
				}
			},
			look: { en: 'Rows: the symbols written (■ = end); columns: your characters; the darker a cell, the larger the weight. It fills in row by row; the slider and the arrows step back through it.', zh: '行：写下的符号（■ 表示结束）；列：你的字符；格子越深，权重越大。它一行一行地填满；滑块和箭头可以往回一步步看。' }, linger: 12000, terms: ['attention']
		},
		{
			id: 'yours', provenance: 'live',
			view: 'tinyseq2seq/align',
			title: { en: 'Your own date, in another format', zh: '你自己的日期，换一种格式' },
			narration: {
				en: 'The same weights handle all {nf} formats the model was trained on; only where it looks changes. Your text is {fmt}, and the model wrote “{wrote}”{verdict}. On {fact:testN|int} test dates in these formats it was right {fact:testExact|pct} of the time, and {fact:testUnseenExact|pct} on the {fact:testUnseenN|int} whose exact text never appeared in training.',
				zh: '同一组权重处理训练时见过的全部 {nf} 种格式，变的只是它看哪里。你的文字{fmt}，模型写下“{wrote}”{verdict}。在这些格式的 {fact:testN|int} 个测试日期上，它答对了 {fact:testExact|pct}；其中 {fact:testUnseenN|int} 个的原文在训练中从没出现过，它答对了 {fact:testUnseenExact|pct}。',
				vars: (rt) => ({ ...dims(rt), fmt: fmtName(rt), wrote: look(rt).written, verdict: verdict(rt) })
			},
			task: {
				prompt: { en: 'Type a date in another training format, say “20 July 1969”, “7/20/1969”, “20.07.1969” or “Sun, 20 Jul 1969”, and compare where the rows look now.', zh: '换一种训练格式输入一个日期，例如“20 July 1969”“7/20/1969”“20.07.1969”或“Sun, 20 Jul 1969”，比较一下现在每一行看的是哪里。' },
				done: (rt) => {
					const f = readDate(textOf(rt.input)).fmt;
					return f !== null && f !== 'long';
				},
				waiting: { en: 'Not yet: type a real date in one of the training formats other than “month day, year”.', zh: '还没有：用“月 日, 年”以外的一种训练格式输入一个真实的日期。' }
			},
			look: { en: 'Compare with step 9: which columns do the rows of the year, the month and the day look at now?', zh: '和第 9 步比一比：现在年、月、日那几行各看的是哪几列？' },
			linger: 12000, terms: ['encoder-decoder']
		},
		{
			id: 'limits', provenance: 'live',
			view: { key: 'tinyseq2seq/align', props: () => ({ tries: true }) },
			kindOf: 'limits',
			title: { en: 'Formats it never saw', zh: '它没见过的格式' },
			narration: {
				en: 'The model only learned its {nf} training formats. Tested on {fact:offN|int} dates of each kind, it gets {fact:offYearFirst|pct} right when the year comes first, {fact:offDashes|pct} with dashes and {fact:offSpelled|pct} with the day as a word; a missing comma hardly matters ({fact:offNoComma|pct}). And it never says it is unsure: for a day that does not exist, such as “{bad}”, it still writes a date. On your text it wrote “{wrote}”{verdict}.',
				zh: '模型只学过训练数据里的 {nf} 种格式。每一类各测 {fact:offN|int} 个日期：年份写在前面的，它只答对 {fact:offYearFirst|pct}；用短横线分隔的答对 {fact:offDashes|pct}；“日”写成英文单词（如 fifth）的答对 {fact:offSpelled|pct}；少一个逗号几乎没有影响（{fact:offNoComma|pct}）。而且它从不说自己没把握：对一个不存在的日子，比如“{bad}”，它照样写出一个日期。对你的文字，它写下“{wrote}”{verdict}。',
				vars: (rt) => ({ ...dims(rt), bad: TRIES[4], wrote: look(rt).written, verdict: verdict(rt) })
			},
			look: { en: 'Each button types a date in a format the model never saw. Read the answer first, then the rows: on “{sp}” most rows still put most of their weight on one character, yet the answer is wrong, so sharp rows are no proof.', zh: '每个按钮会输入一个模型没见过的格式的日期。先看答案，再看每一行：在“{sp}”上，大多数行仍然把大部分权重放在一个字符上，答案却是错的，所以行很清楚并不能证明答案对。', vars: () => ({ sp: TRIES[3] }) }, terms: ['out-of-distribution']
		}
	],
	fixtures: [
		{ id: 'dmy', label: { en: 'day month year', zh: '日 月 年' }, raw: '20 July 1969' },
		{ id: 'us', label: { en: 'US numbers', zh: '美式数字' }, raw: '12/25/2023' },
		{ id: 'eu', label: { en: 'European numbers', zh: '欧式数字' }, raw: '01.10.1949' },
		{ id: 'weekday', label: { en: 'with the weekday', zh: '带星期' }, raw: 'Sunday, February 29, 2032' },
		{ id: 'lower', label: { en: 'lower case, ordinal', zh: '小写、序数' }, raw: 'december 31st, 1999' },
		{ id: 'upper', label: { en: 'upper case, short weekday', zh: '大写、缩写星期' }, raw: 'FRI, 5 MAR 2021' },
		{ id: 'yearFirst', label: { en: 'year first (never seen)', zh: '年份在前（没见过）' }, raw: '2021/03/05' },
		{ id: 'impossible', label: { en: 'March 32 (no such day)', zh: '3 月 32 日（没有这一天）' }, raw: 'March 32, 2021' }
	]
};

export default lesson;
