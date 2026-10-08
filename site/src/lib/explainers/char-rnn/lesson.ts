// char-rnn: the guided lesson (design §7 outline, 11 steps). Every number is a {placeholder}: live values from the
// trace of the learner's text and from the unroll / sample runs, measured ones from facts.json ({fact:...}).
//   1 characters -> 2 h as memory -> 3 forget gate -> 4 input gate + candidate -> 5 c = f·c + i·g -> 6 output gate ->
//   7 next-character bars -> 8 unrolled in time (play) -> 9 a unit that counts letters (task; precomputed contexts) ->
//   10 sampling vs greedy (predict) -> 11 limits: the memory fades (predict).
// Steps 3-6 follow unit {fact:counterUnit} at the space after the longest word (data.ts momentOf). Claims that are not
// computed live are tested in claims.test.ts.
import type { At, ExplainerRuntime, I18n, LessonDef, Tensor } from '../../atlas/types/index.ts';
import { num, plural } from '../../atlas/lesson/say.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import { REF_GATE, alphabetOf, charsOf, fadeAt, longest, momentOf, riskiest, show, unitOf, written } from './views/timeline/data.ts';

type RT = ExplainerRuntime;
const tr = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
const at2 = (rt: RT, id: string, t: number, j: number) => {
	const x = tr(rt, id);
	return x && t >= 0 && t < x.shape[0] ? x.data[t * x.shape[1] + j] : NaN;
};
const H = (rt: RT) => Number(tr(rt, 'lstm')?.shape[1] ?? NaN);
/** the step-3-to-6 moment: unit U at character t (the space after the longest word, else the last character) */
const moment = (rt: RT) => ({ ...momentOf(charsOf(rt)), U: unitOf(rt) });
const gate = (rt: RT) => {
	const { t, U } = moment(rt);
	return { layer: 'lstm', unit: [Math.max(0, t), U] };
};
/** the stage of the steps on the LSTM layer of the trace; steps 3-6 add the zoom on unit U at the moment (one stage, four ideas) */
const LSTM = { source: 'fwd', at: { layer: 'lstm' }, provenance: 'live' } as const;
const GATE = { ...LSTM, select: gate, view: 'zoom.gate' } as const;
/** the gate values of the moment, as the zoom view computes them (explain), else from the trace */
function cell(rt: RT, at: At) {
	const m = moment(rt);
	const v = at.unit?.values ?? {};
	const g = (k: string, id: string) => (typeof v[k] === 'number' ? v[k] : at2(rt, id, m.t, m.U));
	const [f, i, gg] = [g('f', 'lstm#f'), g('i', 'lstm#i'), g('g', 'lstm#g')];
	const where = m.space
		? { en: `the space after “${m.word}” (character ${m.t}, counting from 0), where a word has just ended`, zh: `“${m.word}”后面的空格（第 ${m.t} 个字符，从 0 数起），也就是一个单词刚结束的地方` }
		: { en: `the last character, “${show(charsOf(rt)[m.t])}” (character ${m.t}, counting from 0)`, zh: `最后一个字符“${show(charsOf(rt)[m.t])}”（第 ${m.t} 个字符，从 0 数起）` };
	const cPrev = typeof v.cPrev === 'number' ? v.cPrev : m.t > 0 ? at2(rt, 'lstm#c', m.t - 1, m.U) : 0;
	return { U: m.U, where, f, i, g: gg, o: g('o', 'lstm#o'), c: g('c', 'lstm#c'), h: g('h', 'lstm'), cPrev, zf: v.zf ?? NaN, zi: v.zi ?? NaN, zg: v.zg ?? NaN, zo: v.zo ?? NaN, ig: i * gg, H: H(rt) };
}
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
/** A tiny number prints in calculator notation (5.18e−13): say once what it means; nothing for other numbers. */
function sci(v: number): I18n | string {
	const s = fmt3(v);
	const [m, e] = s.split('e');
	if (!e) return '';
	const p = e.replace('−', '⁻').replace(/\d/g, (d) => SUP[Number(d)]);
	return { en: ` (${s} means ${m} × 10${p}, practically 0)`, zh: `（${s} 就是 ${m} × 10${p}，几乎是 0）` };
}
/** the guess after the whole text: the last row of the softmax */
function guess(rt: RT) {
	const [p, sc] = [tr(rt, 'probs'), tr(rt, 'head')];
	const a = alphabetOf(rt);
	if (!p || !sc) return { top: '…', p: NaN, second: '…', p2: NaN, V: NaN, s1: NaN, s2: NaN, lead: NaN, ratio: NaN };
	const V = p.shape[1];
	const row = Array.from(p.data.subarray(p.data.length - V));
	const order = row.map((_, k) => k).sort((x, y) => row[y] - row[x] || x - y);
	const [s1, s2] = [sc.data[sc.data.length - V + order[0]], sc.data[sc.data.length - V + order[1]]];
	return { top: show(a[order[0]]), p: row[order[0]], second: show(a[order[1]]), p2: row[order[1]], V, s1, s2, lead: s1 - s2, ratio: Math.exp(s1 - s2) };
}
/** one of four texts for a gate value: under 0.1, under 0.5, up to 0.9, above 0.9 */
const grade = (f: number, ...t: [string, string][]): I18n => {
	const [en, zh] = t[f < 0.1 ? 0 : f < 0.5 ? 1 : f <= 0.9 ? 2 : 3];
	return { en, zh };
};
/** a sentence built from numbers (printed as placeholders print them); NaN, so lint reports it, when one is not finite */
const say = (vals: number[], f: (N: (v: number, fmt?: string) => I18n) => I18n): I18n | number => (vals.every(Number.isFinite) ? f((v, fmt) => num(v, fmt) as I18n) : NaN);

const lesson: LessonDef = {
	title: { en: 'A character LSTM, step by step', zh: '一步步看字符级 LSTM' },
	summary: {
		en: 'Follow an LSTM as it reads your text one character at a time: its memory, one unit worked out gate by gate, its guess for the next character, how it writes, and how fast its memory fades.',
		zh: '跟着 LSTM 一次读一个字符地读你的文字：它的记忆、一个单元逐个门的计算、它对下一个字符的猜测、它怎样写字，以及它的记忆消退得有多快。'
	},
	fixtures: [
		{ id: 'quote', label: { en: 'a quote', zh: '一句引语' }, raw: 'Tom said, "Can we go to the park?" Mom smiled and said, "Yes!"' },
		{ id: 'long-word', label: { en: 'a long word', zh: '一个长单词' }, raw: 'She was extraordinarily happy.' },
		{ id: 'no-space', label: { en: 'one word, no space', zh: '只有一个词，没有空格' }, raw: 'Hello' },
		{ id: 'symbols', label: { en: 'characters outside the alphabet', zh: '字母表以外的字符' }, raw: 'Zoë ate 3 crêpes (yum) — “wow”' }
	],
	steps: [
		{
			id: 'chars',
			source: 'fwd',
			at: 'first',
			view: 'kit.tokens',
			provenance: 'live',
			title: { en: 'Your text, one character at a time', zh: '你的文字，一次一个字符' },
			narration: {
				en: 'This model reads text one character at a time: letters, spaces and punctuation alike, so here every token is one character. Each of your {n} characters becomes its id, its place in an alphabet of {fact:vocab} characters (the model’s vocabulary). The id then picks a row of {d} learned numbers, the character’s embedding, and that row is what the next layer reads.',
				zh: '这个模型一次读一个字符：字母、空格和标点都算，所以这里每个词元就是一个字符。你的 {n} 个字符每个都变成它的 id，也就是它在一个有 {fact:vocab} 个字符的字母表（模型的词表）里的位置。这个 id 再取出一行 {d} 个学来的数，叫作这个字符的嵌入，下一层读的就是这一行。',
				vars: (rt) => ({ n: charsOf(rt).length, d: Number(tr(rt, 'emb')?.shape[1] ?? NaN) })
			},
			look: { en: 'Spaces (␣) and punctuation get ids too.', zh: '空格（␣）和标点也有自己的 id。' },
			terms: ['token', 'vocabulary', 'embedding']
		},
		{
			id: 'memory',
			...LSTM,
			view: { key: 'char-rnn/timeline', props: () => ({ mode: 'memory' }) },
			title: { en: 'A memory, updated at every character', zh: '每读一个字符就更新一次的记忆' },
			narration: {
				en: 'An LSTM (long short-term memory) is a recurrent network: it reads the characters in order and carries a memory from each one to the next. The memory is two lists of {H} numbers, one pair for each of its {H} units: the cell state c and the hidden state h. Each column of the picture is h after one more character, so the last column, under “{last}”, is all the model still knows about your {n} characters.',
				zh: 'LSTM（长短期记忆网络）是一种循环网络：它按顺序读字符，并把一份记忆从每个字符带到下一个字符。这份记忆是两组各 {H} 个数，{H} 个单元每个占一对：细胞状态 c 和隐藏状态 h。图中每一列是多读一个字符之后的 h，所以最后一列，也就是“{last}”下面那一列，是模型对你这 {n} 个字符还记得的全部内容。',
				vars: (rt) => ({ H: H(rt), last: show(charsOf(rt).at(-1)), n: charsOf(rt).length })
			},
			look: { en: 'Each row is one unit. The outlined row is unit {U}; the next four steps follow it.', zh: '每一行是一个单元。加框的是单元 {U}；接下来四步都跟着它。', vars: (rt) => ({ U: unitOf(rt) }) },
			terms: ['recurrent-network', 'hidden-state', 'cell-state']
		},
		{
			id: 'forget',
			...GATE,
			title: { en: 'The forget gate: keep or wipe', zh: '遗忘门：保留还是抹掉' },
			narration: {
				en: 'Follow unit {U} at {where}. A gate is a number between 0 and 1: the sigmoid σ squashes a score, made from this character and the previous h with learned weights, into that range. Here the forget gate is f = σ({zf}) = {f}{sciF}: the unit keeps {f|pct} of its old cell value, {cPrev}, so {verdict}.',
				zh: '跟着单元 {U} 看{where}。门是一个 0 到 1 之间的数：sigmoid 函数 σ 把一个分数压进这个范围，这个分数由当前字符和上一步的 h 乘以学来的权重得到。这里遗忘门 f = σ({zf}) = {f}{sciF}：这个单元保留旧细胞值 {cPrev} 的 {f|pct}，所以{verdict}。',
				vars: (rt, at) => {
					const v = cell(rt, at);
					return { U: v.U, where: v.where, zf: v.zf, f: v.f, sciF: sci(v.f), cPrev: v.cPrev, verdict: grade(v.f, ['it all but wipes its memory', '它几乎把记忆抹掉了'], ['it drops most of its memory', '它丢掉了大部分记忆'], ['it keeps part of it', '它保留了一部分'], ['it keeps almost all of it', '它几乎全部保留']) };
				}
			},
			look: {
				en: 'The f bar, then the circle on the solid line below: it marks this character, where unit {U}’s cell value goes from {cPrev} to {c}.',
				zh: '先看 f 条有多长，再看下面实线上的圆点：它标出这个字符，单元 {U} 的细胞值在这里从 {cPrev} 变成 {c}。',
				vars: (rt, at) => {
					const v = cell(rt, at);
					return { U: v.U, cPrev: v.cPrev, c: v.c };
				}
			},
			terms: ['gate', 'activation-function']
		},
		{
			id: 'write',
			...GATE,
			title: { en: 'Input gate and candidate: what to write', zh: '输入门和候选值：写入什么' },
			narration: {
				en: 'Two more numbers decide what gets written. The candidate g = tanh({zg}) = {g|n3} is a new value between −1 and 1 (tanh squashes any score into that range). The input gate i = σ({zi}) = {i|n3} says how much of it to let in, so unit {U} writes i × g = {ig|n3}.{rounded}',
				zh: '还有两个数决定写入什么。候选值 g = tanh({zg}) = {g|n3} 是一个介于 −1 和 1 之间的新值（tanh 把任何分数压进这个范围）。输入门 i = σ({zi}) = {i|n3} 决定放进来多少，所以单元 {U} 写入 i × g = {ig|n3}。{rounded}',
				vars: (rt, at) => {
					const v = cell(rt, at);
					const rounded = [v.g, v.i].some((x) => fmt3(x) === '1.00') ? { en: ' A 1.00 here is rounded: σ and tanh come very close to 1 but never reach it.', zh: '这里的 1.00 是四舍五入的结果：σ 和 tanh 可以非常接近 1，但永远到不了 1。' } : '';
					return { zg: v.zg, g: v.g, zi: v.zi, i: v.i, U: v.U, ig: v.ig, rounded };
				}
			},
			look: { en: 'The bars for i (0 to 1) and g (−1 to 1): a full i bar lets all of g in.', zh: 'i（0 到 1）和 g（−1 到 1）的条形：i 条满了，g 就全部写进去。' },
			terms: ['gate']
		},
		{
			id: 'cell',
			...GATE,
			title: { en: 'The new memory: c = f·c + i·g', zh: '新的记忆：c = f·c + i·g' },
			narration: {
				en: 'Now the new cell value: keep a share of the old one, then add what was written. c = f × c_old + i × g = {f} × {cPrev} + {i|n3} × {g|n3} = {c}. {story}',
				zh: '现在算新的细胞值：先保留旧值的一部分，再加上写入的部分。c = f × c_旧 + i × g = {f} × {cPrev} + {i|n3} × {g|n3} = {c}。{story}',
				vars: (rt, at) => {
					const v = cell(rt, at);
					const story = grade(v.f, ['The old value is almost gone, and the memory starts afresh.', '旧值几乎没了，记忆重新开始。'], ['Most of the old value is dropped.', '旧值的大部分被丢掉了。'], ['Part of the old value is carried forward.', '旧值的一部分被带了下去。'], ['The old value is carried forward almost whole.', '旧值几乎完整地带了下去。']);
					return { f: v.f, cPrev: v.cPrev, i: v.i, g: v.g, c: v.c, story };
				}
			},
			look: { en: 'The boxes from c(t−1) to c(t) show the same sum.', zh: '从 c(t−1) 到 c(t) 的几个方框是同一个算式。' },
			terms: ['cell-state']
		},
		{
			id: 'output',
			...GATE,
			title: { en: 'The output gate: what the memory shows', zh: '输出门：记忆露出多少' },
			narration: {
				en: 'The cell value is not passed on raw. The output gate o = σ({zo}) = {o} decides how much of tanh(c) to show, and that is the hidden state: h = o × tanh(c) = {o} × tanh({c}) = {h}. {verdict} The same four steps run for all {H} units at every character; h goes on to the next character and to the guess.',
				zh: '细胞值不会原样传出去。输出门 o = σ({zo}) = {o} 决定露出多少 tanh(c)，这就是隐藏状态：h = o × tanh(c) = {o} × tanh({c}) = {h}。{verdict}每个字符、全部 {H} 个单元都做同样的四步；h 一边传给下一个字符，一边用来猜下一个字符。',
				vars: (rt, at) => {
					const v = cell(rt, at);
					const s = sci(v.o);
					const verdict = v.o < 0.01 ? { en: `The gate is shut${typeof s === 'string' ? s : s.en}, so the unit keeps its value to itself.`, zh: `这个门是关着的${typeof s === 'string' ? s : s.zh}，所以这个单元把它的值藏在自己里面。` } : grade(v.o, ['The gate is mostly shut.', '这个门大半关着。'], ['The gate is mostly shut.', '这个门大半关着。'], ['The gate is partly open.', '这个门半开着。'], ['The gate is almost fully open.', '这个门几乎全开。']);
					return { zo: v.zo, o: v.o, c: v.c, h: v.h, H: v.H, verdict };
				}
			},
			look: { en: 'The last box, h(t): of this unit, only h reaches the next character and the guess.', zh: '最后一个方框 h(t)：这个单元只有 h 会传给下一个字符和猜测。' },
			terms: ['gate', 'hidden-state']
		},
		{
			id: 'next',
			source: 'fwd',
			at: { layer: 'probs' },
			view: { key: 'char-rnn/timeline', props: () => ({ mode: 'next' }) },
			provenance: 'live',
			title: { en: 'Guessing the next character', zh: '猜下一个字符' },
			narration: {
				en: 'From the last hidden state, a linear layer gives each of the {V} characters a score (a weighted sum of the {H} numbers of h), and softmax turns the scores into chances that add up to 1. After “{tail}”, the top guess is “{top}” with {p|pct}; the runner-up, “{second}”, has {p2|pct}. Softmax rewards a lead: “{top}” scores {s1|n4} and “{second}” {s2|n4}, a lead of {lead}, so “{top}” is e to the power {lead} ≈ {ratio} times as likely.',
				zh: '线性层根据最后的隐藏状态给 {V} 个字符各打一个分（h 里 {H} 个数的加权和），softmax 再把这些分数变成加起来等于 1 的概率。在“{tail}”之后，第一猜测是“{top}”，概率 {p|pct}；第二名“{second}”是 {p2|pct}。softmax 看的是分差：“{top}”得 {s1|n4} 分，“{second}”得 {s2|n4} 分，领先 {lead}，所以“{top}”的可能性是“{second}”的 e 的 {lead} 次方 ≈ {ratio} 倍。',
				vars: (rt) => ({ ...guess(rt), H: H(rt), tail: charsOf(rt).slice(-14).join('').replaceAll('\n', '↵') })
			},
			look: { en: 'The longest bar. Pick a character above the bars to see what the model expected right after it.', zh: '最长的那条。点条形上方的某个字符，看看模型预料它后面紧接着是什么。' },
			terms: ['logit', 'softmax', 'next-token-prediction']
		},
		{
			id: 'unroll',
			source: 'unroll',
			at: 'last',
			play: { from: { phase: 'predict', step: 0 }, to: 'last', phases: ['predict'], interval: 260, unit: { en: 'character', zh: '字符' } },
			view: { key: 'char-rnn/timeline', props: () => ({ mode: 'unroll' }) },
			provenance: 'live',
			title: { en: 'Unrolled: one copy of the LSTM per character', zh: '展开：每个字符一份同样的 LSTM' },
			narration: {
				en: 'The model has just one LSTM layer: the same {H} units with the same weights read every character. Drawn once per character it looks like a chain, the network “unrolled” in time: each copy reads one character plus the memory the copy before left behind. Watch the memory fill up column by column (Play under the picture runs it again). {score}',
				zh: '模型只有一层 LSTM：每个字符都由同样的 {H} 个单元、用同样的权重来读。把它每个字符画一次，就像一条链，也就是网络在时间上“展开”了：每一份读一个字符，再加上前一份留下的记忆。看记忆一列一列地填满（图下面的“播放”可以再放一遍）。{score}',
				vars: (rt, at) => {
					const m = at.tape && at.tape.length ? (at.tape.frame(at.tape.length - 1).metrics ?? {}) : {};
					const [n, hits] = [Number(m.guesses ?? NaN), Number(m.hits ?? NaN)];
					return {
						H: H(rt),
						score:
							n > 0 || !Number.isFinite(n)
								? say([hits, n], (N) => ({ en: `On your text its top guess for the next character was right ${N(hits).en} times out of ${N(n).en}.`, zh: `在你的文字上，它对下一个字符的第一猜测对了 ${N(hits).zh} 次，共猜了 ${N(n).zh} 次。` }))
								: { en: 'With one character there is no next character to check its guess against yet.', zh: '只有一个字符时，还没有下一个字符可以检验它的猜测。' }
					};
				}
			},
			look: { en: 'The bottom row: under each character, the guess the model made one character earlier, green when it was right.', zh: '最下面一行：每个字符下面是模型前一步对它的猜测，猜对的是绿色。' },
			terms: ['recurrent-network']
		},
		{
			id: 'counter',
			...LSTM,
			view: { key: 'char-rnn/timeline', props: () => ({ mode: 'unit' }) },
			title: { en: 'A unit that counts letters', zh: '一个数字母的单元' },
			narration: {
				en: 'Most units mix many signals, but unit {U} is easier to read. Inside a word its cell value usually grows, by about {fact:counterRise|n2} per letter, and at a space it is mostly wiped (typical forget gates: {fact:counterResetF|n2} at a space, {fact:counterLetterF|n2} inside a word). So it roughly counts the letters of the current word. {yours}',
				zh: '大多数单元混着很多信号，但单元 {U} 比较好懂。在单词内部，它的细胞值通常会增加，每个字母大约加 {fact:counterRise|n2}；遇到空格时，它大多会被抹掉（典型的遗忘门：空格处 {fact:counterResetF|n2}，单词内部 {fact:counterLetterF|n2}）。所以它大致在数当前单词有几个字母。{yours}',
				vars: (rt) => {
					const U = unitOf(rt);
					const w = longest(charsOf(rt));
					const c = w ? at2(rt, 'lstm#c', w.to, U) : NaN;
					return {
						U,
						yours: w
							? say([c], (N) => ({ en: `At the end of your longest word, “${w.word}” (${(plural(w.word.length, { one: '{n} letter', many: '{n} letters' }) as I18n).en}), it reaches ${N(c).en}.`, zh: `在你最长的单词“${w.word}”（${w.word.length} 个字母）的末尾，它达到 ${N(c).zh}。` }))
							: { en: 'Your text has no word yet, so there is nothing to count.', zh: '你的文字里还没有单词，所以没什么可数。' }
					};
				}
			},
			look: {
				en: 'The bars climb letter by letter and drop at each space. Below: where it was highest in {fact:evalChars} characters of new stories (precomputed; its value follows the letter count with correlation r = {fact:counterR|n2}, where 1 would be a perfect match and 0 no link at all).',
				zh: '条形一个字母一个字母地升高，遇到空格就掉下来。下面：它在 {fact:evalChars} 个新故事字符里最高的地方（预先算好；它的值和字母数的相关系数 r = {fact:counterR|n2}，1 表示完全一致，0 表示毫无关系）。'
			},
			task: {
				prompt: { en: 'Type a word of at least 10 letters (for example “grandmother”) and watch how high the bars climb.', zh: '输入一个至少 10 个字母的单词（比如“grandmother”），看条形能升多高。' },
				done: (rt) => (longest(charsOf(rt))?.word.length ?? 0) >= 10,
				waiting: { en: 'Not yet: your text needs a word of 10 letters or more.', zh: '还没有：你的文字里需要一个 10 个字母以上的单词。' }
			},
			linger: 12000,
			terms: ['hidden-state']
		},
		{
			id: 'sample',
			source: 'sample',
			at: 'last',
			compare: { a: 'sample', b: 'greedy', label: [{ en: 'Drawing by chance', zh: '按概率抽取' }, { en: 'Always the top character', zh: '总选第一名' }] },
			play: { from: { phase: 'sample', step: 0 }, to: 'last', phases: ['sample'], interval: 220, unit: { en: 'character', zh: '字符' } },
			view: { key: 'char-rnn/timeline', props: () => ({ mode: 'sample' }) },
			provenance: 'live',
			title: { en: 'Writing: draw a character, read it, repeat', zh: '写字：抽一个字符，读进去，再重复' },
			predict: {
				id: 'char-rnn-greedy',
				question: { en: 'The second writer always takes the character with the highest chance. Run it twice on the same text: what do you get?', zh: '第二种写法总是选概率最高的字符。在同一段文字上运行两次，会得到什么？' },
				options: [
					{ en: 'The same text both times', zh: '两次是同样的文字' },
					{ en: 'A different text each time', zh: '每次都不一样' },
					{ en: 'Nothing: it needs a random number to start', zh: '什么也没有：它需要一个随机数才能开始' }
				],
				answer: 0,
				explain: {
					en: 'Without a random draw the same memory always leads to the same choice, so the top-character text is fixed by your text alone. It is also dull: on each of the four example texts it soon writes “said, "I will help”. Drawing gives a different text for each random seed, at the price of an odd character now and then.',
					zh: '没有随机抽取，同样的记忆总会导致同样的选择，所以总选第一名写出的文字完全由你的文字决定。它也很单调：在四段示例文字上，它很快都会写出“said, "I will help”。按概率抽取时，换一个随机种子就会写出不同的文字，代价是时不时冒出一个奇怪的字符。'
				}
			},
			narration: {
				en: 'To write, the model turns its scores into chances and draws a character: lay the chances end to end from 0 to 1, pick a random number u, and take the character where u lands. It then reads that character and repeats. The first draw, u = {u|n3}, picked “{ch}” (chance {p|pct}). Drawing sometimes takes an unlikely character: the least likely of its {n} had a chance of only 1 in {odds|int}.',
				zh: '写字时，模型把分数变成概率，再抽一个字符：把这些概率从 0 到 1 首尾相接排开，取一个随机数 u，u 落在哪个字符上就选哪个。然后它读进这个字符，再重复。第一次抽到 u = {u|n3}，选中了“{ch}”（概率 {p|pct}）。按概率抽取有时会选中不太可能的字符：它写的 {n} 个字符里，最不可能的那个概率只有 {odds|int} 分之一。',
				vars: (rt) => {
					const w = written(rt, 'sample', 0);
					const r = riskiest(rt, 'sample');
					return { u: w?.u ?? NaN, ch: show(alphabetOf(rt)[w?.token ?? -1]), p: w?.p ?? NaN, n: r.n, odds: 1 / r.p };
				}
			},
			look: { en: 'The strip: u is the line, the highlighted piece is the character it picked. Drawing uses the chances as they are (temperature 1); always taking the top character is temperature 0. Play under the texts runs both writers on.', zh: '那条带子：u 是竖线，高亮的那段就是选中的字符。按概率抽取直接用这些概率（温度 1）；总选第一名就是温度 0。文字下面的“播放”让两种写法接着写下去。' },
			terms: ['sampling', 'temperature', 'autoregressive']
		},
		{
			id: 'fade',
			...LSTM,
			view: { key: 'char-rnn/timeline', props: () => ({ mode: 'fade' }) },
			kindOf: 'limits',
			title: { en: 'Limits: the memory fades', zh: '局限：记忆会消退' },
			predict: {
				id: 'char-rnn-fade',
				question: { en: 'Suppose a unit’s forget gate is 0.9 at every character. How much of what it holds now is still there 20 characters later?', zh: '假设某个单元每个字符的遗忘门都是 0.9。它现在存的东西，过 20 个字符后还剩多少？' },
				options: [
					{ en: 'About 90%', zh: '大约 90%' },
					{ en: 'About 12%', zh: '大约 12%' },
					{ en: 'Exactly 0', zh: '正好是 0' }
				],
				answer: 1,
				explain: {
					en: 'Each character multiplies the old value by 0.9 again: 0.9 × 0.9 × … twenty times is 0.9²⁰ ≈ 0.12. So a gate of 0.9 loses most of a memory within a few dozen characters; only gates very close to 1 (0.99 keeps 0.99²⁰ ≈ 0.82) hold it much longer.',
					zh: '每个字符都再乘一次 0.9：0.9 × 0.9 × …… 乘二十次是 0.9²⁰ ≈ 0.12。所以门为 0.9 时，几十个字符之内就会丢掉大部分记忆；只有非常接近 1 的门才能记得久得多（0.99 能保留 0.99²⁰ ≈ 0.82）。'
				}
			},
			narration: {
				en: 'At every character a memory is multiplied by a forget gate, so it shrinks step by step. {fadeLine} This model learned to forget fast: on {fact:evalChars} characters of new stories, no unit follows whether a quote is open (best correlation {fact:quoteR|n2}). A gate near 1, like unit {U}’s inside a word ({fact:counterLetterF|n2}), would keep a memory far longer.',
				zh: '每读一个字符，记忆都要乘一次遗忘门，所以它一步一步地缩小。{fadeLine}这个模型学会了很快地忘记：在 {fact:evalChars} 个新故事字符上，没有一个单元能跟踪引号是否还开着（最好的相关系数是 {fact:quoteR|n2}）。接近 1 的门，比如单元 {U} 在单词内部的遗忘门（{fact:counterLetterF|n2}），能把记忆保留得久得多。',
				vars: (rt) => {
					const f = fadeAt(rt);
					if (!f) return { fadeLine: NaN, U: unitOf(rt) };
					const d = Math.min(20, f.D);
					return {
						U: unitOf(rt),
						fadeLine:
							d > 0
								? say([f.mean, f.keep[d], f.patientCurve[d]], (N) => ({
										en: `On your text the forget gates average ${N(f.mean, 'n2').en}; ${d} characters later the typical (median) unit keeps ${N(f.keep[d], 'pct').en} of a memory, and the most patient one, unit ${f.patient}, keeps ${N(f.patientCurve[d], 'pct').en}.`,
										zh: `在你的文字上，遗忘门的平均值是 ${N(f.mean, 'n2').zh}；过 ${d} 个字符后，典型的（中位数）单元只剩下原来记忆的 ${N(f.keep[d], 'pct').zh}，最能记的单元 ${f.patient} 保留了 ${N(f.patientCurve[d], 'pct').zh}。`
									}))
								: { en: 'With one character there is nothing to carry yet: type more to see it fade.', zh: '只有一个字符时还没有什么要带下去：多输入一些，就能看到它消退。' }
					};
				}
			},
			look: { en: 'Compare the solid line with the thin grey line of a gate that is always {ref}.', zh: '把实线和那条门一直是 {ref} 的细灰线比一比。', vars: () => ({ ref: String(REF_GATE) }) },
			terms: ['gate']
		}
	]
};

export default lesson;
