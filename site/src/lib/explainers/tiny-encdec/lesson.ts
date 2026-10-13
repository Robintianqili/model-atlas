// tiny-encdec: the guided lesson (11 steps). Every number is a {placeholder}: structure from the spec (dims()), live
// values from the trace of the learner's date (source 'fwd') or the 'decode' process, measured ones from facts.json.
// Claims that are not computed live are tested in claims.test.ts.
//   1 characters + positions → 2 encoder self-attention (both ways) → 3 the memory → 4 the decoder's causal
//   self-attention → 5 cross-attention (predict) → 6 one score → 7 softmax and the mix → 8 the next symbol →
//   9 greedy decoding, step by step → 10 your date in another format (task) → 11 formats it never saw (limits)
import type { I18n, LessonDef } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmtSetting, minus } from '../../atlas/core/format.ts';
import { FORMATS, readDate, TGT } from './data.ts';
import { argmax } from './tf.ts';
import { crossAt, head, look, q, rightward, ROW, sharpest, sym, tr, verdict, type RT } from './look.ts';
import { textOf } from './adapters/date-text/adapter.ts';
import { TRIES } from './views/cross/meta.ts';

/** Structural numbers from the spec (never typed into the text). */
export function dims(rt: RT) {
	const L = (id: string) => rt.spec?.layers.find((l) => l.id === id);
	const steps = Number(L('greedy')?.params?.steps ?? NaN);
	return { D: Number(L('src.emb')?.outShape[1] ?? NaN), H: Number(L('enc.scores')?.params?.nHead ?? NaN), dH: Number(L('enc.scores')?.params?.dHead ?? NaN), F: Number(L('enc.up')?.outShape[1] ?? NaN), steps, chars: steps - 1, nsym: TGT.length, nf: FORMATS.length };
}
/** a number with 3 decimals (sums printed this way add up to the printed total within 0.001) */
const fx = (v: number | undefined) => (v === undefined || !Number.isFinite(v) ? NaN : minus(v.toFixed(3)));
/** the decoder's input symbols, one per position (▶ then what it wrote), for the views' row names */
const decTokens = (rt: RT) => look(rt).yin.map(sym);
/** the training format the learner's text is written in, by name */
function fmtName(rt: RT): I18n {
	const f = FORMATS.find((x) => x.id === look(rt).fmt);
	return f ? { en: `in the format “${f.label.en}”`, zh: `用的是“${f.label.zh}”格式` } : { en: 'in none of the training formats', zh: '不属于任何一种训练格式' };
}
/** Decoder self-attention at position ROW: the sharper head, where it looks and how much. */
const selfAt = (rt: RT) => sharpest(rt, 'dec.sattn', ROW);
/** The mix of step ROW, head h: the feature f of h's part with the largest |value|, the mix there and the value of
 *  the top character there. */
function mixAt(rt: RT) {
	const { h, j, w } = crossAt(rt, ROW);
	const M = tr(rt, 'dec.cmix');
	const Vv = tr(rt, 'dec.cv');
	const { D, dH } = dims(rt);
	if (!M || !Vv || h < 0) return { h, j, w, f: -1, m: NaN, v: NaN };
	const part = M.data.subarray(ROW * D + h * dH, ROW * D + (h + 1) * dH);
	const f = argmax(Array.from(part, Math.abs));
	return { h, j, w, f, m: part[f], v: Vv.data[j * D + h * dH + f] };
}

const lesson: LessonDef = {
	title: { en: 'A tiny encoder–decoder Transformer: convert a date with cross-attention', zh: '小型编码器–解码器 Transformer：用交叉注意力转换日期' },
	summary: {
		en: 'An encoder reads your whole date at once with self-attention; a decoder writes it as 2021-03-05, one symbol at a time, reading what it already wrote through a causal mask and your characters through cross-attention. Work one cross-attention step out with real numbers, watch the matrix fill in, then compare it with the GRU on formats it never saw.',
		zh: '编码器用自注意力一次读完你的整个日期；解码器把它写成 2021-03-05 的样子，一次写一个符号：通过因果掩码读自己已经写下的东西，通过交叉注意力读你的字符。用真实的数算出一步交叉注意力，看矩阵一行行填满，再在它没见过的格式上和 GRU 比一比。'
	},
	steps: [
		{
			id: 'chars', provenance: 'live',
			view: 'zoom.elementwise', select: () => ({ layer: 'enc.x0', unit: [0, 0] }),
			title: { en: 'Characters and positions become vectors', zh: '字符和位置变成向量' },
			narration: {
				en: 'The model reads your date as {n}. Each character’s id picks a row of {D} learned numbers, its vector (an embedding). Attention by itself ignores order, so each position adds a learned position vector too. For {c} at position 0 (counting from 0), number 0 is {e} + {p} = {x}.',
				zh: '模型把你的日期读成 {n}。每个字符的 id 取出一行 {D} 个学来的数，也就是它的向量（嵌入）。注意力本身不管顺序，所以每个位置还要加上一个学来的位置向量。位置 0（从 0 数起）上的{c}，编号 0 的数是 {e} + {p} = {x}。',
				vars: (rt) => {
					const L = look(rt);
					const [e, p] = [tr(rt, 'src.emb')?.data[0], tr(rt, 'src.pos')?.data[0]];
					const r = (v: number) => Math.round(v * 1000) / 1000;
					return { ...dims(rt), n: plural(L.T, { one: '{n} character', many: '{n} characters', zh: '{n} 个字符' }), c: q(L.chars[0] ?? ''), e: fx(e), p: fx(p), x: e === undefined || p === undefined ? NaN : fx(r(e) + r(p)) };
				}
			},
			look: { en: 'The stage adds one number of the character vector to one number of the position vector (counting from 0).', zh: '舞台上把字符向量的一个数和位置向量的一个数相加（从 0 数起）。' }, terms: ['embedding', 'positional-encoding']
		},
		{
			id: 'self', provenance: 'live',
			view: { key: 'kit.attention', props: (rt) => { const R = rightward(rt); return { layer: 'enc.attn', head: Math.max(0, R.h), query: Math.max(0, R.i) }; } },
			select: (rt) => { const R = rightward(rt); return { layer: 'enc.attn', unit: [Math.max(0, R.h), Math.max(0, R.i), Math.max(0, R.j)] }; },
			title: { en: 'The encoder: every character reads every other', zh: '编码器：每个字符读所有其他字符' },
			narration: {
				en: 'In self-attention each character’s vector makes a query, scores it against the keys of all characters and reads a weighted mix of their values: in both directions, all at once, where tinyseq2seq’s GRU read strictly left to right. Here {H} heads do this side by side. On your date {r|pct} of the weight goes to later characters; {c} at position {i} puts {w|pct} of head {h}’s weight on {k} at position {j} (counting from 0).',
				zh: '在自注意力里，每个字符的向量给出一个查询，和所有字符的键打分，再按权重读它们的值的混合：两个方向都读，而且同时进行；tinyseq2seq 的 GRU 则严格从左往右读。这里有 {H} 个头并排做这件事。在你的日期上，{r|pct} 的权重流向后面的字符；位置 {i} 上的{c}把头 {h} 的 {w|pct} 权重放在位置 {j} 上的{k}（从 0 数起）。',
				vars: (rt) => {
					const R = rightward(rt);
					const ch = look(rt).chars;
					return { ...dims(rt), r: R.share, i: R.i, j: R.j, h: R.h, w: R.w, c: q(ch[R.i] ?? ''), k: q(ch[R.j] ?? '') };
				}
			},
			look: { en: 'Rows ask, columns answer. The outlined row is the character in the narration; it reads a later character, which a left-to-right reader never could.', zh: '行在提问，列在回答。框出的那一行就是讲解里的那个字符；它读的是后面的字符，这是从左往右读的模型做不到的。' }, terms: ['self-attention', 'query-key-value', 'attention-head']
		},
		{
			id: 'memory', provenance: 'live',
			view: 'layer.tensor', select: () => ({ layer: 'mem' }),
			title: { en: 'The encoder’s output: the memory', zh: '编码器的输出：记忆' },
			narration: {
				en: 'After attention, a feed-forward layer works on each character’s vector on its own ({D} numbers, widened to {F} with ReLU, back to {D}), and a LayerNorm finishes the encoder. Its output, the memory, is one vector per character: {T} × {D} = {TD} numbers. Nothing is squeezed into one last state, and the decoder can read every vector at every step.',
				zh: '注意力之后，一个前馈层分别处理每个字符的向量（{D} 个数，经 ReLU 放大到 {F} 个，再缩回 {D} 个），最后一个层归一化结束编码器。它的输出叫记忆，每个字符一个向量：{T} × {D} = {TD} 个数。没有什么被挤进最后一个状态，解码器每一步都能读到每一个向量。',
				vars: (rt) => { const d = dims(rt); const T = look(rt).T; return { ...d, T, TD: T * d.D }; }
			},
			look: { en: 'One row per character of your date; colour shows each number’s sign and size.', zh: '你日期里的每个字符一行；颜色表示每个数的正负和大小。' }, terms: ['layer-norm', 'residual-connection']
		},
		{
			id: 'dself', provenance: 'live',
			view: { key: 'kit.attention', props: (rt) => ({ layer: 'dec.sattn', tokens: decTokens(rt), head: Math.max(0, selfAt(rt).h), query: ROW }) },
			select: (rt) => { const S = selfAt(rt); return { layer: 'dec.sattn', unit: [Math.max(0, S.h), ROW, Math.max(0, S.j)] }; },
			title: { en: 'The decoder reads what it wrote, never ahead', zh: '解码器只读已经写下的，从不偷看后面' },
			narration: {
				en: 'The decoder writes the answer one symbol at a time: {chars} characters, then an end symbol ■. Its input is a start symbol ▶ and the symbols written so far, each with a position vector. Its self-attention is causal: position t reads only positions 0 to t, because later symbols do not exist yet. Position {t} (reading {s}) puts {w|pct} of head {h}’s weight on position {j} ({s2}).',
				zh: '解码器一次写一个符号：先写 {chars} 个字符，再写一个结束符号 ■。它的输入是起始符号 ▶ 加上已经写下的符号，每个都带一个位置向量。它的自注意力是因果的：位置 t 只读位置 0 到 t，因为后面的符号还不存在。位置 {t}（读入{s}）把头 {h} 的 {w|pct} 权重放在位置 {j}（{s2}）上。',
				vars: (rt) => { const S = selfAt(rt); const y = decTokens(rt); return { ...dims(rt), t: ROW, s: q(y[ROW] ?? ''), h: S.h, j: S.j, w: S.w, s2: q(y[S.j] ?? '') }; }
			},
			look: { en: 'The hatched triangle is the causal mask: weights there are exactly 0.', zh: '打斜线的三角形就是因果掩码：那里的权重正好是 0。' }, terms: ['causal-mask', 'autoregressive']
		},
		{
			id: 'cross', provenance: 'live',
			view: { key: 'tiny-encdec/cross', props: () => ({ row: ROW, hide: true }) },
			title: { en: 'Cross-attention: the decoder reads your date', zh: '交叉注意力：解码器读你的日期' },
			narration: {
				en: 'Cross-attention joins the two halves. The decoder’s vector at each position makes a query; the memory gives a key and a value for every character of your date. Scores, softmax and the mix work as in self-attention, only across the two sequences. Decoding step {t} (counting from 0) writes the year’s last digit.',
				zh: '交叉注意力把两半连起来。解码器每个位置的向量给出一个查询；记忆为你日期里的每个字符提供一个键和一个值。打分、softmax 和混合都和自注意力一样，只是跨在两个序列之间。解码步 {t}（从 0 数起）写的是年份的最后一位数字。',
				vars: () => ({ t: ROW })
			},
			predict: {
				id: 'tiny-encdec-cross',
				question: { en: 'Which characters of your date will that step’s cross-attention read most?', zh: '这一步的交叉注意力会最多地读你日期里的哪些字符？' },
				options: [{ en: 'The letters of the month', zh: '月份的字母' }, { en: 'The year’s last digit itself', zh: '年份最后一位数字本身' }, { en: 'The year’s first digit', zh: '年份的第一位数字' }, { en: 'All characters about the same', zh: '所有字符都差不多' }],
				answer: 1,
				explain: { en: 'The digit itself: cross-attention reads the input directly, so for the year it simply copies. On every test date, the row of the year’s last digit looked hardest at that very digit. Other rows are less simple: the month’s row may read a space next to the month name, whose vector the encoder’s self-attention can fill with the month.', zh: '就是这位数字本身：交叉注意力直接读输入，所以年份只要照抄。在每一个测试日期上，年份最后一位那一行看得最多的都正是这位数字。别的行没这么简单：月份那一行可能读月份名旁边的空格，编码器的自注意力可以把月份的信息混进那个空格的向量。' }
			},
			look: { en: 'Rows: the symbol each step wrote; columns: your characters. The outlined row is decoding step {t}.', zh: '行：每一步写下的符号；列：你的字符。框出的那一行是解码步 {t}。', vars: () => ({ t: ROW }) }, terms: ['cross-attention', 'encoder-decoder']
		},
		{
			id: 'score', provenance: 'live',
			view: 'zoom.weighted-sum', select: (rt) => { const C = crossAt(rt, ROW); return { layer: 'dec.cscores', unit: [Math.max(0, C.h), ROW, Math.max(0, C.j)] }; },
			title: { en: 'One cross-attention score, worked out', zh: '算出一个交叉注意力分数' },
			narration: {
				en: 'Head {h}’s query at decoding step {t} and the key of {c} at position {j} (counting from 0) each have {dH} numbers. Multiply them pair by pair and add: {s|n4}. Times 1/√{dH} = {sc}, the score is {y}. The other characters score lower, down to {lo}.',
				zh: '在解码步 {t}，头 {h} 的查询和位置 {j}（从 0 数起）上{c}的键各有 {dH} 个数。逐对相乘再相加：{s|n4}。乘以 1/√{dH} = {sc}，分数就是 {y}。其他字符的分数都更低，最低到 {lo}。',
				vars: (rt) => {
					const C = crossAt(rt, ROW);
					const S = head(tr(rt, 'dec.cscores'), C.h, ROW);
					const d = dims(rt);
					return { ...d, t: ROW, h: C.h, j: C.j, c: q(look(rt).chars[C.j] ?? ''), s: S ? S[C.j] * Math.sqrt(d.dH) : NaN, sc: fmtSetting(1 / Math.sqrt(d.dH)), y: S ? S[C.j] : NaN, lo: S ? Math.min(...S) : NaN };
				}
			},
			look: { en: 'Each row of the table is one product of a query number and a key number; their sum is the score before scaling.', zh: '表里每一行是查询的一个数乘以键的一个数；它们的和就是缩放前的分数。' }, terms: ['query-key-value', 'dot-product']
		},
		{
			id: 'mix', provenance: 'live',
			view: 'zoom.weighted-sum', select: (rt) => { const M = mixAt(rt); const { dH } = dims(rt); return { layer: 'dec.cmix', unit: [ROW, Math.max(0, M.h) * dH + Math.max(0, M.f)] }; },
			title: { en: 'Softmax, then a weighted mix of values', zh: '先 softmax，再按权重混合值' },
			narration: {
				en: 'Softmax turns head {h}’s {T} scores into weights that add up to 1: {c} gets {w|pct}. The weights then mix the values, one column at a time: number {f} of the mix (counting from 0; one of head {h}’s columns) is the sum of weight × value over all {T} characters, {m}. {near}',
				zh: 'Softmax 把头 {h} 的 {T} 个分数变成加起来等于 1 的权重：{c}得到 {w|pct}。然后用这些权重混合值，一列一列地算：混合结果编号 {f} 的数（从 0 数起；属于头 {h} 的列）是所有 {T} 个字符的“权重 × 值”之和，等于 {m}。{near}',
				vars: (rt) => {
					const M = mixAt(rt);
					const c = q(look(rt).chars[M.j] ?? '');
					const near = pick(M.w > 0.8, { en: `With so much weight on ${c}, that is close to ${c}’s own value there, ${fx(M.v)}.`, zh: `${c}占了这么多权重，所以它很接近${c}自己在这一列的值 ${fx(M.v)}。` }, { en: `The weight is shared, so it mixes several characters’ values; ${c}’s own is ${fx(M.v)}.`, zh: `权重是分摊的，所以它混合了好几个字符的值；${c}自己的值是 ${fx(M.v)}。` });
					return { T: look(rt).T, h: M.h, c, w: M.w, f: M.h * dims(rt).dH + M.f, m: fx(M.m), near };
				}
			},
			look: { en: 'Each row of the table is one character: its weight times its value; the products add up to the mix. {c} is key {j}: scroll the table to its row.', zh: '表里每一行是一个字符：它的权重乘以它的值；这些乘积加起来就是混合结果。{c}是键 {j}：把表格滚动到它那一行。', vars: (rt) => { const M = mixAt(rt); return { c: q(look(rt).chars[M.j] ?? ''), j: M.j }; } }, terms: ['softmax', 'context-vector']
		},
		{
			id: 'next', provenance: 'live',
			view: { key: 'kit.probbars', props: (rt) => ({ layer: 'probs', row: ROW, tokens: decTokens(rt), vocab: TGT.map((_, i) => sym(i)), chosen: look(rt).ids[ROW] ?? null }) },
			title: { en: 'Write the next symbol', zh: '写下一个符号' },
			narration: {
				en: 'Both heads’ mixes are joined, projected back to {D} numbers and added to the decoder’s vector. A feed-forward layer, a last LayerNorm and the output layer turn it into {nsym} scores, and softmax into chances: at decoding step {t}, {y} gets {p|pct}, the runner-up {r} {pr|pct}. Greedy decoding writes the most likely symbol and feeds it back as the next position’s input.',
				zh: '两个头的混合结果拼起来，投影回 {D} 个数，再加到解码器的向量上。一个前馈层、最后一个层归一化和输出层把它变成 {nsym} 个分数，softmax 再变成概率：在解码步 {t}，{y}得到 {p|pct}，第二名{r}得到 {pr|pct}。贪心解码写下概率最大的符号，并把它作为下一个位置的输入送回去。',
				vars: (rt) => {
					const P = tr(rt, 'probs');
					const V = P?.shape[1] ?? 0;
					const row = P ? Array.from(P.data.subarray(ROW * V, (ROW + 1) * V)) : [];
					const order = row.map((_, i) => i).sort((a, b) => row[b] - row[a] || a - b);
					return { ...dims(rt), t: ROW, y: q(sym(order[0] ?? -1)), p: row[order[0]] ?? NaN, r: q(sym(order[1] ?? -1)), pr: row[order[1]] ?? NaN };
				}
			},
			look: { en: 'The bar marked as chosen is the symbol written at this step.', zh: '标为“选中”的那根条，就是这一步写下的符号。' }, terms: ['greedy-decoding', 'logit']
		},
		{
			id: 'decode', provenance: 'live',
			source: 'decode', at: 'last', play: 'view',
			view: { key: 'tiny-encdec/cross', props: () => ({ play: true }) },
			title: { en: 'Repeat: greedy decoding, step by step', zh: '重复：一步一步地贪心解码' },
			narration: {
				en: 'The decoder repeats lesson steps 4 to 8 for every symbol, {n} in all, each time reading ▶ and everything written so far; each row of the matrix is one step’s cross-attention. It wrote “{wrote}”{verdict}. The other lesson steps ran the decoder once over all {steps} positions instead; thanks to the causal mask, their rows equal these decoding steps to within {gap|n2}.',
				zh: '解码器为每个符号重复课程第 4 到第 8 步的计算，一共 {n} 次，每次都读 ▶ 和已经写下的全部符号；矩阵的每一行就是某一步的交叉注意力。它写下“{wrote}”{verdict}。课程的其他步骤则让解码器对全部 {steps} 个位置只运行一次；多亏了因果掩码，那样算出的每一行和这里逐步算出的结果相差最多 {gap|n2}。',
				vars: (rt, at) => {
					const L = look(rt);
					const C = (at.state as { cross?: { data: ArrayLike<number> } } | undefined)?.cross?.data;
					const W = tr(rt, 'dec.cattn');
					const n = L.end >= 0 ? L.end + 1 : L.ids.length;
					let gap = C && W ? 0 : NaN;
					if (C && W) {
						const [H, R, T] = W.shape;
						for (let h = 0; h < H; h++) for (let i = 0; i < n * T; i++) gap = Math.max(gap, Math.abs(C[h * R * T + i] - W.data[h * R * T + i]));
					}
					return { ...dims(rt), n, wrote: L.written, verdict: verdict(rt), gap };
				}
			},
			look: { en: 'It fills in row by row; the slider and the arrows step back through it. The buttons switch between the heads.', zh: '它一行一行地填满；滑块和箭头可以往回一步步看。按钮可以在两个头之间切换。' }, linger: 12000, terms: ['greedy-decoding', 'cross-attention']
		},
		{
			id: 'yours', provenance: 'live',
			view: 'tiny-encdec/cross',
			title: { en: 'Your own date, in another format', zh: '你自己的日期，换一种格式' },
			narration: {
				en: 'The same weights handle all {nf} formats the model was trained on; only where it looks changes. Your text is {fmt}, and the model wrote “{wrote}”{verdict}. On {fact:testN|int} test dates in these formats it was right {fact:testExact|pct} of the time, the GRU {fact:gruTestExact|pct}.',
				zh: '同一组权重处理训练时见过的全部 {nf} 种格式，变的只是它看哪里。你的文字{fmt}，模型写下“{wrote}”{verdict}。在这些格式的 {fact:testN|int} 个测试日期上，它答对了 {fact:testExact|pct}，GRU 是 {fact:gruTestExact|pct}。',
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
			look: { en: 'Compare with lesson step 9: which columns do the rows of the year, the month and the day read now?', zh: '和课程第 9 步比一比：现在年、月、日那几行各读的是哪几列？' },
			linger: 12000, terms: ['encoder-decoder']
		},
		{
			id: 'limits', provenance: 'live',
			view: { key: 'tiny-encdec/cross', props: () => ({ tries: true, gru: true }) },
			kindOf: 'limits',
			title: { en: 'Formats it never saw', zh: '它没见过的格式' },
			narration: {
				en: 'The model only learned its {nf} training formats. On {fact:offN|int} dates of each unseen kind it gets {fact:offYearFirst|pct} right with the year first, {fact:offDashes|pct} with dashes and {fact:offNoComma|pct} without the comma. It copies digits: for a day that does not exist, such as “{bad}”, it copies them in {fact:offImpossible|pct} of cases, never saying it is unsure. On your text it wrote “{wrote}”{verdict}.',
				zh: '模型只学过训练数据里的 {nf} 种格式。每种没见过的格式各测 {fact:offN|int} 个日期：年份在前的答对 {fact:offYearFirst|pct}，用短横线的答对 {fact:offDashes|pct}，少了逗号的答对 {fact:offNoComma|pct}。它会照抄数字：对一个不存在的日子，比如“{bad}”，它在 {fact:offImpossible|pct} 的情况下把数字照抄下来，从不说自己没把握。对你的文字，它写下“{wrote}”{verdict}。',
				vars: (rt) => ({ ...dims(rt), bad: TRIES[4], wrote: look(rt).written, verdict: verdict(rt) })
			},
			look: { en: 'Each button types a date in a format the model never saw. The table puts tinyseq2seq’s GRU beside it on the same dates: the two fail in different places.', zh: '每个按钮会输入一个模型没见过的格式的日期。表格把 tinyseq2seq 的 GRU 在同样日期上的表现放在旁边：两者出错的地方不一样。' }, terms: ['out-of-distribution']
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
