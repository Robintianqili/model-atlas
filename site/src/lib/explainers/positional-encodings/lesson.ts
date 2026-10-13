// positional-encodings: the guided lesson (ten steps; the concept page's primary path). The 'pe' source runs TinyGPT on
// the learner's text (as typed and shuffled, with and without its position table) and the encodings of pe.ts at the
// learner's positions p and k; the extrapolation numbers are precomputed (extrap.ts, facts.json).
//   order (predict) -> positions -> learned -> sinusoidal -> rope -> relative (task) -> alibi -> extrapolate (predict)
//   -> results -> limits
import type { At, I18n } from '../../atlas/types/index.ts';
import { defineLesson } from '../../atlas/lesson/define.ts';
import { fmtNum, fmtPct } from '../../atlas/core/format.ts';
import { CTX, peOf, PRESETS, START } from './adapters/pe-text/adapter.ts';
import { tokText } from './vocab.ts';

const NaNs = new Proxy({} as Record<string, number>, { get: () => NaN });
/** The step's frame metrics (NaN for anything missing, so lint reports it). */
const M = (at: At): Record<string, number> => ({ ...NaNs, ...(at.frame?.metrics ?? {}) }) as Record<string, number>;
const ids = (at: At, f = 'typed') => Array.from(((at.state as Record<string, { data: ArrayLike<number> }> | undefined)?.[f]?.data ?? []) as ArrayLike<number>);
/** A token's text in quotes (a leading space dropped). */
const q = (id: number) => (Number.isFinite(id) ? `“${tokText(id).trim() || '␣'}”` : NaN);
/** True when the shuffle moved something (three or more tokens, and not the same sequence). */
const moved = (m: Record<string, number>) => m.n >= 3 && m.same === 0;
/** "did not change at all" or "changed only by rounding, by 3e−8". */
const none = (d: number): I18n | number =>
	!Number.isFinite(d) ? NaN : d < 1e-12 ? { en: 'did not change at all', zh: '完全没有变化' } : { en: `changed only by rounding, by ${fmtNum(d)}`, zh: `只因舍入误差变化了 ${fmtNum(d)}` };

const lesson = defineLesson({
	title: { en: 'Positional encodings, step by step', zh: '位置编码，分步课程' },
	summary: {
		en: 'Shuffle your sentence and watch attention not notice, read TinyGPT’s learned position table, compute a sinusoidal encoding, turn a query with RoPE, ramp scores with ALiBi, then see which encodings survive text four times longer than training.',
		zh: '打乱你的句子，看注意力察觉不到；读 TinyGPT 学到的位置表，算一个正弦位置编码，用 RoPE 旋转查询，用 ALiBi 给分数加斜坡，再看哪些编码在比训练长四倍的文本上还能用。'
	},
	fixtures: [
		{ id: 'far', label: { en: 'positions past TinyGPT’s table (p 70, k 10)', zh: '超出 TinyGPT 位置表的位置（p 70，k 10）' }, raw: { ...START, p: 70, k: 10, shuffle: 4 } },
		{ id: 'edge-table', label: { en: 'the query on the table’s last row (p 60, k 3)', zh: '查询落在位置表最后一行（p 60，k 3）' }, raw: { ...PRESETS[2].raw, p: 60, k: 3, shuffle: 2 } },
		{ id: 'p0', label: { en: 'the key at position 0 (p 0, k 3)', zh: '键在位置 0（p 0，k 3）' }, raw: { ...PRESETS[1].raw, p: 0, k: 3 } },
		{ id: 'p0-k63', label: { en: 'the widest pair (p 0, k 63)', zh: '相隔最远的一对（p 0，k 63）' }, raw: { ...PRESETS[3].raw, p: 0, k: 63 } }
	],
	steps: [
		{
			id: 'order',
			title: { en: 'Shuffle the words: does attention notice?', zh: '打乱词序：注意力会察觉吗？' },
			source: 'pe',
			at: { phase: 'order' },
			view: { key: 'positional-encodings/order', props: () => ({ mode: 'nopos' }) },
			provenance: 'live',
			terms: ['attention', 'query-key-value', 'positional-encoding'],
			predict: {
				id: 'positional-encodings-shuffle',
				question: {
					en: 'With no positions, the tokens before the last one are shuffled. How much does the last token’s layer-0 attention output change?',
					zh: '没有位置信息时，把最后一个词元之前的词元打乱。最后一个词元在第 0 层的注意力输出会变化多少？'
				},
				options: [
					{ en: 'Not at all', zh: '完全不变' },
					{ en: 'A little: the words are in a different order', zh: '变一点：词的顺序不同了' },
					{ en: 'A lot: the sentence no longer makes sense', zh: '变很多：句子已经说不通了' }
				],
				answer: 0,
				explain: {
					en: 'Not at all. Every earlier token keeps its score and its weight; only their places in the sum change, and a sum does not care about order. To attention without positions, a sentence is a bag of tokens.',
					zh: '完全不变。每个前面的词元都保持原来的分数和权重，变的只是它们在求和里的先后，而求和不在乎顺序。对没有位置信息的注意力来说，一句话就是一袋词元。'
				}
			},
			narration: {
				en: 'Attention scores the last token’s query against the key of every token up to it, itself included, and averages their values by those weights. Nothing in that sum says where a token sits. {setup}',
				zh: '注意力用最后一个词元的查询，去和它自己及前面每个词元的键打分，再按这些权重对它们的值求平均。这个求和里没有任何东西说明一个词元在哪个位置。{setup}',
				vars: (_rt, at) => {
					const m = M(at);
					const t = ids(at);
					if (!Number.isFinite(m.n)) return { setup: NaN };
					const setup: I18n = moved(m)
						? { en: `Here TinyGPT reads your ${m.n} tokens with its position table set to zero, then the same tokens shuffled, the last one, ${q(t[t.length - 1])}, kept in place.`, zh: `这里 TinyGPT 先读你的 ${m.n} 个词元，位置表全部设为零，再读打乱后的同一批词元，最后一个词元 ${q(t[t.length - 1])} 留在原位。` }
						: { en: `Your ${m.n} tokens leave nothing to shuffle (the last one stays in place): type a longer sentence to see it.`, zh: `你的 ${m.n} 个词元没有可以打乱的（最后一个留在原位）：输入一个更长的句子再看。` };
					return { setup };
				}
			},
			look: { en: 'In the stage, pick one token and find it in both rows; compare its colour and percentage.', zh: '在舞台里挑一个词元，在两行里都找到它，比较它的颜色和百分比。' }
		},
		{
			id: 'positions',
			title: { en: 'Add TinyGPT’s position rows, and order matters', zh: '加上 TinyGPT 的位置行，顺序就有了意义' },
			source: 'pe',
			at: { phase: 'order' },
			view: { key: 'positional-encodings/order', props: () => ({ mode: 'pos' }) },
			provenance: 'live',
			terms: ['embedding', 'positional-encoding', 'next-token-prediction'],
			narration: {
				en: 'Without positions, the last token’s attention output {d0}. TinyGPT adds a learned row of numbers for each position to every token’s embedding, so a token looks different at each place. {withPos}',
				zh: '没有位置信息时，最后一个词元的注意力输出{d0}。TinyGPT 给每个词元的嵌入加上它所在位置的一行可学习的数，于是同一个词元在不同位置看起来不一样。{withPos}',
				vars: (_rt, at) => {
					const m = M(at);
					if (!Number.isFinite(m.d1)) return { d0: NaN, withPos: NaN };
					const guess: I18n =
						m.top1 === m.top1s
							? { en: `stays ${q(m.top1)}, but its probability goes from ${fmtPct(m.p1)} to ${fmtPct(m.p1s)}`, zh: `仍是 ${q(m.top1)}，但它的概率从 ${fmtPct(m.p1)} 变成 ${fmtPct(m.p1s)}` }
							: { en: `changes from ${q(m.top1)} (${fmtPct(m.p1)}) to ${q(m.top1s)} (${fmtPct(m.p1s)})`, zh: `从 ${q(m.top1)}（${fmtPct(m.p1)}）变成 ${q(m.top1s)}（${fmtPct(m.p1s)}）` };
					const withPos: I18n = moved(m)
						? { en: `With those rows, shuffling changes that output by up to ${fmtNum(m.d1)}, and the most likely next token ${guess.en}.`, zh: `有了这些位置行，打乱会让这个输出最多变化 ${fmtNum(m.d1)}，最可能的下一个词元${guess.zh}。` }
						: { en: 'With a longer sentence, shuffling would now change that output.', zh: '换一个更长的句子，打乱现在就会改变这个输出。' };
					return { d0: none(m.d0), withPos };
				}
			},
			look: { en: 'Find a token in both rows: with positions, its weight differs between the two orders.', zh: '在两行里找到同一个词元：有了位置信息，它在两种顺序下的权重不一样了。' }
		},
		{
			id: 'learned',
			title: { en: 'TinyGPT’s learned position table', zh: 'TinyGPT 学到的位置表' },
			source: 'pe',
			at: { phase: 'learned' },
			view: 'positional-encodings/table',
			provenance: 'live',
			terms: ['cosine-similarity', 'context-window'],
			narration: {
				en: 'TinyGPT learned {rows} rows of {d} numbers, one per position, counted from 0. {rowP} Training made neighbours alike: rows one apart have a cosine similarity of {mean1} on average, rows {k} apart {meank}, and the first and last rows {far}. There is no row {rows}, so TinyGPT cannot place token number {next}.',
				zh: 'TinyGPT 学到了 {rows} 行、每行 {d} 个数，每个位置一行，位置从 0 数起。{rowP}训练让相邻的行变得很像：相隔一行的两行平均余弦相似度是 {mean1}，相隔 {k} 行的是 {meank}，第一行和最后一行只有 {far}。没有编号 {rows} 的行，所以 TinyGPT 没法安放第 {next} 个词元。',
				vars: (_rt, at) => {
					const m = M(at);
					const rowP: I18n | number = !Number.isFinite(m.p)
						? NaN
						: m.p < m.rows
							? { en: `Row ${m.p} has length (norm) ${fmtNum(m.normP)}, while your tokens’ embeddings have length ${fmtNum(m.tokNorm)} on average.`, zh: `编号 ${m.p} 的行长度（范数）是 ${fmtNum(m.normP)}，而你的词元嵌入平均长度是 ${fmtNum(m.tokNorm)}。` }
							: { en: `Your position p = ${m.p} has no row at all.`, zh: `你选的位置 p = ${m.p} 根本没有对应的行。` };
					return { rows: m.rows, d: m.d, next: m.rows + 1, rowP, mean1: m.mean1, k: m.k, meank: m.meank, far: m.far };
				}
			},
			look: { en: 'Row p is outlined in the table; the strip under it is row p’s similarity with every row.', zh: '表中带框的是第 p 行；下面的色带是第 p 行与每一行的相似度。' }
		},
		{
			id: 'sinusoidal',
			title: { en: 'Sinusoidal: positions from a formula', zh: '正弦编码：用公式算出位置' },
			source: 'pe',
			at: { phase: 'sin' },
			view: 'positional-encodings/table',
			provenance: 'live',
			terms: ['sinusoidal-encoding'],
			narration: {
				en: 'The sinusoidal encoding needs no training: position p gets, in dimensions 2i and 2i + 1 (pair i), sin(p·ω) and cos(p·ω), with ω = {base}^(−2i/{d}). Pair 0 turns fastest, ω = 1: at p = {p} it holds sin {p} = {s0} and cos {p} = {c0}. Pair {last} (counting from 0) turns slowest, ω = {wLast}: it reaches only {aLast} radians, and one full turn takes {turnLast|int} positions.',
				zh: '正弦编码不需要训练：位置 p 在第 2i 和 2i + 1 维（第 i 对）上取 sin(p·ω) 和 cos(p·ω)，其中 ω = {base}^(−2i/{d})。第 0 对转得最快，ω = 1：在 p = {p} 处，它是 sin {p} = {s0} 和 cos {p} = {c0}。第 {last} 对（从 0 数起）转得最慢，ω = {wLast}：它只转到 {aLast} 弧度，转满一圈要 {turnLast|int} 个位置。',
				vars: (_rt, at) => {
					const m = M(at);
					return { base: m.base, d: m.d, p: m.p, s0: m.s0, c0: m.c0, last: m.last, wLast: m.wLast, aLast: m.aLast, turnLast: m.turnLast };
				}
			},
			look: { en: 'In the table, the left columns change from row to row, while the right columns barely change.', zh: '在表中，左边的列每一行都在变，右边的列几乎不变。' }
		},
		{
			id: 'rope',
			title: { en: 'RoPE: turn the query and the key', zh: 'RoPE：旋转查询和键' },
			source: 'pe',
			at: { phase: 'rope' },
			view: 'positional-encodings/rope',
			provenance: 'live',
			terms: ['rope', 'dot-product'],
			narration: {
				en: 'RoPE adds nothing to the embeddings. Inside attention it turns each 2-D pair of a query and a key by the position times an angle θ: 1 radian for pair 0, down to {thetaLast} for pair {lastPair} (counting from 0). TinyGPT was not trained this way; here its real query for {qt}, placed at position {m}, and its key for {kt}, placed at {n}, turn their pair 0 by {m} and {n} radians, and the pair-0 dot product goes from {pairRaw} to {pair}.',
				zh: 'RoPE 不往嵌入上加任何东西。它在注意力内部把查询和键的每一对二维分量旋转“位置 × 角度 θ”：第 0 对每个位置转 1 弧度，第 {lastPair} 对（从 0 数起）只转 {thetaLast}。TinyGPT 并不是这样训练的；这里把它真实的查询（{qt}）放在位置 {m}，把键（{kt}）放在位置 {n}，它们的第 0 对分别转 {m} 和 {n} 弧度，第 0 对的点积从 {pairRaw} 变成 {pair}。',
				vars: (rt, at) => {
					const m = M(at);
					const t = peOf(rt.input).ids;
					return { thetaLast: m.thetaLast, lastPair: m.pairs - 1, qt: t.length ? q(t[t.length - 1]) : NaN, kt: t.length ? q(t[0]) : NaN, m: m.m, n: m.n, pairRaw: m.pairRaw, pair: m.pair };
				}
			},
			look: { en: 'Dashed arrows are before turning, solid ones after; the arcs show how far each one turned.', zh: '虚线箭头是旋转之前，实线是旋转之后；弧线表示各自转了多少。' }
		},
		{
			id: 'relative',
			title: { en: 'With RoPE, only the distance counts', zh: '用 RoPE 时，只有距离起作用' },
			source: 'pe',
			at: { phase: 'relative' },
			view: 'positional-encodings/lines',
			provenance: 'live',
			terms: ['rope', 'attention-head'],
			task: {
				prompt: { en: 'Move the key position p in the input widget and watch your mark slide along the two lines.', zh: '在输入控件里移动键的位置 p，看你的标记沿着两条线滑动。' },
				done: (rt) => peOf(rt.input).p !== START.p,
				hint: { en: 'The slider “Key position p” starts at 5.', zh: '“键的位置 p”滑块从 5 开始。' },
				waiting: { en: 'Not yet: p is still at its start.', zh: '还没有：p 还在起始位置。' },
				doneText: { en: 'Done: on the RoPE line your mark kept its height; on the learned one it moved.', zh: '完成：在 RoPE 那条线上，你的标记高度不变；在可学习位置表那条线上，它的高度变了。' }
			},
			narration: {
				en: 'The query turns by its own position times θ and the key by its own, so wherever the pair sits, the query is turned k × θ further, with k = {k} their distance. Over head 0, the RoPE dot product is {ropeP} with the key at p = {p}, and the same with the key at {ref}. {abs}',
				zh: '查询按自己的位置乘以 θ 旋转，键也按自己的位置旋转，所以无论这一对放在哪里，查询总比键多转 k × θ，k = {k} 是两者的距离。在头 0 上，键在 p = {p} 时 RoPE 点积是 {ropeP}，键在 {ref} 时也一样。{abs}',
				vars: (_rt, at) => {
					const m = M(at);
					const same = Math.abs(m.ropeP - m.ropeRef) < 1e-9;
					const abs: I18n | number = !Number.isFinite(m.abs0)
						? NaN
						: !Number.isFinite(m.absP)
							? { en: `TinyGPT’s learned rows cannot place this pair at all: the query would sit at ${m.p + m.k}, past row ${m.last}.`, zh: `TinyGPT 学到的位置行根本放不下这一对：查询会落在位置 ${m.p + m.k}，超出了第 ${m.last} 行。` }
							: m.refA === m.p
								? { en: `TinyGPT’s learned rows fit this pair in one place only, the key at 0 and the query at ${m.last}, so there is no shift to compare.`, zh: `TinyGPT 学到的位置行只能把这一对放在一个地方：键在 0，查询在 ${m.last}，所以没有平移可以比较。` }
								: { en: `With TinyGPT’s learned rows instead, the same two tokens score ${fmtNum(m.absP)} with the key at p = ${m.p} and ${fmtNum(m.absRef)} at ${m.refA}: an absolute table makes every shift look different.`, zh: `换成 TinyGPT 学到的位置行，同样两个词元在键位于 p = ${m.p} 时得分 ${fmtNum(m.absP)}，在 ${m.refA} 时得分 ${fmtNum(m.absRef)}：绝对位置表让每一次平移看起来都不一样。` };
					return { k: m.k, ropeP: same ? m.ropeP : NaN, p: m.p, ref: m.ref, abs };
				}
			},
			look: { en: 'The flat line is RoPE, the wavy one TinyGPT’s learned rows; the dashed mark is your p.', zh: '平的那条线是 RoPE，起伏的那条是 TinyGPT 学到的位置行；虚线标记是你的 p。' }
		},
		{
			id: 'alibi',
			title: { en: 'ALiBi: a penalty that grows with distance', zh: 'ALiBi：随距离增大的惩罚' },
			source: 'pe',
			at: { phase: 'alibi' },
			view: 'positional-encodings/lines',
			provenance: 'live',
			terms: ['alibi', 'softmax'],
			narration: {
				en: 'ALiBi adds no vectors and turns nothing: it subtracts m × distance from each attention score, where the slope m differs per head: here {m0}, {m1}, {m2} and {m3}. At distance k = {k}, head 0 subtracts {pen0}: of two keys with equal scores, the one {k} back gets {r0|pct} of the weight of the one at distance 0. Head 3 (counting from 0) subtracts only {pen3} and keeps {r3|pct}.',
				zh: 'ALiBi 不加向量，也不旋转：它从每个注意力分数里减去 m × 距离，斜率 m 每个头各不相同：这里是 {m0}、{m1}、{m2} 和 {m3}。在距离 k = {k} 处，头 0 减去 {pen0}：两个分数相同的键里，往前 {k} 个的那个，权重只有距离为 0 的那个的 {r0|pct}。头 3（从 0 数起）只减去 {pen3}，保留 {r3|pct}。',
				vars: (_rt, at) => {
					const m = M(at);
					const frac = (v: number) => (Number.isFinite(v) && v > 0 ? `1/${Math.round(1 / v)}` : NaN);
					return { m0: frac(m.m0), m1: frac(m.m1), m2: frac(m.m2), m3: frac(m.m3), k: m.k, pen0: -m.b0, r0: m.r0, pen3: -m.b3, r3: m.r3 };
				}
			},
			look: { en: 'Each line is one head’s ramp; the dashed mark is your distance k.', zh: '每条线是一个头的斜坡；虚线标记是你的距离 k。' }
		},
		{
			id: 'extrapolate',
			title: { en: 'Train on 64 tokens, read 256', zh: '用 64 个词元训练，读 256 个' },
			source: 'pe',
			at: { phase: 'extrap' },
			view: 'positional-encodings/lines',
			provenance: 'precomputed',
			terms: ['extrapolation', 'cross-entropy', 'nope'],
			predict: {
				id: 'positional-encodings-extrap',
				question: {
					en: 'Which of these three keeps its loss lowest past token 64?',
					zh: '这三个模型中，哪一个在第 64 个词元之后损失保持得最低？'
				},
				options: [
					{ en: 'Sinusoidal', zh: '正弦编码' },
					{ en: 'RoPE', zh: 'RoPE' },
					{ en: 'ALiBi', zh: 'ALiBi' }
				],
				answer: 2,
				explain: {
					en: 'ALiBi. Past token 64 its penalty only grows, which pushes far tokens further down but shows the model nothing new. The sinusoidal and RoPE models meet angles their slow pairs never reached in training.',
					zh: 'ALiBi。过了第 64 个词元，它的惩罚只是继续变大，把远处的词元压得更低，但不会给模型看任何没见过的东西。正弦编码和 RoPE 的模型则会遇到慢速维度对在训练中从未转到过的角度。'
				}
			},
			narration: {
				en: 'Five TinyGPT-sized models, alike but for their positions, trained for {fact:steps} steps on {fact:trainCtx}-token windows of TinyStories, then read {fact:evalCtx} tokens. Within {fact:trainCtx} tokens they score alike, a loss from {fact:inRope} with RoPE to {fact:inNone} with no positions. Guess before the stage shows the loss at every position.',
				zh: '五个大小和 TinyGPT 一样、只有位置编码不同的模型，在 TinyStories 的 {fact:trainCtx} 词元窗口上训练了 {fact:steps} 步，然后读 {fact:evalCtx} 个词元。在 {fact:trainCtx} 个词元以内它们的成绩差不多，损失从 RoPE 的 {fact:inRope} 到无位置编码的 {fact:inNone}。先猜一猜，再看舞台显示的每个位置上的损失。'
			},
			look: { en: 'After you answer, follow each line past the dashed mark at the trained length.', zh: '回答之后，沿着每条线看过训练长度处的虚线标记。' }
		},
		{
			id: 'results',
			title: { en: 'Past the trained length, the encodings split', zh: '超出训练长度后，各种编码分道扬镳' },
			source: 'pe',
			at: { phase: 'extrap' },
			view: 'positional-encodings/lines',
			provenance: 'precomputed',
			terms: ['extrapolation'],
			narration: {
				en: 'Over the last {fact:trainCtx} of the {fact:evalCtx} tokens, ALiBi scores {fact:farAlibi}, better than its {fact:inAlibi} within the first {fact:trainCtx}: more context helps it. RoPE rises to {fact:farRope} and the sinusoidal model to {fact:farSin}. With no positions at all the loss drifts up to {fact:farNone}. The learned table cannot even run there: it has no rows past its trained length.',
				zh: '在 {fact:evalCtx} 个词元的最后 {fact:trainCtx} 个上，ALiBi 的损失是 {fact:farAlibi}，比它在前 {fact:trainCtx} 个以内的 {fact:inAlibi} 还低：更多的上下文对它有帮助。RoPE 升到 {fact:farRope}，正弦编码的模型升到 {fact:farSin}。完全没有位置编码的模型，损失慢慢涨到 {fact:farNone}。可学习的位置表在那里连运行都做不到：超出训练长度就没有行了。'
			},
			look: { en: 'Compare where each line ends on the right of the stage with where it was at the dashed mark.', zh: '比较每条线在舞台最右端的位置和它在虚线标记处的位置。' }
		},
		{
			id: 'limits',
			title: { en: 'Two limits: the mask leaks order, and 64 rows', zh: '两个局限：掩码会泄露顺序，以及 64 行的上限' },
			source: 'pe',
			at: { phase: 'order' },
			view: { key: 'positional-encodings/order', props: () => ({ mode: 'nopos', final: true }) },
			provenance: 'live',
			kindOf: 'limits',
			terms: ['causal-mask', 'nope'],
			narration: {
				en: 'Two limits of what you have seen. {leak} {cut} Nor does this page show how large models stretch RoPE to longer texts, such as position interpolation or YaRN.',
				zh: '你刚才看到的结果有两个局限。{leak}{cut}本页也没有展示大模型怎样把 RoPE 拉伸到更长的文本，例如位置插值或 YaRN。',
				vars: (_rt, at) => {
					const m = M(at);
					if (!Number.isFinite(m.n)) return { leak: NaN, cut: NaN };
					const leak: I18n = moved(m)
						? { en: `No positions does not mean no order: with the table at zero, the last token’s final hidden state still changed by ${fmtNum(m.dF0)} when shuffled, because the causal mask shows each earlier token a different set of tokens before it.`, zh: `没有位置信息不等于没有顺序：位置表设为零时，打乱后最后一个词元的最终隐藏状态仍然变化了 ${fmtNum(m.dF0)}，因为因果掩码让前面每个词元看到的前文都不一样。` }
						: { en: 'No positions does not mean no order: the causal mask shows each earlier token a different set of tokens before it, which deeper layers can use (a longer sentence shows it).', zh: '没有位置信息不等于没有顺序：因果掩码让前面每个词元看到的前文都不一样，更深的层可以利用这一点（换一个更长的句子就能看到）。' };
					const cut: I18n =
						m.cut > 0
							? { en: `And TinyGPT dropped the first ${m.cut} of your tokens: its table stops at ${CTX} rows.`, zh: `另外，TinyGPT 丢掉了你的前 ${m.cut} 个词元：它的位置表只有 ${CTX} 行。` }
							: { en: `And TinyGPT can never read more than ${CTX} tokens: its table stops at ${CTX} rows.`, zh: `另外，TinyGPT 永远读不了超过 ${CTX} 个词元：它的位置表只有 ${CTX} 行。` };
					return { leak, cut };
				}
			},
			look: { en: 'The stage shows the run without positions again: the attention output did not change, the final hidden state did.', zh: '舞台再次显示没有位置信息的那次运行：注意力输出没变，最终隐藏状态变了。' }
		}
	]
});

export default lesson;
