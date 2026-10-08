// word2vec: the guided lesson. Pairs from a sliding window (process `pairs`), one training step slowed down on the first
// pair (source `one`: forward, loss, update), then the whole run (`run`), the window, and a second run (`run2`) for the
// limits. Every number is a {placeholder} from the frames that ran; the other sentences are tested in claims.test.ts.
import type { At, ExplainerRuntime, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmt3 } from '../../atlas/core/format.ts';
import type { Frame } from '../../atlas/types/index.ts';
import { corpusOf, cooccurrence, DEFAULT_WINDOW, SEED, sentencesOf, textOf, windowOf, type CorpusRaw } from './corpus.ts';
import { closestPair, cosine, firstPair, neighbours, wordVectors, worked, workedWeight, finalPca } from './vectors.ts';

type V = Record<string, TmplVal>;
const L = (en: string, zh: string) => ({ en, zh });
const C = (rt: ExplainerRuntime) => corpusOf(rt.input as CorpusRaw);
const q = (w: string | undefined) => (w === undefined ? NaN : `“${w}”`);
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
const net = (show: string) => ({ key: 'word2vec/net', props: () => ({ show }) });
const ONE = (phase: string) => ({ phase, step: 0 });
/** the worked pair's numbers (null on a corpus without pairs, where the step is skipped) */
const W = (rt: ExplainerRuntime, at: At) => worked(rt.input as CorpusRaw, at);
/** the set of context words of each word */
const contexts = (rt: ExplainerRuntime) => cooccurrence(C(rt)).map((r) => new Set(r.flatMap((n, j) => (n ? [j] : []))));
/** the last word vectors of a source */
function lastVecs(rt: ExplainerRuntime, s: string): number[][] {
	const t = rt.sources[s]?.tape;
	return t?.length ? wordVectors(rt.input as CorpusRaw, t.state(t.length - 1)) : [];
}
/** the vocabulary note: the cap (V words kept, d left out), and how Chinese text is cut */
function capOf(V: number, d: number, cjk: boolean) {
	const a = d > 0 ? L(`These ${V} words are the vocabulary; the other ${d} are left out.`, `这 ${V} 个词就是词表，其余 ${d} 个被去掉了。`) : L('This list of words is the vocabulary.', '这张词的列表就是词表。');
	return cjk ? L(`${a.en} Chinese text is cut into single characters, one word each.`, `${a.zh}中文按单个汉字切分，每个汉字算一个词。`) : a;
}
/** the step's center word: the first pair of the run (the same pair `one` slows down) */
const focusId = (rt: ExplainerRuntime) => firstPair(rt.input as CorpusRaw, SEED)?.c ?? -1;
/** step 'pairs': the example frame (the first center word with the most context words); the scrubber opens there */
const EXAMPLE = { phase: 'center', where: (f: Frame) => f.metrics?.ex === 1 };
/** how many context words two words share */
const sharedOf = (ctx: Set<number>[], a: number, b: number) => [...ctx[a]].filter((j) => ctx[b].has(j)).length;
/** step 'window': opposite meanings, same places (when the text has “happy” and “sad” and they ended up alike) */
function happySad(rt: ExplainerRuntime, vs: number[][], ctx: Set<number>[]) {
	const words = C(rt).words, a = words.indexOf('happy'), b = words.indexOf('sad');
	const sh = a >= 0 && b >= 0 ? sharedOf(ctx, a, b) : 0, cos = sh > 0 && vs[a] && vs[b] ? cosine(vs[a], vs[b]) : NaN;
	return cos >= 0.5
		? L(`Sharing context words is not sharing a meaning: “happy” and “sad” mean opposite things, yet they share ${sh} context word${sh === 1 ? '' : 's'}, and their cosine similarity is ${fmt3(cos)}.`, `有相同的上下文词，不等于有相同的意思：“happy”和“sad”意思相反，却有 ${sh} 个共同的上下文词，余弦相似度是 ${fmt3(cos)}。`)
		: L('Words can share context words without sharing a meaning: the model only ever sees which word stands where.', '词可以有相同的上下文词，却没有相同的意思：模型看到的只是哪个词出现在哪个位置。');
}

const lesson: LessonDef = {
	title: L('word2vec: word vectors from your own sentences', 'word2vec：从你自己的句子学出词向量'),
	summary: L(
		'A sliding window turns your sentences into (center, context) pairs; one training step is worked out with its numbers; then 3,000 steps pull words that share context words together, and a second run shows how noisy a tiny corpus is.',
		'滑动窗口把你的句子变成（中心词，上下文词）训练对；带着数字算一遍一步训练；然后 3,000 步训练让上下文相同的词靠在一起；再训练一次，看看很小的语料有多不稳定。'
	),
	steps: [
		{
			id: 'words',
			source: 'pairs',
			at: { phase: 'vocab' },
			view: { key: 'word2vec/window', props: () => ({ mode: 'vocab' }) },
			provenance: 'live',
			title: L('Your sentences become words with ids', '你的句子变成带编号的词'),
			narration: {
				en: 'word2vec learns from nothing but your text. It cuts your {S} sentences into {N} words, {D} of them different, and {give} (ids count from 0, the most frequent word first): {w0} (id 0) appears {n0} times. {cap}',
				zh: 'word2vec 只从你的文本中学习。它把你的 {S} 个句子切成 {N} 个词，其中 {D} 个互不相同，再{give}（编号从 0 开始，出现最多的词排在最前）：{w0}（编号 0）出现了 {n0} 次。{cap}',
				vars: (rt, at) => {
					// counted before the vocabulary cap: the words the learner wrote
					const all = sentencesOf(textOf(rt.input as CorpusRaw)), V = m(at, 'V'), d = m(at, 'dropped');
					const give = pick(d > 0, L(`keeps the ${V} most frequent, giving each an id`,`留下出现最多的 ${V} 个，给每个一个编号`), L('gives each of them an id', '给每个不同的词一个编号'));
					return { S: all.length, N: all.flat().length, D: V + d, give, w0: q(C(rt).words[0]), n0: C(rt).counts[0] ?? NaN, cap: capOf(V, d, C(rt).words.some((w) => /^[^a-z0-9]/.test(w))) };
				}
			},
			look: L('The small number under each word is its id; the list below counts every word.', '每个词下面的小数字是它的编号；下面的列表数出了每个词出现的次数。'),
			terms: ['vocabulary', 'token']
		},
		{
			id: 'pairs',
			source: 'pairs',
			at: EXAMPLE,
			view: 'word2vec/window',
			play: { from: EXAMPLE, to: 'last', phases: ['center'], interval: 300, unit: L('word', '词') },
			provenance: 'live',
			title: L('A window slides over each sentence', '窗口滑过每个句子'),
			narration: {
				en: 'Training needs examples. Each word in turn is the center word, and every word up to {w} places to its left or right, in the same sentence, is one of its context words. Each (center, context) is one training pair: {center} in sentence {s} gives {k} of them, and your whole text gives {P}.',
				zh: '训练需要例子。每个词轮流当中心词；同一句子里、它左右各 {w} 个位置以内的词，都是它的上下文词。每个（中心词，上下文词）就是一个训练对：第 {s} 句里的{center}给出 {k} 对，你的全部文本一共给出 {P} 对。',
				vars: (rt, at) => ({ w: windowOf(rt.input as CorpusRaw), center: q(C(rt).words[m(at, 'c')]), s: m(at, 's') + 1, k: m(at, 'k'), P: C(rt).pairs.length })
			},
			look: L('The filled word is the center, the dashed ones are its window; its pairs are listed under the text. Then the window slides on, word by word (the play button under the text).', '实心的词是中心词，虚线框是它的窗口；它给出的训练对列在文本下面。接着窗口一个词一个词地往后滑（文本下面的播放按钮）。'),
			predict: {
				id: 'word2vec-pairs',
				question: L('With a window of 2, how many (center, context) pairs does one sentence of 5 words give?', '窗口为 2 时，一个 5 个词的句子能给出多少个（中心词，上下文词）对？'),
				options: [L('5', '5'), L('10', '10'), L('14', '14'), L('20', '20')],
				answer: 2,
				explain: L('The five words give 2, 3, 4, 3 and 2 pairs: a word near either end of the sentence has fewer neighbours inside it. 2 + 3 + 4 + 3 + 2 = 14.', '五个词分别给出 2、3、4、3、2 对：靠近句子两端的词，在句子里的邻居更少。2 + 3 + 4 + 3 + 2 = 14。')
			},
			terms: ['skip-gram', 'word-window']
		},
		{
			id: 'onehot',
			source: 'one',
			at: ONE('forward'),
			view: net('onehot'),
			provenance: 'live',
			title: L('The center word as a one-hot vector', '把中心词写成独热向量'),
			narration: {
				en: 'Now one training step, slowed down. Training draws a pair at random: center {c}, context {o}. The network reads the center word as a one-hot vector: {V} numbers, one per word of the vocabulary, all 0 except a 1 at id {cid}.',
				zh: '现在放慢看一步训练。训练随机抽出一个训练对：中心词{c}，上下文词{o}。网络把中心词读成一个独热向量：{V} 个数，词表里每个词一个，除了编号 {cid} 处是 1，其余全是 0。',
				vars: (rt, at): V => { const w = W(rt, at); return w ? { c: q(w.words[w.c]), o: q(w.words[w.o]), V: w.V, cid: w.c } : { c: NaN }; }
			},
			look: L('The column x: a single 1, in the row of the center word.', 'x 这一列：只有一个 1，在中心词那一行。'),
			terms: ['one-hot']
		},
		{
			id: 'row',
			source: 'one',
			at: ONE('forward'),
			view: net('row'),
			provenance: 'live',
			title: L('The hidden vector is one row of W_in', '隐藏向量就是 W_in 的一行'),
			narration: {
				en: 'W_in is a table with one row of {d} numbers for each of the {V} words, small random numbers before training. Multiplying the one-hot vector by W_in multiplies each row by its number in the one-hot vector and adds the rows up; every row is multiplied by 0 except row {cid}, so the result is just that row: h = {h}. This row is the word vector of {c}, the thing word2vec learns.',
				zh: 'W_in 是一张表：{V} 个词每个占一行，每行 {d} 个数，训练前是很小的随机数。独热向量乘以 W_in，就是把每一行乘以独热向量里对应的那个数，再把各行加起来；除了编号 {cid} 的那一行，每一行都乘以 0，所以结果就是这一行：h = {h}。这一行就是{c}的词向量，也就是 word2vec 要学的东西。',
				vars: (rt, at): V => { const w = W(rt, at); return w ? { d: w.d, V: w.V, cid: w.c, h: `(${w.h.map((x) => fmt3(x)).join(', ')})`, c: q(w.words[w.c]) } : { d: NaN }; }
			},
			look: L('The highlighted row of W_in and h above the table: the same numbers.', '高亮的 W_in 那一行和表格上方的 h：是同样的数。'),
			terms: ['embedding', 'weight']
		},
		{
			id: 'scores',
			source: 'one',
			at: ONE('forward'),
			view: net('scores'),
			provenance: 'live',
			title: L('A score for every word', '每个词一个分数'),
			narration: {
				en: 'W_out is a second table, with another vector of {d} numbers for every word: its output vector. A word’s score is the dot product of h with its output vector: multiply number by number, then add. The true context {o} scores {uo}; the highest score, {umax}, goes to {top}.',
				zh: 'W_out 是第二张表，为每个词再存一个 {d} 个数的向量，叫输出向量。一个词的分数就是 h 和它的输出向量的点积：逐个数相乘，再相加。真正的上下文词{o}得分 {uo}；最高分 {umax} 属于{top}。',
				vars: (rt, at): V => { const w = W(rt, at); if (!w) return { d: NaN }; const t = w.u.indexOf(Math.max(...w.u)); return { d: w.d, o: q(w.words[w.o]), uo: w.u[w.o], umax: w.u[t], top: q(w.words[t]) }; }
			},
			look: L('The output vectors, the scores, and the products for the context word written out above the table.', '输出向量、分数，以及表格上方写出的上下文词的乘积。'),
			terms: ['dot-product', 'logit']
		},
		{
			id: 'softmax',
			source: 'one',
			at: ONE('forward'),
			view: net('softmax'),
			provenance: 'live',
			title: L('Softmax turns scores into shares', 'softmax 把分数变成份额'),
			narration: {
				en: 'Softmax turns the {V} scores into shares that add up to 1: for each score u it takes e to the power u, then divides each result by the sum of them all. The shares are the model’s guess at how likely each word is to appear near {c}. Guess first, then look.',
				zh: 'softmax 把 {V} 个分数变成加起来等于 1 的份额：对每个分数 u 先算出 e 的 u 次方，再把每个结果除以它们的总和。这些份额就是模型的猜测：每个词出现在{c}附近的可能性有多大。先猜一猜，再看。',
				vars: (rt, at): V => { const w = W(rt, at); return w ? { V: w.V, c: q(w.words[w.c]) } : { V: NaN }; }
			},
			look: L('After you answer: how different are the bars?', '回答之后：这些条相差多少？'),
			predict: {
				id: 'word2vec-softmax',
				question: L('The weights are still random. Which word gets the largest share?', '权重还是随机的。哪个词会得到最大的份额？'),
				options: [L('The true context word', '真正的上下文词'), L('The most frequent word', '出现最多的词'), L('None stands out: every share is close to 1 / V', '没有哪个突出：每个份额都接近 1 / V')],
				answer: 2,
				explain: L(
					'Small random weights give scores close to 0, and e to the power of a number close to 0 is close to 1, so every word gets about the same share: 1 divided by the number of words. Training has to make the shares differ.',
					'很小的随机权重给出接近 0 的分数，而 e 的接近 0 次幂接近 1，所以每个词的份额都差不多：约为 1 除以词数。训练要做的，就是让这些份额拉开差距。'
				)
			},
			terms: ['softmax']
		},
		{
			id: 'loss',
			source: 'one',
			at: ONE('loss'),
			view: net('loss'),
			provenance: 'live',
			title: L('The loss: how wrong was the guess?', '损失：猜得有多错？'),
			narration: {
				en: 'The true context word was {o}, and the model gave it {p|pct}; an even split over {V} words would give each {even|pct}. The loss is −ln of the true word’s share (ln: the natural logarithm): −ln {pp|n4} = {loss|n4}. A share of 1 would give a loss of 0, and the smaller the share, the bigger the loss. This loss is called cross-entropy.',
				zh: '真正的上下文词是{o}，模型给了它 {p|pct}；如果平均分给 {V} 个词，每个得到 {even|pct}。损失 = −ln(真正那个词的份额)，ln 是自然对数：−ln {pp|n4} = {loss|n4}。份额为 1 时损失为 0；份额越小，损失越大。这种损失叫交叉熵。',
				vars: (rt, at): V => { const w = W(rt, at); return w ? { o: q(w.words[w.o]), p: w.p[w.o], pp: w.p[w.o], V: w.V, even: 1 / w.V, loss: m(at, 'loss') } : { o: NaN }; }
			},
			look: L('The share of the outlined row, and the loss above the table.', '加框那一行的份额，以及表格上方的损失。'),
			terms: ['loss', 'cross-entropy']
		},
		{
			id: 'update',
			source: 'one',
			at: ONE('update'),
			view: net('update'),
			provenance: 'live',
			title: L('One SGD step moves both tables', '一步 SGD 同时改动两张表'),
			narration: {
				en: 'A number’s gradient says how fast the loss grows as that number grows. SGD moves every number a small step against its gradient, scaled by the learning rate {lr}: number {k} (counting from 0) in the row of {c} becomes {old} − {lr} × ({g}) = {new}. That row moves toward the output vector of {o} and a little away from the others, and the output vector of {o} moves toward h. So the score of {o} rises from {u0} to {u1}, and the loss on this pair falls from {l0|n4} to {l1|n4}.',
				zh: '一个数的梯度表示：这个数变大一点时，损失增长得有多快。SGD 让每个数沿梯度的反方向走一小步，步长由学习率 {lr} 缩放：{c}那一行的第 {k} 个数（从 0 数起）从 {old} 变成 {old} − {lr} × ({g}) = {new}。这一行向{o}的输出向量靠近，并稍稍远离其他输出向量；{o}的输出向量也向 h 靠近。于是{o}的分数从 {u0} 升到 {u1}，这一对的损失从 {l0|n4} 降到 {l1|n4}。',
				vars: (rt, at): Record<string, TmplVal> => {
					const w = W(rt, at), x = w && workedWeight(w);
					if (!w || !x) return { lr: NaN };
					return { lr: String(w.lr), k: x.k, c: q(w.words[w.c]), o: q(w.words[w.o]), old: x.old, g: x.g, new: x.new, u0: w.u[w.o], u1: w.u1 ?? NaN, l0: w.loss, l1: w.loss1 ?? NaN };
				}
			},
			look: L('The change columns: in W_in only the center word’s row changes; every output vector changes, the context word’s the most. Above the table: the number worked out here and the context word’s score.', '两列“变化”：W_in 里只有中心词那一行变了；每个输出向量都变了，上下文词的变得最多。表格上方：正文里算出的那个数，以及上下文词的分数。'),
			terms: ['gradient', 'sgd', 'learning-rate']
		},
		{
			id: 'together',
			source: 'run',
			at: 'last',
			view: { key: 'word2vec/map', props: (rt) => ({ focus: focusId(rt), animate: true }) },
			play: 'view',
			// under Play: the map animates for about 7 s while the learner reads; then time to finish the narration
			linger: 12000,
			provenance: 'live',
			title: L('Repeat: words with the same context words move together', '重复下去：上下文相同的词会靠在一起'),
			narration: {
				en: 'Training repeats this step {steps} times, each time on a random pair, and the average loss over all {P} pairs falls from {l0} to {l1}. Words that have the same context words get pulled toward the same output vectors, so their own vectors end up alike. {c} ends closest to {nn}: the cosine similarity of their vectors is {cos} (1 would mean the same direction). They share {shared}, and in your text they {adj}.',
				zh: '训练把这一步重复 {steps} 次，每次用一个随机的训练对；所有 {P} 对的平均损失从 {l0} 降到 {l1}。上下文词相同的词会被拉向相同的输出向量，所以它们自己的向量最后变得相像。{c}最后离{nn}最近：它们的向量的余弦相似度是 {cos}（1 表示方向完全相同）。它们有 {shared}共同的上下文词，而在你的文本里，它们{adj}。',
				vars: (rt, at): V => {
					const vs = wordVectors(rt.input as CorpusRaw, at.state), i = focusId(rt);
					if (i < 0 || !vs[i] || !at.tape) return { steps: NaN };
					const nn = neighbours(vs, i)[0], ctx = contexts(rt), words = C(rt).words;
					const shared = plural(sharedOf(ctx, i, nn.j), { one: '{n} context word', many: '{n} context words', zh: '{n} 个' });
					return { steps: m(at, 'step'), P: C(rt).pairs.length, l0: at.tape.frame(0).metrics?.avgLoss ?? NaN, l1: m(at, 'avgLoss'), c: q(words[i]), nn: q(words[nn.j]), cos: nn.cos, shared, adj: pick(ctx[i].has(nn.j), L('also fall in each other’s window', '也会出现在彼此的窗口里'), L('never fall in each other’s window', '从不出现在彼此的窗口里')) };
				}
			},
			look: {
				en: 'Each dot is a word vector, flattened from {d} numbers to 2 by PCA, which keeps {share|pct} of their spread; the replay shows them move. The dashed line joins {c} to its nearest word.',
				zh: '每个点是一个词向量，用 PCA 从 {d} 个数压成 2 个数，保留了 {share|pct} 的分布；重放可以看到它们怎样移动。虚线把{c}连到离它最近的词。',
				vars: (rt, at): V => { const p = finalPca(rt.input as CorpusRaw, at.tape), i = focusId(rt); return { d: wordVectors(rt.input as CorpusRaw, at.state)[0]?.length ?? NaN, share: p?.share ?? NaN, c: q(C(rt).words[i]) }; }
			},
			terms: ['cosine-similarity', 'pca']
		},
		{
			id: 'window',
			source: 'run',
			at: 'last',
			view: { key: 'word2vec/map', props: (rt, at) => ({ list: true, focus: closestPair(wordVectors(rt.input as CorpusRaw, at.state))?.a ?? -1 }) },
			provenance: 'live',
			title: L('Your turn: change the window', '轮到你：改变窗口'),
			narration: {
				en: 'The window decides which words count as context, so it decides which words end up alike. With a window of {w}, the two most similar words are {a} and {b} (cosine similarity {cos}), and they {same} context words. {meaning}',
				zh: '窗口决定哪些词算上下文，也就决定了哪些词最后会相像。窗口为 {w} 时，最相似的两个词是{a}和{b}（余弦相似度 {cos}），它们{same}。{meaning}',
				vars: (rt, at): V => {
					const vs = wordVectors(rt.input as CorpusRaw, at.state), best = closestPair(vs), ctx = contexts(rt), words = C(rt).words;
					if (!best) return { w: NaN };
					const sh = sharedOf(ctx, best.a, best.b);
					return { w: windowOf(rt.input as CorpusRaw), a: q(words[best.a]), b: q(words[best.b]), cos: best.cos, same: pick(sh > 0, L(`share ${sh} of their`, `有 ${sh} 个共同的上下文词`), L('share none of their', '没有共同的上下文词')), meaning: happySad(rt, vs, ctx) };
				}
			},
			look: L('Under the map: the nearest word of every word. Change the window and compare.', '图下面：每个词最近的词。改一改窗口，再比较。'),
			task: {
				prompt: L('Set the window to another size with the buttons under “Your input”, and watch the pairs, the training and the map run again.', '用“你的输入”下面的按钮把窗口改成另一个大小，看训练对、训练和词向量图重新运行。'),
				done: (rt) => windowOf(rt.input as CorpusRaw) !== DEFAULT_WINDOW,
				hint: L('The buttons 1, 2 and 3 next to “Window”.', '“窗口”旁边的 1、2、3 按钮。'),
				doneText: L('Done: the pairs, the training and the map now use the new window.', '完成了：训练对、训练和词向量图都换成了新的窗口。')
			},
			// under Play: after the window changes, the narration has new numbers to read (the default is 4 s)
			linger: 12000,
			terms: ['word-window']
		},
		{
			id: 'limits',
			source: 'run',
			at: 'last',
			view: { key: 'word2vec/map', props: () => ({ list: true, vs: ['run', 'run2'] }) },
			compare: { a: 'run', b: 'run2', label: [L('Run 1', '第 1 次运行'), L('Run 2: another random start', '第 2 次运行：换一个随机起点')] },
			provenance: 'live',
			kindOf: 'limits',
			title: L('Limits: a tiny corpus gives noisy neighbours', '局限：语料太小，近邻就不稳定'),
			narration: {
				en: 'Train the same text again from another random start (and another random order of pairs), then compare. {keep} of the {V} words keep the same nearest word, and the cosine similarity of a pair of words changes by {d} on average: the closest pair of run 1, {a} and {b}, goes from {c1} to {c2}. With so little text, many neighbours depend on the random start. Real word2vec learns from billions of words, with faster tricks than a full softmax.',
				zh: '从另一个随机起点（训练对的随机顺序也不同）把同样的文本再训练一次，然后比较。在 {V} 个词中，{keep}个词的最近邻和第 1 次运行一样；一对词的余弦相似度平均变化 {d}：第 1 次运行里最接近的一对{a}和{b}，从 {c1} 变成 {c2}。文本这么少，很多词的近邻取决于随机起点。真正的 word2vec 要从数十亿个词中学习，还用了比完整 softmax 更快的技巧。',
				vars: (rt): V => {
					const x = lastVecs(rt, 'run'), y = lastVecs(rt, 'run2'), V = x.length, words = C(rt).words;
					if (V < 2 || y.length !== V) return { V: NaN };
					const k = x.filter((_, i) => neighbours(x, i)[0].j === neighbours(y, i)[0].j).length;
					let sum = 0, n = 0;
					for (let a = 0; a < V; a++) for (let b = a + 1; b < V; b++) (sum += Math.abs(cosine(x[a], x[b]) - cosine(y[a], y[b]))), n++;
					const best = closestPair(x)!;
					return { keep: pick(2 * k < V, L(`Only ${k}`, `只有 ${k} `), L(`${k}`, `有 ${k} `)), V, d: sum / n, a: q(words[best.a]), b: q(words[best.b]), c1: best.cos, c2: cosine(y[best.a], y[best.b]) };
				}
			},
			look: L('The marked words: their nearest word changed between the two runs. Each map is drawn on its own axes (its own PCA), so compare the lists of nearest words, not where the dots sit.', '标记出来的词：两次运行中，它们最近的词变了。两张图各用自己的坐标轴（各自的 PCA）画出，所以要比较最近的词的列表，而不是点的位置。')
		}
	],
	fixtures: [{ id: 'window1', label: L('The pets preset with window 1', '宠物预设，窗口 1'), raw: { text: 'The cat was very happy. The dog was very happy. The cat was sad. The dog was sad.', window: 1 } }]
};

export default lesson;
