// tinyclip: the guided lesson (design §7 outline, 10 steps). Every number is a {placeholder}: structure from the spec,
// live values from the trace of the learner's drawing and captions (fwd) or the batch (nce), measured ones from facts.json.
//   1 what it learned from (synth) → 2 image tower → 3 text tower → 4 length 1 → 5 one score (predict) → 6 the batch
//   table (predict) → 7 row and column softmax → 8 loss and gradient → 9 your captions as classes (task) → 10 word order
//   (breakIt). Claims that are not computed live are tested in claims.test.ts.
import type { At, ExplainerRuntime, I18n, LessonDef, Tensor } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { PRESETS, tokenize } from './scenes.ts';
import { captionsOf, rawOf, type ClipRaw } from './adapters/shapes-captions/adapter.ts';
import { captionsData, pct, vectorsData, wordsData } from './views/match/data.ts';

type RT = ExplainerRuntime;
const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
const I = (en: string, zh: string) => ({ en, zh });
const view = (mode: string, extra: Record<string, unknown> = {}) => ({ key: 'tinyclip/match', props: () => ({ mode, ...extra }) });
/** structural numbers from the spec (never typed into the text) */
function dims(rt: RT) {
	const L = (id: string) => rt.spec?.layers.find((l) => l.id === id)?.outShape ?? [];
	return { d: Number(L('img.proj')[0]), feat: Number(L('img.flat')[0]), w: Number(L('txt.emb')[2]), ch: Number(L('img.pool2')[0]), hw: Number(L('img.pool2')[1]) };
}
/** the image vector's number with the largest size (steps 2 and 4 follow it) */
function topUnit(rt: RT) {
	const y = Array.from(T(rt, 'img.proj')?.data ?? []), u = y.reduce((b, v, i) => (Math.abs(v) > Math.abs(y[b]) ? i : b), 0);
	return { u, y: y[u] ?? NaN };
}
const m = (at: At) => at.frame?.metrics ?? {};
/** metrics of a batch source's frame at a phase (nce: after training; nce0: with the starting weights) */
const after = (rt: RT, phase: string, src = 'nce'): Record<string, number> => {
	const tp = rt.sources[src]?.tape, i = tp ? tp.find({ phase }) : -1;
	return tp && i >= 0 ? (tp.frame(i).metrics ?? {}) : {};
};
/** the shares of row 0 (the learner's drawing) after training if the scale s were 1: a softmax of the cosines alone */
function row0AtScale1(rt: RT): number[] {
	const tp = rt.sources.nce?.tape, i = tp ? tp.find({ phase: 'scores' }) : -1;
	if (!tp || i < 0) return [NaN];
	const { N, scale } = tp.frame(i).metrics ?? {}, L = (tp.state(i) as { L: Tensor }).L.data;
	const e = Array.from({ length: N }, (_, j) => Math.exp(L[j] / scale)), sum = e.reduce((a, b) => a + b, 0);
	return e.map((x) => x / sum);
}
const BEFORE_AFTER = { a: 'nce0', b: 'nce', label: [I('With the starting weights', '用开始时的权重'), I('After training', '训练之后')] as [I18n, I18n] };
export const ORDER = ['a circle above a square', 'a square above a circle', 'a square below a circle'];
/** the presets' captions as the model reads them (word ids): the task needs at least one caption of the learner's own */
export const PRESET_CAPS = new Set(PRESETS.flatMap((f) => f.captions.map((c) => tokenize(c).ids.join(' '))));

const lesson: LessonDef = {
	title: I('TinyCLIP: matching drawings and captions', 'TinyCLIP：给画和描述配对'),
	summary: I('Follow a drawing and a caption through two towers to one score, then a whole batch of scores, the loss that trains them, your own captions, and the word order the model cannot see.', '跟着一幅画和一句描述穿过两座塔，得到一个分数；再看一整批的分数、训练它们的损失、你自己写的描述，以及模型看不到的词序。'),
	steps: [
		{
			id: 'pairs',
			source: 'synth', at: { phase: 'chunk' },
			view: view('pairs'), provenance: 'live',
			title: I('What TinyCLIP learned from', 'TinyCLIP 学习用的材料'),
			narration: {
				en: 'TinyCLIP learned from {fact:trainN|int} pairs like the {n} here: a drawing of one or two shapes and a short caption about it, both made by a small program. Nobody told it what a circle is; all it was trained on is which caption belongs to which drawing. All the captions together use just {fact:vocab} different words.',
				zh: 'TinyCLIP 从 {fact:trainN|int} 对这样的图文对中学习，这里是其中 {n} 对：一幅画着一个或两个形状的画，加上一句关于它的简短描述，都由一个小程序生成。没有人告诉它圆是什么；它训练时唯一知道的，是哪句描述属于哪幅画。所有描述加起来只用了 {fact:vocab} 个不同的词。',
				vars: (_rt, at) => ({ n: m(at).count })
			},
			look: I('Each caption describes its own drawing; some leave out the size or the place. The square frame is part of every drawing: it keeps a shape’s size and place when the drawing is scaled down to the image tower’s grid.', '每句描述都在说它自己那幅画；有的省略了大小或位置。每幅画外面的方框也是画的一部分：画被缩小到图像塔的网格时，它让形状保留自己的大小和位置。'),
			terms: ['contrastive-learning', 'vocabulary']
		},
		{
			id: 'image',
			source: 'fwd', at: { layer: 'img.proj' },
			select: (rt) => ({ layer: 'img.proj', unit: [topUnit(rt).u] }),
			view: 'zoom.weighted-sum', provenance: 'live',
			title: I('The image tower: your drawing becomes 32 numbers', '图像塔：你的画变成 32 个数'),
			narration: {
				en: 'TinyCLIP is two separate networks, called towers, that meet only at the end. The image tower is a small convolutional network like TinyCNN: two rounds of convolution, ReLU and max pooling turn your drawing into {ch} small maps of {hw}×{hw}, {feat} numbers, and a linear layer turns those into {d}, the image vector. Number {u} of it (counting from 0) is a weighted sum of all {feat}: the products add up to {sum}, and adding the bias, {b}, gives {y}.',
				zh: 'TinyCLIP 是两个分开的网络，叫作“塔”，它们只在最后才碰面。图像塔是一个和 TinyCNN 一样的小卷积网络：两轮卷积、ReLU 和最大池化把你的画变成 {ch} 张 {hw}×{hw} 的小图，共 {feat} 个数，再由一个线性层把它们变成 {d} 个数，叫作图像向量。它的第 {u} 个数（从 0 数起）是全部 {feat} 个数用学来的权重做的加权和：乘积加起来是 {sum}，再加上偏置 {b}，得到 {y}。',
				vars: (rt) => {
					const u = rt.sel.unit?.[0] ?? 0, y = T(rt, 'img.proj')?.data[u] ?? NaN, b = rt.model?.weights['img.proj.bias']?.data[u] ?? NaN;
					return { ...dims(rt), u, y, b, sum: y - b };
				}
			},
			look: I('Each small map holds one channel’s products: orange pushes this number up, blue pushes it down.', '每张小图是一个通道的乘积：橙色把这个数往上推，蓝色往下压。'),
			terms: ['convolution', 'pooling', 'dense-layer']
		},
		{
			id: 'caption',
			source: 'fwd', at: { layer: 'txt.proj' },
			view: view('words'), provenance: 'live',
			title: I('The text tower: a caption becomes 32 numbers', '文本塔：一句描述变成 32 个数'),
			narration: {
				en: '{first} Each word picks its own row of {w} learned numbers from a table with one row per known word (an embedding). The rows are averaged into one row, and a linear layer turns it into {d} numbers, the caption vector.',
				zh: '{first}每个词从一张表里取出自己的一行 {w} 个学来的数，表里每个认识的词占一行（这叫嵌入）。这些行被平均成一行，再由一个线性层变成 {d} 个数，叫作描述向量。',
				vars: (rt) => {
					const wd = wordsData(rt), nw = plural(wd.words.length, { one: '{n} word', many: '{n} words', zh: '{n} 个' });
					const first = !wd.words.length ? I('None of your captions has a word the model knows, so there are no rows to average: the average is all zeros, and the caption vector is just the linear layer’s bias.', '你的描述里没有一个模型认识的词，所以没有可以平均的行：平均是全 0，描述向量就只是线性层的偏置。')
						: typeof nw === 'object' ? I(`The text tower reads your first caption, “${wd.caption}”, which has ${nw.en} the model knows.`, `文本塔读你的第一句描述“${wd.caption}”，其中模型认识的词有 ${nw.zh}。`) : NaN;
					return { ...dims(rt), first };
				}
			},
			look: I('One strip per word, then their average, then the caption vector.', '每个词一条，然后是它们的平均，最后是描述向量。'),
			terms: ['embedding', 'vocabulary']
		},
		{
			id: 'normalize',
			source: 'fwd', at: { layer: 'txt.norm' },
			view: view('vectors'), provenance: 'live',
			title: I('Both vectors get length 1', '两个向量的长度都变成 1'),
			narration: {
				en: 'Think of each vector as an arrow in a space with {d} directions. Its length is the square root of the sum of its squared numbers: {lenA|n4} for the image vector and {lenB|n4} for the caption vector. Dividing every number by the length leaves both arrows with length 1, so only their directions can still differ: number {u} of the image vector, {y|n4} in the last step, becomes {y|n4} ÷ {lenA|n4} = {yn}.',
				zh: '把每个向量想成一个 {d} 维空间里的箭头。它的长度是各数平方和的平方根：图像向量是 {lenA|n4}，描述向量是 {lenB|n4}。把每个数都除以长度之后，两个箭头的长度都是 1，剩下能不同的只有方向：比如图像向量的第 {u} 个数，上一步里是 {y|n4}，现在变成 {y|n4} ÷ {lenA|n4} = {yn}。',
				vars: (rt) => {
					const v = vectorsData(rt), { u, y } = topUnit(rt);
					return { ...dims(rt), lenA: v.lenA, lenB: v.lenB, u, y, yn: v.an[u] ?? NaN };
				}
			},
			look: I('The strips keep their pattern; only the size of the numbers changes.', '这些条的图案不变，只是数的大小变了。')
		},
		{
			id: 'cosine',
			source: 'fwd', at: { layer: 'sim' },
			view: view('pair'), provenance: 'live',
			title: I('One score: how well two arrows agree', '一个分数：两个箭头有多一致'),
			predict: {
				id: 'tinyclip-cos-max',
				question: I('Both vectors now have length 1. Multiply them number by number and add up the products. What is the largest this sum can be?', '两个向量现在长度都是 1。把它们逐个数相乘，再把乘积加起来。这个和最大能是多少？'),
				options: [I('1, when both arrows point the same way', '1，当两个箭头方向相同时'), I('32, one for each pair of numbers', '32，每一对数贡献 1'), I('There is no limit', '没有上限')],
				answer: 0,
				explain: I('For two arrows of length 1 the sum of products is the cosine of the angle between them: 1 when they point the same way, 0 at a right angle, −1 when they point opposite ways. That is why it is called cosine similarity, and why a scale is needed to spread the scores apart.', '对两个长度为 1 的箭头，乘积之和就是它们夹角的余弦：方向相同时是 1，垂直时是 0，方向相反时是 −1。所以它叫余弦相似度，也正因为如此，需要一个缩放把分数拉开。')
			},
			narration: {
				en: 'To score your drawing against your first caption, TinyCLIP multiplies the two vectors number by number and adds up the {d} products: here they add up to {cos}. Then it multiplies by a scale s that was learned along with every other weight, s = {s}, which gives the score {score}.',
				zh: '为了给你的画和你的第一句描述打分，TinyCLIP 把两个向量逐个数相乘，再把 {d} 个乘积加起来：这里加起来是 {cos}。然后再乘以一个和其他权重一起学出来的缩放 s，s = {s}，得到分数 {score}。',
				vars: (rt) => {
					const v = vectorsData(rt);
					return { ...dims(rt), cos: v.cos, s: v.scale, score: v.score };
				}
			},
			look: I('The third strip holds the products; the score is their sum, times s.', '第三条是乘积；分数就是它们的和再乘以 s。'),
			terms: ['cosine-similarity', 'temperature']
		},
		{
			id: 'matrix',
			source: 'nce0', at: { phase: 'scores' },
			view: view('matrix', { show: 'L' }),
			compare: BEFORE_AFTER, provenance: 'live',
			title: I('A batch: every drawing against every caption', '一批：每幅画对每句描述'),
			predict: {
				id: 'tinyclip-collapse',
				question: I('Suppose training only raised the scores of the true pairs and ignored the other cells. What would be the easiest way for the model to do that?', '假设训练只提高正确的对的分数，而不管其他格子。模型最省事的做法是什么？'),
				options: [I('Learn what every shape looks like', '学会每种形状长什么样'), I('Give every drawing and every caption the same vector', '让每幅画和每句描述都得到同一个向量'), I('Make all the vectors shorter', '把所有向量都变短')],
				answer: 1,
				explain: I('Equal vectors make every cosine 1, so every score is the largest possible, s, for the wrong pairs too, and the table can no longer tell anything apart. So the loss also pushes the wrong pairs below the true ones: it only gets small when each true pair beats every wrong pair in its row and column. How high the scores are does not matter, only these gaps. Learning by such comparisons is called contrastive learning.', '向量都一样，每个余弦就都是 1，每个分数都是最大的 s，错误的对也一样，这张表就什么都分不出来了。所以损失还要把错误的对压到正确的对下面：只有当每个正确的对胜过它所在行和列中的每个错误的对时，损失才会变小。分数本身有多高不要紧，要紧的只是这些差距。靠这种比较来学习，就叫对比学习。')
			},
			narration: {
				en: 'Training never scores one pair alone: it takes a batch (here {N} pairs; yours is row and column 0, the other five are fixed) and scores every drawing against every caption, a {N}×{N} table with the {N} true pairs on its diagonal. With the starting weights the true pairs score {diag0} on average and the other {off} cells {off0}: they do not stand out. After training the true pairs score {diag1} and the others {off1}{rose}, so the true pairs lead by {gap1}.',
				zh: '训练从不单独给一对打分：它取一批图文对（这里有 {N} 对：你的是第 0 行、第 0 列，另外五对是固定的），让每幅画和每句描述都打一次分，得到一张 {N}×{N} 的表，{N} 个正确的对在对角线上。用开始时的权重，正确的对平均得 {diag0} 分，其余 {off} 个格子平均 {off0} 分：正确的对并不突出。训练之后，正确的对平均 {diag1} 分，其余的平均 {off1} 分{rose}，正确的对领先 {gap1} 分。',
				vars: (rt, at) => {
					const a = after(rt, 'scores');
					const rose = pick(a.offMean > m(at).offMean, I(' (the others rose too, but far less)', '（其余的也升高了，只是少得多）'), I('', ''));
					return { N: m(at).N, off: m(at).N * m(at).N - m(at).N, diag0: m(at).diagMean, off0: m(at).offMean, diag1: a.diagMean, off1: a.offMean, rose, gap1: a.diagMean - a.offMean };
				}
			},
			look: I('Rows are drawings (pictured on the left), columns are captions (numbered, listed below); the outlined diagonal is each drawing with its own caption.', '行是画（画在左边），列是描述（编号，列在下面）；加框的对角线是每幅画和它自己的描述。'),
			terms: ['batch', 'contrastive-learning']
		},
		{
			id: 'softmax',
			source: 'nce0', at: { phase: 'softmax' },
			view: view('matrix', { show: 'P' }),
			compare: BEFORE_AFTER, provenance: 'live',
			title: I('Softmax along rows, and along columns', '沿行做 softmax，也沿列做'),
			narration: {
				en: 'Softmax turns each row into shares that add up to 1: which caption fits that drawing; down each column, which drawing fits that caption. With the starting weights your caption gets {p0} of your drawing’s row and your drawing {q0} of your caption’s column; after training, {p1} and {q1}. The scale s = {s1} multiplies the cosines first and makes the shares sharper: with s = 1, the shares in your row would only range from {lo} to {hi}. CLIP calls 1/s the temperature.',
				zh: 'Softmax 把每一行变成加起来等于 1 的份额，说明哪句描述配这幅画；沿每一列做，则说明哪幅画配这句描述。用开始时的权重，你的描述在你的画那一行里占 {p0}，你的画在你的描述那一列里占 {q0}；训练之后，前者是 {p1}，后者是 {q1}。缩放 s = {s1} 先乘在余弦上，让份额更尖锐：如果 s = 1，你那一行的份额只会在 {lo} 到 {hi} 之间。CLIP 把 1/s 叫作温度。',
				vars: (rt, at) => {
					const a = after(rt, 'softmax');
					const r1 = row0AtScale1(rt);
					return { p0: pct(m(at).p00), q0: pct(m(at).q00), p1: pct(a.p00), q1: pct(a.q00), s1: after(rt, 'scores').scale, lo: pct(Math.min(...r1)), hi: pct(Math.max(...r1)) };
				}
			},
			look: I('Switch between rows and columns (both tables switch), and compare the diagonal before and after training.', '在行和列之间切换（两张表一起切换），比较训练前后的对角线。'),
			terms: ['softmax', 'temperature']
		},
		{
			id: 'gradient',
			source: 'nce0', at: { phase: 'grad' },
			view: view('matrix', { show: 'G' }),
			compare: BEFORE_AFTER, provenance: 'live',
			title: I('The loss, and how it pulls on every score', '损失，以及它怎样拉动每个分数'),
			narration: {
				en: 'The InfoNCE loss averages −log of the true pair’s share over every row and column (0 when every true pair gets the whole share). With the starting weights it is {loss0}, {cmp} an even guess (ln {N} = {chance}). Its gradient says which way each score should move to lower the loss: up for the true pairs, whose gradient is negative (yours: {g00}), down for the wrong ones, whose gradient is positive (up to {gOff}). These flow back through both towers into all {fact:params|int} weights; after training the loss is {loss1}.',
				zh: 'InfoNCE 损失把每一行、每一列中正确那一对份额的 −log 求平均（每个正确的对都拿到全部份额时是 0）。用开始时的权重，它是 {loss0}，{cmp}平均乱猜（ln {N} = {chance}）。它的梯度说明每个分数该往哪个方向移动才能降低损失：正确的对往上，它们的梯度是负的（你的：{g00}）；错误的对往下，它们的梯度是正的（最大到 {gOff}）。这些数沿两座塔往回传，进入全部 {fact:params|int} 个权重；训练之后，这里的损失是 {loss1}。',
				vars: (rt, at) => {
					const s0 = after(rt, 'softmax', 'nce0');
					return { loss0: s0.loss, cmp: pick(s0.loss > s0.chance, I('worse than', '还不如'), I('better than', '好于')), N: after(rt, 'scores', 'nce0').N, chance: s0.chance, g00: m(at).g00, gOff: m(at).gOffMax, loss1: after(rt, 'softmax').loss };
				}
			},
			look: { en: 'Blue cells (negative) are pushed up, orange cells (positive) pushed down; {left}.', zh: '蓝色格子（负的）往上推，橙色格子（正的）往下压；{left}。',
				vars: (rt) => ({ left: pick(after(rt, 'softmax').loss < 0.01, I('after training almost nothing is left to pull', '训练之后几乎没什么可拉的了'), I('after training some cells still pull, because this batch is not solved', '训练之后还有格子在拉，因为这一批还没有解决')) })
			},
			terms: ['infonce', 'gradient', 'loss']
		},
		{
			id: 'captions',
			source: 'fwd', at: { layer: 'probs' },
			view: view('captions'), provenance: 'live',
			title: I('Your turn: captions as classes', '轮到你：把描述当作类别'),
			narration: {
				en: 'Score one drawing against several captions and take a softmax, and the captions become the classes of a classifier that was never trained on them as classes (zero-shot). {verdict} On the {fact:oneN} one-shape test drawings, the five captions “a circle” to “a star” pick the right shape {fact:shapeAcc|pct} of the time, and captions for the place pick the right place {fact:placeAcc|pct} of the time.',
				zh: '让一幅画和几句描述分别打分，再做 softmax，这些描述就成了一个分类器的类别，而这个分类器从没把它们当作类别训练过（零样本）。{verdict}在 {fact:oneN} 幅只有一个形状的测试画上，“a circle”到“a star”这五句描述选对形状的比例是 {fact:shapeAcc|pct}，描述位置的句子选对位置的比例是 {fact:placeAcc|pct}。',
				vars: (rt) => {
					const rows = captionsData(rt);
					const b = rows.reduce((x, r, k) => (r.share > rows[x].share ? k : x), 0);
					const q = rows[b]?.text ?? '', p = pct(rows[b]?.share ?? NaN);
					if (typeof p !== 'object') return { verdict: NaN };
					const verdict = rows.length > 1 ? I(`Of your ${rows.length} captions, “${q}” wins with ${p.en}.`, `在你的 ${rows.length} 句描述里，“${q}”胜出，份额是 ${p.zh}。`)
						: q ? I(`With a single caption, “${q}”, there is nothing to compare: the softmax gives it the whole share, so write more captions.`, `只有一句描述“${q}”时没什么可比的：softmax 把全部份额都给它，所以多写几句吧。`)
						: I('Without a caption that has a known word there is nothing to compare: the one empty caption gets the whole share.', '没有含认识的词的描述，就没什么可比的：那一句空描述拿到全部份额。');
					return { verdict };
				}
			},
			look: I('The longest bar is the caption the model matches best with your drawing.', '最长的条，就是模型认为和你的画最相配的描述。'),
			task: {
				prompt: I('Write three different captions for your drawing: one true, one with the wrong size or place, one with the wrong shape. Which one wins, and by how much?', '给你的画写三句不同的描述：一句正确的，一句大小或位置写错的，一句形状写错的。哪一句胜出，领先多少？'),
				done: (rt) => {
					const c = captionsOf(rt.input).map((x) => x.ids.join(' '));
					return !(rt.input as ClipRaw | null)?.preset && c.length === 3 && new Set(c).size === 3 && c.some((k) => !PRESET_CAPS.has(k));
				},
				waiting: I('Not yet: three different captions, at least one written by you.', '还没有：要三句不同的描述，至少一句是你自己写的。'),
				hint: I('Type into the caption boxes; the model knows only the words listed under them.', '在描述框里输入；模型只认识框下面列出的词。')
			},
			terms: ['zero-shot', 'classifier']
		},
		{
			id: 'order',
			source: 'fwd', at: { layer: 'probs' },
			view: view('captions', { texts: ORDER }), provenance: 'live',
			kindOf: 'breakIt',
			title: I('Where it breaks: word order', '它在哪里出错：词序'),
			narration: {
				en: 'The text tower averages word rows, and an average ignores order: “{t0}” and “{t1}” have the same words, so they get the same vector and the same score with any drawing (here {s0} and {s1}). “{t2}” means the same as the first, but it has “below” where the first has “above”, so its words differ and it scores {s2}. On all {fact:twoN} two-shape test drawings, swapping the shapes in the true caption left its score unchanged; a text tower that reads order, such as CLIP’s Transformer, could tell them apart.',
				zh: '文本塔把词行求平均，而平均不管顺序：“{t0}”和“{t1}”用的词完全一样，所以它们得到同一个向量，和任何画的分数也都一样（这里是 {s0} 和 {s1}）。“{t2}”和第一句意思相同，但第一句用“above”的地方它用了“below”，词不一样了，得分是 {s2}。在全部 {fact:twoN} 幅有两个形状的测试画上，把正确描述里的两个形状对调，得分都没有变；能读出词序的文本塔，比如 CLIP 的 Transformer，就能把它们分开。',
				vars: (rt) => {
					const r = captionsData(rt, ORDER);
					return { t0: ORDER[0], t1: ORDER[1], t2: ORDER[2], s0: r[0].score, s1: r[1].score, s2: r[2].score };
				}
			},
			look: I('The first two bars are exactly equal, whatever you draw: on the circle-above-square drawing the false caption scores as high as the true one.', '前两条完全一样长，不管你画什么：在“圆在正方形上面”这幅画上，错误的描述和正确的描述得分一样高。'),
			terms: ['bag-of-words', 'transformer']
		}
	],
	fixtures: PRESETS.map((f) => ({ id: f.id, label: f.label, raw: rawOf(f) }))
};

export default lesson;
