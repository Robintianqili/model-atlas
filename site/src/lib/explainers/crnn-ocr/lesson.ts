// crnn-ocr: the guided lesson (11 steps). Every number is a {placeholder}: live values from the trace of the learner's
// writing, the decode process (= its Python reference) and ctc.ts, measured ones from facts.json ({fact:...}).
//   1 the strip -> 2 conv feature maps -> 3 columns -> 4 LSTM left to right -> 5 both directions -> 6 chances per column
//   (predict: the blank) -> 7 greedy decoding (play) -> 8 many paths, one text (interactive) -> 9 the CTC loss ->
//   10 task: make it misread -> 11 limits: letters and messier writing.
// Claims that are not computed live are tested in claims.test.ts.
import type { At, ExplainerRuntime, I18n, LessonDef, Tensor } from '../../atlas/types/index.ts';
import { plural } from '../../atlas/lesson/say.ts';
import { layerStat } from '../../atlas/lesson/facts.ts';
import { unflatten } from '../../atlas/runtime/selection.ts';
import { fmt3, fmtPct } from '../../atlas/core/format.ts';
import { bestPath, collapse, forward, probsOf, textOf } from './ctc.ts';
import { stripOf, variant } from './views/ctc/data.ts';
import { lossOf } from './process.ts';
import { textOf as wroteOf } from './adapters/word-strokes/adapter.ts';

type RT = ExplainerRuntime;
const tr = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
const dim = (rt: RT, id: string, i: number) => Number(tr(rt, id)?.shape[i] ?? NaN);
const strip = (rt: RT) => stripOf({ rt, at: { index: 0 }, props: {} });
/** a text in quotes, or "nothing" when every column is blank */
export const show = (s: string | undefined): I18n => (s ? { en: `“${s}”`, zh: `“${s}”` } : { en: 'nothing (every column is blank)', zh: '什么也没有（每一列都是空白）' });
const m = (at: At, k: string) => Number(at.frame?.metrics?.[k] ?? NaN);
/** a chance with two decimals (99.98%, where {x|pct} would print 100.0%) */
const pct2 = (v: number) => (Number.isFinite(v) ? fmtPct(v, 2) : NaN);
/** sizes of the network, read from the trace */
const dims = (rt: RT) => ({
	h: dim(rt, 'input', 1), w: dim(rt, 'input', 2), k1: dim(rt, 'conv1', 0), k2: dim(rt, 'conv2', 0), ph: dim(rt, 'pool1', 1), pw: dim(rt, 'pool1', 2),
	T: dim(rt, 'cols', 0), f: dim(rt, 'cols', 1), hid: dim(rt, 'fwd', 1), d: dim(rt, 'both', 1), V: dim(rt, 'probs', 1), digits: dim(rt, 'probs', 1) - 1
});
/** step 9: the CTC loss of the text the learner meant (when it has a path), else of the text read (process.ts lossOf) */
const loss = (rt: RT, at: At) => {
	const mt = (at.frame?.metrics ?? {}) as Record<string, number>;
	const l = lossOf(mt);
	return { want: show(Number.isFinite(mt.logTarget) ? wroteOf(rt.input) : (at.state as { text?: string } | undefined)?.text), paths: l.paths, loss: sci(l.loss) };
};
/** a small loss as the formula on the stage prints it (1.71 × 10⁻⁴, not 1.71e−4); other numbers as fmt3 */
const sci = (v: number) => (Number.isFinite(v) ? fmt3(v).replace(/e(−?)(\d+)$/, (_, s, d: string) => ` × 10${s ? '⁻' : ''}${[...d].map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('')}`) : NaN);
const DECODE = { source: 'decode', at: 'last', provenance: 'live' } as const;
const ctcView = (mode: string, extra: Record<string, unknown> = {}) => ({ key: 'crnn-ocr/ctc', props: () => ({ mode, ...extra }) });

const lesson: LessonDef = {
	title: { en: 'Reading handwriting with a CRNN and CTC', zh: '用 CRNN 和 CTC 读手写字' },
	summary: {
		en: 'Follow a tiny CRNN as it reads your handwritten number: the strip of pixels, convolution features cut into columns, two LSTMs reading them both ways, a symbol for every column, and CTC turning the columns into text.',
		zh: '跟着一个微型 CRNN 读你手写的数：像素长条、切成一列列的卷积特征、从两头读这些列的两个 LSTM、每一列的符号，以及 CTC 怎样把这些列变成文字。'
	},
	fixtures: [{ id: 'messy-1100', label: { en: '1100 with an extra stroke', zh: '1100 多一笔' }, raw: { strokes: [{ points: [{ x: 0.1, y: 0.5 }, { x: 0.3, y: 0.45 }] }, { points: [{ x: 0.5, y: 0.3 }, { x: 0.5, y: 0.7 }] }], text: '1100' } }],
	steps: [
		{
			id: 'strip',
			source: 'fwd',
			at: { layer: 'input' },
			view: 'layer.tensor',
			provenance: 'live',
			title: { en: 'Your writing becomes a strip of numbers', zh: '你写的字变成一长条数' },
			narration: {
				en: 'OCR (optical character recognition) turns writing into text. This model never sees your pen strokes: they are drawn again into a strip of {h} rows by {w} columns of pixels, {n} numbers from 0 (paper) to 1 (full ink). Your ink touches {cells} of them.',
				zh: 'OCR（光学字符识别）把写下的字读成计算机里的文本。这个模型看不到你的笔画：笔画会被重新画进一个 {h} 行、{w} 列像素的长条，共 {n} 个数，0 表示白纸，1 表示满墨。其中有 {cells} 个沾到了你的墨迹。',
				vars: (rt) => ({ ...dims(rt), n: tr(rt, 'input')?.data.length ?? NaN, cells: tr(rt, 'input')?.data.filter((v) => v > 0).length ?? NaN })
			},
			look: { en: 'The strip is much wider than it is tall: it holds the whole number, to be read from left to right.', zh: '这个长条宽远大于高：它装下整个数，要从左往右读。' },
			terms: ['ocr', 'tensor']
		},
		{
			id: 'features',
			source: 'fwd',
			at: { layer: 'relu2' },
			view: 'layer.tensor',
			provenance: 'live',
			title: { en: 'Convolutions find pieces of strokes', zh: '卷积找出笔画的片段' },
			narration: {
				en: 'Two convolution layers look for small patterns. {k1} kernels of {kk} × {kk} pixels slide over the strip, max pooling halves it to {ph} × {pw}, and {k2} more kernels combine what the first ones found. The result is {k2} feature maps; each lights up where its learned pattern appears in your writing.',
				zh: '两层卷积寻找小图案。{k1} 个 {kk} × {kk} 像素的卷积核在长条上滑动，最大池化把高和宽各减半，变成 {ph} × {pw}，再由 {k2} 个卷积核把前面找到的东西组合起来。结果是 {k2} 张特征图；你的字里哪里出现某张图学到的图案，那张图在哪里就亮起来。',
				vars: (rt) => ({ ...dims(rt), kk: Number(rt.model?.weights['conv1.weight']?.shape[2] ?? NaN) })
			},
			// open the stage at the cell of the largest value: layer.tensor's own default channel (largest sum) can differ
			focus: (rt) => (tr(rt, 'relu2') ? [{ field: 'relu2', index: unflatten(tr(rt, 'relu2')!.shape, Number(layerStat(tr(rt, 'relu2')!, 'argmax'))), role: 'accent' }] : []),
			look: { en: 'The enlarged map is channel {layer.relu2.strongestChannel} (counting from 0): it holds the strongest response of all, {layer.relu2.max}, in the marked cell.', zh: '放大的是通道 {layer.relu2.strongestChannel}（从 0 数起）：所有反应里最强的 {layer.relu2.max} 就在它被标出的格子里。' },
			terms: ['convolution', 'kernel', 'pooling', 'feature-map']
		},
		{
			id: 'columns',
			source: 'fwd',
			at: { layer: 'cols' },
			view: 'layer.tensor',
			provenance: 'live',
			title: { en: 'The picture becomes a row of columns', zh: '图被切成一列一列' },
			narration: {
				en: 'Now the picture becomes a sequence. The {k2} maps are cut into {T} columns, left to right, and each column is laid out as one list of {f} numbers ({k2} maps × {ph} rows). A column stands for a slice {px} pixels wide of your strip, though through the kernels its numbers saw {rf} pixels across.',
				zh: '现在图变成一个序列。{k2} 张特征图从左到右切成 {T} 列，每一列摊平成一串 {f} 个数（{k2} 张图 × {ph} 行）。一列对应你长条上 {px} 像素宽的一窄条，不过经过两层卷积核，它的数其实看到了 {rf} 像素宽的范围。',
				vars: (rt) => {
					const d = dims(rt);
					const k = Number(rt.model?.weights['conv1.weight']?.shape[2] ?? NaN);
					return { ...d, px: d.w / d.T, rf: k + 1 + (k - 1) * (d.w / d.pw) };
				}
			},
			look: { en: 'Each row of the grid is one column of your writing; row 0 is its leftmost column.', zh: '网格的每一行是你写的字的一列；第 0 行（从 0 数起）是最左边的一列。' },
			terms: ['feature-map', 'flatten']
		},
		{
			id: 'forward',
			source: 'fwd',
			at: { layer: 'fwd' },
			view: 'layer.tensor',
			provenance: 'live',
			title: { en: 'An LSTM reads the columns, left to right', zh: '一个 LSTM 从左往右读这些列' },
			narration: {
				en: 'An LSTM reads the {T} columns one at a time, left to right, and carries a memory of {hid} numbers from each column to the next. So its state at a column sums up that column and everything to its left, but nothing to its right: at column {mid} (counting from 0) it has not yet seen the {right} columns after it.',
				zh: '一个 LSTM 从左往右逐列读这 {T} 列，每读完一列，就把一份 {hid} 个数的记忆带到下一列。所以它在某一列的状态概括了这一列和它左边的全部内容，却对右边一无所知：在第 {mid} 列（从 0 数起），它还没看到后面的 {right} 列。',
				vars: (rt) => {
					const d = dims(rt);
					const mid = Math.floor(d.T / 2);
					return { ...d, mid, right: d.T - 1 - mid };
				}
			},
			look: { en: 'Each row is the hidden state h after one more column.', zh: '每一行是多读一列之后的隐藏状态 h。' },
			terms: ['lstm', 'recurrent-network', 'hidden-state']
		},
		{
			id: 'both',
			source: 'fwd',
			at: { layer: 'both' },
			view: 'layer.tensor',
			provenance: 'live',
			title: { en: 'A second LSTM reads right to left', zh: '第二个 LSTM 从右往左读' },
			narration: {
				en: 'A second LSTM reads the same columns right to left: the columns are put in reverse order, read, and its states reversed back, so row t again belongs to column t. Joined, the two give every column {d} numbers: {hid} about what lies to its left and {hid} about what lies to its right. This pair is a bidirectional LSTM.',
				zh: '第二个 LSTM 从右往左读同样的列：先把列倒过来排，读完再把它的状态倒回来，这样第 t 行又对应第 t 列。两者拼在一起，每一列有 {d} 个数：{hid} 个说它左边有什么，{hid} 个说它右边有什么。这一对就是双向 LSTM。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'The left half of each row comes from the left-to-right LSTM, the right half from the right-to-left one.', zh: '每一行的左半来自从左往右的 LSTM，右半来自从右往左的那个。' },
			terms: ['bidirectional-rnn', 'concatenation']
		},
		{
			id: 'chances',
			source: 'fwd',
			at: { layer: 'probs' },
			view: ctcView('chances'),
			provenance: 'live',
			title: { en: 'Every column scores every symbol', zh: '每一列给每个符号打分' },
			narration: {
				en: 'A linear layer and a softmax turn each column’s {d} numbers into chances for {V} symbols: the {digits} digits and one extra symbol, the blank, written –. In {nBlank} of your {T} columns the blank is the top symbol; over all columns of the test words it is {fact:blankShare|pct}.',
				zh: '一个线性层加 softmax 把每一列的 {d} 个数变成 {V} 个符号的概率：{digits} 个数字，外加一个特殊符号——空白，写作 –。你的 {T} 列中有 {nBlank} 列排第一的是空白；在测试词的全部列里，这个比例是 {fact:blankShare|pct}。',
				vars: (rt) => {
					const q = probsOf(tr(rt, 'probs'));
					return { ...dims(rt), nBlank: q ? bestPath(q).filter((k) => k === 0).length : NaN };
				}
			},
			look: { en: 'Each column of the grid sits under its slice of the strip; its most strongly colored cell is its top symbol.', zh: '网格的每一列都在长条上对应的那一窄条下面；颜色最浓的格子就是这一列排第一的符号。' },
			predict: {
				id: 'crnn-ocr-blank',
				question: { en: 'What is the blank for?', zh: '空白符号是做什么用的？' },
				options: [
					{ en: 'It marks the gap between two numbers, like a space between words', zh: '它标出两个数之间的间隔，就像词与词之间的空格' },
					{ en: 'It means “no new character here”, so doubled digits can stay apart', zh: '它表示“这里没有新字符”，让重复的数字能分开' },
					{ en: 'It marks a column the model could not read, so that column is skipped', zh: '它标出模型读不懂的列，好让这一列被跳过' }
				],
				answer: 1,
				explain: {
					en: 'The blank is CTC’s “nothing new here”. Columns around and between digits pick it, and it is thrown away when the columns become text. It also keeps doubled digits apart: the path 1 1 reads as one 1, while 1 – 1 reads as 11.',
					zh: '空白是 CTC 的“这里没有新东西”。数字周围和数字之间的列选它，列变成文字时它会被丢掉。它还能把重复的数字分开：路径 1 1 读成一个 1，而 1 – 1 读成 11。'
				}
			},
			terms: ['softmax', 'ctc-blank']
		},
		{
			id: 'decode',
			...DECODE,
			view: ctcView('decode'),
			play: { from: 0, to: 'last' },
			title: { en: 'Greedy decoding: merge repeats, drop blanks', zh: '贪心解码：合并重复，去掉空白' },
			narration: {
				en: 'Greedy decoding takes the top symbol of each column, left to right, and CTC’s rule turns them into text: first merge repeats, then drop blanks. Of your {T} columns, {blanks} hold a blank and {repeats} repeat the symbol before them; what is left reads {text}.',
				zh: '贪心解码从左往右取每一列排第一的符号，再用 CTC 的规则把它们变成文字：先合并重复，再去掉空白。你的 {T} 列里，{blanks} 列是空白，{repeats} 列重复了前一列的符号；剩下的读作{text}。',
				vars: (rt, at) => {
					const p = Array.from((at.state as { path?: Tensor } | undefined)?.path?.data ?? []);
					return {
						T: m(at, 'T'),
						blanks: p.length ? p.filter((k) => k === 0).length : NaN,
						repeats: p.length ? p.filter((k, t) => k > 0 && t > 0 && p[t - 1] === k).length : NaN,
						text: show((at.state as { text?: string } | undefined)?.text)
					};
				}
			},
			look: { en: 'Press Play to read the columns one at a time: = marks a merged repeat, · a dropped blank.', zh: '按“播放”一列一列地读：= 表示合并掉的重复，· 表示去掉的空白。' },
			terms: ['ctc', 'greedy-decoding']
		},
		{
			id: 'paths',
			...DECODE,
			view: ctcView('paths'),
			title: { en: 'Many paths, one text', zh: '许多条路径，同一段文字' },
			narration: {
				en: 'The row of top symbols is only one path through the columns, and many paths spell the same text: a digit can start a column earlier, or more blanks can sit between digits. Your path has a chance of {pPath}, the {T} top chances multiplied; the chances of {allPaths} spelling {text} add up to {pText}, so the other paths bring {extra}.',
				zh: '排第一的符号连成的那一行只是穿过这些列的一条路径，许多条路径会拼出同一段文字：一个数字可以早一列开始，数字之间也可以多几个空白。你这条路径的概率是 {pPath}，也就是 {T} 列各自最高概率的乘积；拼出{text}的{allPaths}概率加起来是 {pText}，也就是说其余路径又添了 {extra}。',
				vars: (rt, at) => {
					const [pPath, pText] = [Math.exp(m(at, 'logPath')), Math.exp(m(at, 'logText'))];
					const allPaths = plural(m(at, 'paths'), { one: 'the only path', many: 'all {n} paths', zh: '全部 {n} 条路径' }, 'int');
					return { T: m(at, 'T'), pPath: pct2(pPath), allPaths, pText: pct2(pText), extra: pct2(pText - pPath), text: show(strip(rt)?.text) };
				}
			},
			look: { en: 'Click a column in the stage and give it another symbol: the path’s chance changes at once, and so do the text and its total when the change spells another text.', zh: '在舞台上点一列，给它换一个符号：这条路径的概率立刻改变；如果改完拼出的是另一段文字，文字和它的总概率也跟着变。' },
			terms: ['ctc-path']
		},
		{
			id: 'loss',
			...DECODE,
			view: 'zoom.formula',
			title: { en: 'Training adds up every path', zh: '训练把所有路径加起来' },
			narration: {
				en: 'That sum is how CTC trains without anyone marking where each digit sits. The CTC loss of a word is −log (the natural log) of the total chance of its right text over all its paths; for {want}, summed over {paths} paths, it is {loss} here. Training lowered it on {fact:trainWords} synthetic words, and the model now reads {fact:testAcc|pct} of {fact:testWords} neatly written test words exactly.',
				zh: '正是靠这个总和，CTC 训练时不需要任何人标出每个数字在哪里。一个词的 CTC 损失，是它正确文字在所有路径上的总概率取 −log（log 是自然对数）；这里{want}在 {paths} 条路径上加起来，损失是 {loss}。训练在 {fact:trainWords} 个合成词上不断降低这个损失，现在模型能把 {fact:testWords} 个写得工整的测试词中的 {fact:testAcc|pct} 一字不差地读对。',
				vars: (rt, at) => loss(rt, at)
			},
			look: { en: 'In the formula, π runs over every path that spells y = {want}.', zh: '公式里的 π 走遍所有拼出 y = {want} 的路径。', vars: (rt, at) => loss(rt, at) },
			terms: ['loss', 'ctc']
		},
		{
			id: 'task',
			...DECODE,
			view: ctcView('decode'),
			title: { en: 'Your turn: make it misread', zh: '轮到你：让它读错' },
			narration: {
				en: 'Under “Your input”, press “Change it here”, write a number on the pad and type the digits you meant in the box; keep trying until what the model reads differs from what you wrote. Right now it reads {text}{vs}.',
				zh: '在“你的输入”下按“在这里修改”，在画板上写一个数，再在框里输入你想写的数字；一直试到模型读出的和你写的不一样为止。现在它读作{text}{vs}。',
				vars: (rt) => {
					const w = wroteOf(rt.input);
					return { text: show(strip(rt)?.text), vs: w ? { en: `, and you meant “${w}”`, zh: `，而你想写的是“${w}”` } : '' };
				}
			},
			look: { en: 'Squeezed or touching digits, a heavy slant or an unusual shape are worth a try.', zh: '挤在一起或连在一起的数字、很斜的写法或少见的字形都值得试试。' },
			task: {
				prompt: { en: 'Write a number the model reads wrong, and type the digits you meant.', zh: '写一个模型会读错的数，并输入你想写的数字。' },
				done: (rt) => {
					const w = wroteOf(rt.input);
					const s = strip(rt);
					return !!w && !!s && s.text !== w;
				},
				waiting: { en: 'Not yet: the model reads what you meant (or the box is empty).', zh: '还没有：模型读出的就是你想写的（或者框是空的）。' },
				doneText: { en: 'Done: it misread your writing. The stage shows the columns that led it astray.', zh: '完成：它读错了你写的字。舞台上是把它带偏的那些列。' }
			},
			terms: ['ctc']
		},
		{
			id: 'limits',
			source: 'fwd',
			at: { layer: 'probs' },
			view: ctcView('decode', { variant: 'hex' }),
			provenance: 'live',
			kindOf: 'limits',
			title: { en: 'What it cannot read', zh: '它读不了什么' },
			narration: {
				en: 'This model knows {digits} digits and the blank, nothing else. Letters were never in its training data, yet every column must pick one of its {V} symbols: the letters HEX come out as {hex}, a reading it gives a chance of {pHex|pct}. It also learned only from synthetic writing, and messiness costs it: it reads {fact:testAcc|pct} of the neatest test words, but only {fact:wildAcc|pct} of words as messy as the messiest it trained on.',
				zh: '这个模型只认识 {digits} 个数字和空白，别的都不认识。字母从没出现在它的训练数据里，可每一列都必须从 {V} 个符号里选一个：字母 HEX 被读成{hex}，它给这个读法的概率是 {pHex|pct}。它也只学过合成的字迹，而且字越潦草越吃力：最工整的测试词它能读对 {fact:testAcc|pct}，而和它训练时见过的最潦草的字一样潦草的词，只能读对 {fact:wildAcc|pct}。',
				vars: (rt) => {
					const v = variant(rt, 'hex');
					const q = probsOf(v?.probs);
					const ids = q ? collapse(bestPath(q)) : [];
					return { ...dims(rt), hex: q ? show(textOf(ids)) : '…', pHex: q ? Math.exp(forward(q, ids).logP) : NaN };
				}
			},
			look: { en: 'The stage shows the HEX preset, not your writing: every column still picks a digit or the blank.', zh: '舞台上是预设 HEX，不是你写的字：每一列仍然只能选一个数字或空白。' },
			terms: ['out-of-distribution']
		}
	]
};

export default lesson;
