// llava-toy: the guided lesson (11 steps). Every number is a {placeholder}: structure from the spec (dims()), live values
// from the trace of the learner's drawing and question, measured ones from facts.json ({fact:...}).
//   1 two inputs → 2 the frozen TinyViT (16 image features) → 3 the projector (image tokens) → 4 the word rows →
//   5 one sequence → 6 the causal mask → 7 where the answer row looks (predict) → 8 its numbers → 9 the answer →
//   10 your own question (task) → 11 a tiny vocabulary (limits). Claims that are not computed live: claims.test.ts.
import type { ExplainerRuntime, LessonDef, Tensor } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { HOUSE, N_IMG, VOCAB, rowNames } from './adapters/drawing-question/adapter.ts';
import { gridOf, inkedPatches } from './views/patch-grid/grid.ts';

type RT = ExplainerRuntime;
const T = (rt: RT, id: string): Tensor | undefined => rt.sources.fwd?.trace?.t[id];
const argmax = (a: ArrayLike<number>, from = 0, to = a.length) => {
	let b = from;
	for (let i = from + 1; i < to; i++) if (a[i] > a[b]) b = i;
	return b;
};
const PATCHES = Array.from({ length: N_IMG }, (_, k) => `▦${k}`);
const VIT_ROWS = ['[CLS]', ...PATCHES];

/** Structural numbers, read from the spec and the input (never typed into the text). */
export function dims(rt: RT) {
	const L = (id: string) => rt.spec?.layers.find((l) => l.id === id);
	const ids = rt.pre?.tensors.text?.data;
	const rows = ids ? ids.length : NaN;
	const V = Number(L('lm.head')?.outShape[1] ?? NaN);
	const k = Number((L('vit.patch')?.params as { kernel?: number } | undefined)?.kernel ?? NaN);
	return {
		n: N_IMG, k, rows, L: rows - N_IMG, V, V1: V - 1,
		dv: Number(L('feat')?.outShape[1] ?? NaN), dh: Number(L('proj.fc1')?.outShape[1] ?? NaN), dl: Number(L('proj.fc2')?.outShape[1] ?? NaN),
		vb: Number((rt.spec as { vitBlocks?: number } | null)?.vitBlocks ?? NaN), vbAll: rt.num('fact:vitBlocksAll'),
		heads: Number((L('lm.L1.scores')?.params as { nHead?: number } | undefined)?.nHead ?? NaN),
		C: VOCAB.length - VOCAB.indexOf('apple'), others: VOCAB.length - VOCAB.indexOf('apple') - 1
	};
}
/** the words the model reads, as text */
const words = (rt: RT) => (rt.pre?.tokens ?? []).slice(N_IMG);
const qText = (rt: RT) => words(rt).join(' ');
/** the answer of the last row of a [T, V] share table: word, share, id */
function answerOf(p: Tensor | undefined) {
	if (!p) return { label: '…', p: NaN, id: -1 };
	const [Tn, V] = p.shape;
	const row = p.data.subarray((Tn - 1) * V, Tn * V);
	const id = argmax(row);
	return { label: VOCAB[id], p: row[id], id };
}
const grid = (rt: RT, block: number) => gridOf({ rt, source: 'fwd', at: { index: 0 }, props: { block } });
const inked = (rt: RT) => {
	const x = T(rt, 'image');
	return x ? inkedPatches(x.data).filter(Boolean).length : NaN;
};
/** mean over heads of block 1's weight from row q to the image tokens */
function imgShareOfRow(rt: RT, q: number) {
	const a = T(rt, 'lm.L1.attn');
	if (!a) return NaN;
	const [H, Tq, Tk] = a.shape;
	let s = 0;
	for (let h = 0; h < H; h++) for (let k = 0; k < N_IMG; k++) s += a.data[(h * Tq + q) * Tk + k];
	return s / H;
}
const pct = (v: number) => (Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : '…');

const lesson: LessonDef = {
	title: { en: 'llava-toy: glue an image encoder to a language model', zh: 'llava-toy：把图像编码器接到语言模型上' },
	summary: {
		en: 'Follow your drawing through a frozen TinyViT and a small projector into the sequence of a tiny language model, see where the answer row looks, then ask your own question.',
		zh: '跟着你的画穿过冻结的 TinyViT 和一个小投影层，进入一个很小的语言模型的序列，看写答案的那一行看向哪里，再问你自己的问题。'
	},
	steps: [
		{
			auto: 'input',
			override: {
				title: { en: 'Two inputs: a drawing and a question', zh: '两个输入：一幅画和一个问题' },
				narration: {
					en: 'llava-toy gets two inputs. Your drawing becomes {px|int} numbers in a {side}×{side} grid, 0 for paper and 1 for ink, exactly what TinyViT reads on its own page. Your question “{q}” becomes {ids}, {idList}, from a vocabulary of {V1} words; any other word is skipped.',
					zh: 'llava-toy 有两个输入。你的画变成 {side}×{side} 网格里的 {px|int} 个数，0 是纸，1 是墨，和 TinyViT 在它自己的页面上读到的完全一样。你的问题“{q}”变成 {ids}：{idList}，这些词来自一个 {V1} 个词的词表；别的词都会被跳过。',
					vars: (rt) => ({ ...dims(rt), px: T(rt, 'image')?.data.length ?? NaN, side: T(rt, 'image')?.shape[2] ?? NaN, q: qText(rt), idList: Array.from(rt.pre?.tensors.text?.data ?? []).slice(N_IMG).join(' '), ids: plural(dims(rt).L, { one: '{n} word id', many: '{n} word ids', zh: '{n} 个词的 id' }) })
				},
				look: { en: 'The bright cells are your ink: these {px|int} numbers are all TinyViT gets.', zh: '亮的格子是你的墨迹：TinyViT 拿到的只有这 {px|int} 个数。', vars: (rt) => ({ px: T(rt, 'image')?.data.length ?? NaN }) }
			},
			terms: ['tensor', 'token', 'vocabulary']
		},
		{
			id: 'vit',
			source: 'fwd',
			at: { layer: 'feat' },
			view: { key: 'layer.tensor', props: () => ({ tokens: PATCHES }) },
			provenance: 'live',
			title: { en: 'A frozen TinyViT reads the drawing', zh: '冻结的 TinyViT 读这幅画' },
			narration: {
				en: 'TinyViT cuts the drawing into {n} patches of {k}×{k} pixels, puts its learned summary row, [CLS], in front and runs {vb} of its {vbAll} blocks. The {fact:vitRead|int} weights of these parts are frozen: trained earlier to name drawings, they never changed while llava-toy learned. As in LLaVA, the real vision-language model this toy copies, it is read one block before the end and only the patch rows are kept: {n} image features of {dv} numbers.',
				zh: 'TinyViT 把画切成 {n} 个 {k}×{k} 像素的图块，在前面放上它学来的汇总行 [CLS]，再运行它 {vbAll} 个块中的 {vb} 个。这些部分的 {fact:vitRead|int} 个权重是冻结的：它们早先为了给画命名而训练好，在 llava-toy 学习时从没改变。和这个玩具模仿的真实视觉语言模型 LLaVA 一样，这里在最后一块之前读出，只保留图块行：{n} 个图像特征，每个 {dv} 个数。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'Row ▦k is patch k of the drawing, counted from 0 row by row from the top left.', zh: '▦k 行就是画中编号 k 的图块：从左上角开始逐行编号，从 0 数起。' },
			terms: ['frozen-layer', 'image-patch', 'feature-extraction', 'vision-language-model']
		},
		{
			id: 'proj',
			source: 'fwd',
			at: { layer: 'proj.fc2' },
			select: (rt) => {
				const y = T(rt, 'proj.fc2');
				const k = grid(rt, 1)?.top ?? 0;
				if (!y) return { layer: 'proj.fc2', unit: [k, 0] };
				const D = y.shape[1];
				return { layer: 'proj.fc2', unit: [k, argmax(Array.from(y.data.subarray(k * D, (k + 1) * D), Math.abs))] };
			},
			view: { key: 'zoom.weighted-sum', props: () => ({ tokens: PATCHES }) },
			provenance: 'live',
			title: { en: 'The projector turns patches into image tokens', zh: '投影层把图块变成图像词元' },
			narration: {
				en: 'The language model’s rows are {dl} numbers wide; TinyViT’s are {dv}. The projector, the glue, is a small two-layer network applied to each patch row: {dv} → {dh}, GELU, {dh} → {dl}. Number {c} of image token ▦{p} (counting from 0) is a weighted sum of the {dh} numbers this patch has after the first layer and GELU: {dot} plus the bias {b} gives {y}.',
				zh: '语言模型的每一行有 {dl} 个数，TinyViT 的每一行只有 {dv} 个。投影层就是“胶水”：一个作用在每个图块行上的两层小网络，{dv} → {dh}，GELU，{dh} → {dl}。图像词元 ▦{p} 中编号 {c} 的数（从 0 数起）是这个图块经过第一层和 GELU 之后那 {dh} 个数的加权和：{dot} 加上偏置 {b}，得到 {y}。',
				vars: (rt) => {
					const [p, c] = rt.sel.unit ?? [];
					const g = T(rt, 'proj.gelu');
					const y = T(rt, 'proj.fc2');
					const W = rt.model?.weights['proj.fc2.weight'];
					const B = rt.model?.weights['proj.fc2.bias'];
					const D = dims(rt);
					let dot = NaN;
					if (g && W && Number.isFinite(p)) {
						dot = 0;
						for (let i = 0; i < D.dh; i++) dot += g.data[p * D.dh + i] * W.data[c * D.dh + i];
					}
					return { ...D, p, c, dot, b: B ? B.data[c] : NaN, y: y ? y.data[p * D.dl + c] : NaN };
				}
			},
			look: { en: 'Each product is one input times its learned weight; together with the bias they add up to this one number.', zh: '每个乘积是一个输入乘以它学来的权重；它们再加上偏置，合起来就是这一个数。' },
			terms: ['projector', 'dense-layer', 'activation-function']
		},
		{
			id: 'words',
			source: 'fwd',
			at: { layer: 'lm.tok' },
			select: () => ({ layer: 'lm.tok', unit: [N_IMG, 0] }),
			view: { key: 'zoom.lookup', props: (rt) => ({ tokens: rowNames(Array.from(rt.pre?.tensors.text?.data ?? []).slice(N_IMG)) }) },
			provenance: 'live',
			title: { en: 'The words are looked up', zh: '查出每个词的那一行' },
			narration: {
				en: 'Each word id picks one row of the language model’s embedding table: {V} rows of {dl} numbers, learned together with the projector. The first word, “{w0}”, is id {id0}, so it gets row {id0} (counting from 0). The {n} placeholders in front, id 0 (<image>), get row 0, but only for a moment: the next step replaces them.',
				zh: '每个词的 id 从语言模型的嵌入表里取出一行：表里有 {V} 行，每行 {dl} 个数，和投影层一起学出来。第一个词“{w0}”的 id 是 {id0}，所以它取编号 {id0} 的那一行（从 0 数起）。前面的 {n} 个占位符 id 是 0（<image>），取编号 0 的那一行，但只停留一会儿：下一步就把它们换掉。',
				vars: (rt) => {
					const ids = rt.pre?.tensors.text?.data;
					const id0 = ids ? Number(ids[N_IMG]) : NaN;
					return { ...dims(rt), id0, w0: VOCAB[id0] ?? '…' };
				}
			},
			look: { en: 'The row of the first word, copied from the table.', zh: '第一个词的那一行，从表里原样复制。' },
			terms: ['embedding', 'token']
		},
		{
			id: 'seq',
			source: 'fwd',
			at: { layer: 'lm.embed' },
			view: { key: 'layer.tensor', props: (rt) => ({ tokens: rt.pre?.tokens }) },
			provenance: 'live',
			title: { en: 'One sequence: image tokens, then words', zh: '一个序列：先是图像词元，再是词' },
			narration: {
				en: 'The {n} placeholder rows are replaced by the {n} image tokens, and a learned position row is added to each of the {rows} rows. From here on there is no separate image input: the language model reads one sequence of {rows} rows of {dl} numbers, image tokens first. LLaVA fills its <image> placeholder the same way.',
				zh: '{n} 个占位行被换成 {n} 个图像词元，再给这 {rows} 行各加上一行学来的位置编码。从这里开始就没有单独的图像输入了：语言模型读的是一个 {rows} 行、每行 {dl} 个数的序列，图像词元在前。LLaVA 也是这样填充它的 <image> 占位符的。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'Rows ▦0 to ▦{last} come from your drawing, {lastRows} from your question.', zh: '▦0 到 ▦{last} 行来自你的画，{lastRows}来自你的问题。', vars: (rt) => ({ last: N_IMG - 1, lastRows: plural(dims(rt).L, { one: 'the last row', many: 'the last {n}', zh: '最后 {n} 行' }) }) },
			terms: ['image-token', 'positional-encoding']
		},
		{
			id: 'mask',
			source: 'fwd',
			at: { layer: 'lm.L1.attn' },
			view: { key: 'kit.attention', props: (rt) => ({ tokens: rt.pre?.tokens }) },
			provenance: 'live',
			title: { en: 'Words can read the image, not the other way round', zh: '词能读图像，反过来不行' },
			narration: {
				en: 'Each of the {heads} heads of a block lets every row read itself and the rows before it, never the rows after: a causal mask, as in any GPT-style model. The image tokens come first, so {qRows} read all {n} of them, while no image token can read the question. In block 1 the first word, “{w0}”, already gives {wImg|pct} of its attention to the image tokens.',
				zh: '一个块的 {heads} 个头都让每一行只读它自己和它前面的行，从不读后面的行：这就是因果掩码，和所有 GPT 式模型一样。图像词元排在前面，所以{qRows}读到全部 {n} 个图像词元，而图像词元读不到问题。在第 1 块里，第一个词“{w0}”已经把 {wImg|pct} 的注意力给了图像词元。',
				vars: (rt) => {
						const D = dims(rt);
						const qRows = pick(D.L === 1, { en: 'the one question row can', zh: '唯一的问题行能' }, { en: `each of the ${D.L} question rows can`, zh: ` ${D.L} 个问题行都能` });
						return { ...D, qRows, w0: words(rt)[0] ?? '…', wImg: imgShareOfRow(rt, N_IMG) };
					}
			},
			look: { en: 'The empty triangle above the diagonal is the mask: those weights are exactly 0.', zh: '对角线上方的空三角就是掩码：那些权重正好是 0。' },
			terms: ['causal-mask', 'attention', 'attention-head']
		},
		{
			id: 'look',
			source: 'fwd',
			at: { layer: 'lm.L1.attn' },
			view: { key: 'llava-toy/patch-grid', props: () => ({ block: 1 }) },
			provenance: 'live',
			title: { en: 'Where does the answer row look?', zh: '写答案的那一行看向哪里？' },
			predict: {
				id: 'llava-toy-look',
				question: { en: 'In block 1, where does most of the last row’s attention go?', zh: '在第 1 块里，最后一行的注意力大部分去了哪里？' },
				options: [
					{ en: 'To the words of the question', zh: '去了问题里的词' },
					{ en: 'To the image tokens', zh: '去了图像词元' },
					{ en: 'To itself, the last row', zh: '去了它自己，也就是最后一行' }
				],
				answer: 1,
				explain: {
					en: 'To the image tokens: the words only say what is asked, while the answer has to come from the drawing. For “what is this ?”, a patch with ink usually gets more weight than an empty one, though the empty ones are not ignored; the next step gives the numbers.',
					zh: '去了图像词元：词只说明问的是什么，答案却必须来自画。问“what is this ?”时，有墨的图块通常比空图块得到更多权重，不过空图块也没有被忽略；下一步给出具体的数。'
				}
			},
			narration: {
				en: 'The answer is the word that comes after the last row, so the last row’s attention decides what the answer can read. In block 1 it has {rows} weights that add up to 1: {n} for the image tokens and {wordW}.',
				zh: '答案是最后一行之后的那个词，所以最后一行的注意力决定了答案能读到什么。在第 1 块里，它有 {rows} 个加起来等于 1 的权重：{n} 个给图像词元，{wordW}。',
				vars: (rt) => ({ ...dims(rt), wordW: plural(dims(rt).L, { one: '1 for itself, the only word', many: '{n} for the words up to and including itself', zh: '{n} 个给到它自己为止的各个词' }) })
			},
			look: { en: 'Each patch is shaded by its image token’s weight (mean of the heads); the bars are the words.', zh: '每个图块按它的图像词元的权重上色（各头平均）；条形是各个词。' },
			terms: ['attention', 'next-token-prediction']
		},
		{
			id: 'where',
			source: 'fwd',
			at: { layer: 'lm.L2.attn' },
			view: { key: 'llava-toy/patch-grid', props: () => ({ block: 1, pickBlock: true }) },
			provenance: 'live',
			title: { en: 'The answer row reads the image', zh: '写答案的那一行在读图像' },
			narration: {
				en: 'In block 1 the answer row gives {img1|pct} of its attention to the image tokens (spread evenly, they would get {even|pct}) and {word1|pct} to the words. {inkLine} Block 2 gives {img2|pct} to the image tokens. Over the {fact:testN} test drawings asked “what is this ?”, block 1 gives {fact:imgShare1|pct} to the image tokens, and a token with ink gets on average {fact:inkLean1|n2} times the weight of an empty one.',
				zh: '在第 1 块里，写答案的那一行把 {img1|pct} 的注意力给了图像词元（平均分配的话是 {even|pct}），{word1|pct} 给了词。{inkLine}第 2 块给图像词元 {img2|pct}。在 {fact:testN} 幅被问“what is this ?”的测试画上，第 1 块平均给图像词元 {fact:imgShare1|pct}，一个有墨的词元得到的权重平均是空词元的 {fact:inkLean1|n2} 倍。',
				vars: (rt) => {
					const g1 = grid(rt, 1);
					const g2 = grid(rt, 2);
					const n = inked(rt);
					const ink = pct(g1?.inkShare ?? NaN);
					const inkLine = pick(
						n > 0,
						{ en: `${n === 1 ? 'The 1 patch with ink gets' : `The ${n} patches with ink get`} ${ink} of all its attention, and the most-read token is ▦${g1?.top}.`, zh: `${n === 1 ? '唯一有墨的那个图块' : `${n} 个有墨的图块一共`}得到全部注意力的 ${ink}，读得最多的是 ▦${g1?.top}。` },
						{ en: 'Your pad is blank, so no patch has ink.', zh: '画板是空的，所以没有图块有墨。' }
					);
					return { even: N_IMG / dims(rt).rows, img1: g1?.imgShare ?? NaN, word1: g1 ? 1 - g1.imgShare : NaN, img2: g2?.imgShare ?? NaN, inkLine: g1 ? inkLine : NaN };
				}
			},
			look: { en: 'Switch blocks and heads: each head spreads its weight its own way.', zh: '切换块和头：每个头按自己的方式分配权重。' },
			terms: ['attention-head']
		},
		{
			id: 'answer',
			source: 'fwd',
			at: { layer: 'probs' },
			view: 'kit.probbars',
			provenance: 'live',
			title: { en: 'The answer: one word', zh: '答案：一个词' },
			narration: {
				en: 'After block 2 and a last layer normalization, the last row becomes {V} scores, one per id of the vocabulary, and softmax turns them into shares: {top.label} with {top.p|pct}, then {runnerUp.label} with {runnerUp.p|pct}. On the {fact:testN} test drawings, “what is this ?” gets the right word {fact:whatAcc|pct} of the time, close to TinyViT’s own class head ({fact:vitAcc|pct}), so the projector passes on almost all it needs.',
				zh: '经过第 2 块和最后一次层归一化，最后一行变成 {V} 个分数，词表里每个 id 一个，softmax 再把它们变成份额：{top.label}{top.p|pct}，其次是{runnerUp.label}{runnerUp.p|pct}。在 {fact:testN} 幅测试画上，“what is this ?”答对的比例是 {fact:whatAcc|pct}，接近 TinyViT 自己的类别头（{fact:vitAcc|pct}），所以投影层几乎把需要的都传过去了。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'The longest bar is the answer; every bar is one word of the vocabulary.', zh: '最长的条就是答案；每一条都是词表里的一个词。' },
			terms: ['logit', 'softmax', 'layer-norm']
		},
		{
			id: 'ask',
			source: 'fwd',
			at: { layer: 'probs' },
			view: 'kit.probbars',
			provenance: 'live',
			title: { en: 'Ask your own question', zh: '问你自己的问题' },
			narration: {
				en: 'A new question changes only the word rows; the {n} image tokens stay exactly the same. Now the question is “{q}” and the answer {top.label} ({top.p|pct}). Asked about a test drawing’s own thing (“is this a house ?” for a house), it says yes {fact:yesTrue|pct} of the time; asked about one of the {others} other things, it says no {fact:noOther|pct} of the time.',
				zh: '换一个问题只会改变词行；{n} 个图像词元完全不变。现在的问题是“{q}”，答案是{top.label}（{top.p|pct}）。拿测试画本来画的东西来问（比如对房子问“is this a house ?”），它有 {fact:yesTrue|pct} 的时候说 yes；换成另外 {others} 样东西之一来问，它有 {fact:noOther|pct} 的时候说 no。',
				vars: (rt) => ({ ...dims(rt), q: qText(rt) })
			},
			look: { en: 'The bars of yes and no: the same drawing, a different question.', zh: 'yes 和 no 的条：同一幅画，不同的问题。' },
			task: {
				prompt: {
					en: 'Make the model answer “yes”: ask whether your drawing is what it shows, for example “is this a house ?” for the house. Type it under Your input, or press an example question.',
					zh: '让模型回答“yes”：问它你的画是不是它画的那样东西，比如对房子问“is this a house ?”。在“你的输入”下面输入，或者点一个示例问题。'
				},
				done: (rt) => answerOf(T(rt, 'probs')).label === 'yes',
				waiting: { en: 'Not yet: the answer is not “yes”.', zh: '还没有：答案不是“yes”。' },
				hint: { en: 'Use the words it knows: is, this, a or an, and one of its 10 things, then “?”.', zh: '用它认识的词：is、this、a 或 an，加上它那 10 样东西之一，最后是“?”。' },
				doneText: { en: 'Done: same image tokens, a new answer.', zh: '完成：同样的图像词元，新的答案。' }
			},
			terms: ['vocabulary']
		},
		{
			id: 'limits',
			source: 'fwd',
			at: { layer: 'probs' },
			view: 'kit.probbars',
			provenance: 'live',
			kindOf: 'limits',
			title: { en: 'A tiny vocabulary, not a chatbot', zh: '很小的词表，不是聊天机器人' },
			narration: {
				en: 'llava-toy knows {V1} words and two kinds of question, and it always answers with one of its words. It has no word for a ladder or for “I do not know”: asked “what is this ?” about the ladder preset, it says “{lad}” ({ladP|pct}), and on {fact:heldOutN} test ladders and envelopes {fact:heldOutClassWord|pct} of its answers were one of its {C} class words. Real LLaVA puts the same kind of image tokens in front of a large pretrained language model.',
				zh: 'llava-toy 只认识 {V1} 个词和两种问法，而且总是用它的某个词来回答。它没有“梯子”这个词，也没有“我不知道”：对梯子预设问“what is this ?”，它回答“{lad}”（{ladP|pct}）；在 {fact:heldOutN} 幅梯子和信封的测试画上，它 {fact:heldOutClassWord|pct} 的回答都是它那 {C} 个类别词之一。真正的 LLaVA 用同样的办法，把图像词元放在一个大型预训练语言模型的前面。',
				vars: (rt) => ({ ...dims(rt), lad: VOCAB[rt.num('fact:ladderAnswer')] ?? '…', ladP: rt.num('fact:ladderP') })
			},
			look: { en: 'Its {V} ids (its {V1} words and <image>) are all it can answer with; nothing else has a bar.', zh: '它能用来回答的只有这 {V} 个 id（{V1} 个词加上 <image>）；别的都没有条。', vars: (rt) => dims(rt) },
			terms: ['vocabulary', 'out-of-distribution']
		}
	],
	fixtures: [
		{ id: 'house-yes', label: { en: 'the house, “is this a house ?”', zh: '房子，“is this a house ?”' }, raw: { strokes: HOUSE, question: 'is this a house ?' } },
		{ id: 'house-car', label: { en: 'the house, “is it a car ?”', zh: '房子，“is it a car ?”' }, raw: { strokes: HOUSE, question: 'is it a car ?' } }
	]
};

export default lesson;
export { answerOf };
