// positional-encodings: the concept page around the lesson (atlas/ui/concept/panels.ts ConceptDef). hook (text), toy,
// real and compare are StepDefs on the lesson's own runtime and source 'pe', shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';

const M = (at: { frame?: { metrics?: Record<string, number> } }, k: string) => at.frame?.metrics?.[k] ?? NaN;

export const hook: I18n = {
	en: 'Attention compares every token with every other token and mixes them by those scores. That arithmetic has no idea where a token sits: shuffle the words and, without extra information, attention sees the same bag of tokens. This page shows that on a real model, then four ways to give attention positions (a learned table, a sinusoidal formula, rotations called RoPE and a distance penalty called ALiBi) and tests which still work on text longer than any they trained on.',
	zh: '注意力把每个词元和其他所有词元比较，再按这些分数把它们混合起来。这套运算完全不知道词元在哪个位置：打乱词序，如果没有额外信息，注意力看到的还是同一袋词元。本页先在一个真实模型上展示这一点，再介绍给注意力提供位置的四种方法（可学习的位置表、正弦公式、叫作 RoPE 的旋转，以及叫作 ALiBi 的距离惩罚），并检验在比训练时更长的文本上，哪些方法依然有效。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Your query and key, any positions', zh: '你的查询和键，任意位置' },
	source: 'pe',
	at: { phase: 'rope' },
	view: 'positional-encodings/rope',
	provenance: 'live',
	narration: {
		en: 'Pair 0 of head 0: the query at position {m} and the key at {n}, k = {k} apart. Move p and k in the input widget: the arrows turn, and the whole-head dot product, now {score}, changes only when k does.',
		zh: '头 0 的第 0 对：查询在位置 {m}，键在位置 {n}，相隔 k = {k}。在输入控件里移动 p 和 k：箭头会旋转，而整个头的点积（现在是 {score}）只在 k 改变时才变。',
		vars: (_rt, at) => ({ m: M(at, 'm'), n: M(at, 'n'), k: M(at, 'k'), score: M(at, 'score') })
	},
	look: { en: 'Change only p and the solid arrows turn together; change k and the angle between them changes.', zh: '只改 p，两支实线箭头一起转；改 k，它们之间的夹角就变了。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'TinyGPT’s own position table', zh: 'TinyGPT 自己的位置表' },
	source: 'pe',
	at: { phase: 'learned' },
	view: 'positional-encodings/table',
	provenance: 'live',
	narration: {
		en: 'The {rows} rows TinyGPT learned, one per position. Neighbouring rows are alike, cosine similarity {mean1} on average one row apart, and the first and last rows are almost unrelated, {far}. Move p in the input widget to outline another row.',
		zh: 'TinyGPT 学到的 {rows} 行，每个位置一行。相邻的行很像，相隔一行的平均余弦相似度是 {mean1}；第一行和最后一行几乎毫无关系，只有 {far}。在输入控件里移动 p，就能框出另一行。',
		vars: (_rt, at) => ({ rows: M(at, 'rows'), mean1: M(at, 'mean1'), far: M(at, 'far') })
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Five encodings on text four times too long', zh: '五种编码，面对四倍长的文本' },
	source: 'pe',
	at: { phase: 'extrap' },
	view: 'positional-encodings/lines',
	provenance: 'precomputed',
	narration: {
		en: 'Five TinyGPT-sized models trained on {fact:trainCtx}-token windows, scored on {fact:evalCtx} tokens of {fact:windows} validation texts. Over the last {fact:trainCtx} tokens: ALiBi {fact:farAlibi}, no positions {fact:farNone}, RoPE {fact:farRope}, sinusoidal {fact:farSin}; the learned table has no rows there.',
		zh: '五个大小和 TinyGPT 一样的模型，在 {fact:trainCtx} 词元的窗口上训练，在 {fact:windows} 段验证文本的 {fact:evalCtx} 个词元上评测。最后 {fact:trainCtx} 个词元上的损失：ALiBi {fact:farAlibi}，无位置编码 {fact:farNone}，RoPE {fact:farRope}，正弦编码 {fact:farSin}；可学习的位置表在那里没有行。'
	}
};

export const pitfalls = [
	{
		claim: { en: 'Without positional encodings, a decoder cannot tell word order at all.', zh: '没有位置编码，解码器就完全分不出词序。' },
		truth: {
			en: 'Attention by itself cannot, but a decoder’s causal mask shows every token a different set of earlier tokens, and deeper layers can use that. Trained with no positions, the model on this page learns almost as well as the others within 64 tokens.',
			zh: '单靠注意力确实分不出，但解码器的因果掩码让每个词元看到不同的一组前文词元，更深的层可以利用这一点。本页那个完全不用位置编码训练的模型，在 64 个词元以内学得几乎和其他模型一样好。'
		},
		step: 'limits'
	},
	{
		claim: { en: 'The sinusoidal formula gives a vector for every position, so a sinusoidal model handles any length.', zh: '正弦公式能给任何位置算出向量，所以正弦编码的模型能处理任意长度。' },
		truth: {
			en: 'The formula does, but the model never saw those vectors in training. Trained on 64 tokens, the sinusoidal model’s loss more than doubles over tokens 192 to 255.',
			zh: '公式确实能算，但模型在训练中从没见过那些向量。用 64 个词元训练的正弦编码模型，在第 192 到 255 个词元上的损失翻了一倍还多。'
		},
		step: 'results'
	},
	{
		claim: { en: 'ALiBi extrapolates because the model reads the far tokens as well as the near ones.', zh: 'ALiBi 能外推，是因为模型读远处的词元和读近处的一样好。' },
		truth: {
			en: 'The opposite: its penalty grows with distance, so far tokens get very little weight. Head 0 subtracts 16 from the score of a token 64 positions back. ALiBi extrapolates by looking mostly nearby.',
			zh: '恰恰相反：它的惩罚随距离增大，所以远处的词元得到的权重很小。头 0 给 64 个位置之前的词元的分数减去 16。ALiBi 能外推，是因为它主要只看附近。'
		},
		step: 'alibi'
	}
];

export const further = [
	{ label: { en: 'Attention Is All You Need: the sinusoidal encoding (Vaswani et al., 2017)', zh: 'Attention Is All You Need：正弦位置编码（Vaswani 等，2017）' }, href: 'https://arxiv.org/abs/1706.03762' },
	{ label: { en: 'RoFormer: rotary position embedding (Su et al., 2021)', zh: 'RoFormer：旋转位置编码（Su 等，2021）' }, href: 'https://arxiv.org/abs/2104.09864' },
	{ label: { en: 'Train Short, Test Long: ALiBi (Press et al., 2021)', zh: 'Train Short, Test Long：ALiBi（Press 等，2021）' }, href: 'https://arxiv.org/abs/2108.12409' },
	{ label: { en: 'Transformer language models without positional encodings still learn positional information (Haviv et al., 2022)', zh: '不用位置编码的 Transformer 语言模型仍能学到位置信息（Haviv 等，2022）' }, href: 'https://arxiv.org/abs/2203.16634' },
	{ label: { en: 'The impact of positional encoding on length generalization in transformers (Kazemnejad et al., 2023)', zh: '位置编码对 Transformer 长度泛化的影响（Kazemnejad 等，2023）' }, href: 'https://arxiv.org/abs/2305.19466' },
	{ label: { en: 'Extending the context window by position interpolation (Chen et al., 2023)', zh: '用位置插值扩展上下文窗口（Chen 等，2023）' }, href: 'https://arxiv.org/abs/2306.15595' },
	{ label: { en: 'YaRN: efficient context window extension (Peng et al., 2023)', zh: 'YaRN：高效扩展上下文窗口（Peng 等，2023）' }, href: 'https://arxiv.org/abs/2309.00071' }
];
