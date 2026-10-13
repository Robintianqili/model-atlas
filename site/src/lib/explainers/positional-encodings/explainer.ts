// positional-encodings: the manifest of a CONCEPT package: routed at /train/positional-encodings/ by the concept route
// (AutoConceptPage: goals, hook, the lesson, toy, real, compare, pitfalls, checkpoint, links). Why attention needs
// positions, and four ways to give them: TinyGPT's learned table, the sinusoidal formula, RoPE and ALiBi, then a
// precomputed comparison beyond the trained length. Sources:
//   gpt   TinyGPT's spec and weights (static/models/tinygpt, read only), run on the learner's text   (trace, inputs: input)
//   pe    TinyGPT and the encodings on the learner's text and positions (process.ts 'pe'; one frame per phase)
// The page around the lesson is concept.ts; the arithmetic is pe.ts; the sweep is extrap.ts (pipeline/positional-encodings).
import type { ExplainerManifest, ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { peOf, START } from './adapters/pe-text/adapter.ts';
import type { PeInput } from './process.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

/** TinyGPT's spec and weights, copied once per loaded model (each tensor its own buffer, so the worker can receive them;
 *  pattern of the quantization package), with the learner's settings. */
const copies = new WeakMap<object, { spec: NonNullable<PeInput>['spec']; weights: Record<string, Tensor> }>();
export function peInput(rt: ExplainerRuntime): PeInput {
	const m = rt.sources.gpt?.model;
	if (!m) return null;
	let c = copies.get(m);
	if (!c) {
		const weights: Record<string, Tensor> = {};
		for (const [k, t] of Object.entries(m.weights)) weights[k] = { shape: t.shape.slice(), data: (t.data as Float32Array).slice() };
		c = { spec: JSON.parse(JSON.stringify(m.spec)), weights };
		copies.set(m, c);
	}
	return { ...c, ...peOf(rt.input) };
}

const manifest: ConceptManifest = {
	slug: 'positional-encodings',
	type: 'concept',
	kinds: ['process', 'trace'],
	quality: 'reviewed',
	input: { adapter: 'pe-text', presetsFromAdapter: true, start: START },
	assets: { dir: 'models/positional-encodings', auto: ['facts.json'] },
	sources: {
		gpt: { kind: 'trace', assets: 'models/tinygpt', inputs: ['input'] },
		pe: { kind: 'process', process: 'pe', input: peInput, reference: false }
	},
	page: {
		goals: [
			{
				en: 'Show with a real model that attention without positions treats a sentence as a bag of tokens, and say how the causal mask leaks some order anyway.',
				zh: '用一个真实模型说明：没有位置信息时，注意力把一句话当成一袋词元；并说出因果掩码为什么还是会漏出一些顺序。'
			},
			{
				en: 'Compute a sinusoidal encoding by hand, and read TinyGPT’s learned position table.',
				zh: '亲手算出一个正弦位置编码，并读懂 TinyGPT 学到的位置表。'
			},
			{
				en: 'Explain how RoPE turns queries and keys so that their dot product depends only on the distance, and how ALiBi’s bias ramp favours near tokens.',
				zh: '解释 RoPE 怎样旋转查询和键，使它们的点积只取决于距离；以及 ALiBi 的偏置斜坡怎样偏向近处的词元。'
			},
			{
				en: 'Compare the encodings on text four times longer than training, and say which ones extrapolate.',
				zh: '在比训练长四倍的文本上比较各种编码，并说出哪些能外推。'
			}
		],
		prequestion: {
			id: 'positional-encodings-pre',
			question: {
				en: 'A language model trained only on windows of 64 tokens is given 256 tokens. What happens to its predictions after token 64?',
				zh: '一个只在 64 个词元的窗口上训练过的语言模型，拿到了 256 个词元。第 64 个词元之后，它的预测会怎样？'
			},
			options: [
				{ en: 'They keep improving: more context always helps', zh: '继续变好：上下文越多总是越好' },
				{ en: 'It depends on how the model is given positions: some ways break down, others hold up', zh: '取决于模型怎样获得位置信息：有的方式会崩溃，有的方式能撑住' },
				{ en: 'Every model fails the same way past its training length', zh: '超出训练长度后，所有模型都以同样的方式失败' }
			],
			answer: 1,
			explain: {
				en: 'On this page, five TinyGPT-sized models that differ only in their positions read 256 tokens. The sinusoidal and RoPE models get much worse past token 64, the model with no positions somewhat worse, and ALiBi keeps improving; a learned table simply has no row for position 64.',
				zh: '本页让五个大小和 TinyGPT 一样、只有位置编码不同的模型读 256 个词元。正弦编码和 RoPE 的模型在第 64 个词元之后明显变差，不用位置编码的模型也变差一些，而 ALiBi 还在继续变好；可学习的位置表则根本没有第 64 个位置的那一行。'
			}
		},
		quiz: [
			{
				q: { en: 'RoPE turns a query at position 10 and a key at position 4. What does their dot product depend on, besides the two vectors?', zh: 'RoPE 旋转位置 10 上的查询和位置 4 上的键。除了这两个向量本身，它们的点积还取决于什么？' },
				options: [
					{ en: 'Only the distance, 10 − 4 = 6', zh: '只取决于距离 10 − 4 = 6' },
					{ en: 'Both positions, 10 and 4, separately', zh: '分别取决于两个位置 10 和 4' },
					{ en: 'Nothing: a rotation never changes a dot product', zh: '什么都不取决：旋转从不改变点积' }
				],
				answer: 0,
				explain: {
					en: 'The query turns by 10·θ and the key by 4·θ in each pair, so the angle between them changes by (10 − 4)·θ. Moving both by the same amount leaves the dot product as it was.',
					zh: '在每一对维度里，查询转了 10·θ，键转了 4·θ，所以两者之间的夹角改变了 (10 − 4)·θ。把两个位置同时移动同样的量，点积保持不变。'
				}
			},
			{
				q: { en: 'An ALiBi head has the slope m = 1/4. What does it add to the score of a key 8 positions back?', zh: '一个 ALiBi 头的斜率 m = 1/4。它给 8 个位置之前的那个键的分数加上多少？' },
				options: [
					{ en: '−2', zh: '−2' },
					{ en: '−0.25', zh: '−0.25' },
					{ en: '−32', zh: '−32' }
				],
				answer: 0,
				explain: {
					en: 'The bias is −m × distance = −(1/4) × 8 = −2. After the softmax, that key gets e^(−2), about 14%, of the weight it would get at distance 0 with the same score.',
					zh: '偏置是 −m × 距离 = −(1/4) × 8 = −2。经过 softmax 之后，在原始分数相同的情况下，这个键得到的权重是距离为 0 时的 e^(−2)，约 14%。'
				}
			}
		],
		misconception: {
			claim: { en: 'A positional encoding is always a vector added to the token embeddings.', zh: '位置编码一定是加到词元嵌入上的一个向量。' },
			truth: {
				en: 'Only learned tables and the sinusoidal formula are. RoPE turns the queries and keys inside every attention layer, and ALiBi subtracts a penalty from the attention scores; neither adds anything to the embeddings.',
				zh: '只有可学习的位置表和正弦公式是这样。RoPE 在每个注意力层里旋转查询和键，ALiBi 从注意力分数里减去一个惩罚；它们都不往嵌入上加任何东西。'
			},
			step: 'rope'
		},
		lessonNote: {
			en: 'Ten steps: shuffle your sentence, read TinyGPT’s own position table, work the sinusoidal formula, turn a query with RoPE, ramp scores with ALiBi, then see which encodings hold up on text four times longer than training.',
			zh: '十步：打乱你的句子，读 TinyGPT 自己的位置表，代入正弦公式，用 RoPE 旋转一个查询，用 ALiBi 给分数加上斜坡，再看哪些编码在比训练长四倍的文本上还能撑住。'
		}
	},
	credits: [
		{
			name: 'TinyStories (TinyStoriesV2-GPT4)',
			url: 'https://huggingface.co/datasets/roneneldan/TinyStories',
			license: 'CDLA-Sharing-1.0',
			usedFor: 'training and scoring the five models of the extrapolation sweep (TinyGPT’s tokenized copy: the first 100 MiB of the training split, 500 windows of the validation split)',
			attribution: 'TinyStories by Ronen Eldan and Yuanzhi Li (2023)'
		},
		{
			name: 'Vaswani et al., Attention Is All You Need (NeurIPS 2017)',
			url: 'https://arxiv.org/abs/1706.03762',
			license: 'Paper; ideas only (no text, figures or code used)',
			usedFor: 'the sinusoidal encoding PE(p, 2i) = sin(p / 10000^(2i/d)), PE(p, 2i+1) = cos(…), and scaling the token embeddings by √d before adding it',
			attribution: 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin'
		},
		{
			name: 'Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)',
			url: 'https://arxiv.org/abs/2104.09864',
			license: 'Paper; ideas only (no text, figures or code used)',
			usedFor: 'RoPE: each 2-D pair of a query and a key turned by position × θ_i with θ_i = 10000^(−2i/d), so that their dot product depends on the distance only',
			attribution: 'Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen and Yunfeng Liu'
		},
		{
			name: 'Press et al., Train Short, Test Long: Attention with Linear Biases (ALiBi, ICLR 2022)',
			url: 'https://arxiv.org/abs/2108.12409',
			license: 'Paper; ideas only (no text, figures or code used)',
			usedFor: 'ALiBi’s bias −m × distance with the head slopes 2^(−8/H), 2^(−16/H), …, and the train-short, test-long comparison',
			attribution: 'Ofir Press, Noah A. Smith and Mike Lewis'
		},
		{
			name: 'nanoGPT (ideas)',
			url: 'https://github.com/karpathy/nanoGPT',
			license: 'MIT (ideas only; no code used)',
			usedFor: 'the small GPT trained in the extrapolation sweep (as for TinyGPT); code written fresh',
			attribution: 'Andrej Karpathy'
		}
	],
	glossary: [
		{
			id: 'sinusoidal-encoding',
			term: { en: 'sinusoidal positional encoding', zh: '正弦位置编码' },
			def: {
				en: 'A fixed table, no training: position p gets sin(p·ω) and cos(p·ω) for a set of frequencies ω, from one turn per 2π positions down to very slow ones. It is added to the token embeddings.',
				zh: '一张固定的表，不需要训练：位置 p 得到一组频率 ω 下的 sin(p·ω) 和 cos(p·ω)，频率从每 2π 个位置转一圈一直到非常慢的。它被加到词元嵌入上。'
			},
			related: ['positional-encoding']
		},
		{
			id: 'rope',
			term: { en: 'RoPE (rotary position embedding)', zh: 'RoPE（旋转位置编码）' },
			def: {
				en: 'Positions as rotations: inside attention, each 2-D pair of a query or a key is turned by its position times a fixed angle, so a query–key dot product depends only on how far apart the two tokens are.',
				zh: '把位置变成旋转：在注意力里，查询和键的每一对二维分量都按“位置 × 固定角度”旋转，于是查询与键的点积只取决于两个词元相隔多远。'
			},
			related: ['positional-encoding', 'query-key-value']
		},
		{
			id: 'alibi',
			term: { en: 'ALiBi (attention with linear biases)', zh: 'ALiBi（线性偏置注意力）' },
			def: {
				en: 'No position vectors at all: each attention score gets −m × distance, with one slope m per head, so far tokens get less weight.',
				zh: '完全不用位置向量：每个注意力分数加上 −m × 距离，每个头有自己的斜率 m，于是远处的词元得到的权重更小。'
			},
			related: ['positional-encoding', 'attention-head']
		},
		{
			id: 'nope',
			term: { en: 'NoPE (no positional encoding)', zh: 'NoPE（无位置编码）' },
			def: {
				en: 'A decoder with no position information at all. Its causal mask still shows each token a different set of earlier tokens, which deeper layers can use to tell positions apart.',
				zh: '完全没有位置信息的解码器。它的因果掩码仍让每个词元看到不同的一组前文词元，更深的层可以借此分辨位置。'
			},
			related: ['causal-mask', 'positional-encoding']
		}
	]
};

export default manifest;
