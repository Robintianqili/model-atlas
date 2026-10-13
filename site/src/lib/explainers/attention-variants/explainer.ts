// attention-variants: the manifest of a CONCEPT package: routed at /train/attention-variants/ by the concept route
// (AutoConceptPage: goals, hook, the lesson, toy, real, compare, pitfalls, checkpoint, links). Multi-head, multi-query,
// grouped-query and latent attention: which heads share keys and values, what that does to the KV cache (an analytic
// calculator on the learner's configuration and on published ones), and what sharing does to a real model (TinyGPT's
// trained heads converted in the browser; the uptraining that repairs them measured in PyTorch, facts.json). Sources:
//   cache   the learner's configuration: bytes per token and for the sequence, against MHA and MQA   (process 'cache')
//   gpt     TinyGPT's spec and weights (static/models/tinygpt, read only); no forward pass here     (trace, inputs: ids)
//   share   TinyGPT with block 2's, then every block's, keys and values shared, scored on four stories (process 'share')
// The page around the lesson is concept.ts; the arithmetic is data.ts.
import type { ExplainerManifest, ExplainerRuntime, Tensor } from '../../atlas/types/index.ts';
import { START } from './data.ts';
import type { GptInput } from './process.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

/** TinyGPT's spec and weights for the share process, copied once per loaded model (each tensor gets its own buffer, so the
 *  worker receives 1.46 MB, and the same object every time, so a new learner input keeps the run). From quantization. */
const copies = new WeakMap<object, GptInput>();
export function gptInput(rt: ExplainerRuntime): GptInput {
	const m = rt.sources.gpt?.model;
	if (!m) return null;
	let c = copies.get(m);
	if (!c) {
		const weights: Record<string, Tensor> = {};
		for (const [k, t] of Object.entries(m.weights)) weights[k] = { shape: t.shape.slice(), data: (t.data as Float32Array).slice() };
		c = { spec: JSON.parse(JSON.stringify(m.spec)), weights };
		copies.set(m, c);
	}
	return c;
}

const PAPER = 'Paper; ideas and published numbers only (no text, figures or code used)';
const paper = (name: string, url: string, usedFor: string, attribution: string) => ({ name, url, license: PAPER, usedFor, attribution });

const manifest: ConceptManifest = {
	slug: 'attention-variants',
	type: 'concept',
	kinds: ['process', 'trace'],
	quality: 'reviewed',
	input: { adapter: 'config', start: START, presetsFromAdapter: true },
	assets: { dir: 'models/attention-variants', auto: ['facts.json'] },
	sources: {
		cache: { kind: 'process', process: 'cache' },
		gpt: { kind: 'trace', assets: 'models/tinygpt', inputs: ['ids'] },
		share: { kind: 'process', process: 'share', input: gptInput }
	},
	page: {
		goals: [
			{
				en: 'Say what the KV cache stores and compute its size: 2 × layers × KV heads × head width × tokens × bytes.',
				zh: '说出 KV 缓存存的是什么，并算出它的大小：2 × 层数 × KV 头数 × 头宽 × 词元数 × 字节数。'
			},
			{
				en: 'Draw which query heads share a key head and a value head in MHA, GQA and MQA, and read the saving off the drawing.',
				zh: '画出 MHA、GQA 和 MQA 中哪些查询头共用一个键头和一个值头，并从图上读出省了多少。'
			},
			{
				en: 'Follow one attention computation on TinyGPT in which two query heads share their keys and values, and explain why their weights still differ.',
				zh: '在 TinyGPT 上跟着走一遍两个查询头共用键和值的注意力计算，并解释为什么它们的权重仍然不同。'
			},
			{
				en: 'Explain what sharing costs a trained model, how much uptraining repairs, and what MLA caches instead.',
				zh: '解释共享会让训练好的模型付出什么代价、继续训练能修复多少，以及 MLA 改为缓存什么。'
			}
		],
		prequestion: {
			id: 'attention-variants-pre',
			question: {
				en: 'A chatbot slows down and runs out of memory in long conversations, although the model’s weights fit easily. What fills the memory?',
				zh: '一个聊天机器人在长对话里变慢、内存不够用，可模型的权重明明放得下。是什么占满了内存？'
			},
			options: [
				{ en: 'The keys and values it keeps for every earlier token, in every layer', zh: '它在每一层为之前每个词元保存的键和值' },
				{ en: 'A second copy of the weights for every new token', zh: '每生成一个新词元就多出的一份权重' },
				{ en: 'The text of the conversation itself', zh: '对话文本本身' }
			],
			answer: 0,
			explain: {
				en: 'Each new token attends to every earlier token, so the model keeps their keys and values: the KV cache. It grows with every token, and with long contexts it can outgrow the weights. Sharing key and value heads is one way to shrink it.',
				zh: '每个新词元都要关注之前所有的词元，所以模型把它们的键和值存下来，这就是 KV 缓存。它随每个词元增长，上下文很长时甚至会比权重还大。让多个查询头共用键头和值头，是缩小它的办法之一。'
			}
		},
		quiz: [
			{
				q: { en: 'A model has 32 query heads and 8 KV heads. How large is its KV cache compared with the same model using MHA?', zh: '一个模型有 32 个查询头、8 个 KV 头。和同一个模型用 MHA 相比，它的 KV 缓存有多大？' },
				options: [
					{ en: 'A quarter', zh: '四分之一' },
					{ en: 'An eighth', zh: '八分之一' },
					{ en: 'The same: there are still 32 query heads', zh: '一样大：查询头仍然是 32 个' }
				],
				answer: 0,
				explain: { en: 'The cache holds one key and one value per KV head: 8 instead of 32, so 8 / 32 = a quarter. Each group of four query heads reads the same keys and values.', zh: '缓存为每个 KV 头存一个键和一个值：8 个而不是 32 个，所以是 8 / 32 = 四分之一。每组四个查询头读同一份键和值。' }
			},
			{
				q: { en: 'Two query heads share one key head. Do they put the same attention weights on the tokens?', zh: '两个查询头共用一个键头。它们给各个词元的注意力权重一样吗？' },
				options: [
					{ en: 'No: each head still has its own query, so its scores differ', zh: '不一样：每个头仍有自己的查询，所以分数不同' },
					{ en: 'Yes: the same keys give the same weights', zh: '一样：同样的键给出同样的权重' },
					{ en: 'Only if the values are shared too', zh: '只有值也共用时才一样' }
				],
				answer: 0,
				explain: { en: 'A score is the dot product of a head’s query with a key. Shared keys meet different queries, so the two heads can still look at different tokens.', zh: '分数是某个头的查询与键的点积。共用的键遇上不同的查询，两个头仍然可以看向不同的词元。' }
			}
		],
		misconception: {
			claim: { en: 'Sharing keys and values makes every head attend to the same tokens.', zh: '共用键和值会让所有头都关注同样的词元。' },
			truth: {
				en: 'Only the keys and values are shared; every query head keeps its own query, so the heads still score the tokens differently: on TinyGPT, two heads that share one key head peak at different tokens. What sharing costs is the separate keys each head had learned, which a converted model has to relearn.',
				zh: '共用的只是键和值；每个查询头保留自己的查询，所以各个头给词元打的分仍然不同：在 TinyGPT 上，共用一个键头的两个头，峰值落在不同的词元上。共享真正的代价，是每个头原先学到的那一套键，转换后的模型得重新学。'
			},
			step: 'shared'
		},
		lessonNote: {
			en: 'Eleven steps: what the KV cache holds, which heads share keys and values, one shared-key attention computation on TinyGPT, the caches of published models, latent attention, and what sharing costs.',
			zh: '十一步：KV 缓存存了什么，哪些头共用键和值，在 TinyGPT 上做一次共用键的注意力计算，已发表模型的缓存，潜在注意力，以及共享的代价。'
		}
	},
	credits: [
		paper(
			'Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (EMNLP 2023)',
			'https://arxiv.org/abs/2305.13245',
			'Grouped-query attention, converting a multi-head checkpoint by mean pooling each group’s key and value heads, and uptraining with 5% of the pretraining steps; the page repeats both on TinyGPT',
			'Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón and Sumit Sanghai'
		),
		paper('Shazeer, Fast Transformer Decoding: One Write-Head is All You Need (2019)', 'https://arxiv.org/abs/1911.02150', 'Multi-query attention: one key head and one value head shared by every query head', 'Noam Shazeer'),
		paper(
			'DeepSeek-AI, DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2024)',
			'https://arxiv.org/abs/2405.04434',
			'Multi-head latent attention and the DeepSeek-V2 preset: 60 layers, 128 heads of width 128, a cached latent of 512 plus a 64-wide RoPE key per token and layer',
			'DeepSeek-AI'
		),
		paper('Touvron et al., Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)', 'https://arxiv.org/abs/2307.09288', 'The Llama 2 7B (32 layers, 32 heads, MHA) and Llama 2 70B (80 layers, 64 heads, 8 KV heads) presets, head width 128', 'Hugo Touvron, Louis Martin, Kevin Stone and the Llama 2 team (Meta)'),
		paper('Llama Team, AI @ Meta, The Llama 3 Herd of Models (2024)', 'https://arxiv.org/abs/2407.21783', 'The Llama 3 8B preset: 32 layers, 32 heads, 8 KV heads, head width 128', 'Aaron Grattafiori, Abhimanyu Dubey and the Llama 3 team (Meta)'),
		paper('Almazrouei et al., The Falcon Series of Open Language Models (2023)', 'https://arxiv.org/abs/2311.16867', 'The Falcon-7B preset: 32 layers, 71 query heads of width 64, multi-query attention', 'Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi and the Falcon team (TII)'),
		paper('Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2, 2019)', 'https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf', 'The GPT-2 small preset: 12 layers, 12 heads of width 64', 'Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever'),
		{
			name: 'TinyStories (TinyStoriesV2-GPT4)',
			url: 'https://huggingface.co/datasets/roneneldan/TinyStories',
			license: 'CDLA-Sharing-1.0',
			usedFor: 'The text TinyGPT is scored on (four validation stories in the browser, 1,000 validation windows in PyTorch) and the training tokens of the uptraining runs (TinyGPT’s tokenized copy)',
			attribution: 'TinyStories by Ronen Eldan and Yuanzhi Li (2023)'
		}
	],
	glossary: [
		{
			id: 'grouped-query-attention',
			term: { en: 'grouped-query attention (GQA)', zh: '分组查询注意力（GQA）' },
			def: {
				en: 'Attention in which the query heads are split into groups and each group shares one key head and one value head. With one query head per group it is multi-head attention; with a single group, multi-query attention.',
				zh: '把查询头分成若干组、每组共用一个键头和一个值头的注意力。每组只有一个查询头时就是多头注意力；只有一组时就是多查询注意力。'
			},
			related: ['attention-head', 'kv-cache', 'multi-query-attention']
		},
		{
			id: 'multi-query-attention',
			term: { en: 'multi-query attention (MQA)', zh: '多查询注意力（MQA）' },
			def: {
				en: 'Attention in which every query head reads the same single key head and value head, so the KV cache holds one key and one value per token and layer.',
				zh: '所有查询头都读同一个键头和值头的注意力，所以 KV 缓存每个词元、每层只存一个键和一个值。'
			},
			related: ['grouped-query-attention', 'kv-cache']
		},
		{
			id: 'latent-attention',
			term: { en: 'multi-head latent attention (MLA)', zh: '多头潜在注意力（MLA）' },
			def: {
				en: 'Attention that caches one short latent vector per token and layer, plus a small position key shared by the heads, and rebuilds every head’s keys and values from the latent with learned up-projections.',
				zh: '每个词元、每层只缓存一个较短的潜向量和一个各头共用的小位置键，再用学到的上投影矩阵从潜向量还原每个头的键和值的注意力。'
			},
			related: ['kv-cache', 'grouped-query-attention']
		},
		{
			id: 'uptraining',
			term: { en: 'uptraining', zh: '继续训练（uptraining）' },
			def: {
				en: 'Training an already trained model a little longer after changing its architecture, here after sharing its key and value heads, so that it adapts to the change.',
				zh: '改动已训练模型的结构之后（这里是让它的键头和值头共用），再让它多训练一小段，使它适应这个改动。'
			},
			related: ['fine-tuning', 'pre-training']
		}
	],
	// The catalog entry promote writes (data only the CLI reads; not page code). It replaces the planned entry's
	// "TinyGPT trained with MHA vs GQA": the page converts TinyGPT's trained heads live and shows a precomputed uptraining.
	catalog: {
		interaction: {
			en: 'Pick a published model or set your own layers, heads, KV heads, head width, tokens and bytes, and see the KV cache of MHA, GQA, MQA and MLA next to a head-sharing diagram. Follow one shared-key attention computation on TinyGPT, convert its trained heads to 2 and 1 KV heads in the browser and measure the damage, then see how much uptraining repairs (measured in PyTorch).',
			zh: '选一个已发表的模型，或自己设定层数、头数、KV 头数、头宽、词元数和字节数，对照头共享示意图看 MHA、GQA、MQA 和 MLA 的 KV 缓存。在 TinyGPT 上跟着做一次共用键的注意力计算，在浏览器里把它训练好的头转换成 2 个和 1 个 KV 头并测出损失，再看继续训练能修复多少（用 PyTorch 测得）。'
		},
		engine: { en: 'Analytic + TinyGPT live + precomputed', zh: '解析计算 + TinyGPT 实时 + 预计算' },
		mode: 'both',
		links: ['tinygpt']
	} as unknown as ExplainerManifest['catalog']
};

export default manifest;
