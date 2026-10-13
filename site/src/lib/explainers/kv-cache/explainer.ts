// kv-cache: the manifest of a CONCEPT package (K2): routed at /train/kv-cache/ by the concept route, which renders
// AutoConceptPage (goals, hook, the lesson, toy, real, compare, pitfalls, checkpoint, links). TinyGPT continues the learner's
// prompt in the browser twice, without and with a KV cache, and the page counts the work, the memory and the differences.
// Sources:
//   gpt     TinyGPT's spec and weights (static/models/tinygpt, read only); no forward pass of its own  (trace, inputs: none)
//   gen     prompt A continued for 24 tokens, without and with the cache, in lockstep            (process 'gen')
//   prefix  prompt B's prefill from scratch and with prompt A's cached prefix                     (process 'prefix')
// The page around the lesson is concept.ts; the forward pass with a cache is gpt.ts.
import type { ExplainerManifest, ExplainerRuntime, I18n, Tensor } from '../../atlas/types/index.ts';
import type { Gpt } from './gpt.ts';
import { promptA, promptB, START, type PromptsRaw } from './data.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it),
 *  and the concept catalog fields promote prints (interaction, engine, mode, links), as self-supervised-objectives. */
type ConceptManifest = Omit<ExplainerManifest, 'type' | 'catalog'> & {
	type: 'concept';
	catalog?: ExplainerManifest['catalog'] & { interaction?: I18n; engine?: I18n; mode?: string; links?: string[] };
};

/** TinyGPT's spec and weights, copied once per loaded model (each tensor gets its own buffer, so the worker receives
 *  1.46 MB rather than one copy of the whole weights file per tensor; the same object every time). As quantization does. */
const copies = new WeakMap<object, Gpt>();
export function gptOf(rt: ExplainerRuntime): Gpt | null {
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
const raw = (rt: ExplainerRuntime) => rt.input as PromptsRaw;

const manifest: ConceptManifest = {
	slug: 'kv-cache',
	type: 'concept',
	kinds: ['process', 'trace'],
	quality: 'reviewed',
	input: { adapter: 'prompts', presetsFromAdapter: true, start: START },
	assets: { dir: 'models/kv-cache', auto: ['facts.json'] },
	sources: {
		gpt: { kind: 'trace', assets: 'models/tinygpt', inputs: ['weightsOnly'] },
		// no Python reference: the uncached run, the cached run and the kit's own forward pass check each other live
		gen: {
			kind: 'process',
			process: 'gen',
			reference: false,
			input: (rt) => {
				const g = gptOf(rt);
				return g ? { ...g, ids: promptA(raw(rt)) } : null;
			}
		},
		prefix: {
			kind: 'process',
			process: 'prefix',
			reference: false,
			input: (rt) => {
				const g = gptOf(rt);
				return g ? { ...g, a: promptA(raw(rt)), b: promptB(raw(rt)) } : null;
			}
		}
	},
	page: {
		goals: [
			{
				en: 'Say why generating text one token at a time repeats work, and which numbers a new token can and cannot change.',
				zh: '说明为什么逐个词元生成文本会重复计算，以及一个新词元能改变哪些数、不能改变哪些数。'
			},
			{
				en: 'Compute a KV cache’s size as layers × heads × tokens × head size × 2 × bytes, for TinyGPT and for a large model.',
				zh: '按“层数 × 头数 × 词元数 × 每头维度 × 2 × 字节数”算出 KV 缓存的大小，分别算 TinyGPT 和一个大模型的。'
			},
			{
				en: 'Compare the work per new token with and without the cache, and check live that both give the same tokens.',
				zh: '比较有无缓存时每个新词元的计算量，并实时验证两者给出相同的词元。'
			},
			{
				en: 'Explain when two prompts can share cached keys and values, and why the sharing stops at the first different token.',
				zh: '解释两个提示词什么时候能共享缓存的键和值，以及为什么共享在第一个不同的词元处就停止。'
			}
		],
		prequestion: {
			id: 'kv-cache-pre',
			question: {
				en: 'A language model writes a reply one token at a time. To pick each new token, what does it have to compute?',
				zh: '语言模型一次写一个词元。为了选出每个新词元，它需要计算什么？'
			},
			options: [
				{ en: 'The whole text again, from the first token to the last', zh: '把整段文字从第一个词元到最后一个重新算一遍' },
				{ en: 'Only the newest token, plus a look back at numbers it saved for the earlier ones', zh: '只算最新的词元，再回头看它为前面的词元存下的数' },
				{ en: 'Only the newest token, with no look back at all', zh: '只算最新的词元，完全不回头看' }
			],
			answer: 1,
			explain: {
				en: 'Attention lets the new token look at every earlier token’s key and value. Those never change once computed, because no token looks ahead, so a model can save them (the KV cache) and compute only the new token. Without the cache it would recompute the whole text each time, with the same result.',
				zh: '注意力让新词元查看前面每个词元的键和值。这些数一旦算出就不会再变，因为没有哪个词元会往后看，所以模型可以把它们存起来（KV 缓存），每次只算新词元。没有缓存时，它每次都要把整段文字重算一遍，结果完全相同。'
			}
		},
		quiz: [
			{
				q: { en: 'A model has 2 layers, 2 heads of size 8, and stores fp32 (4 bytes). How big is its KV cache for 10 tokens?', zh: '一个模型有 2 层、2 个头、每头维度 8，用 fp32（4 字节）存储。10 个词元的 KV 缓存有多大？' },
				options: [
					{ en: '2 × 2 × 10 × 8 × 2 × 4 = 2,560 bytes', zh: '2 × 2 × 10 × 8 × 2 × 4 = 2,560 字节' },
					{ en: '2 × 2 × 10 × 8 × 4 = 1,280 bytes', zh: '2 × 2 × 10 × 8 × 4 = 1,280 字节' },
					{ en: '2 × 2 × 8 × 2 × 4 = 256 bytes, whatever the length', zh: '2 × 2 × 8 × 2 × 4 = 256 字节，与长度无关' }
				],
				answer: 0,
				explain: { en: 'Every token keeps a key and a value (× 2) in every head of every layer, each of head size 8, at 4 bytes a number. The cache grows with every token.', zh: '每个词元在每一层的每个头里都存一个键和一个值（× 2），各有 8 维，每个数 4 字节。缓存随每个词元增长。' }
			},
			{
				q: { en: 'Prompts A and B are identical except for their first word. How many of B’s tokens can reuse A’s cached keys and values?', zh: '提示词 A 和 B 只有第一个词不同。B 有多少个词元能复用 A 缓存的键和值？' },
				options: [
					{ en: 'None', zh: '一个也不能' },
					{ en: 'All but the first', zh: '除了第一个都能' },
					{ en: 'Only the last one', zh: '只有最后一个' }
				],
				answer: 0,
				explain: { en: 'From layer 1 on, every later token’s keys and values depend on all the tokens before it, the first one included, so none of them match A’s. Prefix caching reuses only the tokens before the first difference, and here there are none.', zh: '从第 1 层起，后面每个词元的键和值都取决于它前面的所有词元（包括第一个），所以没有一个与 A 的相同。前缀缓存只复用第一个不同词元之前的部分，而这里一个也没有。' }
			}
		],
		misconception: {
			claim: { en: 'A KV cache is an approximation: it trades a little accuracy for speed.', zh: 'KV 缓存是一种近似：它牺牲一点精度换取速度。' },
			truth: {
				en: 'The cache stores numbers that would be computed again exactly the same way: no earlier token ever looks at a later one. On this page both runs pick the same tokens, and their logits match.',
				zh: '缓存存下的数，重算一遍也会得到完全一样的结果：前面的词元从不查看后面的词元。在本页，两次运行选出的词元相同，logit 也一致。'
			},
			step: 'same'
		},
		lessonNote: {
			en: 'Ten steps: watch TinyGPT redo its work for every new token, then cache the keys and values, count the memory and the work, check the answers match, and share a prefix between two prompts.',
			zh: '十步：先看 TinyGPT 每生成一个词元就重做一遍计算，再缓存键和值，算出内存和计算量，验证答案一致，最后在两个提示词之间共享前缀。'
		}
	},
	credits: [
		{
			name: 'TinyStories (TinyStoriesV2-GPT4)',
			url: 'https://huggingface.co/datasets/roneneldan/TinyStories',
			license: 'CDLA-Sharing-1.0',
			usedFor: 'the training text of TinyGPT, the model this page runs; the example prompts are written for this page in its style',
			attribution: 'TinyStories by Ronen Eldan and Yuanzhi Li (2023)'
		},
		{
			name: 'Prompt caching (ideas)',
			url: 'https://ngrok.com/blog/prompt-caching/',
			license: 'ideas only (no code or text used; no licence stated)',
			usedFor: 'an attention grid that fills as tokens are generated, with the cached work shown as skipped; code written fresh',
			attribution: 'Sam Rose, ngrok blog (2025)'
		}
	],
	glossary: [
		{
			id: 'prefill-decode',
			term: { en: 'prefill and decode', zh: '预填充与解码' },
			def: {
				en: 'The two phases of generating text. Prefill runs the whole prompt in one pass and fills the KV cache; decode then adds one token per pass, reading the cache.',
				zh: '生成文本的两个阶段。预填充一次处理整个提示词，并填满 KV 缓存；解码随后每一遍只添加一个词元，读取缓存。'
			},
			related: ['kv-cache', 'autoregressive']
		},
		{
			id: 'prefix-caching',
			term: { en: 'prefix caching (prompt caching)', zh: '前缀缓存（提示缓存）' },
			def: {
				en: 'Reusing the cached keys and values of a prompt’s beginning for another prompt that starts with exactly the same tokens. The reuse stops at the first token that differs.',
				zh: '另一个提示词的开头与某个提示词的词元完全相同时，复用前者开头部分缓存的键和值。复用在第一个不同的词元处停止。'
			},
			related: ['kv-cache']
		},
		{
			id: 'multiply-add',
			term: { en: 'multiply-add', zh: '乘加' },
			def: {
				en: 'One multiplication and one addition, s += a × b: the step every matrix product repeats. Counting them measures the work of a forward pass.',
				zh: '一次乘法加一次加法，s += a × b：每个矩阵乘法都在重复这一步。数一数它们，就能衡量一次前向计算的工作量。'
			},
			related: ['forward-pass']
		}
	],
	// The concept entry's interaction, which promote writes into concepts.ts (engine, mode and links stay as planned).
	// The planned text said "recomputed cells grayed out": here recomputed cells are orange, gray is work the cache skips.
	catalog: {
		interaction: {
			en: 'TinyGPT continues your prompt twice, without and with a KV cache: an attention grid shows the work done again in orange and the work the cache skips in gray, with live counts of multiply-adds and cache memory (GPT-2 small and Llama 2 7B by the same formula) and a check that both runs pick the same tokens. A second prompt reuses the first one’s cached prefix up to the first different token.',
			zh: 'TinyGPT 把你的提示词续写两遍，一遍不用 KV 缓存，一遍用：注意力网格里橙色是重复计算的部分，灰色是缓存省下的计算；实时统计乘加次数和缓存内存（用同一公式算出 GPT-2 small 和 Llama 2 7B 的），并核对两遍选出的词元相同。第二个提示词复用第一个提示词缓存的前缀，直到第一个不同的词元为止。'
		}
	}
};

export default manifest;
