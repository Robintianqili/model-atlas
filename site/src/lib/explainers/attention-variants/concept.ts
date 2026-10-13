// attention-variants: the concept page around the lesson (atlas/ui/concept/panels.ts ConceptDef). hook (text), toy, real
// and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { bytesText, configOf, perToken, total } from './data.ts';

export const hook: I18n = {
	en: 'A language model writes one token at a time, and every new token looks back at all the earlier ones through their keys and values. The model keeps those in a KV cache, which grows with every token and can end up bigger than the weights. Multi-query, grouped-query and latent attention all shrink it by letting query heads share what is cached. This page counts what each variant stores, runs one shared-key attention on TinyGPT, and measures what sharing costs.',
	zh: '语言模型一次写一个词元，每个新词元都要通过之前所有词元的键和值回头看它们。模型把这些键和值存在 KV 缓存里；缓存随每个词元增长，最后可能比权重还大。多查询注意力、分组查询注意力和潜在注意力，都是让查询头共用缓存的内容来缩小它。本页数一数每种变体存了什么，在 TinyGPT 上做一次共用键的注意力计算，再测一测共享的代价。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Your configuration, three ways', zh: '你的配置，三种做法' },
	source: 'cache',
	at: 'first',
	view: { key: 'attention-variants/heads', props: () => ({ rows: ['mha', 'gqa', 'mqa'] }) },
	provenance: { mode: 'illustrative', note: { en: 'A schematic of which heads share; the sizes are computed from your configuration.', zh: '示意图只画出哪些头共用；大小按你的配置计算。' } },
	narration: {
		en: 'Your configuration as MHA, GQA and MQA. Yours caches {per} per token and {all} in total. Change the layers, heads, KV heads, head width, tokens or bytes in the input widget, or start from a published model.',
		zh: '把你的配置分别做成 MHA、GQA 和 MQA。你的配置每个词元缓存 {per}，一共 {all}。在输入控件里可以改层数、头数、KV 头数、头宽、词元数或字节数，也可以从一个已发表的模型开始。',
		vars: (rt) => {
			const c = configOf(rt.input);
			return { per: bytesText(perToken(c)), all: bytesText(total(c)) };
		}
	},
	look: { en: 'The three rows share your query heads; only the key-and-value boxes change.', zh: '三行的查询头都一样；变的只是键值方框。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'TinyGPT with one key head for all four queries', zh: '四个查询只用一个键头的 TinyGPT' },
	source: 'share',
	at: { phase: 'shared', step: 1 },
	view: { key: 'attention-variants/attn', props: () => ({ head: 2 }) },
	provenance: 'live',
	narration: {
		en: 'TinyGPT’s block {layer} (counting from 0) with multi-query attention: all four heads read the mean of their trained keys and values, {kv} key/value head. The block’s output for the last token moves by {change|pct} of its size. Pick a head in the stage to see its scores and weights.',
		zh: 'TinyGPT 编号 {layer} 的块（从 0 数起）改用多查询注意力：四个头都读它们训练出的键和值的平均，只剩 {kv} 个键值头。这一块对最后一个词元的输出变化了自身大小的 {change|pct}。在舞台里选一个头，看它的分数和权重。',
		vars: (_rt, at) => ({ layer: at.frame?.metrics?.layer ?? NaN, kv: at.frame?.metrics?.kv ?? NaN, change: at.frame?.metrics?.outChange ?? NaN })
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Converted, then uptrained: MHA, GQA and MQA', zh: '转换后再继续训练：MHA、GQA 与 MQA' },
	source: 'share',
	at: { phase: 'eval', step: 2 },
	view: { key: 'attention-variants/cost', props: () => ({ mode: 'uptrain' }) },
	provenance: 'precomputed',
	narration: {
		en: 'Validation perplexity of TinyGPT converted to 4, 2 and 1 key/value heads, then uptrained for {fact:upSteps} steps on {fact:upTokens} tokens. After uptraining: MHA {fact:pplMha1500}, GQA {fact:pplGqa1500}, MQA {fact:pplMqa1500}.',
		zh: 'TinyGPT 转换成 4 个、2 个和 1 个键值头，再在 {fact:upTokens} 个词元上继续训练 {fact:upSteps} 步，验证集困惑度如下。继续训练之后：MHA {fact:pplMha1500}，GQA {fact:pplGqa1500}，MQA {fact:pplMqa1500}。'
	}
};

export const pitfalls = [
	{
		claim: { en: 'GQA and MQA make attention faster by doing less arithmetic.', zh: 'GQA 和 MQA 通过减少计算量让注意力变快。' },
		truth: {
			en: 'Every query head still scores every cached key. What shrinks is the cache that has to be stored and read for each new token, which is what limits the speed of generation on long contexts.',
			zh: '每个查询头仍要给每个缓存的键打分。缩小的是每生成一个新词元都要存储和读取的缓存，而长上下文生成的速度恰恰受它限制。'
		},
		step: 'limits'
	},
	{
		claim: { en: 'A trained MHA model can switch to GQA for free: just average its heads.', zh: '训练好的 MHA 模型可以无代价地换成 GQA：把头平均一下就行。' },
		truth: {
			en: 'Averaging the heads without training makes TinyGPT much worse. A short uptraining repairs most of the damage, but not all of it.',
			zh: '不经训练就把头取平均，会让 TinyGPT 变差很多。短暂的继续训练能修复大部分损失，但不是全部。'
		},
		step: 'convert'
	}
];

export const further = [
	{ label: { en: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)', zh: 'GQA：从多头检查点训练通用的多查询 Transformer 模型（Ainslie 等，2023）' }, href: 'https://arxiv.org/abs/2305.13245' },
	{ label: { en: 'Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019)', zh: '快速 Transformer 解码：一个写入头就够了（Shazeer，2019）' }, href: 'https://arxiv.org/abs/1911.02150' },
	{ label: { en: 'DeepSeek-V2, with multi-head latent attention (DeepSeek-AI, 2024)', zh: 'DeepSeek-V2，含多头潜在注意力（DeepSeek-AI，2024）' }, href: 'https://arxiv.org/abs/2405.04434' }
];
