// kv-cache: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef).
// hook (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { fmtBytes } from '../../atlas/core/format.ts';

export const hook: I18n = {
	en: 'A chatbot writes its reply one token at a time, and every new token has to look back at everything before it: the whole conversation and the reply so far. Done naively, that means reading the entire text again for every single token. Language models avoid it with one trick, the KV cache: they keep the numbers that never change. This page runs a real language model, TinyGPT, in your browser with and without that cache, and counts what it saves and what it costs.',
	zh: '聊天机器人一次写一个词元，而每个新词元都要回头查看它前面的全部内容：整段对话，以及到目前为止写出的回复。如果照直去做，每写一个词元就得把全文重新读一遍。语言模型用一个技巧避免这件事，那就是 KV 缓存：把那些不会再变的数存起来。本页在你的浏览器里运行一个真实的语言模型 TinyGPT，分别在有缓存和没有缓存时生成文字，算出缓存省下了什么、又花掉了什么。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Your prompt, pass by pass', zh: '你的提示词，一遍一遍地看' },
	source: 'gen',
	at: 'last',
	view: { key: 'kv-cache/grid', props: () => ({ controls: true }) },
	provenance: 'live',
	narration: {
		en: 'TinyGPT continues your prompt A for {N} passes. In the stage, switch the cache off and on, drag through the passes and pick the layer whose attention colors the cells. Edit prompt A in the input widget and every pass runs again.',
		zh: 'TinyGPT 把你的提示词 A 接着写了 {N} 遍。在舞台里可以开关缓存、拖动查看每一遍，还可以选择用哪一层的注意力给单元上色。在输入控件里修改提示词 A，每一遍都会重新运行。',
		vars: (_rt, at) => ({ N: at.frame?.metrics?.maxNew ?? NaN })
	},
	look: { en: 'Purple: computed for the first time; orange: computed again; gray: not needed thanks to the cache.', zh: '紫色：第一次计算；橙色：又算了一遍；灰色：有了缓存就不用算。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'The work and the memory of your run', zh: '你这次运行的计算量和内存' },
	source: 'gen',
	at: 'last',
	view: { key: 'kv-cache/cost', props: () => ({ memory: true }) },
	provenance: 'live',
	terms: ['multiply-add', 'kv-cache'],
	narration: {
		en: 'For your prompt, TinyGPT did {cumF} multiply-adds without the cache and {cumC} with it, while the cache grew to {bytes}. Switch the chart in the stage between each pass and the running totals.',
		zh: '对你的提示词，TinyGPT 不用缓存做了 {cumF} 次乘加，用缓存只做了 {cumC} 次，而缓存增长到了 {bytes}。在舞台里可以把图表在“每一遍”和“到目前为止的总和”之间切换。',
		vars: (_rt, at) => {
			const m = at.frame?.metrics;
			return { cumF: m?.cumFull ?? NaN, cumC: m?.cumCache ?? NaN, bytes: m && Number.isFinite(m.bytes) ? fmtBytes(m.bytes) : NaN };
		}
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Prompt B next to prompt A', zh: '提示词 B 与提示词 A 对照' },
	source: 'prefix',
	at: 'last',
	view: { key: 'kv-cache/prefix', props: () => ({ layers: true }) },
	provenance: 'live',
	terms: ['prefix-caching'],
	narration: {
		en: 'Your two prompts share {s} leading tokens, so prompt B’s prefill can skip {saved} multiply-adds by reusing prompt A’s cache. Edit either prompt in the input widget to move the first difference.',
		zh: '你的两个提示词开头有 {s} 个词元相同，所以提示词 B 的预填充复用提示词 A 的缓存，可以省下 {saved} 次乘加。在输入控件里修改任一提示词，可以挪动第一个不同的位置。',
		vars: (_rt, at) => ({ s: at.frame?.metrics?.s ?? NaN, saved: at.frame?.metrics?.saved ?? NaN })
	}
};

export const pitfalls = [
	{
		claim: { en: 'With a cache, a long prompt is cheap too.', zh: '有了缓存，很长的提示词也很便宜。' },
		truth: {
			en: 'The cache only helps after the prompt has been read once. The prefill computes every prompt token in full; only a later request that starts with exactly the same tokens can reuse that work.',
			zh: '缓存只在提示词读过一遍之后才有用。预填充要把每个提示词词元完整算一遍；只有之后开头词元完全相同的请求才能复用这些计算。'
		},
		step: 'limits'
	},
	{
		claim: { en: 'Two prompts that differ in one early word can still share most of their cache.', zh: '两个提示词只在开头差一个词，仍然可以共享大部分缓存。' },
		truth: {
			en: 'From layer 1 on, every later token’s keys and values depend on the tokens before it, so one early difference changes them all. Prefix caching stops at the first different token.',
			zh: '从第 1 层起，后面每个词元的键和值都取决于它前面的词元，所以开头的一处不同会让它们全都改变。前缀缓存在第一个不同的词元处就停止了。'
		},
		step: 'break'
	}
];

export const further = [
	{ label: { en: 'Prompt caching: an interactive explainer (Sam Rose, ngrok blog)', zh: '提示缓存：交互式讲解（Sam Rose，ngrok 博客）' }, href: 'https://ngrok.com/blog/prompt-caching/' },
	{ label: { en: 'Efficiently scaling Transformer inference (Pope et al., 2022)', zh: '高效扩展 Transformer 推理（Pope 等，2022）' }, href: 'https://arxiv.org/abs/2211.05102' },
	{ label: { en: 'Efficient memory management for LLM serving with PagedAttention (Kwon et al., 2023)', zh: '用 PagedAttention 为大模型服务高效管理内存（Kwon 等，2023）' }, href: 'https://arxiv.org/abs/2309.06180' }
];
