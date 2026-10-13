<!--
  Sources (redesign spec §5.9; a lazy panel of the tokenization page, #sources, the old #links and #further): the related
  lessons as catalog rows (built at build time by the route's server load), further reading, and the tokenizer files'
  licences.
-->
<script lang="ts" module>
	import type { I18n } from '$lib/i18n/types';
	import type { CatalogIndex, LessonItem } from '$lib/kit/catalog/types';
	export interface Related {
		sprites: CatalogIndex['sprites'];
		groups: { title: I18n; lessons: LessonItem[]; planned: { title: I18n; href: string }[] }[];
	}
</script>

<script lang="ts">
	import { base } from '$app/paths';
	import Section from '$lib/kit/Section.svelte';
	import LessonRow from '$lib/kit/catalog/LessonRow.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmtInt } from '$lib/kit/format';

	let { vp }: { vp: { related?: Related }; section?: string } = $props();
	const related = $derived(vp.related);
</script>

<div class="src">
	{#if related}
		{#each related.groups as g, gi (gi)}
			<Section id={gi === 0 ? 'sources-next' : `sources-${gi}`} title={g.title}>
				{#if g.lessons.length}
					<div class="rows">{#each g.lessons as l (l.slug)}<LessonRow lesson={l} sprites={related.sprites} />{/each}</div>
				{/if}
				{#if g.planned.length}
					<p class="planned">
						<b>{t({ en: 'Planned', zh: '规划中' })}:</b>
						{#each g.planned as p, i (p.href)}{#if i}{t({ en: ', ', zh: '、' })}{/if}<a href="{base}{p.href}">{t(p.title)}</a>{/each}
					</p>
				{/if}
			</Section>
		{/each}
	{/if}

	<Section id="sources-further" title={{ en: 'Further reading', zh: '延伸阅读' }}>
		<ul class="further">
			<li><a href="https://arxiv.org/abs/1508.07909" rel="external noopener">Sennrich, Haddow & Birch (2016), Neural Machine Translation of Rare Words with Subword Units</a>: {t({ en: 'the paper that brought BPE to neural networks (the low / lower / newest / widest example).', zh: '把 BPE 引入神经网络的论文（low / lower / newest / widest 的例子出自这里）。' })}</li>
			<li><a href="https://github.com/karpathy/minbpe" rel="external noopener">Andrej Karpathy, minbpe</a>: {t({ en: 'a short, readable BPE implementation, with a video lecture.', zh: '简短易读的 BPE 实现，附视频讲解。' })}</li>
			<li><a href="https://huggingface.co/learn/llm-course/chapter6/5" rel="external noopener">Hugging Face LLM course, chapter 6</a>: {t({ en: 'BPE, WordPiece and Unigram side by side.', zh: '并排讲解 BPE、WordPiece 和 Unigram。' })}</li>
			<li><a href="https://zh.d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html" rel="external noopener">{t({ en: 'Dive into Deep Learning: subword embedding (BPE), Chinese edition', zh: '《动手学深度学习》：子词嵌入（BPE）' })}</a> · <a href="https://d2l.ai/chapter_natural-language-processing-pretraining/subword-embedding.html" rel="external noopener">English</a></li>
			<li><a href="https://arxiv.org/abs/2305.15425" rel="external noopener">Petrov et al. (2023), Language Model Tokenizers Introduce Unfairness Between Languages</a>: {t({ en: 'measures the “same meaning, different price” effect across many languages.', zh: '在许多语言上测量了“同样的意思、不同的价钱”这一现象。' })}</li>
			<li><a href="https://github.com/openai/tiktoken" rel="external noopener">OpenAI tiktoken</a> · <a href="https://huggingface.co/spaces/Xenova/the-tokenizer-playground" rel="external noopener">The Tokenizer Playground</a></li>
		</ul>
	</Section>

	<Section id="sources-credits" title={{ en: 'Tokenizer files and licences', zh: '分词器文件与许可证' }}>
		<p class="credits">
			{t({
				en: `Tokenizer files on this page: GPT-2 (MIT), bert-base-uncased (Apache-2.0), cl100k_base (MIT, tiktoken), Qwen2.5 (Apache-2.0); ${fmtInt(578)} test strings checked against the reference implementations. About lists every source.`,
				zh: `本页用到的分词器文件：GPT-2（MIT）、bert-base-uncased（Apache-2.0）、cl100k_base（MIT，tiktoken）、Qwen2.5（Apache-2.0）；用 ${fmtInt(578)} 条测试文本与参考实现核对过。完整来源见“关于”页面。`
			})}
			<a href="{base}/about/">{t({ en: 'About', zh: '关于' })}</a>
		</p>
	</Section>
</div>

<style>
	.src {
		display: grid;
		gap: var(--s7);
	}
	.rows {
		display: grid;
		gap: 4px;
	}
	.planned {
		margin: var(--s3) 0 0;
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.further {
		margin: 0;
		padding-left: 1.2em;
		display: grid;
		gap: 8px;
		max-width: var(--measure);
	}
	.credits {
		margin: 0;
		max-width: var(--measure);
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
</style>
