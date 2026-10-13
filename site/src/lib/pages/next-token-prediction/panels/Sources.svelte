<!--
  Sources (redesign spec §5.9; a lazy panel of the next-token-prediction page, #sources, the old #links and #further):
  the related lessons as catalog rows (built at build time by the route's server load) and further reading.
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

	let { vp }: { vp: { related?: Related }; section?: string } = $props();
	const related = $derived(vp.related);
</script>

<div class="src">
	{#if related}
		{#each related.groups as g, gi (gi)}
			<Section id={`sources-${gi}`} title={g.title}>
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
			<li>
				<a href="http://gltr.io/" rel="noopener">GLTR</a>:
				{t({ en: 'Gehrmann, Strobelt and Rush (2019): coloring text by the rank a language model gives each token (the idea behind the rank view in Explore).', zh: 'Gehrmann、Strobelt 与 Rush（2019）：按语言模型给每个词元的排名给文本上色（“探索”里排名视图的思路来源）。' })}
			</li>
			<li>
				<a href="https://arxiv.org/abs/2305.07759" rel="noopener">TinyStories</a>:
				{t({ en: 'Eldan and Li (2023): the children’s-story dataset TinyGPT was trained on, and why small models can write coherent English.', zh: 'Eldan 与 Li（2023）：TinyGPT 所用的儿童故事数据集，以及为什么小模型也能写出连贯的英文。' })}
			</li>
			<li>
				{t({ en: 'Dive into Deep Learning, “Language models”:', zh: '《动手学深度学习》“语言模型”一节：' })}
				<a href="https://d2l.ai/chapter_recurrent-neural-networks/language-model.html" rel="noopener">English</a> ·
				<a href="https://zh.d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html" rel="noopener">中文</a>
			</li>
			<li>
				<a href="https://github.com/karpathy/nanoGPT" rel="noopener">nanoGPT</a>:
				{t({ en: 'Andrej Karpathy’s minimal GPT training code; TinyGPT follows the same recipe at a much smaller size.', zh: 'Andrej Karpathy 的极简 GPT 训练代码；TinyGPT 用的是同样的做法，只是小得多。' })}
			</li>
			<li>
				<a href="https://animatedllm.github.io/" rel="noopener">AnimatedLLM</a>:
				{t({ en: 'animated walk-through of how a decoder language model predicts the next token.', zh: '用动画逐步展示解码器语言模型如何预测下一个词元。' })}
			</li>
		</ul>
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
</style>
