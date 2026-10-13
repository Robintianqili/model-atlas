<!--
  Sources (redesign spec §5.9; a lazy panel of the transfer-learning page, #sources, the old #links and #further): the
  related lessons as catalog rows (built at build time by the route's server load), and where to go deeper (our BERT
  reproduction, the glossary).
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
	import { BERT_SLIDES } from '../bert';

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

	<Section id="sources-further" title={{ en: 'Go deeper', zh: '继续深入' }}>
		<ul class="further">
			<li>
				<a href={BERT_SLIDES} rel="noopener">{t({ en: 'Our BERT reproduction (slides)', zh: '我们的 BERT 复现（幻灯片）' })}</a>:
				{t({ en: '218 fine-tuning runs: GLUE results, pre-training claims, fine-tuning claims.', zh: '218 次微调：GLUE 结果、预训练结论、微调结论。' })}
			</li>
			<li><a href="{base}/glossary/#transfer-learning">{t({ en: 'Glossary: transfer learning, fine-tuning, linear probe, frozen layer', zh: '术语表：迁移学习、微调、线性探针、冻结层' })}</a></li>
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
