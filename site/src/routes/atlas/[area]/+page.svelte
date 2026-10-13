<!--
  An area page (redesign spec §6.5): the area's one-line intro and counts, its learning path when one exists, all its
  live lessons as rows, every planned model with what it will show (no tier codes), and the training methods whose
  lessons use these models. Data from the build-time index (+page.server.ts).
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '$lib/kit/Icon.svelte';
	import LessonRow from '$lib/kit/catalog/LessonRow.svelte';
	import PathLine from '$lib/kit/catalog/PathLine.svelte';
	import SectionChip from '$lib/kit/catalog/SectionChip.svelte';
	import { glyphFor, left, right } from '$lib/kit/catalog/glyphs';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import { ct } from '$lib/kit/catalog/text';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const g = $derived(data.group);
	const SHOWN = 6;
	let allPlanned = $state(false);
	const planned = $derived(allPlanned ? data.planned : data.planned.slice(0, SHOWN));
</script>

<svelte:head><title>{t(g.name)} · {ct('allLessons')} · Model Atlas</title></svelte:head>

<div class="container area">
	<nav class="crumbs" aria-label={t({ en: 'Breadcrumb', zh: '面包屑导航' })}>
		<a href="{base}/atlas/">{ct('allLessons')}</a><span class="sep" aria-hidden="true">/</span><span aria-current="page">{t(g.name)}</span>
	</nav>
	<header class="head">
		<h1><Icon d={glyphFor(g.slug)} glyph size={28} />{t(g.name)}</h1>
		<p class="lede">{t(g.blurb)}</p>
		<p class="meta">
			{g.live ? ct('liveAndPlanned', { n: g.live, m: g.planned }) : t({ en: `${g.planned} planned, none live yet`, zh: `${g.planned} 节计划中，暂无可学` })}
			{#if g.examples}{' · '}{t(g.examples)}{/if}
		</p>
	</header>

	{#if data.path}
		<section class="block" aria-labelledby="path-h">
			<h2 id="path-h" class="sub">{t({ en: 'A path through this area', zh: '这个领域的一条路径' })}</h2>
			<PathLine path={data.path} />
		</section>
	{/if}

	{#if data.lessons.length}
		<section class="block" aria-labelledby="live-h">
			<h2 id="live-h" class="sub">{t({ en: `Lessons you can run now · ${data.lessons.length}`, zh: `现在可以学的课程 · ${data.lessons.length}` })}</h2>
			<ul class="rows">
				{#each data.lessons as l (l.slug)}
					<li><LessonRow lesson={l} sprites={data.sprites} /></li>
				{/each}
			</ul>
		</section>
	{/if}

	{#if data.planned.length}
		<section class="block" aria-labelledby="planned-h">
			<h2 id="planned-h" class="sub">{ct('plannedN', { n: data.planned.length })}</h2>
			<ul class="plist">
				{#each planned as p (p.slug)}
					<li>
						<a href="{base}{p.href}">{t(p.title)}</a>
						{#if p.shows}<span>{t(p.shows)}</span>{/if}
					</li>
				{/each}
			</ul>
			{#if data.planned.length > SHOWN && !allPlanned}
				<button type="button" class="btn more" onclick={() => (allPlanned = true)}>{t({ en: `Show ${data.planned.length - SHOWN} more`, zh: `再显示 ${data.planned.length - SHOWN} 个` })}</button>
			{/if}
		</section>
	{/if}

	{#if data.sections.length}
		<section class="block" aria-labelledby="methods-h">
			<h2 id="methods-h" class="sub">{t({ en: 'Training methods that use these models', zh: '用到这些模型的训练方法' })}</h2>
			<ul class="secs">
				{#each data.sections as s (s.slug)}
					<li><SectionChip group={s} /></li>
				{/each}
			</ul>
		</section>
	{/if}

	<nav class="pn" aria-label={ui('areas')}>
		<a href="{base}/atlas/{data.prev.slug}/"><Icon d={left} size={16} />{t(data.prev.name)}</a>
		<a href="{base}/atlas/{data.next.slug}/">{t(data.next.name)}<Icon d={right} size={16} /></a>
	</nav>
</div>

<style>
	.area ul {
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.crumbs {
		padding-top: 24px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.crumbs a {
		color: var(--ink-2);
	}
	.sep {
		margin: 0 8px;
	}
	.head {
		padding: 12px 0 4px;
	}
	h1 {
		display: flex;
		align-items: center;
		gap: 12px;
		margin: 0 0 6px;
	}
	.lede {
		margin: 0 0 6px;
	}
	.meta {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.block {
		padding-top: 32px;
	}
	.sub {
		margin: 0 0 12px;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
	}
	.block :global(.path) {
		border-top: 1px solid var(--rule);
	}
	.rows {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 2px 28px;
	}
	.plist {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 10px 32px;
	}
	.plist li {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding-left: 12px;
		border-left: 2px dashed var(--line-control);
	}
	.plist a {
		font-weight: var(--w-ui);
		color: var(--ink-2);
	}
	.plist span {
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.more {
		margin-top: 14px;
	}
	.secs {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 12px;
	}
	.pn {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		margin-top: 40px;
		padding-top: 16px;
		border-top: 1px solid var(--rule);
	}
	.pn a {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		min-height: 44px;
		font-weight: var(--w-ui);
		color: var(--ink-2);
	}
	@media (max-width: 959px) {
		.rows,
		.plist {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
