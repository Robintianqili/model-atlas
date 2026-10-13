<!--
  The planned model page (redesign spec §6.9): one catalog page in the site chrome, at most about a screen and a half.
  The area glyph, the name, "Planned" in text, the family, the hook, what the lesson will show, one line of facts
  (inputs, runtime, size), the live lessons to take first (as lesson rows when the route passes them, else as links),
  and the way back to the area. No "Get notified" (there is no backend).

  The /m/[slug] route (lane W2's) renders it for a planned slug from the root layout's model slice:
    <PlannedPage catalog={page.data.catalog} before={data.before} sprites={data.sprites} />
  where the route's +page.server.ts may add, for the rows:
    before: liveLessons().filter((l) => MODEL_BY_SLUG[slug].prereqs.includes(l.slug)), sprites: catalogIndex().sprites
  (kit/catalog/lessons.ts). It sets its own <title> and description.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '../Icon.svelte';
	import LessonRow from './LessonRow.svelte';
	import { glyphFor, right } from './glyphs';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import { ct } from './text';
	import type { InputKind, ModelPageCatalog } from '$lib/catalog/types';
	import type { CatalogIndex, LessonItem } from './types';

	interface Props {
		catalog: ModelPageCatalog;
		/** the live prerequisites as lesson rows (from the route's server load) */
		before?: LessonItem[];
		sprites?: CatalogIndex['sprites'];
	}
	let { catalog: c, before, sprites = {} }: Props = $props();
	const m = $derived(c.model);
	const liveFirst = $derived(c.prereqs.filter((p) => p.status === 'live'));
	const inputName = (k: InputKind) => ui(`in_${k}` as Parameters<typeof ui>[0]);
</script>

<svelte:head>
	<title>{t(m.name)} ({ui('statusPlanned')}) · Model Atlas</title>
	<meta name="description" content={t(m.hook)} />
</svelte:head>

<div class="container planned">
	<nav class="crumbs" aria-label={t({ en: 'Breadcrumb', zh: '面包屑导航' })}>
		<a href="{base}/atlas/">{ct('allLessons')}</a><span class="sep" aria-hidden="true">/</span><a href="{base}/atlas/{m.area}/">{t(c.area)}</a>
	</nav>
	<header class="head">
		<p class="kick"><Icon d={glyphFor(m.area)} glyph size={18} />{t(c.area)} · {t(m.family)}</p>
		<h1>{t(m.name)} <span class="tag">{ui('statusPlanned')}</span></h1>
		<p class="lede">{t(m.hook)}</p>
	</header>

	<section class="block" aria-labelledby="shows-h">
		<h2 id="shows-h">{ct('willShow')}</h2>
		<p class="shows">{t(m.shows)}</p>
		<p class="facts">
			{#if m.inputs.length}<span><b>{ui('input')}</b> {m.inputs.map(inputName).join(' · ')}</span>{/if}
			<span><b>{ui('runtime')}</b> {t(c.tech.runtime)}</span>
			{#if c.tech.size}<span><b>{ui('download')}</b> <span class="num">{t(c.tech.size)}</span></span>{/if}
		</p>
		<p class="note">
			{t({
				en: 'This lesson is not built yet. It is listed so you can see where the atlas is going.',
				zh: '这节课还没有做好。列在这里，是为了让你看到图鉴接下来的方向。'
			})}
		</p>
	</section>

	{#if (before && before.length) || liveFirst.length}
		<section class="block" aria-labelledby="first-h">
			<h2 id="first-h">{ct('liveFirst')}</h2>
			{#if before && before.length}
				<ul class="rows">
					{#each before as l (l.slug)}<li><LessonRow lesson={l} {sprites} /></li>{/each}
				</ul>
			{:else}
				<ul class="links">
					{#each liveFirst as p (p.slug)}
						<li><a href="{base}/{p.kind === 'model' ? 'm' : 'train'}/{p.slug}/">{t(p.title)}</a>{#if p.family}<span>{t(p.family)}</span>{/if}</li>
					{/each}
				</ul>
			{/if}
		</section>
	{/if}

	<p class="back">
		<a href="{base}/atlas/{m.area}/">{t({ en: `More in ${c.area.en}`, zh: `${c.area.zh ?? c.area.en}领域的更多课程` })}<Icon d={right} size={16} /></a>
		<a href="{base}/atlas/">{ct('allLessons')}<Icon d={right} size={16} /></a>
	</p>
</div>

<style>
	.planned ul {
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
		padding: 12px 0 0;
		max-width: 760px;
	}
	.kick {
		display: flex;
		align-items: center;
		gap: 8px;
		margin: 0 0 8px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		color: var(--muted);
	}
	h1 {
		margin: 0 0 8px;
	}
	.tag {
		margin-left: 6px;
		padding: 2px 10px;
		border: 1px dashed var(--line-control);
		border-radius: var(--r-pill);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		letter-spacing: 0;
		color: var(--muted);
		vertical-align: middle;
	}
	.lede {
		margin: 0;
	}
	.block {
		max-width: 760px;
		padding-top: 28px;
	}
	.block h2 {
		margin: 0 0 8px;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
	}
	.shows {
		margin: 0;
	}
	.facts {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 20px;
		margin: 12px 0 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.facts b {
		font-weight: var(--w-ui);
		color: var(--muted);
		margin-right: 4px;
	}
	.note {
		margin: 16px 0 0;
		padding: 10px 14px;
		border: 1px dashed var(--line-control);
		border-radius: var(--r3);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.rows {
		display: grid;
		gap: 2px;
	}
	.links {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.links li {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 2px 10px;
	}
	.links a {
		font-weight: var(--w-ui);
	}
	.links span {
		font-size: var(--t-sm);
		color: var(--muted);
	}
	.back {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 24px;
		margin: 32px 0 0;
	}
	.back a {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		min-height: 44px;
		font-weight: var(--w-ui);
		color: var(--ink-2);
	}
</style>
