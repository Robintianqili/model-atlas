<!--
  /train/<slug>/: an included concept package renders its auto page (the lesson app, a lazy chunk from +page.ts); every
  other catalog concept is a planned page (redesign spec §6.9): one catalog page with the site chrome, at most 1.5
  screens. Its data comes from +page.server.ts, so this route ships no catalog module.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import CatalogCard from '$lib/kit/CatalogCard.svelte';
	import Icon from '$lib/kit/Icon.svelte';
	import { glyphFor } from '$lib/kit/icons';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const p = $derived(data.planned);
	const modeText = $derived(
		p?.mode === 'live'
			? { en: 'Live: computed in your browser', zh: '实时：在你的浏览器中计算' }
			: p?.mode === 'precomputed'
				? { en: 'Precomputed: replayed from saved runs', zh: '预计算：回放保存的训练结果' }
				: { en: 'Both live and precomputed parts', zh: '包含实时与预计算两部分' }
	);
</script>

<svelte:head>
	{#if p}
		<title>{t(p.title)} ({ui('statusPlanned')}) · Model Atlas</title>
		<meta name="description" content={t(p.interaction)} />
	{/if}
</svelte:head>

{#if data.component && data.auto}
	<!-- an included concept package: its auto page (a lazy chunk returned by +page.ts) -->
	{@const Auto = data.component}
	<Auto slug={data.auto.slug} autoBytes={data.auto.autoBytes} text={data.auto.text} />
{:else if p}
	<article class="planned container">
		<p class="eyebrow"><Icon d={glyphFor(p.section.slug)} glyph size={18} /><a href="{base}/train/#{p.section.slug}">{p.section.n} · {t(p.section.name)}</a></p>
		<h1>{t(p.title)} <span class="tag">{ui('statusPlanned')}</span></h1>
		<p class="lede">{t(p.interaction)}</p>
		<p class="note" role="note"><strong>{ui('plannedPageTitle')}.</strong> {ui('plannedPageBody')}</p>
		<dl class="facts">
			<div><dt>{ui('section')}</dt><dd>{p.section.n} · {t(p.section.name)}</dd></div>
			<div><dt>{ui('engine')}</dt><dd>{t(p.engine)}</dd></div>
			<div><dt>{t({ en: 'Numbers', zh: '数字来源' })}</dt><dd>{t(modeText)}</dd></div>
			{#if p.v1}<div><dt>{ui('tier')}</dt><dd>{ui('inV1')}</dd></div>{:else if p.tier === 'P2'}<div><dt>{ui('tier')}</dt><dd>{ui('tierP2')}</dd></div>{/if}
		</dl>
		{#if p.models.length}
			<section aria-labelledby="mod-h">
				<h2 id="mod-h">{ui('linksToModels')}</h2>
				<div class="cards">{#each p.models as c (c.slug)}<CatalogCard card={c} compact />{/each}</div>
			</section>
		{/if}
		{#if p.siblings.length}
			<section aria-labelledby="sib-h">
				<h2 id="sib-h">{t({ en: 'More in this section', zh: '本章节的其他概念' })}</h2>
				<div class="cards">{#each p.siblings as c (c.slug)}<CatalogCard card={c} compact />{/each}</div>
			</section>
		{/if}
	</article>
{/if}

<style>
	.planned {
		display: grid;
		gap: var(--s4);
		padding-top: var(--s6);
		padding-bottom: var(--s7);
		max-width: 880px;
	}
	.planned > * {
		margin: 0;
	}
	.eyebrow {
		display: flex;
		align-items: center;
		gap: 8px;
		color: var(--ink-2);
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
	}
	.eyebrow a {
		color: inherit;
	}
	h1 {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 12px;
	}
	.tag {
		padding: 2px 10px;
		border: 1.5px dashed var(--line-control);
		border-radius: var(--r-pill);
		color: var(--ink-2);
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
		letter-spacing: 0;
	}
	.lede {
		max-width: var(--measure);
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
	}
	.note {
		padding: 12px 16px;
		border: 1.5px dashed var(--line-control);
		border-radius: var(--r3);
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	h2 {
		margin-bottom: var(--s2);
		font-size: var(--t-lg);
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 180px), 1fr));
		gap: 12px 20px;
		padding: 14px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
	}
	.facts dt {
		color: var(--muted);
		font-size: var(--t-xs);
		font-weight: var(--w-ui);
	}
	.facts dd {
		margin: 0;
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 230px), 1fr));
		gap: 10px;
	}
</style>
