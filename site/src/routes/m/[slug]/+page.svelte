<!--
  /m/<slug>/: an included explainer package renders its auto page (the lesson app, a lazy chunk from +page.ts); every
  other catalog model is a planned page (redesign spec §6.9): one catalog page with the site chrome, at most 1.5 screens:
  the area glyph, the name and family, "Planned" in words, the hook, what the lesson will show, the facts, the live
  prerequisites and more in the area. Everything comes from the page data (catalog slices computed at build time), so
  this route ships no catalog module.
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import CatalogCard from '$lib/kit/CatalogCard.svelte';
	import Icon from '$lib/kit/Icon.svelte';
	import { glyphFor } from '$lib/kit/icons';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { UIKey } from '$lib/i18n/strings';
	import type { ModelPageCatalog } from '$lib/catalog/types';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const cat = $derived((page.data as { catalog?: ModelPageCatalog | null }).catalog ?? null);
	const model = $derived(cat?.model);
	const tierKey: Record<string, UIKey> = { P0: 'tierP0', V1: 'tierV1', P1: 'tierP1', P2: 'tierP2' };
	const modeText = $derived(
		model?.mode === 'live'
			? { en: 'Live: computed in your browser', zh: '实时：在你的浏览器中计算' }
			: model?.mode === 'precomputed'
				? { en: 'Precomputed: replayed from saved model outputs', zh: '预计算：回放保存的模型输出' }
				: { en: 'Both: live where possible, precomputed for the heavy parts', zh: '两者兼有：能实时则实时，较重部分预计算' }
	);
	const livePrereqs = $derived((cat?.prereqs ?? []).filter((c) => c.status === 'live'));
	/** a lesson listed under "Learn these first" is not repeated under "More in <area>" (QA 2026-10-13) */
	const siblings = $derived(data.siblings.filter((c) => !livePrereqs.some((p) => p.slug === c.slug)));
</script>

<svelte:head>
	{#if model && !data.component}
		<title>{t(model.name)} ({ui('statusPlanned')}) · Model Atlas</title>
		<meta name="description" content={t(model.hook)} />
	{/if}
</svelte:head>

{#if data.component && data.auto}
	<!-- an included explainer package: its auto page (a lazy chunk returned by +page.ts) -->
	{@const Auto = data.component}
	<Auto slug={data.auto.slug} spec={data.auto.spec} autoBytes={data.auto.autoBytes} />
{:else if model && cat}
	<article class="planned container">
		<p class="eyebrow"><Icon d={glyphFor(model.area)} glyph size={18} /><a href="{base}/atlas/{model.area}/">{t(cat.area)}</a> · {t(model.family)}</p>
		<h1>{t(model.name)} <span class="tag">{ui('statusPlanned')}</span></h1>
		<p class="lede">{t(model.hook)}</p>
		<p class="note" role="note"><strong>{ui('plannedPageTitle')}.</strong> {ui('plannedPageBody')}</p>
		<section aria-labelledby="shows-h">
			<h2 id="shows-h">{ui('whatItWillShow')}</h2>
			<p class="shows">{t(model.shows)}</p>
			{#if model.notes}<p class="warn">{t(model.notes)}</p>{/if}
		</section>
		<dl class="facts">
			<div><dt>{ui('inputs')}</dt><dd>{model.inputs.map((k) => ui(`in_${k}` as UIKey)).join(', ')}</dd></div>
			<div><dt>{ui('runtime')}</dt><dd>{t(cat.tech.runtime)}</dd></div>
			{#if cat.tech.params}<div><dt>{ui('params')}</dt><dd class="num">{t(cat.tech.params)}</dd></div>{/if}
			{#if cat.tech.size}<div><dt>{ui('download')}</dt><dd class="num">{t(cat.tech.size)}</dd></div>{/if}
			<div><dt>{t({ en: 'Numbers', zh: '数字来源' })}</dt><dd>{t(modeText)}</dd></div>
			<div><dt>{t({ en: 'Model', zh: '模型来源' })}</dt><dd>{model.own ? t({ en: 'A tiny model we train ourselves', zh: '我们自己训练的小模型' }) : t({ en: 'An existing pretrained model', zh: '已有的预训练模型' })}</dd></div>
			<div><dt>{ui('tier')}</dt><dd>{ui(tierKey[model.tier])}{#if model.tierNote}<br /><span class="muted">{t(model.tierNote)}</span>{/if}</dd></div>
		</dl>
		{#if livePrereqs.length}
			<section aria-labelledby="pre-h">
				<h2 id="pre-h">{t({ en: 'Learn these first', zh: '可以先学这些' })}</h2>
				<div class="cards">{#each livePrereqs as c (c.slug)}<CatalogCard card={c} compact />{/each}</div>
			</section>
		{/if}
		{#if siblings.length}
			<section aria-labelledby="sib-h">
				<h2 id="sib-h">{t({ en: `More in ${cat.area.en}`, zh: `${cat.area.zh ?? cat.area.en}领域的更多模型` })}</h2>
				<div class="cards">{#each siblings as c (c.slug)}<CatalogCard card={c} compact showArea={false} />{/each}</div>
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
	.shows {
		max-width: var(--measure);
		margin: 0;
	}
	.warn {
		margin: 8px 0 0;
		padding-left: 10px;
		border-left: 3px solid var(--illus);
		color: var(--ink-2);
		font-size: var(--t-sm);
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
