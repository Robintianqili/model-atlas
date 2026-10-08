<script lang="ts">
	import { base } from '$app/paths';
	import ModelCard from '$lib/kit/ModelCard.svelte';
	import ConceptCard from '$lib/kit/ConceptCard.svelte';
	import { AREAS } from '$lib/catalog/areas';
	import { MODELS, sortModels } from '$lib/catalog/models';
	import { CONCEPTS } from '$lib/catalog/concepts';
	import type { Tier } from '$lib/catalog/types';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { UIKey } from '$lib/i18n/strings';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const area = $derived(data.area);
	const models = $derived(sortModels(MODELS.filter((m) => m.area === area.slug)));
	const live = $derived(models.filter((m) => m.status === 'live'));
	const start = $derived(live[0] ?? models.find((m) => m.tier === 'P0' || m.tier === 'V1') ?? models[0]);
	const tiers: Tier[] = ['P0', 'V1', 'P1', 'P2'];
	const tierKey: Record<Tier, UIKey> = { P0: 'tierP0', V1: 'tierV1', P1: 'tierP1', P2: 'tierP2' };
	const groups = $derived(
		tiers.map((tr) => ({ tier: tr, items: models.filter((m) => m.status !== 'live' && m.tier === tr) })).filter((g) => g.items.length)
	);
	const slugs = $derived(new Set(models.map((m) => m.slug)));
	const concepts = $derived(CONCEPTS.filter((c) => c.links.some((l) => slugs.has(l))).slice(0, 8));
	const idx = $derived(AREAS.findIndex((a) => a.slug === area.slug));
	const prev = $derived(AREAS[(idx - 1 + AREAS.length) % AREAS.length]);
	const next = $derived(AREAS[(idx + 1) % AREAS.length]);
</script>

<svelte:head><title>{t(area.name)} · {ui('navAtlas')} · Model Atlas</title></svelte:head>

<div class="container page">
	<nav class="crumbs" aria-label="Breadcrumb">
		<a href="{base}/atlas/">{ui('navAtlas')}</a> / <span>{t(area.name)}</span>
	</nav>
	<header class="head">
		<p class="eyebrow num">{String(area.n).padStart(2, '0')} / {AREAS.length}</p>
		<h1>{t(area.name)}</h1>
		<p class="lede">{t(area.blurb)}</p>
		<p class="muted">
			{ui('countModels', { n: models.length })} · {ui('countLive', { n: live.length })} · {ui('countPlanned', { n: models.length - live.length })}
			· {t(area.examples)}
		</p>
	</header>

	{#if start}
		<section class="block" aria-labelledby="start-h">
			<h2 id="start-h">{t({ en: 'Start here', zh: '从这里开始' })}</h2>
			<div class="start"><ModelCard model={start} showArea={false} /></div>
		</section>
	{/if}

	{#if live.length > 1 || (live.length === 1 && live[0] !== start)}
		<section class="block" aria-labelledby="live-h">
			<h2 id="live-h">{ui('livePages')}</h2>
			<div class="cards">{#each live as m (m.slug)}<ModelCard model={m} showArea={false} />{/each}</div>
		</section>
	{/if}

	{#each groups as g (g.tier)}
		<section class="block" aria-labelledby="tier-{g.tier}">
			<h2 id="tier-{g.tier}">{ui('plannedPages')} · <span class="num">{ui(tierKey[g.tier])}</span></h2>
			<div class="cards">{#each g.items as m (m.slug)}<ModelCard model={m} showArea={false} />{/each}</div>
		</section>
	{/each}

	{#if concepts.length}
		<section class="block" aria-labelledby="concepts-h">
			<h2 id="concepts-h">{ui('seeAlsoConcepts')}</h2>
			<div class="cards">{#each concepts as c (c.slug)}<ConceptCard concept={c} compact showSection />{/each}</div>
		</section>
	{/if}

	<nav class="pn" aria-label={ui('areas')}>
		<a href="{base}/atlas/{prev.slug}/">← {t(prev.name)}</a>
		<a href="{base}/atlas/{next.slug}/">{t(next.name)} →</a>
	</nav>
</div>

<style>
	.page {
		padding-top: var(--s5);
	}
	.crumbs {
		font-size: var(--fs-sm);
		color: var(--muted);
		margin-bottom: var(--s3);
	}
	.head {
		display: grid;
		gap: 6px;
	}
	.head h1,
	.head p {
		margin: 0;
	}
	.block {
		padding-top: var(--s5);
	}
	.block h2 {
		font-size: var(--fs-xl);
	}
	.start {
		max-width: 420px;
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 260px), 1fr));
		gap: 12px;
	}
	.pn {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		margin-top: var(--s7);
		padding-top: var(--s4);
		border-top: 1px solid var(--rule);
	}
</style>
