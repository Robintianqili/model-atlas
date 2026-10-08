<script lang="ts">
	import ConceptCard from '$lib/kit/ConceptCard.svelte';
	import { CONCEPTS, conceptsBySection } from '$lib/catalog/concepts';
	import { t, ui } from '$lib/i18n/i18n.svelte';

	const groups = conceptsBySection();
	const live = CONCEPTS.filter((c) => c.status === 'live');
	let onlyV1 = $state(false);
</script>

<svelte:head><title>{ui('navTrain')} · Model Atlas</title></svelte:head>

<div class="container page">
	<header class="head">
		<p class="eyebrow">/train/</p>
		<h1>{ui('navTrain')}</h1>
		<p class="lede">
			{t({
				en: `How models are trained, pre-trained, fine-tuned, aligned, compressed and served: ${CONCEPTS.length} concepts in ${groups.length} sections. Each page pairs a small toy, where every variable is visible, with the same idea on a real model page.`,
				zh: `模型如何训练、预训练、微调、对齐、压缩与部署：${groups.length} 个章节中的 ${CONCEPTS.length} 个概念。每个页面都把一个所有变量都可见的小玩具，和真实模型页面上的同一个想法配对。`
			})}
		</p>
	</header>

	<section class="block" aria-labelledby="live-h">
		<h2 id="live-h">{ui('livePages')}</h2>
		<div class="cards">{#each live as c (c.slug)}<ConceptCard concept={c} showSection />{/each}</div>
	</section>

	<div class="filter">
		<label><input type="checkbox" bind:checked={onlyV1} /> {t({ en: 'Show only the v1 pages', zh: '只显示 v1 页面' })}</label>
	</div>

	<nav class="toc" aria-label={ui('onThisPage')}>
		{#each groups as g (g.section.slug)}<a href="#{g.section.slug}"><span class="num">{g.section.n}</span> {t(g.section.name)}</a>{/each}
	</nav>

	{#each groups as g (g.section.slug)}
		{@const items = onlyV1 ? g.items.filter((c) => c.v1) : g.items}
		{#if items.length}
			<section class="sec" id={g.section.slug} aria-labelledby="{g.section.slug}-h">
				<h2 id="{g.section.slug}-h"><span class="n num">{g.section.n}</span> {t(g.section.name)}</h2>
				<p class="muted">{t(g.section.blurb)} · {ui('countConcepts', { n: g.items.length })}</p>
				<div class="cards">{#each items as c (c.slug)}<ConceptCard concept={c} />{/each}</div>
			</section>
		{/if}
	{/each}
</div>

<style>
	.page {
		padding-top: var(--s6);
	}
	.head {
		display: grid;
		gap: 8px;
	}
	.head h1,
	.head p {
		margin: 0;
	}
	.block {
		padding-top: var(--s5);
	}
	.filter {
		margin-top: var(--s5);
		font-size: var(--fs-sm);
	}
	.filter input {
		accent-color: var(--accent);
	}
	.toc {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 16px;
		margin: var(--s3) 0;
		font-size: var(--fs-sm);
	}
	.sec {
		padding-top: var(--s5);
		scroll-margin-top: calc(var(--header-h) + 12px);
	}
	.sec h2 {
		display: flex;
		align-items: baseline;
		gap: 10px;
		margin-bottom: 4px;
		font-size: var(--fs-xl);
	}
	.n {
		font-size: var(--fs-sm);
		color: var(--accent);
	}
	.sec p {
		margin: 0 0 var(--s3);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 280px), 1fr));
		gap: 12px;
	}
</style>
