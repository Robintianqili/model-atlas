<script lang="ts">
	import ConceptPageShell from '$lib/kit/ConceptPageShell.svelte';
	import ConceptCard from '$lib/kit/ConceptCard.svelte';
	import { CONCEPT_BY_SLUG, CONCEPTS, SECTION_BY_SLUG } from '$lib/catalog/concepts';
	import { techText } from '$lib/catalog/tech-zh';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const concept = $derived(CONCEPT_BY_SLUG[data.slug]);
	const section = $derived(SECTION_BY_SLUG[concept.section]);
	const siblings = $derived(CONCEPTS.filter((c) => c.section === concept.section && c.slug !== concept.slug));
	const modeText = $derived(
		concept.mode === 'live'
			? { en: 'Live: computed in your browser', zh: '实时：在你的浏览器中计算' }
			: concept.mode === 'precomputed'
				? { en: 'Precomputed: replayed from saved runs', zh: '预计算：回放保存的训练结果' }
				: { en: 'Both live and precomputed parts', zh: '包含实时与预计算两部分' }
	);
</script>

<!-- <title> and <meta name="description"> are set by ConceptPageShell -->
{#if data.component && data.auto}
	<!-- an included concept package: its auto page (a lazy chunk returned by +page.ts) -->
	{@const Auto = data.component}
	<Auto slug={data.auto.slug} autoBytes={data.auto.autoBytes} text={data.auto.text} />
{:else}
<ConceptPageShell slug={concept.slug} lede={section.blurb} titles={{ toy: ui('howYouPlay') }}>
	{#snippet toy()}
		<div class="planned-note" role="note">
			<strong>{ui('plannedPageTitle')}.</strong>
			{ui('plannedPageBody')}
		</div>
		<p class="big">{t(concept.interaction)}</p>
		<dl class="facts">
			<div><dt>{ui('section')}</dt><dd>{section.n} · {t(section.name)}</dd></div>
			<div><dt>{ui('engine')}</dt><dd>{t(techText(concept.engine))}</dd></div>
			<div><dt>{t({ en: 'Numbers', zh: '数字来源' })}</dt><dd>{t(modeText)}</dd></div>
			{#if concept.v1}<div><dt>{ui('tier')}</dt><dd>{ui('inV1')}</dd></div>{/if}
			{#if concept.tier === 'P2'}<div><dt>{ui('tier')}</dt><dd>{ui('tierP2')}</dd></div>{/if}
		</dl>
	{/snippet}
	{#snippet further()}
		{#if siblings.length}
			<h3>{t({ en: 'More in this section', zh: '本章节的其他概念' })}</h3>
			<div class="cards">{#each siblings as c (c.slug)}<ConceptCard concept={c} compact />{/each}</div>
		{/if}
	{/snippet}
</ConceptPageShell>
{/if}

<style>
	.planned-note {
		padding: 12px 16px;
		border: 1.5px dashed var(--planned);
		border-radius: var(--r3);
		background: var(--planned-soft);
		color: var(--ink-2);
		font-size: var(--fs-sm);
		margin-bottom: var(--s4);
	}
	.big {
		font-size: var(--fs-lg);
		max-width: var(--measure);
	}
	.facts {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 150px), 1fr));
		gap: 12px 20px;
		margin: 0;
		padding: 14px 16px;
		background: var(--surface);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
	}
	dt {
		font-size: var(--fs-xs);
		color: var(--muted);
		text-transform: uppercase;
		letter-spacing: 0.06em;
		font-family: var(--font-mono);
	}
	dd {
		margin: 0;
		font-size: var(--fs-sm);
		font-weight: 600;
	}
	h3 {
		font-size: var(--fs-md);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 240px), 1fr));
		gap: 10px;
		max-width: none;
	}
</style>
