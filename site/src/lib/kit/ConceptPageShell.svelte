<!--
  ConceptPageShell: the fixed concept-page template (brief §4.4). Sections are snippet props;
  missing ones are omitted.

   1 goals + predict-first   2 hook (the question)   2b lesson (guided Walkthrough; adds a hero
   "Start the lesson" button and an "Explore freely" heading before the sections below)
   3 toy (every variable visible)
   4 real (the idea on a real model page)              5 compare (side-by-side runs)
   6 pitfalls and misconceptions                       7 post (back to your prediction)
   8 links (automatic: models from the catalog; append further reading with `further`)

  <ConceptPageShell slug="tokenization" goals={[…]} prequestion={q}>
    {#snippet hook()}…{/snippet}
    {#snippet toy()}…{/snippet}
  </ConceptPageShell>
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { base } from '$app/paths';
	import Section from './Section.svelte';
	import StatusBadge from './StatusBadge.svelte';
	import PredictFirst from './PredictFirst.svelte';
	import PredictReveal from './PredictReveal.svelte';
	import ModelCard from './ModelCard.svelte';
	import { CONCEPT_BY_SLUG, SECTION_BY_SLUG } from '$lib/catalog/concepts';
	import { MODEL_BY_SLUG } from '$lib/catalog/models';
	import { ui, t, colon, type Text } from '$lib/i18n/i18n.svelte';
	import type { UIKey } from '$lib/i18n/strings';
	import type { PredictQuestion, Provenance } from './types';

	type SectionKey = 'goals' | 'hook' | 'lesson' | 'toy' | 'real' | 'compare' | 'pitfalls' | 'post' | 'links';

	interface Props {
		slug: string;
		goals?: Text[];
		prequestion?: PredictQuestion;
		/** lede under the title; defaults to the catalog interaction summary */
		lede?: Text;
		hook?: Snippet;
		/** the guided lesson (kit Walkthrough), shown after the hook */
		lesson?: Snippet;
		/** one line under the hero's "Start the step-by-step lesson" button */
		lessonNote?: Text;
		/** intro under the "Explore freely" heading */
		exploreIntro?: Text;
		toy?: Snippet;
		real?: Snippet;
		compare?: Snippet;
		pitfalls?: Snippet;
		post?: Snippet;
		links?: Snippet;
		further?: Snippet;
		titles?: Partial<Record<SectionKey, Text>>;
		intros?: Partial<Record<SectionKey, Text>>;
		badges?: Partial<Record<SectionKey, Provenance>>;
	}
	let {
		slug,
		goals = [],
		prequestion,
		lede,
		hook,
		lesson,
		lessonNote,
		exploreIntro,
		toy,
		real,
		compare,
		pitfalls,
		post,
		links,
		further,
		titles = {},
		intros = {},
		badges = {}
	}: Props = $props();

	const concept = $derived(CONCEPT_BY_SLUG[slug]);
	const section = $derived(concept ? SECTION_BY_SLUG[concept.section] : undefined);
	const models = $derived((concept?.links ?? []).map((s) => MODEL_BY_SLUG[s]).filter(Boolean));
	const defaultTitle: Record<SectionKey, UIKey> = {
		goals: 'secGoals',
		hook: 'secHook',
		lesson: 'secLesson',
		toy: 'secToy',
		real: 'secReal',
		compare: 'secCompare',
		pitfalls: 'secPitfalls',
		post: 'secPost',
		links: 'secFurther'
	};
	const title = (k: SectionKey) => titles[k] ?? ui(defaultTitle[k]);
	const present = $derived(
		(
			[
				['goals', goals.length > 0 || !!prequestion],
				['hook', !!hook],
				['lesson', !!lesson],
				['toy', !!toy],
				['real', !!real],
				['compare', !!compare],
				['pitfalls', !!pitfalls],
				['post', !!prequestion || !!post],
				['links', true]
			] as [SectionKey, boolean][]
		)
			.filter(([, ok]) => ok)
			.map(([k]) => k)
	);
	const num = (k: SectionKey) => present.indexOf(k) + 1;
	const firstExplore = $derived(lesson ? present.find((k) => k !== 'goals' && k !== 'hook' && k !== 'lesson') : undefined);
</script>

<svelte:head>
	{#if concept}
		<title>{t(concept.title)}{concept.status === 'live' ? '' : ` (${ui('statusPlanned')})`} · Model Atlas</title>
		<meta name="description" content={t(concept.interaction)} />
	{/if}
</svelte:head>

{#if concept}
	<article class="cps container">
		<header class="hero">
			<p class="eyebrow">
				<a href="{base}/train/#{concept.section}">{section ? `${section.n} · ${t(section.name)}` : concept.section}</a>
			</p>
			<div class="title-row">
				<h1>{t(concept.title)}</h1>
				<StatusBadge status={concept.status} />
			</div>
			<p class="lede">{t(lede ?? concept.interaction)}</p>
			{#if lesson}
				<p class="start">
					<a class="btn primary big" href="#lesson">
						<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M4.5 2.8v10.4L13 8z" fill="currentColor" /></svg>
						{ui('lessonStart')}
					</a>
					{#if lessonNote}<span class="muted">{t(lessonNote)}</span>{/if}
				</p>
			{/if}
			{#if present.length > 2}
				<nav class="toc" aria-label={ui('onThisPage')}>
					<span class="muted">{ui('onThisPage')}{colon()}</span>
					{#each present as k (k)}
						{#if k === firstExplore}<span class="muted sep">· {ui('exploreFreely')}{colon()}</span>{/if}
						<a href="#{k}" class:strong={k === 'lesson'}>{t(title(k))}</a>
					{/each}
				</nav>
			{/if}
		</header>

		{#if present.includes('goals')}
			<Section id="goals" n={num('goals')} title={title('goals')} intro={intros.goals}>
				<div class="goals">
					{#if goals.length}
						<ul class="goal-list">{#each goals as g, i (i)}<li>{t(g)}</li>{/each}</ul>
					{/if}
					{#if prequestion}<PredictFirst q={prequestion} />{/if}
				</div>
			</Section>
		{/if}
		{#if hook}<Section id="hook" n={num('hook')} title={title('hook')} intro={intros.hook} provenance={badges.hook}>{@render hook()}</Section>{/if}
		{#if lesson}
			<Section id="lesson" n={num('lesson')} title={title('lesson')} intro={intros.lesson} provenance={badges.lesson}>{@render lesson()}</Section>
			{#if firstExplore}
				<div class="explore" id="explore">
					<h2>{ui('exploreFreely')}</h2>
					<p class="muted">{t(exploreIntro ?? ui('exploreFreelyIntro'))}</p>
				</div>
			{/if}
		{/if}
		{#if toy}<Section id="toy" n={num('toy')} title={title('toy')} intro={intros.toy} provenance={badges.toy}>{@render toy()}</Section>{/if}
		{#if real}<Section id="real" n={num('real')} title={title('real')} intro={intros.real} provenance={badges.real}>{@render real()}</Section>{/if}
		{#if compare}<Section id="compare" n={num('compare')} title={title('compare')} intro={intros.compare} provenance={badges.compare}>{@render compare()}</Section>{/if}
		{#if pitfalls}<Section id="pitfalls" n={num('pitfalls')} title={title('pitfalls')} intro={intros.pitfalls}><div class="prose">{@render pitfalls()}</div></Section>{/if}
		{#if present.includes('post')}
			<Section id="post" n={num('post')} title={title('post')} intro={intros.post}>
				<div class="post">
					{#if prequestion}<PredictReveal q={prequestion} />{/if}
					{#if post}{@render post()}{/if}
				</div>
			</Section>
		{/if}
		<Section id="links" n={num('links')} title={title('links')} intro={intros.links}>
			{#if links}
				{@render links()}
			{:else if models.length}
				<h3>{ui('linksToModels')}</h3>
				<div class="cards">{#each models as m (m.slug)}<ModelCard model={m} compact />{/each}</div>
			{/if}
			{#if further}<div class="further prose">{@render further()}</div>{/if}
		</Section>
	</article>
{:else}
	<p class="container">Unknown concept slug: {slug}</p>
{/if}

<style>
	.cps {
		padding-top: var(--s6);
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
	}
	.hero {
		display: grid;
		gap: 8px;
	}
	.eyebrow a {
		color: inherit;
	}
	.title-row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 8px 16px;
	}
	h1,
	.lede {
		margin: 0;
	}
	.toc {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
		font-size: var(--fs-sm);
		margin-top: 6px;
	}
	.toc .strong {
		font-weight: 700;
	}
	.start {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 14px;
		margin: 6px 0 0;
		font-size: var(--fs-sm);
	}
	.big {
		gap: 8px;
		min-height: 44px;
		padding-inline: 18px;
		font-size: var(--fs-md);
	}
	.explore {
		margin-top: var(--s5);
		padding: var(--s5) 0 0;
		border-top: 3px double var(--rule-strong);
	}
	.explore h2 {
		margin: 2px 0 4px;
		font-size: var(--fs-2xl);
	}
	.explore p {
		margin: 0;
		max-width: var(--measure);
	}
	.goals {
		display: grid;
		gap: var(--s4);
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
		align-items: start;
	}
	.goal-list {
		margin: 0;
		padding-left: 1.2em;
	}
	.goal-list li + li {
		margin-top: 6px;
	}
	.post {
		display: grid;
		gap: var(--s4);
	}
	h3 {
		font-size: var(--fs-md);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 230px), 1fr));
		gap: 10px;
	}
	.further {
		margin-top: var(--s4);
	}
</style>
