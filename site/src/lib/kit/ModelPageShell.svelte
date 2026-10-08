<!--
  ModelPageShell: the fixed per-model page template (brief §4.2). Every section is a snippet
  prop; sections a page does not provide are omitted (and left out of the "On this page" list).

   1 goals + predict-first   (goals: Text[], prequestion: PredictQuestion)
   2 facts                   (automatic from the catalog entry + spec.json; override with `facts`)
   3 input                   3b lesson (guided Walkthrough; adds a hero "Start the lesson" button and an
                                        "Explore freely" heading before the sections below)
   4 overview                5 inspector       6 unit (arithmetic)
   7 output                  8 tryThis         9 training       10 article
  11 code                   12 links (automatic: concepts using this model, prerequisites, next;
                                     override with `links`, or append with `moreLinks`)
  13 checkpoint (PredictReveal + your `checkpoint` snippet) + credits

  <ModelPageShell slug="tinycnn" {spec} goals={[…]} prequestion={q} provenance="live">
    {#snippet input()}<DrawPad … />{/snippet}
    {#snippet overview()}<ModelOverview … />{/snippet}
    …
  </ModelPageShell>
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import Section from './Section.svelte';
	import FactStrip from './FactStrip.svelte';
	import StatusBadge from './StatusBadge.svelte';
	import PredictFirst from './PredictFirst.svelte';
	import PredictReveal from './PredictReveal.svelte';
	import CatalogCard from './CatalogCard.svelte';
	import type { ModelPageCatalog } from '$lib/catalog/types';
	import { ui, t, colon, type Text } from '$lib/i18n/i18n.svelte';
	import type { UIKey } from '$lib/i18n/strings';
	import { fmtBytes, fmtInt, fmtShape } from './format';
	import type { ModelSpec } from './spec';
	import type { PredictQuestion, Provenance } from './types';

	interface Props {
		slug: string;
		/** the catalog entry and its cards (default: page.data.catalog, which the root layout's server load computes for every
		 *  /m/<slug>/ route and the Lesson Lab; catalog/slices.ts). The shell imports no catalog module, so a model page's JS
		 *  does not grow with the catalog. */
		catalog?: ModelPageCatalog | null;
		spec?: ModelSpec | null;
		/** measured download size of the files the page loads (defaults to the build-time measure) */
		downloadBytes?: number;
		provenance?: Provenance;
		goals?: Text[];
		prequestion?: PredictQuestion;
		/** sections */
		facts?: Snippet;
		input?: Snippet;
		/** the guided lesson (kit Walkthrough), shown right after the input */
		lesson?: Snippet;
		/** one line under the hero's "Start the step-by-step lesson" button, e.g. "10 steps, with your drawing" */
		lessonNote?: Text;
		/** intro under the "Explore freely" heading (default: a generic sentence) */
		exploreIntro?: Text;
		overview?: Snippet;
		inspector?: Snippet;
		unit?: Snippet;
		output?: Snippet;
		tryThis?: Snippet;
		training?: Snippet;
		article?: Snippet;
		code?: Snippet;
		links?: Snippet;
		moreLinks?: Snippet;
		checkpoint?: Snippet;
		credits?: Snippet;
		/** optional section titles/intros overriding the defaults */
		titles?: Partial<Record<SectionKey, Text>>;
		intros?: Partial<Record<SectionKey, Text>>;
		/** per-section provenance badges */
		badges?: Partial<Record<SectionKey, Provenance>>;
	}
	type SectionKey =
		| 'goals'
		| 'input'
		| 'lesson'
		| 'overview'
		| 'inspector'
		| 'unit'
		| 'output'
		| 'tryThis'
		| 'training'
		| 'article'
		| 'code'
		| 'links'
		| 'checkpoint';

	let {
		slug,
		catalog,
		spec = null,
		downloadBytes,
		provenance,
		goals = [],
		prequestion,
		facts,
		input,
		lesson,
		lessonNote,
		exploreIntro,
		overview,
		inspector,
		unit,
		output,
		tryThis,
		training,
		article,
		code,
		links,
		moreLinks,
		checkpoint,
		credits,
		titles = {},
		intros = {},
		badges = {}
	}: Props = $props();

	const cat = $derived.by(() => {
		const c = catalog ?? (page.data as { catalog?: ModelPageCatalog | null }).catalog;
		return c && c.model.slug === slug ? c : undefined;
	});
	const model = $derived(cat?.model);
	const area = $derived(cat?.area);
	const mode = $derived<Provenance>(provenance ?? (model?.status === 'live' ? 'live' : 'precomputed'));
	const measured = $derived(downloadBytes ?? (page.data as { liveSizes?: Record<string, number> })?.liveSizes?.[slug]);
	// Facts: the fetched spec once it is loaded; before that (and in the prerendered HTML) the
	// spec.json facts read at build time by the root +layout.server.ts.
	type SpecFacts = Pick<ModelSpec, 'params' | 'precision' | 'parity'> & { input?: { shape: (number | string)[] }; inputs?: { kind: string; shape: (number | string)[] }[] };
	const sf = $derived<SpecFacts | undefined>(
		spec ?? (page.data as { liveSpecs?: Record<string, SpecFacts> })?.liveSpecs?.[slug]
	);
	const usedBy = $derived(cat?.usedBy ?? []);
	const prereqs = $derived(cat?.prereqs ?? []);
	const next = $derived(cat?.next ?? []);

	const defaultTitle: Record<SectionKey, UIKey> = {
		goals: 'secGoals',
		input: 'secInput',
		lesson: 'secLesson',
		overview: 'secOverview',
		inspector: 'secInspector',
		unit: 'secUnit',
		output: 'secOutput',
		tryThis: 'secTryThis',
		training: 'secTraining',
		article: 'secArticle',
		code: 'secCode',
		links: 'secLinks',
		checkpoint: 'secCheckpoint'
	};
	const title = (k: SectionKey) => titles[k] ?? ui(defaultTitle[k]);

	/** an input kind's name (a multi-input spec's kinds; DevKit K1) */
	const inName = (k: string) => {
		try {
			return ui(`in_${k}` as UIKey) || k;
		} catch {
			return k;
		}
	};
	const inputLabel = $derived(
		model
			? sf?.inputs && sf.inputs.length > 1
				? [...new Set(sf.inputs.map((x) => x.kind))].map((k) => `${inName(k)} → ${sf.inputs!.filter((x) => x.kind === k).map((x) => fmtShape(x.shape)).join(', ')}`).join('; ')
				: model.inputs.map((k) => ui(`in_${k}` as UIKey)).join(', ') + (sf?.input?.shape ? ` → ${fmtShape(sf.input.shape)}` : '')
			: ''
	);
	const factList = $derived(
		model
			? [
					{ label: ui('area'), value: area ?? model.area },
					{ label: ui('family'), value: model.family },
					{ label: ui('params'), value: sf?.params ? fmtInt(sf.params) : (cat?.tech.params ?? '—'), mono: true },
					{ label: ui('download'), value: measured ? fmtBytes(measured) : (cat?.tech.size ?? '—'), mono: true },
					{ label: ui('runtime'), value: cat?.tech.runtime ?? model.runtime },
					{ label: ui('input'), value: inputLabel },
					{ label: ui('precision'), value: sf?.precision ?? '—', mono: true }
				]
			: []
	);

	// numbered sections in template order
	const present = $derived(
		(
			[
				['goals', goals.length > 0 || !!prequestion],
				['input', !!input],
				['lesson', !!lesson],
				['overview', !!overview],
				['inspector', !!inspector],
				['unit', !!unit],
				['output', !!output],
				['tryThis', !!tryThis],
				['training', !!training],
				['article', !!article],
				['code', !!code],
				['links', true],
				['checkpoint', !!prequestion || !!checkpoint || !!credits]
			] as [SectionKey, boolean][]
		)
			.filter(([, ok]) => ok)
			.map(([k]) => k)
	);
	const num = (k: SectionKey) => present.indexOf(k) + 1;
	/** with a lesson, the free-exploration sections start here (after the lesson) */
	const firstExplore = $derived(lesson ? present.find((k) => k !== 'goals' && k !== 'input' && k !== 'lesson') : undefined);
</script>

{#snippet exploreHead()}
	<div class="explore" id="explore">
		<h2>{ui('exploreFreely')}</h2>
		<p class="muted">{t(exploreIntro ?? ui('exploreFreelyIntro'))}</p>
	</div>
{/snippet}

<svelte:head>
	{#if model}
		<title>{t(spec?.name ?? model.name)}{model.status === 'live' ? '' : ` (${ui('statusPlanned')})`} · Model Atlas</title>
		<meta name="description" content={t(model.hook)} />
	{/if}
</svelte:head>

{#if model}
	<article class="mps container">
		<header class="hero">
			<p class="eyebrow">
				<a href="{base}/atlas/{model.area}/">{area ? t(area) : model.area}</a> · {t(model.family)}
			</p>
			<div class="title-row">
				<h1>{t(spec?.name ?? model.name)}</h1>
				<StatusBadge status={model.status} tier={model.tier} tooltip />
			</div>
			<p class="lede">{t(model.hook)}</p>
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

		{#if facts}
			{@render facts()}
		{:else}
			<FactStrip
				facts={factList}
				provenance={mode}
				precision={sf?.precision}
				tolerance={sf?.parity?.max_abs_diff}
				against={sf?.parity?.against}
			/>
		{/if}

		{#if present.includes('goals')}
			<Section id="goals" n={num('goals')} title={title('goals')} intro={intros.goals}>
				<div class="goals">
					{#if goals.length}
						<ul class="goal-list">
							{#each goals as g, i (i)}<li>{t(g)}</li>{/each}
						</ul>
					{/if}
					{#if prequestion}<PredictFirst q={prequestion} />{/if}
				</div>
			</Section>
		{/if}
		{#if input}
			<Section id="input" n={num('input')} title={title('input')} intro={intros.input} provenance={badges.input}>{@render input()}</Section>
		{/if}
		{#if lesson}
			<Section id="lesson" n={num('lesson')} title={title('lesson')} intro={intros.lesson} provenance={badges.lesson}>{@render lesson()}</Section>
			{#if firstExplore}{@render exploreHead()}{/if}
		{/if}
		{#if overview}
			<Section id="overview" n={num('overview')} title={title('overview')} intro={intros.overview} provenance={badges.overview}>{@render overview()}</Section>
		{/if}
		{#if inspector}
			<Section id="inspector" n={num('inspector')} title={title('inspector')} intro={intros.inspector} provenance={badges.inspector}>{@render inspector()}</Section>
		{/if}
		{#if unit}
			<Section id="unit" n={num('unit')} title={title('unit')} intro={intros.unit} provenance={badges.unit}>{@render unit()}</Section>
		{/if}
		{#if output}
			<Section id="output" n={num('output')} title={title('output')} intro={intros.output} provenance={badges.output}>{@render output()}</Section>
		{/if}
		{#if tryThis}
			<Section id="tryThis" n={num('tryThis')} title={title('tryThis')} intro={intros.tryThis} provenance={badges.tryThis}>{@render tryThis()}</Section>
		{/if}
		{#if training}
			<Section id="training" n={num('training')} title={title('training')} intro={intros.training} provenance={badges.training}>{@render training()}</Section>
		{/if}
		{#if article}
			<Section id="article" n={num('article')} title={title('article')} intro={intros.article}>
				<div class="prose">{@render article()}</div>
			</Section>
		{/if}
		{#if code}
			<Section id="code" n={num('code')} title={title('code')} intro={intros.code}>{@render code()}</Section>
		{/if}
		<Section id="links" n={num('links')} title={title('links')} intro={intros.links}>
			{#if links}
				{@render links()}
			{:else}
				<div class="links">
					{#if usedBy.length}
						<div>
							<h3>{ui('seeAlsoConcepts')}</h3>
							<div class="cards">
								{#each usedBy as c (c.slug)}<CatalogCard card={c} compact />{/each}
							</div>
						</div>
					{/if}
					{#if prereqs.length}
						<div>
							<h3>{ui('prerequisites')}</h3>
							<div class="cards">
								{#each prereqs as m (m.slug)}<CatalogCard card={m} compact />{/each}
							</div>
						</div>
					{/if}
					{#if next.length}
						<div>
							<h3>{t({ en: 'Next', zh: '下一步' })}</h3>
							<div class="cards">
								{#each next as m (m.slug)}<CatalogCard card={m} compact />{/each}
							</div>
						</div>
					{/if}
				</div>
			{/if}
			{#if moreLinks}{@render moreLinks()}{/if}
		</Section>
		{#if present.includes('checkpoint')}
			<Section id="checkpoint" n={num('checkpoint')} title={title('checkpoint')} intro={intros.checkpoint}>
				<div class="cp">
					{#if prequestion}<PredictReveal q={prequestion} />{/if}
					{#if checkpoint}{@render checkpoint()}{/if}
					{#if credits}
						<div class="credits">
							<h3>{ui('credits')}</h3>
							{@render credits()}
						</div>
					{/if}
				</div>
			</Section>
		{/if}
	</article>
{:else}
	<p class="container">Unknown model slug: {slug}</p>
{/if}

<style>
	.mps {
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
	h1 {
		margin: 0;
	}
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
	.links {
		display: grid;
		gap: var(--s5);
	}
	.links h3 {
		font-size: var(--fs-md);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 230px), 1fr));
		gap: 10px;
	}
	.cp {
		display: grid;
		gap: var(--s4);
	}
	.credits h3 {
		font-size: var(--fs-md);
	}
</style>
