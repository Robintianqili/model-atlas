<!--
  ModelPageShell (redesign spec §5, R1): a model page is its lesson. The shell is a view host: the Lesson (the
  Walkthrough of the `lesson` snippet, which becomes the whole page: top bar, stage | narration, action bar, cover) and,
  one tab away, Explore, How it works, Code and Training (hash views: #explore, #article, #code, #training; the old
  #overview, #inspector, #unit, #output, #tryThis, #checkpoint and #credits open the matching view). A view is a lazy
  panel (`views`, rendered only when it opens; `viewProps` is what it receives as `vp`) or, during the migration of the
  hand-built pages, the old section snippets, which render inside the matching view:
     Explore   overview, inspector, unit, output, tryThis, checkpoint (with PredictReveal), links, moreLinks
     Article   article                     Code   code                     Training   training, credits
  The facts strip and the credits move into the About sheet (ⓘ), the input into the input sheet (the stage header's
  chip opens it; `inputChip` / `inputSheet`, else the old `input` snippet), the goals and the predict-first question onto
  the lesson's cover. A tab exists only if its view has content.

  <ModelPageShell slug="tinycnn" {spec} goals={[…]} prequestion={q} views={{ explore: () => import('./ExplorePanel.svelte') }} viewProps={{ S }}>
    {#snippet lesson()}<Walkthrough … />{/snippet}
  </ModelPageShell>

  A page whose route returns { chrome: 'lesson' } is the lesson app (no site header; app mode at 960 x 600 and up);
  without it (a hand-built page before its migration) the lesson is a card under the site header.
-->
<script lang="ts">
	import { onMount, type Component, type Snippet } from 'svelte';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import Sheet from './Sheet.svelte';
	import PredictReveal from './PredictReveal.svelte';
	import CatalogCard from './CatalogCard.svelte';
	import Section from './Section.svelte';
	import { book, chart, code as codeIcon, compass, lesson as lessonIcon } from './icons';
	import type { ModelPageCatalog } from '$lib/catalog/types';
	import { lui, ui, t, type Text } from '$lib/i18n/i18n.svelte';
	import type { UIKey } from '$lib/i18n/strings';
	import { fmtBytes, fmtInt, fmtShape } from './format';
	import type { ModelSpec } from './spec';
	import type { PredictQuestion, Provenance, WalkContext } from './types';
	import { setLessonApp, type LessonApp, type ViewTab } from './lesson/app';
	import { parseHash, type ViewId } from './lesson/views';
	import ViewHost from './lesson/ViewHost.svelte';

	type Loader = () => Promise<{ default: Component<any> }>;
	interface Props {
		slug: string;
		/** the catalog entry and its cards (default: page.data.catalog, computed at build time; catalog/slices.ts) */
		catalog?: ModelPageCatalog | null;
		spec?: ModelSpec | null;
		/** measured download size of the files the page loads (defaults to the build-time measure) */
		downloadBytes?: number;
		provenance?: Provenance;
		goals?: Text[];
		prequestion?: PredictQuestion;
		/** the cover's promise (default: the catalog's lesson summary) */
		summary?: Text;
		/** reading time in minutes (default: the catalog's) */
		minutes?: number;
		/** R1: the views as lazy panels; each receives `vp` (viewProps) and `section` */
		views?: Partial<Record<Exclude<ViewId, 'lesson'>, Loader>>;
		viewProps?: Record<string, unknown>;
		/** R1: the stage header's input chip (it calls `open` to open the input sheet) and the sheet's content */
		inputChip?: Snippet<[() => void]>;
		inputSheet?: Snippet;
		/** R1: the cover's stage */
		cover?: Snippet<[WalkContext]>;
		/** the About sheet's body (default: the facts, the parity check and the credits) */
		about?: Snippet;
		/** sections (accepted during the migration; they render inside the matching view) */
		facts?: Snippet;
		input?: Snippet;
		lesson?: Snippet;
		lessonNote?: Text;
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
		titles?: Partial<Record<SectionKey, Text>>;
		intros?: Partial<Record<SectionKey, Text>>;
		badges?: Partial<Record<SectionKey, Provenance>>;
	}
	type SectionKey = 'goals' | 'input' | 'lesson' | 'overview' | 'inspector' | 'unit' | 'output' | 'tryThis' | 'training' | 'article' | 'code' | 'links' | 'checkpoint';

	let {
		slug,
		catalog,
		spec = null,
		downloadBytes,
		provenance,
		goals = [],
		prequestion,
		summary,
		minutes,
		views = {},
		viewProps = {},
		inputChip,
		inputSheet,
		cover,
		about,
		facts,
		input,
		lesson,
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
	type SpecFacts = Pick<ModelSpec, 'params' | 'precision' | 'parity'> & { input?: { shape: (number | string)[] }; inputs?: { kind: string; shape: (number | string)[] }[] };
	const sf = $derived<SpecFacts | undefined>(spec ?? (page.data as { liveSpecs?: Record<string, SpecFacts> })?.liveSpecs?.[slug]);
	const name = $derived<Text>(spec?.name ?? model?.name ?? slug);
	/** the page is the lesson app (its route returned chrome: 'lesson') */
	const embedded = $derived((page.data as { chrome?: string }).chrome !== 'lesson');

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
				: model.inputs.map((k) => inName(k)).join(', ') + (sf?.input?.shape ? ` → ${fmtShape(sf.input.shape)}` : '')
			: ''
	);
	const factList = $derived(
		model
			? [
					{ label: ui('area'), value: area ? t(area) : model.area },
					{ label: ui('family'), value: t(model.family) },
					{ label: ui('params'), value: sf?.params ? fmtInt(sf.params) : cat?.tech.params ? t(cat.tech.params) : '—', mono: true },
					{ label: ui('download'), value: measured ? fmtBytes(measured) : cat?.tech.size ? t(cat.tech.size) : '—', mono: true },
					{ label: ui('runtime'), value: cat ? t(cat.tech.runtime) : t(model.runtime) },
					{ label: ui('input'), value: inputLabel },
					{ label: ui('precision'), value: sf?.precision ?? '—', mono: true }
				]
			: []
	);
	const parity = $derived(sf?.parity && typeof sf.parity.max_abs_diff === 'number' ? sf.parity : null);

	// ------------------------------------------------------------------ views
	const hasExplore = $derived(!!views.explore || !!(overview || inspector || unit || output || tryThis || checkpoint || links || moreLinks));
	const tabs = $derived<ViewTab[]>(
		[
			{ id: 'lesson' as const, label: { en: 'Lesson', zh: '课程' }, icon: lessonIcon, note: { en: 'Step by step, on your input', zh: '一步一步，用你的输入' }, ok: true },
			{ id: 'explore' as const, label: { en: 'Explore', zh: '探索' }, icon: compass, note: { en: 'Every layer and number, freely', zh: '自由查看每一层、每个数' }, ok: hasExplore },
			{ id: 'article' as const, label: { en: 'How it works', zh: '原理' }, icon: book, note: { en: 'The idea in one article', zh: '一篇文章讲清原理' }, ok: !!views.article || !!article },
			{ id: 'code' as const, label: { en: 'Code', zh: '代码' }, icon: codeIcon, note: { en: 'The code, and how it was checked', zh: '代码，以及如何核对' }, ok: !!views.code || !!code },
			{ id: 'training' as const, label: { en: 'Training', zh: '训练' }, icon: chart, note: { en: 'How it learned, and from what', zh: '它怎么学的，用了什么数据' }, ok: !!views.training || !!training || !!credits }
		]
			.filter((v) => v.ok)
			.map(({ ok: _ok, ...v }) => v)
	);
	const upNext = $derived.by(() => {
		const c = (cat?.next ?? []).find((x) => x.status === 'live') ?? (cat?.usedBy ?? []).find((x) => x.status === 'live');
		return c ? { href: `${base}/${c.kind === 'concept' ? 'train' : 'm'}/${c.slug}/`, title: c.title, note: c.family ?? c.area } : undefined;
	});
	const inputOpen = $state({ open: false });
	const nav = $state<{ view: ViewId; section?: string }>({ view: 'lesson' });
	const pos = $state({ at: 0, n: 0, cover: false, id: 'lesson' } as LessonApp['pos']);
	const app: LessonApp = {
		get slug() {
			return slug;
		},
		kind: 'model',
		get name() {
			return name;
		},
		get crumbs() {
			return model ? [{ href: `${base}/atlas/`, label: { en: 'Atlas', zh: '图鉴' } }, { href: `${base}/atlas/${model.area}/`, label: area ?? model.area }] : [];
		},
		get tabs() {
			return tabs;
		},
		get embedded() {
			return embedded;
		},
		nav,
		pos,
		get goals() {
			return goals;
		},
		get prequestion() {
			return prequestion;
		},
		get summary() {
			return summary ?? model?.lesson?.summary;
		},
		get minutes() {
			return minutes ?? model?.lesson?.minutes;
		},
		get inputChip() {
			return inputChip;
		},
		get inputSheet() {
			return inputSheet ?? input;
		},
		get cover() {
			return cover;
		},
		get about() {
			return about ?? aboutBody;
		},
		aboutTitle: { en: 'About this model', zh: '关于这个模型' },
		get upNext() {
			return upNext;
		},
		input: inputOpen,
		go: (v, s) => {
			location.hash = v === 'lesson' ? 'lesson' : s ? `${v}-${s}` : v;
		}
	};
	setLessonApp(app);

	function sync() {
		const was = nav.view;
		const r = parseHash(location.hash, { kind: 'model', lessonId: pos.id, has: tabs.map((x) => x.id) });
		if (!r) return;
		nav.view = r.view;
		nav.section = r.section;
		if (r.input) inputOpen.open = true;
		// back in the lesson from a scrolled view: the lesson starts at the top of the page (flow mode)
		if (r.view === 'lesson' && was !== 'lesson' && !embedded) window.scrollTo({ top: 0 });
	}
	onMount(() => {
		sync();
		addEventListener('hashchange', sync);
		return () => removeEventListener('hashchange', sync);
	});
	const current = $derived(tabs.find((x) => x.id === nav.view) ?? tabs[0]);
	const sec = (k: SectionKey, fallback: Text) => titles[k] ?? fallback;
	const linkGroups = $derived(
		[
			{ h: ui('seeAlsoConcepts'), cards: cat?.usedBy ?? [] },
			{ h: ui('prerequisites'), cards: cat?.prereqs ?? [] },
			{ h: t({ en: 'Next', zh: '下一步' }), cards: cat?.next ?? [] }
		].filter((g) => g.cards.length)
	);
</script>

{#snippet aboutBody()}
	<dl class="facts">
		{#each factList as f (f.label)}<div><dt>{f.label}</dt><dd class:num={f.mono}>{f.value}</dd></div>{/each}
	</dl>
	<p class="ab-note">
		<strong>{mode === 'live' ? ui('provLive') : ui('provPrecomputed')}.</strong>
		{mode === 'live' ? ui('provLiveHelp') : ui('provPrecomputedHelp')}
		{#if parity}{t({ en: `Checked against ${parity.against ?? 'PyTorch'}: every layer within ${(parity.max_abs_diff as number).toExponential(2)}.`, zh: `与 ${parity.against ?? 'PyTorch'} 对照：每一层的误差都在 ${(parity.max_abs_diff as number).toExponential(2)} 以内。` })}{/if}
	</p>
	{#if facts}<div class="ab-more">{@render facts()}</div>{/if}
	{#if credits}<div class="ab-more"><h3>{ui('credits')}</h3>{@render credits()}</div>{/if}
{/snippet}

{#snippet exploreContent()}
	{#if overview}<Section id="explore-overview" title={sec('overview', ui('secOverview'))} intro={intros.overview} provenance={badges.overview}>{@render overview()}</Section>{/if}
	{#if inspector}<Section id="explore-inspector" title={sec('inspector', ui('secInspector'))} intro={intros.inspector} provenance={badges.inspector}>{@render inspector()}</Section>{/if}
	{#if unit}<Section id="explore-unit" title={sec('unit', ui('secUnit'))} intro={intros.unit} provenance={badges.unit}>{@render unit()}</Section>{/if}
	{#if output}<Section id="explore-output" title={sec('output', ui('secOutput'))} intro={intros.output} provenance={badges.output}>{@render output()}</Section>{/if}
	{#if tryThis}<Section id="explore-tryThis" title={sec('tryThis', ui('secTryThis'))} intro={intros.tryThis} provenance={badges.tryThis}>{@render tryThis()}</Section>{/if}
	{#if checkpoint || prequestion}
		<Section id="explore-checkpoint" title={sec('checkpoint', ui('secCheckpoint'))} intro={intros.checkpoint}>
			<div class="cp">
				{#if prequestion}<PredictReveal q={prequestion} />{/if}
				{#if checkpoint}{@render checkpoint()}{/if}
			</div>
		</Section>
	{/if}
	<Section id="explore-links" title={sec('links', ui('secLinks'))} intro={intros.links}>
		{#if links}
			{@render links()}
		{:else}
			<div class="links">
				{#each linkGroups as g (g.h)}
					<div>
						<h3>{g.h}</h3>
						<div class="cards">{#each g.cards as c (c.slug)}<CatalogCard card={c} compact />{/each}</div>
					</div>
				{/each}
			</div>
		{/if}
		{#if moreLinks}{@render moreLinks()}{/if}
	</Section>
{/snippet}
{#snippet articleContent()}
	{#if article}<div class="prose">{@render article()}</div>{/if}
{/snippet}
{#snippet codeContent()}
	{#if code}{@render code()}{/if}
{/snippet}
{#snippet trainingContent()}
	{#if training}<Section id="training-data" title={sec('training', ui('secTraining'))} intro={intros.training} provenance={badges.training}>{@render training()}</Section>{/if}
	{#if credits}<Section id="training-credits" title={ui('credits')}>{@render credits()}</Section>{/if}
{/snippet}

<svelte:head>
	{#if model}
		<title>{t(name)}{model.status === 'live' ? '' : ` (${ui('statusPlanned')})`} · Model Atlas</title>
		<meta name="description" content={t(model.hook)} />
	{/if}
</svelte:head>

{#if model}
	<div class="lapp" class:embedded data-view={nav.view}>
		<div class="lv" hidden={nav.view !== 'lesson' || undefined}>
			{#if lesson}{@render lesson()}{/if}
		</div>
		{#if nav.view !== 'lesson' && current}
			<ViewHost
				{app}
				view={nav.view}
				section={nav.section}
				title={current.label}
				intro={nav.view === 'explore' ? (exploreIntro ?? current.note) : current.note}
				loader={views[nav.view as Exclude<ViewId, 'lesson'>]}
				vp={viewProps}
				content={nav.view === 'explore' ? exploreContent : nav.view === 'article' ? articleContent : nav.view === 'code' ? codeContent : trainingContent}
				width={nav.view === 'article' ? 'article' : nav.view === 'training' ? 'narrow' : 'wide'}
			/>
		{/if}
	</div>
	{#if inputSheet || input}
		<Sheet bind:open={inputOpen.open} title={lui('yourInput')} width={720}>
			{#if inputSheet}{@render inputSheet()}{:else if input}{@render input()}{/if}
		</Sheet>
	{/if}
{:else}
	<p class="container">Unknown model slug: {slug}</p>
{/if}

<style>
	.lapp {
		background: var(--panel);
	}
	.lapp.embedded {
		max-width: var(--page-max);
		margin: var(--s4) auto var(--s6);
		padding: 0 var(--gutter);
		background: none;
	}
	.lv[hidden] {
		display: none;
	}
	.facts {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 10px 16px;
		margin: 0;
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
		overflow-wrap: anywhere;
	}
	.ab-note {
		margin: var(--s4) 0 0;
		padding-top: var(--s3);
		border-top: 1px solid var(--rule);
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.ab-more {
		margin-top: var(--s4);
		padding-top: var(--s3);
		border-top: 1px solid var(--rule);
		font-size: var(--t-sm);
	}
	.ab-more h3 {
		font-size: var(--t-md);
	}
	.cp {
		display: grid;
		gap: var(--s4);
	}
	.links {
		display: grid;
		gap: var(--s5);
	}
	.links h3 {
		font-size: var(--t-md);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 230px), 1fr));
		gap: 10px;
	}
</style>
