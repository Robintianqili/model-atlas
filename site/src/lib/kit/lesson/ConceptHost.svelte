<!--
  ConceptHost (redesign W2): the concept page's view host, from resolved data (no catalog module): the Lesson (the
  Walkthrough of the `lesson` snippet: the whole page, with its cover) and, one tab away, Explore, Pitfalls and Sources
  (#explore, #pitfalls, #sources; the old #toy, #real, #compare open Explore at that section, #post and #checkpoint the
  Pitfalls view, #links and #further the Sources view). A view is a lazy panel (`views`; `viewProps` is what it
  receives as `vp`) or the page's own section snippets, which render inside the matching view:
     Explore   hook, toy, real, compare         Pitfalls   pitfalls, post (with PredictReveal)
     Sources   links (the concept's models), further
  ConceptPageShell (the hand-built pages) and AutoConceptPage (concept packages, with the data from the route's server
  load, so the lesson's JS carries no catalog module) both render it.
-->
<script lang="ts" module>
	import type { CardData } from '$lib/catalog/types';
	import type { I18n } from '$lib/i18n/types';
	/** What the page shows about its concept (resolved: bilingual engine text, the section's name, model cards). */
	export interface ConceptPageData {
		slug: string;
		title: I18n;
		status: 'live' | 'planned';
		interaction: I18n;
		engine: I18n;
		mode: string;
		section: { slug: string; n: string | number; name: I18n } | null;
		lesson?: { summary?: I18n; minutes?: number };
		models: CardData[];
	}
</script>

<script lang="ts">
	import { onMount, type Component, type Snippet } from 'svelte';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import Sheet from '../Sheet.svelte';
	import Section from '../Section.svelte';
	import PredictReveal from '../PredictReveal.svelte';
	import CatalogCard from '../CatalogCard.svelte';
	import { compass, help, lesson as lessonIcon, sources } from '../icons';
	import { lui, ui, t, type Text } from '$lib/i18n/i18n.svelte';
	import type { UIKey } from '$lib/i18n/strings';
	import type { PredictQuestion, Provenance, WalkContext } from '../types';
	import { setLessonApp, type LessonApp, type ViewTab } from './app';
	import { parseHash, type ViewId } from './views';
	import ViewHost from './ViewHost.svelte';

	type SectionKey = 'goals' | 'hook' | 'lesson' | 'toy' | 'real' | 'compare' | 'pitfalls' | 'post' | 'links';
	type Loader = () => Promise<{ default: Component<any> }>;

	interface Props {
		concept: ConceptPageData | null;
		goals?: Text[];
		prequestion?: PredictQuestion;
		lede?: Text;
		summary?: Text;
		minutes?: number;
		views?: Partial<Record<Exclude<ViewId, 'lesson'>, Loader>>;
		viewProps?: Record<string, unknown>;
		inputChip?: Snippet<[() => void]>;
		inputSheet?: Snippet;
		cover?: Snippet<[WalkContext]>;
		about?: Snippet;
		hook?: Snippet;
		lesson?: Snippet;
		lessonNote?: Text;
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
		concept,
		goals = [],
		prequestion,
		lede,
		summary,
		minutes,
		views = {},
		viewProps = {},
		inputChip,
		inputSheet,
		cover,
		about,
		hook,
		lesson,
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

	const slug = $derived(concept?.slug ?? '');
	const section = $derived(concept?.section ?? null);
	const models = $derived(concept?.models ?? []);
	const embedded = $derived((page.data as { chrome?: string }).chrome !== 'lesson');
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

	const tabs = $derived<ViewTab[]>(
		[
			{ id: 'lesson' as const, label: { en: 'Lesson', zh: '课程' }, icon: lessonIcon, note: { en: 'Step by step, with live numbers', zh: '一步一步，带实时数字' }, ok: true },
			{ id: 'explore' as const, label: { en: 'Explore', zh: '探索' }, icon: compass, note: { en: 'Turn the dials yourself', zh: '自己动手调一调' }, ok: !!views.explore || !!(hook || toy || real || compare) },
			{ id: 'pitfalls' as const, label: { en: 'Pitfalls', zh: '常见误区' }, icon: help, note: { en: 'What people get wrong, and a check', zh: '常见误解，以及自测' }, ok: !!views.pitfalls || !!pitfalls || !!post || !!prequestion },
			{ id: 'sources' as const, label: { en: 'Sources', zh: '资料来源' }, icon: sources, note: { en: 'Models that use it, further reading, credits', zh: '用到它的模型、延伸阅读、致谢' }, ok: true }
		]
			.filter((v) => v.ok)
			.map(({ ok: _ok, ...v }) => v)
	);
	const upNext = $derived.by(() => {
		const m = models.find((x) => x.status === 'live');
		return m ? { href: `${base}/m/${m.slug}/`, title: m.title, note: m.family } : undefined;
	});
	const inputOpen = $state({ open: false });
	const nav = $state<{ view: ViewId; section?: string }>({ view: 'lesson' });
	const pos = $state({ at: 0, n: 0, cover: false, id: 'lesson' } as LessonApp['pos']);
	const app: LessonApp = {
		get slug() {
			return slug;
		},
		kind: 'concept',
		get name() {
			return concept?.title ?? slug;
		},
		crumbs: [{ href: `${base}/train/`, label: { en: 'Training methods', zh: '训练方法' } }],
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
			return summary ?? lede ?? concept?.lesson?.summary ?? concept?.interaction;
		},
		get minutes() {
			return minutes ?? concept?.lesson?.minutes;
		},
		get inputChip() {
			return inputChip;
		},
		get inputSheet() {
			return inputSheet;
		},
		get cover() {
			return cover;
		},
		get about() {
			return about ?? aboutBody;
		},
		aboutTitle: { en: 'About this lesson', zh: '关于这节课' },
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
		const r = parseHash(location.hash, { kind: 'concept', lessonId: pos.id, has: tabs.map((x) => x.id) });
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
</script>

{#snippet aboutBody()}
	{#if concept}
		<dl class="facts">
			{#if section}<div><dt>{ui('section')}</dt><dd>{section.n} · {t(section.name)}</dd></div>{/if}
			<div><dt>{ui('engine')}</dt><dd>{t(concept.engine)}</dd></div>
			<div><dt>{t({ en: 'Numbers', zh: '数字来源' })}</dt><dd>{concept.mode === 'live' ? ui('provLive') : concept.mode === 'precomputed' ? ui('provPrecomputed') : `${ui('provLive')} + ${ui('provPrecomputed')}`}</dd></div>
		</dl>
		<p class="ab-note">{t(concept.interaction)}</p>
	{/if}
{/snippet}

{#snippet exploreContent()}
	{#if hook}<Section id="explore-hook" title={title('hook')} intro={intros.hook} provenance={badges.hook}>{@render hook()}</Section>{/if}
	{#if toy}<Section id="explore-toy" title={title('toy')} intro={intros.toy} provenance={badges.toy}>{@render toy()}</Section>{/if}
	{#if real}<Section id="explore-real" title={title('real')} intro={intros.real} provenance={badges.real}>{@render real()}</Section>{/if}
	{#if compare}<Section id="explore-compare" title={title('compare')} intro={intros.compare} provenance={badges.compare}>{@render compare()}</Section>{/if}
{/snippet}
{#snippet pitfallsContent()}
	{#if pitfalls}<Section id="pitfalls-list" title={title('pitfalls')} intro={intros.pitfalls}><div class="prose">{@render pitfalls()}</div></Section>{/if}
	{#if post || prequestion}
		<Section id="pitfalls-post" title={title('post')} intro={intros.post}>
			<div class="post">
				{#if prequestion}<PredictReveal q={prequestion} />{/if}
				{#if post}{@render post()}{/if}
			</div>
		</Section>
	{/if}
{/snippet}
{#snippet sourcesContent()}
	{#if links}
		{@render links()}
	{:else if models.length}
		<Section id="sources-models" title={ui('linksToModels')}>
			<div class="cards">{#each models as m (m.slug)}<CatalogCard card={m} compact />{/each}</div>
		</Section>
	{/if}
	{#if further}<div class="further prose">{@render further()}</div>{/if}
{/snippet}

<svelte:head>
	{#if concept}
		<title>{t(concept.title)}{concept.status === 'live' ? '' : ` (${ui('statusPlanned')})`} · Model Atlas</title>
		<meta name="description" content={t(concept.interaction)} />
	{/if}
</svelte:head>

{#if concept}
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
				content={nav.view === 'explore' ? exploreContent : nav.view === 'pitfalls' ? pitfallsContent : sourcesContent}
				width={nav.view === 'explore' ? 'wide' : 'narrow'}
			/>
		{/if}
	</div>
	{#if inputSheet}
		<Sheet bind:open={inputOpen.open} title={lui('yourInput')} width={720}>{@render inputSheet()}</Sheet>
	{/if}
{:else}
	<p class="container">Unknown concept</p>
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
	}
	.ab-note {
		margin: var(--s4) 0 0;
		padding-top: var(--s3);
		border-top: 1px solid var(--rule);
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.post {
		display: grid;
		gap: var(--s4);
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 240px), 1fr));
		gap: 10px;
	}
	.further {
		margin-top: var(--s4);
	}
</style>
