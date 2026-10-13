<!--
  The atlas of lessons (redesign spec §6.4; mockup final/atlas.html), shared by /atlas/ (all lessons) and /train/ (Kind =
  Training methods): a title row with the Rows/Pictures segment, one count sentence, the filter bar, the index rail and
  the groups: glyph + heading + "6 live · 12 planned" + its path, the lesson rows in two columns, and the planned lessons
  as one line that expands in place. Groups with nothing live are one "Not live yet" block per kind. Phones: the search,
  a sticky jump bar, two rows per group with "All 6" in the group header.
  The URL keeps the filters: ?q=, ?input=draw|type|points|play|graphs, ?kind=models|methods, ?view=pictures.
    <AtlasView index={data.index} kind="all" />
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import { replaceState } from '$app/navigation';
	import Icon from '../Icon.svelte';
	import LessonRow from './LessonRow.svelte';
	import LessonCard from './LessonCard.svelte';
	import FilterBar from './FilterBar.svelte';
	import AtlasIndex from './AtlasIndex.svelte';
	import { glyphFor, right, rows as rowsIcon, grid } from './glyphs';
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import { ct } from './text';
	import type { I18n } from '$lib/i18n/types';
	import type { CatalogIndex, GroupItem, InputBucket, LessonItem, PlannedItem } from './types';

	interface Props {
		index: CatalogIndex;
		/** the kind the page opens with (/train/: 'concept') */
		kind?: 'all' | 'model' | 'concept';
		title: I18n;
		/** the sentence under the title; the count sentence when absent */
		lede?: I18n;
	}
	let { index, kind: startKind = 'all', title, lede }: Props = $props();

	// svelte-ignore state_referenced_locally
	let kind = $state<'all' | 'model' | 'concept'>(startKind);
	let q = $state('');
	let input = $state<InputBucket | null>(null);
	let view = $state<'rows' | 'pictures'>('rows');
	let open = $state<Record<string, boolean>>({});
	let more = $state<Record<string, boolean>>({});
	let current = $state('');
	let ready = false;

	const PHONE_ROWS = 2;
	const PLANNED_SHOWN = 5;
	const BUCKETS: InputBucket[] = ['draw', 'type', 'points', 'play', 'graphs'];
	const KIND_Q: Record<string, 'model' | 'concept'> = { models: 'model', methods: 'concept' };

	const groupName = $derived(new Map(index.groups.map((g) => [g.slug, g.name])));
	const words = $derived(
		q
			.trim()
			.toLowerCase()
			.split(/\s+/)
			.filter(Boolean)
	);
	const hay = (l: { title: I18n; slug: string; group: string; hook?: I18n }) =>
		[l.title.en, l.title.zh, l.slug, l.hook?.en, l.hook?.zh, groupName.get(l.group)?.en, groupName.get(l.group)?.zh].join(' ').toLowerCase();
	const hays = $derived(new Map<string, string>([...index.lessons, ...index.planned].map((l) => [l.slug, hay(l)])));
	const hit = (l: LessonItem | PlannedItem) => words.every((w) => hays.get(l.slug)?.includes(w));
	const kindOk = (k: 'model' | 'concept') => kind === 'all' || kind === k;
	const filtered = $derived(words.length > 0 || input !== null);

	const visible = $derived(index.lessons.filter((l) => kindOk(l.kind) && (!input || l.buckets.includes(input)) && hit(l)));
	const shown = $derived(
		index.groups
			.filter((g) => kindOk(g.kind))
			.map((g) => ({
				g,
				items: visible.filter((l) => l.group === g.slug),
				planned: index.planned.filter((p) => p.group === g.slug && (!input || p.buckets.includes(input)) && hit(p))
			}))
	);
	const liveGroups = $derived(shown.filter((s) => s.items.length));
	const quiet = $derived(filtered ? [] : shown.filter((s) => !s.g.live));
	const railGroups = $derived<GroupItem[]>([...liveGroups, ...quiet].map((s) => s.g).sort((a, b) => index.groups.indexOf(a) - index.groups.indexOf(b)));

	onMount(() => {
		const sp = page.url.searchParams;
		q = sp.get('q') ?? '';
		const inp = sp.get('input');
		if (inp && (BUCKETS as string[]).includes(inp)) input = inp as InputBucket;
		const k = sp.get('kind');
		if (k && KIND_Q[k]) kind = KIND_Q[k];
		let v = sp.get('view');
		try {
			v ??= localStorage.getItem('atlas.view');
		} catch {
			/* storage blocked */
		}
		if (v === 'pictures') view = 'pictures';
		ready = true;
	});

	// keep the filters in the URL (a shared link opens the same view); shallow, so no navigation happens
	$effect(() => {
		const sp = new URLSearchParams();
		if (q.trim()) sp.set('q', q.trim());
		if (input) sp.set('input', input);
		if (kind !== startKind) sp.set('kind', kind === 'model' ? 'models' : kind === 'concept' ? 'methods' : 'all');
		if (view === 'pictures') sp.set('view', 'pictures');
		if (!ready) return;
		const s = sp.toString();
		if (s === page.url.searchParams.toString()) return;
		try {
			replaceState(`${page.url.pathname}${s ? `?${s}` : ''}${page.url.hash}`, page.state);
		} catch {
			/* the router is not ready yet: the next change writes it */
		}
	});

	function setView(v: 'rows' | 'pictures') {
		view = v;
		try {
			localStorage.setItem('atlas.view', v);
		} catch {
			/* storage blocked */
		}
	}
	function clear() {
		q = '';
		input = null;
		kind = startKind;
	}

	const countLine = $derived(
		t({
			en: `${index.counts.lessons} lessons you can run now: ${index.counts.models} models in ${index.counts.liveAreas} areas and ${index.counts.methods} training methods. ${index.counts.planned} more are planned.`,
			zh: `现在可以学的 ${index.counts.lessons} 节课：${index.counts.liveAreas} 个领域的 ${index.counts.models} 个模型，以及 ${index.counts.methods} 种训练方法。另有 ${index.counts.planned} 节在计划中。`
		})
	);
	const sep = $derived(i18n.lang === 'zh' ? '、' : ', ');
</script>

<div class="container atlas">
	<header class="ahead">
		<div class="ah-row">
			<h1>{t(title)}</h1>
			<div class="seg vtog" role="group" aria-label={ct('layout')}>
				<button type="button" aria-pressed={view === 'rows'} onclick={() => setView('rows')}><Icon d={rowsIcon} size={16} />{ct('rows')}</button>
				<button type="button" aria-pressed={view === 'pictures'} onclick={() => setView('pictures')}><Icon d={grid} size={16} />{ct('pictures')}</button>
			</div>
		</div>
		<p class="lede">{lede ? t(lede) : countLine}</p>
	</header>

	<div class="tools">
		<FilterBar bind:q bind:kind bind:input counts={index.counts} inputs={index.inputs} />
	</div>

	<p class="status" aria-live="polite">
		{#if filtered}
			{t({ en: `Showing ${visible.length} of ${index.counts.lessons} lessons.`, zh: `显示 ${visible.length} / ${index.counts.lessons} 节课。` })}
			<button type="button" class="linkbtn" onclick={clear}>{ct('clearFilters')}</button>
		{/if}
	</p>

	<nav class="jump" aria-label={ct('jumpTo')}>
		{#each liveGroups as s (s.g.slug)}
			<a class="chip" href="#{s.g.slug}" aria-current={current === s.g.slug ? 'true' : undefined}>
				<Icon d={glyphFor(s.g.slug)} glyph size={15} />{t(s.g.name)} <span class="c num">{s.items.length}</span>
			</a>
		{/each}
	</nav>

	<div class="layout">
		<AtlasIndex groups={railGroups} bind:current />
		<div class="groups">
			{#if !liveGroups.length}
				<div class="empty">
					<p>{q.trim() ? ct('noMatch', { q: q.trim() }) : t({ en: 'No lesson matches these filters.', zh: '没有符合筛选条件的课程。' })}</p>
					<button type="button" class="btn" onclick={clear}>{ct('clearFilters')}</button>
				</div>
			{/if}
			{#each ['model', 'concept'] as k (k)}
				{#each liveGroups.filter((s) => s.g.kind === k) as s (s.g.slug)}
					{@const g = s.g}
					{@const all = open[g.slug] || filtered}
					<section class="grp" id={g.slug} data-group={g.slug} aria-labelledby="h-{g.slug}">
						<div class="gh">
							<h2 id="h-{g.slug}"><Icon d={glyphFor(g.slug)} glyph size={22} />{t(g.name)}</h2>
							<p class="gm">{ct('liveAndPlanned', { n: g.live, m: g.planned })}</p>
							{#if s.items.length > PHONE_ROWS && !filtered}
								<button type="button" class="gall" aria-expanded={!!open[g.slug]} onclick={() => (open[g.slug] = !open[g.slug])}>
									{open[g.slug] ? t({ en: 'Fewer', zh: '收起' }) : ct('allN', { n: s.items.length })}<Icon d={right} size={14} />
								</button>
							{/if}
							{#if g.path}
								<a class="gp" href="{base}/#path-{g.path.id}">{t({ en: 'Path', zh: '路径' })}{i18n.lang === 'zh' ? '：' : ': '}{t(g.path.title)}<Icon d={right} size={14} /></a>
							{/if}
						</div>
						{#if view === 'pictures'}
							<ul class="cards">
								{#each s.items as l, i (l.slug)}
									<li class:ph-more={!all && i >= PHONE_ROWS}><LessonCard lesson={l} sprites={index.sprites} /></li>
								{/each}
							</ul>
						{:else}
							<ul class="rows">
								{#each s.items as l, i (l.slug)}
									<li class:ph-more={!all && i >= PHONE_ROWS}><LessonRow lesson={l} sprites={index.sprites} /></li>
								{/each}
							</ul>
						{/if}
						{#if s.planned.length}
							<p class="planned">
								<span class="pk">{ct('plannedN', { n: s.planned.length })}{i18n.lang === 'zh' ? '：' : ':'}</span>
								{(more[g.slug] ? s.planned : s.planned.slice(0, PLANNED_SHOWN)).map((p) => t(p.title)).join(sep)}{#if s.planned.length > PLANNED_SHOWN && !more[g.slug]}{' '}<button type="button" class="linkbtn" onclick={() => (more[g.slug] = true)}>{ct('andMore', { n: s.planned.length - PLANNED_SHOWN })}</button>{/if}
							</p>
						{/if}
					</section>
				{/each}
				{@const q0 = quiet.filter((s) => s.g.kind === k)}
				{#if q0.length}
					<section class="grp soon" aria-labelledby="soon-{k}">
						<h2 class="soon-h" id="soon-{k}">{ct('notLiveYet')}{#if kind === 'all'}{' · '}{k === 'model' ? ct('modelsByArea') : ct('methods')}{/if}</h2>
						<ul>
							{#each q0 as s (s.g.slug)}
								<li id={s.g.slug} data-group={s.g.slug}>
									<Icon d={glyphFor(s.g.slug)} glyph size={16} />
									<span><b>{t(s.g.name)}</b>{' · '}{t({ en: `${s.planned.length} planned: `, zh: `${s.planned.length} 节计划中：` })}{s.planned.slice(0, 3).map((p) => t(p.title)).join(sep)}{s.planned.length > 3 ? t({ en: ', …', zh: '等' }) : ''}</span>
								</li>
							{/each}
						</ul>
					</section>
				{/if}
			{/each}
		</div>
	</div>
</div>

<style>
	.atlas ul {
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.ahead {
		padding: 36px 0 14px;
	}
	.ah-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 16px;
	}
	h1 {
		margin: 0;
	}
	.lede {
		margin: 6px 0 0;
		max-width: none;
	}
	.tools {
		position: sticky;
		top: var(--header-h);
		z-index: 10;
		padding: 14px 0 18px;
		border-bottom: 1px solid var(--rule);
		background: var(--paper);
	}
	.status {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.status:not(:empty) {
		padding-top: 14px;
	}
	.linkbtn {
		padding: 0;
		border: 0;
		background: none;
		color: var(--ink-2);
		font: inherit;
		text-decoration: underline;
		text-underline-offset: 3px;
		cursor: pointer;
	}
	.linkbtn:hover {
		color: var(--ink);
	}
	.jump {
		display: none;
	}
	.layout {
		display: grid;
		grid-template-columns: 208px minmax(0, 1fr);
		gap: 36px;
		padding-top: 24px;
	}
	.empty {
		padding: 24px 0;
	}
	.empty p {
		margin-bottom: 12px;
	}
	.grp {
		padding: 8px 0 28px;
		scroll-margin-top: calc(var(--header-h) + 90px);
	}
	.grp + .grp {
		padding-top: 24px;
		border-top: 1px solid var(--rule);
	}
	.gh {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 4px 14px;
		margin-bottom: 10px;
	}
	.gh h2 {
		display: flex;
		align-items: center;
		gap: 10px;
		margin: 0;
		font-size: var(--t-xl);
		line-height: var(--lh-xl);
	}
	.gm {
		margin: 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.gp {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		margin-left: auto;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		color: var(--ink-2);
	}
	.gall {
		display: none;
	}
	.rows {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 2px 28px;
	}
	.cards {
		display: grid;
		grid-template-columns: repeat(3, minmax(0, 1fr));
		gap: 16px;
	}
	.cards li {
		display: flex;
	}
	.cards li > :global(*) {
		flex: 1;
	}
	.planned {
		margin: 10px 0 0;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
	}
	.pk {
		font-weight: var(--w-title);
		color: var(--ink-2);
	}
	.soon-h {
		margin: 0 0 8px;
		font-size: var(--t-lg);
		line-height: var(--lh-lg);
	}
	.soon ul {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 8px 32px;
	}
	.soon li {
		display: flex;
		align-items: flex-start;
		gap: 10px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--muted);
		scroll-margin-top: calc(var(--header-h) + 90px);
	}
	.soon li :global(.icon) {
		margin-top: 2px;
	}
	.soon b {
		color: var(--ink-2);
		font-weight: var(--w-ui);
	}
	@media (max-width: 1099px) {
		.layout {
			grid-template-columns: minmax(0, 1fr);
		}
		.layout > :global(.rail) {
			display: none;
		}
		.rows {
			gap: 4px 24px;
		}
		.cards {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
	@media (max-width: 959px) {
		.rows {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	@media (max-width: 759px) {
		.ahead {
			padding: 24px 0 8px;
		}
		.lede {
			font-size: var(--t-md);
			line-height: var(--lh-md);
		}
		.vtog {
			display: none;
		}
		.tools {
			position: static;
			padding: 8px 0 4px;
			border: 0;
		}
		.jump {
			position: sticky;
			top: var(--header-h);
			z-index: 10;
			display: flex;
			gap: 8px;
			margin: 0 calc(-1 * var(--gutter));
			padding: 10px var(--gutter);
			overflow-x: auto;
			border-bottom: 1px solid var(--rule);
			background: var(--paper);
			scrollbar-width: none;
		}
		.jump .chip {
			flex: none;
		}
		.layout {
			padding-top: 8px;
		}
		.grp {
			padding-bottom: 12px;
			scroll-margin-top: calc(var(--header-h) + 64px);
		}
		.grp + .grp {
			padding-top: 12px;
		}
		.gh {
			margin-bottom: 2px;
		}
		.gh h2 {
			font-size: var(--t-lg);
			line-height: var(--lh-lg);
		}
		.gp,
		.planned {
			display: none;
		}
		.gall {
			display: inline-flex;
			align-items: center;
			gap: 2px;
			min-height: 44px;
			margin-left: auto;
			padding: 0 2px;
			border: 0;
			background: none;
			color: var(--ink-2);
			font-size: var(--t-sm);
			font-weight: var(--w-ui);
			text-decoration: underline;
			text-underline-offset: 3px;
			cursor: pointer;
		}
		.ph-more {
			display: none;
		}
		.cards {
			grid-template-columns: minmax(0, 1fr);
		}
		.soon ul {
			grid-template-columns: minmax(0, 1fr);
		}

	}
</style>
