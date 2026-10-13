<!--
  The DevKit section of the site (/devkit/...): one route (src/routes/devkit/[...path]/), so the client router grows by a
  single entry; this component is the docs layout (redesign spec §6.8) and picks the page. Every page's data was
  generated at build time (data.server.ts).

  Layout: the site header, then a 248 px navigation tree (DocsNav) beside the page. Reading pages (a guide section,
  Contribute) keep their text to a 720 px column at 16/26, with "On this page" as a rail at 1280 px and wider; overview
  pages (the landing, the guide's index, the tutorial, the reference) use the whole width beside the tree. Below 960 px
  the tree moves into a menu sheet, opened from a bar under the site header.
-->
<script lang="ts">
	import { page as app } from '$app/state';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import Glyph from './Glyph.svelte';
	import { code, list } from './icons.ts';
	import type { PageData, SectionRef, T2 } from './types.ts';
	import DocsNav from './DocsNav.svelte';
	import DocsMenu from './DocsMenu.svelte';
	import OnThisPage from './OnThisPage.svelte';
	import Home from './Home.svelte';
	import GuideIndex from './GuideIndex.svelte';
	import GuideSection from './GuideSection.svelte';
	import RefIndex from './RefIndex.svelte';
	import OpsRef from './OpsRef.svelte';
	import ViewsRef from './ViewsRef.svelte';
	import Tutorial from './Tutorial.svelte';
	import Contribute from './Contribute.svelte';

	let { page, nav }: { page: PageData; nav: SectionRef[] } = $props();

	/** the navigation item of this page */
	const current = $derived.by((): string => {
		switch (page.kind) {
			case 'home':
				return 'overview';
			case 'section':
				return `guide/${page.section.slug}`;
			case 'ops':
				return 'reference/ops';
			case 'views':
				return 'reference/views';
			default:
				return page.kind;
		}
	});
	const title = $derived.by((): T2 => {
		switch (page.kind) {
			case 'home':
				return { en: 'Build your own lesson', zh: '自己动手做一节课' };
			case 'guide':
				return { en: 'Guide', zh: '指南' };
			case 'section':
				return { en: `${page.section.n}. ${page.section.title}`, zh: `${page.section.n}. ${page.section.zh}` };
			case 'reference':
				return { en: 'Reference', zh: '参考' };
			case 'ops':
				return { en: 'Op reference', zh: '算子参考' };
			case 'views':
				return { en: 'View reference', zh: '视图参考' };
			case 'tutorial':
				return { en: 'Tutorial: build hello-xor', zh: '教程：做出 hello-xor' };
			case 'contribute':
				return { en: 'Contribute', zh: '参与贡献' };
		}
	});
	/** reading pages: a 720 px column (and "On this page"); the others use the width beside the tree */
	const reading = $derived(page.kind === 'section' || page.kind === 'contribute');
	const toc = $derived.by((): { items: { id: string; text: string }[]; lang?: string } => {
		if (page.kind === 'section') return { items: page.section.subs, lang: 'en' };
		if (page.kind === 'contribute')
			return {
				items: [
					{ id: 'how', text: t({ en: 'How to contribute', zh: '怎样贡献' }) },
					{ id: 'checklist', text: t({ en: 'The checklist', zh: '检查清单' }) },
					{ id: 'gates', text: t({ en: 'The gates', zh: '关卡' }) },
					{ id: 'error-codes', text: t({ en: 'Error codes', zh: '错误代码' }) }
				]
			};
		return { items: [] };
	});
	const hasToc = $derived(reading && toc.items.length > 1);

	// the menu sheet (below 960 px): it closes when a link in it is followed or the page changes
	let menu = $state(false);
	$effect(() => {
		void app.url.pathname;
		menu = false;
	});
</script>

<svelte:head><title>{t(title)} · DevKit · Model Atlas</title></svelte:head>

<div class="dk">
	<div class="bar">
		<button type="button" class="menu" aria-haspopup="dialog" onclick={() => (menu = true)}>
			<Glyph d={list} size={18} />
			<span class="where"><strong>DevKit</strong><span class="sep" aria-hidden="true">·</span><span class="cur">{t(title)}</span></span>
			<span class="all">{t({ en: 'All pages', zh: '全部页面' })}</span>
		</button>
	</div>
	<DocsMenu bind:open={menu} title="DevKit">
		{#if menu}<DocsNav {nav} {current} idp="dns" onnavigate={() => (menu = false)} />{/if}
	</DocsMenu>

	<div class="side">
		<p class="kit">
			<span class="tile" aria-hidden="true"><Glyph d={code} size={16} /></span>
			<span><strong>DevKit</strong><span class="tag">{t({ en: 'Build a lesson for the atlas', zh: '为图鉴做一节课' })}</span></span>
		</p>
		<DocsNav {nav} {current} />
	</div>

	<div class="main" class:reading class:has-toc={hasToc}>
		<div class="page">
			{#if hasToc}<OnThisPage items={toc.items} lang={toc.lang} variant="inline" />{/if}
			{#if page.kind === 'home'}
				<Home {page} />
			{:else if page.kind === 'guide'}
				<GuideIndex {page} />
			{:else if page.kind === 'section'}
				<GuideSection {page} />
			{:else if page.kind === 'reference'}
				<RefIndex {page} />
			{:else if page.kind === 'ops'}
				<OpsRef {page} />
			{:else if page.kind === 'views'}
				<ViewsRef {page} />
			{:else if page.kind === 'tutorial'}
				<Tutorial {page} />
			{:else if page.kind === 'contribute'}
				<Contribute {page} />
			{/if}
		</div>
		{#if hasToc}
			<div class="toc"><OnThisPage items={toc.items} lang={toc.lang} variant="rail" /></div>
		{/if}
	</div>
</div>

<style>
	/* code colours of the generated code blocks on --well, light and dark (each >= 4.5:1; no teal) */
	.dk {
		--dk-com: #5d636a;
		--dk-str: #1a6b3c;
		--dk-kw: #16181b;
		--dk-num: #8a4b08;
		--dk-key: #3a5596;
		max-width: var(--page-max);
		margin-inline: auto;
		padding: 0 var(--gutter) var(--s7);
		display: grid;
		grid-template-columns: 248px minmax(0, 1fr);
		column-gap: var(--s6);
		align-items: start;
	}
	@media (prefers-color-scheme: dark) {
		:global(:root:not([data-theme='light'])) .dk {
			--dk-com: #9aa09a;
			--dk-str: #7fd39b;
			--dk-kw: #eceee9;
			--dk-num: #f1b65c;
			--dk-key: #9db3ec;
		}
	}
	:global(:root[data-theme='dark']) .dk {
		--dk-com: #9aa09a;
		--dk-str: #7fd39b;
		--dk-kw: #eceee9;
		--dk-num: #f1b65c;
		--dk-key: #9db3ec;
	}

	/* the tree: sticky under the site header; it scrolls on its own when it is taller than the window */
	.side {
		grid-column: 1;
		grid-row: 1;
		position: sticky;
		top: var(--header-h);
		max-height: calc(100vh - var(--header-h));
		min-height: calc(100vh - var(--header-h));
		overflow-y: auto;
		overscroll-behavior: contain;
		padding: var(--s6) var(--s3) var(--s6) 0;
		border-right: 1px solid var(--rule);
		scrollbar-width: thin;
	}
	.kit {
		display: flex;
		align-items: center;
		gap: var(--s3);
		margin: 0 0 var(--s4);
		padding-left: var(--s3);
		line-height: var(--lh-sm);
	}
	.kit strong {
		display: block;
		font-size: var(--t-md);
		line-height: var(--lh-md);
	}
	.tag {
		display: block;
		font-size: var(--t-sm);
		color: var(--muted);
	}
	.tile {
		flex: none;
		display: inline-grid;
		place-items: center;
		width: 32px;
		height: 32px;
		border-radius: var(--r2);
		background: var(--act);
		color: var(--act-ink);
	}
	.main {
		grid-column: 2;
		grid-row: 1;
		min-width: 0;
		padding-top: var(--s6);
	}
	.main.reading {
		display: grid;
		grid-template-columns: minmax(0, 720px);
	}
	.page {
		min-width: 0;
	}
	/* code reads as typed everywhere in the DevKit: no ligatures (=> stays two characters) */
	.dk :global(:is(code, pre, kbd)) {
		font-variant-ligatures: none;
	}
	.toc {
		display: none;
	}
	@media (min-width: 1280px) {
		.main.reading.has-toc {
			grid-template-columns: minmax(0, 720px) minmax(0, 1fr);
			column-gap: var(--s6);
		}
		.toc {
			display: block;
			align-self: stretch;
		}
	}

	/* below 960 px: the tree is a sheet, opened from a sticky bar under the site header */
	.bar {
		display: none;
	}
	@media (max-width: 959px) {
		.dk {
			grid-template-columns: minmax(0, 1fr);
		}
		.side {
			display: none;
		}
		.main {
			grid-column: 1;
			grid-row: 2;
			padding-top: var(--s5);
		}
		.bar {
			grid-row: 1;
			display: block;
			position: sticky;
			top: var(--header-h);
			z-index: 20;
			margin: 0 calc(-1 * var(--gutter));
			padding: 0 var(--gutter);
			background: color-mix(in srgb, var(--paper) 92%, transparent);
			backdrop-filter: blur(10px);
			-webkit-backdrop-filter: blur(10px);
			border-bottom: 1px solid var(--rule);
		}
		/* headings a link jumps to clear the header and this bar */
		:global(html:has(.dk)) {
			scroll-padding-top: calc(var(--header-h) + 60px);
		}
	}
	.menu {
		display: flex;
		align-items: center;
		gap: var(--s2);
		width: 100%;
		min-height: 48px;
		padding: 0;
		border: 0;
		background: transparent;
		color: var(--ink);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		text-align: left;
	}
	.where {
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.sep {
		margin: 0 6px;
		color: var(--muted);
	}
	.cur {
		color: var(--ink-2);
	}
	.all {
		flex: none;
		padding: 4px 12px;
		border: 1px solid var(--line-control);
		border-radius: var(--r-pill);
		font-weight: var(--w-ui);
	}
</style>
