<!--
  Site header (redesign spec §4.5) on catalog and site pages; lesson routes render their own lesson top bar instead (the
  root layout's chrome switch). 64 px, sticky, paper at 90 % with a blur and a hairline:
    1100 px and up   brand · nav (Lessons, Paths, Glossary, About; current = here style) · search · EN|中文 · theme
    760-1099 px      brand · search · EN|中文 · theme · menu (the nav moves into a sheet)
    below 760 px     brand · 中文 flip · theme · search icon · menu (56 px)
  A skip link comes first. DevKit is linked from the footer and About only, never from the learner nav.
-->
<script lang="ts">
	import type { Component } from 'svelte';
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import Icon from './Icon.svelte';
	import LangToggle from './LangToggle.svelte';
	import Mark from './Mark.svelte';
	import ThemeToggle from './ThemeToggle.svelte';
	import SearchPalette from './catalog/SearchPalette.svelte';
	import { menu } from './icons';
	import { ui } from '$lib/i18n/i18n.svelte';
	import type { UIKey } from '$lib/i18n/strings';

	const links: { href: string; key: UIKey; routes: RegExp | null }[] = [
		{ href: '/atlas/', key: 'navLessons', routes: /^\/(atlas|train|m)(\/|$)/ },
		{ href: '/#paths', key: 'navPaths', routes: null },
		{ href: '/glossary/', key: 'navGlossary', routes: /^\/glossary(\/|$)/ },
		{ href: '/about/', key: 'navAbout', routes: /^\/about(\/|$)/ }
	];
	// compare route ids (independent of the base path): '/atlas/[area]' is under Lessons
	const isCurrent = (l: (typeof links)[number]) => !!l.routes && l.routes.test(page.route.id ?? '');

	// the menu sheet loads on first use (it is not part of every page's shell)
	let Sheet: Component<any> | null = $state(null);
	let menuOpen = $state(false);
	let menuBtn: HTMLButtonElement | undefined = $state();
	async function openMenu() {
		Sheet ??= (await import('./Sheet.svelte')).default;
		menuOpen = true;
	}

	$effect(() => {
		void page.url.pathname;
		menuOpen = false;
	});
</script>

<a class="skip" href="#main">{ui('skipToContent')}</a>
<header class="hdr">
	<div class="inner">
		<a class="brand" href="{base}/">
			<Mark size={30} />
			<span class="nm">Model Atlas</span>
			<span class="zh" lang="zh-CN">模型图鉴</span>
		</a>
		<nav class="nav" aria-label={ui('navMain')}>
			<ul>
				{#each links as l (l.href)}
					<li><a href="{base}{l.href}" aria-current={isCurrent(l) ? 'page' : undefined}>{ui(l.key)}</a></li>
				{/each}
			</ul>
		</nav>
		<div class="find"><SearchPalette /></div>
		<div class="tools">
			<LangToggle />
			<ThemeToggle />
			<button bind:this={menuBtn} type="button" class="icon-btn menu" aria-haspopup="dialog" aria-expanded={menuOpen} aria-label={ui('menu')} onclick={openMenu}>
				<Icon d={menu} />
			</button>
		</div>
	</div>
</header>

{#if Sheet}
	<Sheet bind:open={menuOpen} title={ui('menu')} placement="end" width={360}>
		<nav aria-label={ui('navMain')}>
			<ul class="sheet-nav">
				{#each links as l (l.href)}
					<li><a href="{base}{l.href}" aria-current={isCurrent(l) ? 'page' : undefined} onclick={() => (menuOpen = false)}>{ui(l.key)}</a></li>
				{/each}
			</ul>
		</nav>
	</Sheet>
{/if}

<style>
	.skip {
		position: absolute;
		left: 8px;
		top: -60px;
		z-index: 100;
		padding: 8px 14px;
		border-radius: var(--r3);
		background: var(--act);
		color: var(--act-ink);
		font-weight: var(--w-ui);
	}
	.skip:focus {
		top: 8px;
	}
	.hdr {
		position: sticky;
		top: 0;
		z-index: 30;
		height: var(--header-h);
		background: color-mix(in srgb, var(--paper) 90%, transparent);
		backdrop-filter: blur(10px);
		-webkit-backdrop-filter: blur(10px);
		border-bottom: 1px solid var(--rule);
	}
	.inner {
		max-width: var(--page-max);
		height: 100%;
		margin: 0 auto;
		padding: 0 var(--gutter);
		display: flex;
		align-items: center;
		gap: var(--s5);
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 10px;
		flex: none;
		color: var(--ink);
		text-decoration: none;
		font-size: var(--t-md);
		font-weight: var(--w-title);
		white-space: nowrap;
	}
	.brand .zh {
		font-family: var(--font-cjk);
		font-weight: var(--w-body);
		color: var(--muted);
	}
	.nav ul {
		display: flex;
		align-items: center;
		gap: 2px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.nav a {
		display: inline-flex;
		align-items: center;
		min-height: 40px;
		padding: 0 12px;
		border-radius: 10px;
		color: var(--ink-2);
		font-size: var(--t-sm);
		font-weight: var(--w-ui);
		text-decoration: none;
		white-space: nowrap;
	}
	.nav a:hover {
		background: var(--well);
		color: var(--ink);
	}
	.nav a[aria-current='page'] {
		background: var(--here-bg);
		color: var(--ink);
	}
	.find {
		flex: 1;
		max-width: 340px;
		margin-left: auto;
		display: flex;
		justify-content: flex-end;
	}
	.tools {
		display: flex;
		align-items: center;
		gap: 4px;
		flex: none;
	}
	.menu {
		display: none;
	}
	.sheet-nav {
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.sheet-nav a {
		display: flex;
		align-items: center;
		min-height: 48px;
		padding: 0 var(--s3);
		border-radius: var(--r3);
		color: var(--ink);
		font-size: var(--t-lg);
		font-weight: var(--w-ui);
		text-decoration: none;
	}
	.sheet-nav a:hover {
		background: var(--well);
	}
	.sheet-nav a[aria-current='page'] {
		background: var(--here-bg);
		box-shadow: inset 3px 0 0 var(--here);
	}
	@media (max-width: 1099px) {
		.nav {
			display: none;
		}
		.menu {
			display: inline-grid;
		}
		.find {
			max-width: 300px;
		}
	}
	@media (max-width: 759px) {
		.inner {
			gap: var(--s2);
			padding: 0 6px 0 var(--gutter);
		}
		.brand .zh {
			display: none;
		}
		.find {
			flex: 0 0 auto;
			order: 1;
			margin-left: 0;
		}
		.tools {
			margin-left: auto;
			gap: 0;
		}
		.menu {
			order: 2;
		}
		.tools {
			display: contents;
		}
		.brand {
			margin-right: auto;
		}
	}
	@media (max-width: 359px) {
		.brand .nm {
			font-size: var(--t-sm);
		}
	}
</style>
