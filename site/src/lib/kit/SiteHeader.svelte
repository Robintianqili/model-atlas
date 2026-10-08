<!-- Site header: brand, main navigation, language and theme toggles. Collapses to a menu on phones. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import LangToggle from './LangToggle.svelte';
	import ThemeToggle from './ThemeToggle.svelte';
	import { ui } from '$lib/i18n/i18n.svelte';
	import type { UIKey } from '$lib/i18n/strings';

	let menuOpen = $state(false);
	const links: { href: string; key: UIKey; short?: UIKey }[] = [
		{ href: '/atlas/', key: 'navAtlas' },
		{ href: '/train/', key: 'navTrain', short: 'navTrainShort' },
		{ href: '/glossary/', key: 'navGlossary' },
		{ href: '/devkit/', key: 'navDevkit', short: 'navDevkitShort' },
		{ href: '/about/', key: 'navAbout' }
	];
	// compare route ids (independent of the base path): '/atlas/[area]' is under '/atlas/'
	const isCurrent = (href: string) => (page.route.id ?? '').startsWith(href.replace(/\/$/, ''));

	$effect(() => {
		void page.url.pathname;
		menuOpen = false;
	});
</script>

<a class="skip" href="#main">{ui('skipToContent')}</a>
<header class="hdr">
	<div class="inner container">
		<a class="brand" href="{base}/" aria-label="{ui('siteName')} — {ui('navHome')}">
			<svg viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">
				<rect width="32" height="32" rx="7" fill="var(--accent)" />
				<g fill="var(--accent-ink)">
					<rect x="6" y="6" width="5" height="5" rx="1" /><rect x="13.5" y="6" width="5" height="5" rx="1" opacity=".55" /><rect x="21" y="6" width="5" height="5" rx="1" opacity=".3" />
					<rect x="6" y="13.5" width="5" height="5" rx="1" opacity=".55" /><rect x="13.5" y="13.5" width="5" height="5" rx="1" /><rect x="21" y="13.5" width="5" height="5" rx="1" opacity=".55" />
					<rect x="6" y="21" width="5" height="5" rx="1" opacity=".3" /><rect x="13.5" y="21" width="5" height="5" rx="1" opacity=".55" /><rect x="21" y="21" width="5" height="5" rx="1" />
				</g>
			</svg>
			<span class="name">Model Atlas <span class="zh" lang="zh-CN">模型图鉴</span></span>
		</a>
		<nav class="nav" class:open={menuOpen} aria-label="Main">
			<ul>
				{#each links as l (l.href)}
					<li>
						<a href="{base}{l.href}" aria-current={isCurrent(l.href) ? 'page' : undefined}>
							<span class="full">{ui(l.key)}</span>{#if l.short}<span class="short">{ui(l.short)}</span>{/if}
						</a>
					</li>
				{/each}
			</ul>
		</nav>
		<div class="tools">
			<LangToggle />
			<ThemeToggle />
			<button
				type="button"
				class="btn ghost icon menu"
				aria-expanded={menuOpen}
				aria-label={menuOpen ? ui('closeMenu') : ui('menu')}
				onclick={() => (menuOpen = !menuOpen)}
			>
				<svg viewBox="0 0 16 16" width="18" height="18" aria-hidden="true" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
					{#if menuOpen}<path d="M3.5 3.5l9 9M12.5 3.5l-9 9" />{:else}<path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" />{/if}
				</svg>
			</button>
		</div>
	</div>
</header>

<style>
	.skip {
		position: absolute;
		left: 8px;
		top: -60px;
		z-index: 100;
		padding: 8px 14px;
		background: var(--accent);
		color: var(--accent-ink);
		border-radius: var(--r2);
	}
	.skip:focus {
		top: 8px;
	}
	.hdr {
		position: sticky;
		top: 0;
		z-index: 30;
		height: var(--header-h);
		background: color-mix(in srgb, var(--bg) 88%, transparent);
		backdrop-filter: blur(10px);
		-webkit-backdrop-filter: blur(10px);
		border-bottom: 1px solid var(--rule);
	}
	.inner {
		height: 100%;
		display: flex;
		align-items: center;
		gap: 16px;
	}
	.brand {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		text-decoration: none;
		color: var(--ink);
		flex: 0 0 auto;
	}
	.name {
		font-family: var(--font-display);
		font-weight: 650;
		font-size: 1.15rem;
		letter-spacing: -0.01em;
		white-space: nowrap;
	}
	.zh {
		font-family: var(--font-cjk);
		font-weight: 500;
		font-size: 0.85em;
		color: var(--muted);
		margin-left: 4px;
	}
	.nav {
		margin-left: auto;
	}
	.nav ul {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		gap: 2px;
	}
	.nav a {
		display: inline-flex;
		align-items: center;
		min-height: 36px;
		padding: 0 12px;
		border-radius: var(--r2);
		color: var(--ink-2);
		text-decoration: none;
		font-size: var(--fs-sm);
		font-weight: 600;
	}
	.nav a:hover {
		background: var(--surface-2);
		color: var(--ink);
	}
	.nav a[aria-current='page'] {
		color: var(--ink);
		background: var(--surface-2);
		box-shadow: inset 0 -2px 0 var(--accent);
	}
	.short {
		display: none;
	}
	.tools {
		display: flex;
		align-items: center;
		gap: 2px;
	}
	.menu {
		display: none;
	}
	@media (max-width: 960px) {
		.full:has(+ .short) {
			display: none;
		}
		.short {
			display: inline;
		}
	}
	@media (max-width: 760px) {
		.zh {
			display: none;
		}
		.menu {
			display: inline-flex;
		}
		.nav {
			position: absolute;
			top: var(--header-h);
			left: 0;
			right: 0;
			display: none;
			background: var(--bg);
			border-bottom: 1px solid var(--rule);
			box-shadow: var(--shadow-2);
		}
		.nav.open {
			display: block;
		}
		.nav ul {
			flex-direction: column;
			padding: 8px var(--gutter) 14px;
		}
		.nav a {
			width: 100%;
			min-height: 44px;
			font-size: var(--fs-md);
		}
		.full:has(+ .short) {
			display: inline;
		}
		.short {
			display: none;
		}
		.tools {
			margin-left: auto;
		}
	}
	@media (max-width: 380px) {
		.name {
			font-size: 1rem;
		}
		.inner {
			gap: 8px;
		}
	}
</style>
