<!--
  The DevKit section of the site (/devkit/...): one route (src/routes/devkit/[...path]/), so the client router grows by a
  single entry; this component picks the page. Every page's data was generated at build time (data.server.ts).
-->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { PageData, T2 } from './types.ts';
	import Home from './Home.svelte';
	import GuideIndex from './GuideIndex.svelte';
	import GuideSection from './GuideSection.svelte';
	import RefIndex from './RefIndex.svelte';
	import OpsRef from './OpsRef.svelte';
	import ViewsRef from './ViewsRef.svelte';
	import Tutorial from './Tutorial.svelte';
	import Contribute from './Contribute.svelte';

	let { page }: { page: PageData } = $props();

	const tabs: { id: string; href: string; label: T2 }[] = [
		{ id: 'home', href: '/devkit/', label: { en: 'Overview', zh: '概览' } },
		{ id: 'guide', href: '/devkit/guide/', label: { en: 'Guide', zh: '指南' } },
		{ id: 'tutorial', href: '/devkit/tutorial/', label: { en: 'Tutorial', zh: '教程' } },
		{ id: 'reference', href: '/devkit/reference/', label: { en: 'Reference', zh: '参考' } },
		{ id: 'contribute', href: '/devkit/contribute/', label: { en: 'Contribute', zh: '参与贡献' } }
	];
	const current = $derived(page.kind === 'section' ? 'guide' : page.kind === 'ops' || page.kind === 'views' ? 'reference' : page.kind);
	const title = $derived.by((): string => {
		switch (page.kind) {
			case 'home':
				return t({ en: 'Build your own', zh: '自己动手' });
			case 'guide':
				return t({ en: 'Guide', zh: '指南' });
			case 'section':
				return `${page.section.n}. ${t({ en: page.section.title, zh: page.section.zh })}`;
			case 'reference':
				return t({ en: 'Reference', zh: '参考' });
			case 'ops':
				return t({ en: 'Op reference', zh: '算子参考' });
			case 'views':
				return t({ en: 'View reference', zh: '视图参考' });
			case 'tutorial':
				return t({ en: 'Tutorial: build hello-xor', zh: '教程：做出 hello-xor' });
			case 'contribute':
				return t({ en: 'Contribute', zh: '参与贡献' });
		}
	});
</script>

<svelte:head><title>{title} · DevKit · Model Atlas</title></svelte:head>

<div class="dk">
	<nav class="dk-nav" aria-label={t({ en: 'DevKit', zh: 'DevKit 开发工具包' })}>
		<div class="container row">
			<a class="brand" href="{base}/devkit/"><span class="mark" aria-hidden="true">{'{ }'}</span> DevKit</a>
			<ul>
				{#each tabs as tab (tab.id)}
					<li><a href="{base}{tab.href}" aria-current={current === tab.id ? 'page' : undefined}>{t(tab.label)}</a></li>
				{/each}
			</ul>
		</div>
	</nav>

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

<style>
	/* code colors of the generated code blocks, light and dark (contrast >= 4.5 on --surface-2) */
	.dk {
		--dk-com: #67635a;
		--dk-str: #1b6b3a;
		--dk-kw: #074f53;
		--dk-num: #9a4300;
		--dk-key: #1d4f91;
		padding-bottom: var(--s7);
	}
	@media (prefers-color-scheme: dark) {
		:global(:root:not([data-theme='light'])) .dk {
			--dk-com: #a19c90;
			--dk-str: #7fd39b;
			--dk-kw: #7fd9d5;
			--dk-num: #ffb877;
			--dk-key: #9cc3ff;
		}
	}
	:global(:root[data-theme='dark']) .dk {
		--dk-com: #a19c90;
		--dk-str: #7fd39b;
		--dk-kw: #7fd9d5;
		--dk-num: #ffb877;
		--dk-key: #9cc3ff;
	}
	.dk-nav {
		border-bottom: 1px solid var(--rule);
		background: var(--surface);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 4px 18px;
		min-height: 48px;
	}
	.brand {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-family: var(--font-mono);
		font-weight: 700;
		font-size: var(--fs-sm);
		color: var(--ink);
		text-decoration: none;
	}
	.mark {
		display: inline-grid;
		place-items: center;
		width: 26px;
		height: 22px;
		border-radius: 6px;
		background: var(--accent);
		color: var(--accent-ink);
		font-size: 0.72rem;
		letter-spacing: -0.04em;
	}
	ul {
		display: flex;
		flex-wrap: wrap;
		gap: 2px 4px;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	ul a {
		display: inline-block;
		padding: 6px 10px;
		border-radius: var(--r2);
		color: var(--ink-2);
		font-size: var(--fs-sm);
		text-decoration: none;
	}
	ul a:hover {
		background: var(--surface-2);
		color: var(--ink);
	}
	ul a[aria-current='page'] {
		background: var(--accent-soft);
		color: var(--ink);
		font-weight: 600;
	}
	@media (max-width: 560px) {
		.row {
			gap: 2px 10px;
			padding-block: 6px;
		}
		ul a {
			padding: 5px 8px;
		}
	}
</style>
