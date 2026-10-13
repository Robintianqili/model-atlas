<!--
  The DevKit's docs navigation (redesign spec §6.8): a tree of the section's pages with collapsible groups (Guide,
  Reference, Contribute). The current page has the "here" style (an ink bar on a pale row, aria-current="page"). Used in
  the 248 px sidebar on wide screens and in the menu sheet on phones (DevkitPage.svelte). A group opens by itself when
  the current page is in it; the learner's own open/close choices are kept while they move between DevKit pages.
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import { base } from '$app/paths';
	import { i18n, t } from '$lib/i18n/i18n.svelte';
	import Glyph from './Glyph.svelte';
	import { right } from './icons.ts';
	import type { SectionRef, T2 } from './types.ts';

	interface Props {
		/** the guide's sections */
		nav: SectionRef[];
		/** the current item: 'overview', 'tutorial', 'guide', 'guide/<slug>', 'reference', 'reference/ops', 'reference/views', 'contribute' */
		current: string;
		/** called when a link is followed (the menu sheet closes) */
		onnavigate?: () => void;
		/** an id prefix, so the sidebar and the sheet copy have different ids */
		idp?: string;
	}
	let { nav, current, onnavigate, idp = 'dn' }: Props = $props();

	interface Item {
		id: string;
		href: string;
		label: T2;
		/** a section number, shown before the label */
		n?: number;
		/** the English label under a Chinese one */
		en?: string;
	}
	interface Group {
		id: 'guide' | 'reference' | 'contribute';
		href: string;
		label: T2;
		items: Item[];
	}

	const top: Item[] = [
		{ id: 'overview', href: '/devkit/', label: { en: 'Overview', zh: '概览' } },
		{ id: 'tutorial', href: '/devkit/tutorial/', label: { en: 'Tutorial: build hello-xor', zh: '教程：做出 hello-xor' } }
	];
	const groups = $derived<Group[]>([
		{
			id: 'guide',
			href: '/devkit/guide/',
			label: { en: 'Guide', zh: '指南' },
			items: nav.map((s) => ({ id: `guide/${s.slug}`, href: `/devkit/guide/${s.slug}/`, label: { en: s.title, zh: s.zh }, n: s.n }))
		},
		{
			id: 'reference',
			href: '/devkit/reference/',
			label: { en: 'Reference', zh: '参考' },
			items: [
				{ id: 'reference/ops', href: '/devkit/reference/ops/', label: { en: 'Ops', zh: '算子' } },
				{ id: 'reference/views', href: '/devkit/reference/views/', label: { en: 'Views', zh: '视图' } }
			]
		},
		{
			id: 'contribute',
			href: '/devkit/contribute/',
			label: { en: 'Contribute', zh: '参与贡献' },
			items: [
				{ id: 'contribute#how', href: '/devkit/contribute/#how', label: { en: 'How to contribute', zh: '怎样贡献' } },
				{ id: 'contribute#checklist', href: '/devkit/contribute/#checklist', label: { en: 'The checklist', zh: '检查清单' } },
				{ id: 'contribute#gates', href: '/devkit/contribute/#gates', label: { en: 'The gates', zh: '关卡' } },
				{ id: 'contribute#error-codes', href: '/devkit/contribute/#error-codes', label: { en: 'Error codes', zh: '错误代码' } }
			]
		}
	]);

	const groupOf = (id: string) => id.split(/[/#]/)[0];
	/** the learner's own choices; a group without one is open when the current page is in it (the guide also on the
	 *  overview and the tutorial, where it is the next thing to read) */
	let chosen = $state<Record<string, boolean>>({});
	const isOpen = (g: string) => chosen[g] ?? (groupOf(current) === g || (g === 'guide' && (current === 'overview' || current === 'tutorial')));
	// moving to a page inside a group the learner had closed opens it again, so the current page is never hidden
	$effect(() => {
		const g = groupOf(current);
		untrack(() => {
			if (chosen[g] === false) chosen[g] = true;
		});
	});
	const label = (x: Item) => (i18n.lang === 'zh' ? x.label.zh : x.label.en);
</script>

{#snippet link(x: Item, sub: boolean)}
	<a
		class="item"
		class:sub
		href="{base}{x.href}"
		aria-current={x.id === current ? 'page' : undefined}
		onclick={() => onnavigate?.()}
	>
		{#if x.n !== undefined}<span class="n">{x.n}</span>{/if}
		<span class="lab">{label(x)}</span>
	</a>
{/snippet}

<nav class="dn" aria-label={t({ en: 'DevKit pages', zh: 'DevKit 页面' })}>
	<ul class="top">
		{#each top as x (x.id)}<li>{@render link(x, false)}</li>{/each}
	</ul>
	{#each groups as g (g.id)}
		{@const open = isOpen(g.id)}
		<div class="group">
			<div class="gh">
				<a class="item head" href="{base}{g.href}" aria-current={current === g.id ? 'page' : undefined} onclick={() => onnavigate?.()}>
					<span class="lab">{t(g.label)}</span>
					<span class="count">{g.items.length}</span>
				</a>
				<button
					type="button"
					class="tog"
					aria-expanded={open}
					aria-controls="{idp}-{g.id}"
					aria-label={open ? t({ en: `Hide the ${g.label.en.toLowerCase()} pages`, zh: `收起${g.label.zh}` }) : t({ en: `Show the ${g.label.en.toLowerCase()} pages`, zh: `展开${g.label.zh}` })}
					onclick={() => (chosen[g.id] = !open)}
				>
					<Glyph d={right} size={16} class={open ? 'down' : ''} />
				</button>
			</div>
			<ul id="{idp}-{g.id}" class="items" hidden={!open}>
				{#each g.items as x (x.id)}<li>{@render link(x, true)}</li>{/each}
			</ul>
		</div>
	{/each}
</nav>

<style>
	.dn {
		display: grid;
		gap: var(--s3);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	ul {
		margin: 0;
		padding: 0;
		list-style: none;
		display: grid;
		gap: 1px;
	}
	.group {
		display: grid;
		gap: 2px;
		padding-top: var(--s3);
		border-top: 1px solid var(--rule);
	}
	.item {
		display: flex;
		align-items: baseline;
		gap: var(--s2);
		min-height: 34px;
		padding: 7px var(--s3);
		border-radius: var(--r2);
		color: var(--ink-2);
		text-decoration: none;
		transition: background var(--dur-1) var(--ease);
	}
	.item:hover {
		background: color-mix(in srgb, var(--here-bg) 70%, transparent);
		color: var(--ink);
	}
	/* here: an ink bar on a pale row */
	.item[aria-current='page'] {
		background: var(--here-bg);
		box-shadow: inset 3px 0 0 var(--here);
		color: var(--ink);
		font-weight: var(--w-ui);
	}
	.n {
		flex: none;
		min-width: 1.4em;
		color: var(--muted);
		font-variant-numeric: tabular-nums;
		text-align: right;
	}
	.item[aria-current='page'] .n {
		color: var(--ink);
	}
	.lab {
		min-width: 0;
	}
	.gh {
		display: flex;
		align-items: center;
		gap: 2px;
	}
	.head {
		flex: 1;
		color: var(--ink);
		font-weight: var(--w-ui);
	}
	.count {
		margin-left: auto;
		color: var(--muted);
		font-size: var(--t-xs);
		font-weight: var(--w-body);
		font-variant-numeric: tabular-nums;
	}
	.tog {
		flex: none;
		display: inline-grid;
		place-items: center;
		width: 34px;
		height: 34px;
		padding: 0;
		border: 0;
		border-radius: var(--r2);
		background: transparent;
		color: var(--muted);
	}
	.tog:hover {
		background: var(--here-bg);
		color: var(--ink);
	}
	.tog :global(.icon) {
		transition: transform var(--dur-2) var(--ease);
	}
	.tog :global(.icon.down) {
		transform: rotate(90deg);
	}
	.items {
		padding-left: var(--s2);
	}
	.items[hidden] {
		display: none;
	}
	@media (pointer: coarse) {
		.item,
		.tog {
			min-height: 44px;
		}
		.tog {
			width: 44px;
			height: 44px;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		.tog :global(.icon) {
			transition: none;
		}
	}
</style>
