<!--
  "On this page" for a DevKit page (redesign spec §6.8): the page's headings as a sticky rail beside the 720 px text column
  at 1280 px and wider, with the heading being read in the "here" style (scroll-spy: the last heading that has passed
  the top of the window); below 1280 px the same list folds into a <details> at the top of the text.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { ui } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import Glyph from './Glyph.svelte';
	import { right } from './icons.ts';

	interface Props {
		items: { id: string; text: string }[];
		variant: 'rail' | 'inline';
		/** the language of the item texts (the guide's headings are English) */
		lang?: string;
	}
	let { items, variant, lang }: Props = $props();
	let here = $state<string | null>(null);

	onMount(() => {
		if (variant !== 'rail') return;
		let raf = 0;
		const spy = () => {
			raf = 0;
			const top = (parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header-h')) || 64) + 96;
			let cur: string | null = null;
			for (const it of items) {
				const el = document.getElementById(it.id);
				if (!el) continue;
				if (el.getBoundingClientRect().top <= top) cur = it.id;
				else break;
			}
			// at the very bottom the last heading may never reach the top: it is the one being read
			if (innerHeight + scrollY >= document.documentElement.scrollHeight - 2 && items.length) cur = items[items.length - 1].id;
			here = cur;
		};
		const on = () => {
			if (!raf) raf = requestAnimationFrame(spy);
		};
		spy();
		addEventListener('scroll', on, { passive: true });
		addEventListener('resize', on);
		return () => {
			removeEventListener('scroll', on);
			removeEventListener('resize', on);
			if (raf) cancelAnimationFrame(raf);
		};
	});
</script>

{#snippet list()}
	<ul {lang}>
		{#each items as it (it.id)}
			<li><a href="#{it.id}" aria-current={variant === 'rail' && here === it.id ? 'location' : undefined}>{it.text}</a></li>
		{/each}
	</ul>
{/snippet}

{#if variant === 'rail'}
	<nav class="rail" aria-label={ui('onThisPage')} use:keyScroll>
		<p class="h">{ui('onThisPage')}</p>
		{@render list()}
	</nav>
{:else}
	<details class="inline">
		<summary>{ui('onThisPage')} <span class="c">{items.length}</span><Glyph d={right} size={16} class="chev" /></summary>
		<nav aria-label={ui('onThisPage')}>{@render list()}</nav>
	</details>
{/if}

<style>
	.rail {
		position: sticky;
		top: calc(var(--header-h) + var(--s5));
		max-height: calc(100vh - var(--header-h) - var(--s6));
		overflow-y: auto;
		overscroll-behavior: contain;
		padding-bottom: var(--s4);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.h {
		margin: 0 0 var(--s2);
		font-weight: var(--w-ui);
		color: var(--ink);
	}
	ul {
		margin: 0;
		padding: 0;
		list-style: none;
		border-left: 1px solid var(--rule);
	}
	a {
		display: block;
		margin-left: -1px;
		padding: 5px 0 5px var(--s3);
		border-left: 2px solid transparent;
		color: var(--muted);
		text-decoration: none;
		overflow-wrap: anywhere;
	}
	a:hover {
		color: var(--ink);
	}
	/* here: the ink bar, ink text on the pale row */
	a[aria-current='location'] {
		border-left-color: var(--here);
		background: var(--here-bg);
		color: var(--ink);
		font-weight: var(--w-ui);
	}
	.inline {
		margin: 0 0 var(--s5);
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	summary {
		display: flex;
		align-items: center;
		gap: var(--s2);
		min-height: 44px;
		padding: 0 var(--s4);
		cursor: pointer;
		list-style: none;
		font-weight: var(--w-ui);
	}
	summary::-webkit-details-marker {
		display: none;
	}
	summary :global(.chev) {
		margin-left: auto;
		color: var(--muted);
		transition: transform var(--dur-2) var(--ease);
	}
	.inline[open] summary :global(.chev) {
		transform: rotate(90deg);
	}
	.c {
		color: var(--muted);
		font-weight: var(--w-body);
		font-variant-numeric: tabular-nums;
	}
	.inline nav {
		padding: 0 var(--s4) var(--s3);
		max-height: 50vh;
		overflow-y: auto;
	}
	.inline a {
		padding-block: 7px;
	}
	@media (min-width: 1280px) {
		.inline {
			display: none;
		}
	}
</style>
