<!--
  Site search in the header (redesign spec §4.5). The field is a search form to /atlas/?q=<query> (it works before any
  script loads); `/` or Ctrl+K focuses it. On first focus or typing it loads SearchPanel (a lazy chunk: the results, the
  index fetch and the combobox keys), so no page shell carries the palette. Phones: the search icon opens the panel in a
  bottom sheet (a link to /atlas/ without scripts).
    <SearchPalette />
-->
<script lang="ts">
	import type { Component } from 'svelte';
	import { base } from '$app/paths';
	import { goto } from '$app/navigation';
	import Icon from '../Icon.svelte';
	import { search } from '../icons';
	import { ui } from '$lib/i18n/i18n.svelte';

	let input: HTMLInputElement | undefined = $state();
	let q = $state('');
	let open = $state(false);
	let mode = $state<'inline' | 'sheet'>('inline');
	let Panel: Component<any> | null = $state(null);
	const id = $props.id();

	async function wake(m: 'inline' | 'sheet' = 'inline') {
		mode = m;
		Panel ??= (await import('./SearchPanel.svelte')).default;
		open = true;
	}

	function submit(e: SubmitEvent) {
		e.preventDefault();
		const s = q.trim();
		void goto(`${base}/atlas/${s ? `?q=${encodeURIComponent(s)}` : ''}`);
	}

	/** typing in a field, or a key a widget handles itself: leave it alone */
	function typing(e: KeyboardEvent): boolean {
		const el = e.target as HTMLElement | null;
		return !!el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
	}

	function onkey(e: KeyboardEvent) {
		const slash = e.key === '/' && !e.ctrlKey && !e.metaKey && !e.altKey;
		const ctrlK = (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k';
		if (!(slash || ctrlK) || (slash && typing(e)) || e.defaultPrevented) return;
		e.preventDefault();
		// the field is hidden on phones: the sheet opens instead
		if (!input || input.offsetParent === null) return void wake('sheet');
		input.focus();
		input.select();
	}
</script>

<svelte:window onkeydown={onkey} />

<form class="search" role="search" action="{base}/atlas/" method="get" onsubmit={submit}>
	<label for={id} class="visually-hidden">{ui('searchLabel')}</label>
	<span class="glass" aria-hidden="true"><Icon d={search} size={18} /></span>
	<input bind:this={input} bind:value={q} {id} type="search" name="q" placeholder={ui('searchLabel')} autocomplete="off" enterkeyhint="search" onfocus={() => wake()} />
	<kbd class="kbd hint" aria-hidden="true">/</kbd>
</form>
<a class="icon-btn search-icon" href="{base}/atlas/" aria-label={ui('searchOpen')} onclick={(e) => (e.preventDefault(), wake('sheet'))}><Icon d={search} /></a>
{#if Panel}<Panel {input} {mode} bind:q bind:open />{/if}

<style>
	.search {
		position: relative;
		display: flex;
		align-items: center;
		width: 100%;
		margin: 0;
	}
	.glass {
		position: absolute;
		left: 12px;
		color: var(--muted);
		pointer-events: none;
	}
	input {
		width: 100%;
		height: 40px;
		padding: 0 40px 0 38px;
		border: 1px solid var(--line-control);
		border-radius: var(--r3);
		background: var(--panel);
		color: var(--ink);
		font-size: var(--t-sm);
	}
	input::placeholder {
		color: var(--muted);
		opacity: 1;
	}
	input::-webkit-search-cancel-button {
		display: none;
	}
	.kbd {
		position: absolute;
		right: 10px;
		pointer-events: none;
	}
	input:focus + .kbd,
	input:not(:placeholder-shown) + .kbd {
		display: none;
	}
	.search-icon {
		display: none;
	}
	@media (max-width: 759px) {
		.search {
			display: none;
		}
		.search-icon {
			display: inline-grid;
		}
	}
</style>
