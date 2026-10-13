<!--
  An area tile (redesign spec §4.5): the glyph and name, a few examples, a live/planned meter and "6 live · 12 planned".
  On phones it is a one-line row with a 4 px meter. Areas get glyphs, not colours.
    <AreaTile group={g} href="/atlas/vision/" />
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '../Icon.svelte';
	import { glyphFor } from './glyphs';
	import { t } from '$lib/i18n/i18n.svelte';
	import { ct } from './text';
	import type { GroupItem } from './types';

	let { group: g, href }: { group: GroupItem; href: string } = $props();
	const share = $derived(g.live + g.planned ? g.live / (g.live + g.planned) : 0);
</script>

<a class="tile" class:none={!g.live} href="{base}{href}">
	<span class="name"><Icon d={glyphFor(g.slug)} glyph size={20} /><span>{t(g.name)}</span></span>
	{#if g.examples}<span class="ex">{t(g.examples)}</span>{/if}
	<span class="meter" aria-hidden="true"><i style:width="{(share * 100).toFixed(0)}%"></i></span>
	<span class="count">{g.live ? ct('liveAndPlanned', { n: g.live, m: g.planned }) : t({ en: `${g.planned} planned`, zh: `${g.planned} 节计划中` })}</span>
</a>

<style>
	.tile {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 14px 16px;
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
		color: inherit;
		text-decoration: none;
	}
	.tile:hover {
		color: inherit;
		border-color: var(--rule-strong);
		box-shadow: var(--shadow-1);
	}
	.name {
		display: flex;
		align-items: center;
		gap: 10px;
		font-size: var(--t-md);
		line-height: var(--lh-md);
		font-weight: var(--w-title);
	}
	.tile:hover .name span {
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.none .name {
		color: var(--ink-2);
	}
	.ex {
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	.meter {
		display: block;
		height: 6px;
		margin-top: 4px;
		border-radius: var(--r-pill);
		background: var(--rule);
		overflow: hidden;
	}
	.meter i {
		display: block;
		height: 100%;
		border-radius: var(--r-pill);
		background: var(--ink-2);
	}
	.count {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	@media (max-width: 759px) {
		.tile {
			display: grid;
			grid-template-columns: minmax(0, 1fr) auto;
			gap: 4px 12px;
			min-height: 44px;
			padding: 9px 0;
			border: 0;
			border-bottom: 1px solid var(--rule);
			border-radius: 0;
			background: none;
		}
		.tile:hover {
			box-shadow: none;
		}
		.ex {
			display: none;
		}
		.meter {
			grid-column: 1 / -1;
			height: 4px;
			margin: 0;
		}
		.count {
			grid-row: 1;
			grid-column: 2;
			align-self: center;
		}
	}
</style>
