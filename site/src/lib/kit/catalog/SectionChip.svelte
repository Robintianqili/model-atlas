<!--
  A training-method section as a chip (redesign spec §4.5): glyph, name and "6 live"; a section with nothing live is a
  dashed chip that says "planned". Links to the section's group on /train/.
    <SectionChip group={g} />
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '../Icon.svelte';
	import { glyphFor } from './glyphs';
	import { t } from '$lib/i18n/i18n.svelte';
	import { ct } from './text';
	import type { GroupItem } from './types';

	let { group: g }: { group: GroupItem } = $props();
</script>

<a class="sc" class:none={!g.live} href="{base}/train/#{g.slug}">
	<Icon d={glyphFor(g.slug)} glyph size={18} />
	<span class="nm">{t(g.name)}</span>
	<span class="c">{g.live ? ct('liveN', { n: g.live }) : t({ en: 'planned', zh: '计划中' })}</span>
</a>

<style>
	.sc {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		align-items: center;
		gap: 10px;
		min-height: 44px;
		padding: 8px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r3);
		background: var(--panel);
		color: inherit;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		font-weight: var(--w-ui);
		text-decoration: none;
	}
	.sc:hover {
		color: inherit;
		border-color: var(--rule-strong);
	}
	.sc:hover .nm {
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.c {
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-body);
		color: var(--muted);
		white-space: nowrap;
	}
	.none {
		color: var(--muted);
		background: transparent;
		border-style: dashed;
		border-color: var(--line-control);
	}
	@media (max-width: 759px) {
		.sc {
			display: inline-flex;
			gap: 8px;
			min-height: 40px;
			padding: 6px 12px;
			border-radius: var(--r-pill);
		}
	}
</style>
