<!--
  A lesson card (redesign spec §4.5, §6.2): the poster on the stage colour, the area or section with its glyph, the title,
  the hook (written to fit two lines, never clamped) and the meta line "10 steps · ≈ 8 min · Draw". The whole card is one
  link. There is no Live pill: live is the rule.
    <LessonCard lesson={l} group={groupName} sprites={index.sprites} />
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '../Icon.svelte';
	import Thumb from './Thumb.svelte';
	import { glyphFor } from './glyphs';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { I18n } from '$lib/i18n/types';
	import type { CatalogIndex, LessonItem } from './types';

	let { lesson: l, group, sprites }: { lesson: LessonItem; group?: I18n; sprites: CatalogIndex['sprites'] } = $props();
</script>

<a class="card" href="{base}{l.href}">
	{#if l.card}<Thumb tile={l.card} sprite={sprites.cards} group={l.group} />{:else}<Thumb tile={l.thumb} sprite={sprites.rows} group={l.group} />{/if}
	<span class="body">
		{#if group}<span class="kick"><Icon d={glyphFor(l.group)} glyph size={14} />{t(group)}</span>{/if}
		<span class="title">{t(l.title)}</span>
		<span class="hook">{t(l.hook)}</span>
		<span class="meta">
			<span>{ui('lessonSteps', { n: l.steps })}</span><span>{ui('minutes', { m: l.minutes })}</span>{#if l.inputs.length}<span>{l.inputs.map((w) => t(w)).join(' · ')}</span>{/if}
		</span>
	</span>
</a>

<style>
	.card {
		display: flex;
		flex-direction: column;
		min-width: 0;
		border: 1px solid var(--rule);
		border-radius: var(--r4);
		background: var(--panel);
		color: inherit;
		text-decoration: none;
		overflow: hidden;
		transition:
			border-color var(--dur-1) var(--ease),
			box-shadow var(--dur-1) var(--ease);
	}
	.card:hover {
		color: inherit;
		border-color: var(--rule-strong);
		box-shadow: var(--shadow-1);
	}
	.card :global(.thumb) {
		border-radius: 0;
		border-bottom: 1px solid var(--rule);
	}
	.body {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 14px 16px 16px;
	}
	.kick {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-ui);
		color: var(--muted);
	}
	.title {
		font-size: var(--t-md);
		line-height: var(--lh-md);
		font-weight: var(--w-title);
	}
	.card:hover .title {
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.hook {
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		color: var(--ink-2);
	}
	:global(:root:lang(zh-CN)) .hook {
		line-height: 1.7;
	}
	.meta {
		margin-top: auto;
		padding-top: 4px;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
</style>
