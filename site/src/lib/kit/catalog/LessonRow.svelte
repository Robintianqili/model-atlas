<!--
  A lesson row (redesign spec §4.5): the atlas default. A 112 × 70 poster, the title, the hook and the meta line; the
  whole row is one link.
    <LessonRow lesson={l} sprites={index.sprites} />
-->
<script lang="ts">
	import { base } from '$app/paths';
	import Thumb from './Thumb.svelte';
	import { t, ui } from '$lib/i18n/i18n.svelte';
	import type { CatalogIndex, LessonItem } from './types';

	let { lesson: l, sprites }: { lesson: LessonItem; sprites: CatalogIndex['sprites'] } = $props();
</script>

<a class="row" href="{base}{l.href}">
	<span class="pic"><Thumb tile={l.thumb} sprite={sprites.rows} group={l.group} /></span>
	<span class="body">
		<span class="title">{t(l.title)}</span>
		<span class="hook">{t(l.hook)}</span>
		<span class="meta">
			<span>{ui('lessonSteps', { n: l.steps })}</span><span>{ui('minutes', { m: l.minutes })}</span>{#if l.inputs.length}<span>{l.inputs.map((w) => t(w)).join(' · ')}</span>{/if}
		</span>
	</span>
</a>

<style>
	.row {
		display: grid;
		grid-template-columns: 112px minmax(0, 1fr);
		gap: 14px;
		align-items: center;
		padding: 8px;
		margin: 0 -8px;
		border-radius: var(--r3);
		color: inherit;
		text-decoration: none;
	}
	.row:hover {
		color: inherit;
		background: var(--panel);
	}
	.pic {
		display: block;
		border-radius: 10px;
		overflow: hidden;
		outline: 1px solid var(--rule);
		outline-offset: -1px;
	}
	.body {
		display: flex;
		flex-direction: column;
		gap: 3px;
		min-width: 0;
	}
	.title {
		font-size: var(--t-md);
		line-height: var(--lh-md);
		font-weight: var(--w-title);
	}
	.row:hover .title {
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
		display: flex;
		flex-wrap: wrap;
		gap: 2px 12px;
		padding-top: 2px;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		color: var(--muted);
	}
	@media (max-width: 759px) {
		.row {
			gap: 12px;
			padding: 8px 0;
			margin: 0;
			border-radius: 0;
			border-bottom: 1px solid var(--rule);
		}
		.row:hover {
			background: none;
		}
	}
</style>
