<!--
  The atlas index rail (redesign spec §4.5): sticky on the left at 1100 px and up, "Models by area" and "Training methods",
  glyph + name + live count; the group in view is marked with the here style (scroll-spy over the page's
  [data-group] sections). Below 1100 px the page shows its jump chips instead.
    <AtlasIndex groups={shown} />
-->
<script lang="ts">
	import Icon from '../Icon.svelte';
	import { glyphFor } from './glyphs';
	import { t } from '$lib/i18n/i18n.svelte';
	import { ct } from './text';
	import type { GroupItem } from './types';

	let { groups, current = $bindable('') }: { groups: GroupItem[]; current?: string } = $props();

	const parts = $derived(
		[
			{ kind: 'model', title: ct('modelsByArea'), items: groups.filter((g) => g.kind === 'model') },
			{ kind: 'concept', title: ct('methods'), items: groups.filter((g) => g.kind === 'concept') }
		].filter((p) => p.items.length)
	);

	// scroll-spy: the last group whose top has passed the line below the sticky bars
	$effect(() => {
		void groups;
		const els = [...document.querySelectorAll<HTMLElement>('[data-group]')];
		if (!els.length) return;
		let raf = 0;
		const spy = () => {
			raf = 0;
			const line = Math.max((parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header-h')) || 64) + 120, innerHeight * 0.35);
			let id = els[0].dataset.group ?? '';
			for (const el of els) if (el.getBoundingClientRect().top <= line) id = el.dataset.group ?? id;
			current = id;
		};
		const on = () => {
			if (!raf) raf = requestAnimationFrame(spy);
		};
		spy();
		void document.fonts?.ready.then(on);
		addEventListener('scroll', on, { passive: true });
		addEventListener('resize', on, { passive: true });
		return () => {
			removeEventListener('scroll', on);
			removeEventListener('resize', on);
			if (raf) cancelAnimationFrame(raf);
		};
	});
</script>

<nav class="rail" aria-label={t({ en: 'Areas and training methods', zh: '领域与训练方法' })}>
	{#each parts as p (p.kind)}
		<p class="h">{p.title}</p>
		<ul>
			{#each p.items as g (g.slug)}
				<li>
					<a href="#{g.slug}" class:none={!g.live} aria-current={current === g.slug ? 'location' : undefined}>
						<Icon d={glyphFor(g.slug)} glyph size={16} /><span>{t(g.name)}</span><span class="c num">{g.live}</span>
					</a>
				</li>
			{/each}
		</ul>
	{/each}
</nav>

<style>
	.rail {
		position: sticky;
		top: calc(var(--header-h) + 96px);
		align-self: start;
		max-height: calc(100vh - var(--header-h) - 120px);
		overflow: auto;
		overscroll-behavior: contain;
		padding: 2px 4px 8px 2px;
	}
	.h {
		margin: 16px 0 6px 10px;
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
		font-weight: var(--w-title);
		color: var(--muted);
	}
	.h:first-child {
		margin-top: 0;
	}
	ul {
		margin: 0;
		padding: 0;
		list-style: none;
		display: flex;
		flex-direction: column;
		gap: 1px;
	}
	a {
		position: relative;
		display: grid;
		grid-template-columns: 16px minmax(0, 1fr) auto;
		align-items: center;
		gap: 10px;
		padding: 6px 10px;
		border-radius: var(--r2);
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		text-decoration: none;
	}
	a:hover {
		color: var(--ink);
		background: var(--well);
	}
	a.none {
		color: var(--muted);
	}
	.c {
		font-size: var(--t-xs);
		color: var(--muted);
	}
	a[aria-current] {
		background: var(--here-bg);
		color: var(--ink);
		font-weight: var(--w-title);
	}
	a[aria-current]::before {
		content: '';
		position: absolute;
		left: 0;
		top: 7px;
		bottom: 7px;
		width: 3px;
		border-radius: 3px;
		background: var(--here);
	}
</style>
