<!--
  Capture of the kit 'graph' adapter (K1, promoted from the gcn and gnn-variants packages): the starting graphs (the
  club, member 11 alone, no friendships), the member in focus and, in the full copy, the club drawn with its adjacency
  table: click a member to focus on it, a square to add or remove a friendship. The compact copy beside the lesson keeps
  the buttons and the focus menu. Labelled members (GraphRaw.labelled) are kept as the input has them.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../types/explainer.ts';
	import { adjacency, graphOf, N, toggleEdge, type GraphRaw } from '../../data/karate.ts';
	import { GRAPH_PRESETS } from './adapter.ts';
	import Club from './Club.svelte';

	let { raw, onchange, compact = false }: CaptureProps<GraphRaw> = $props();
	const g = $derived(graphOf(raw));
	const set = (p: Partial<GraphRaw>) => onchange?.({ ...g, ...p });
	const same = (e: GraphRaw['edges']) => JSON.stringify(graphOf({ ...g, edges: e }).edges) === JSON.stringify(g.edges);
</script>

<div class="cap" class:compact>
	<div class="row" role="group" aria-label={t({ en: 'Friendships', zh: '朋友关系' })}>
		{#each GRAPH_PRESETS as p (p.id)}<button type="button" class="btn" aria-pressed={same(p.raw.edges)} onclick={() => set({ edges: p.raw.edges })}>{t(p.label)}</button>{/each}
	</div>
	<div class="row">
		<label
			>{t({ en: 'Member in focus', zh: '关注的成员' })}
			<select value={g.focus} onchange={(e) => set({ focus: Number(e.currentTarget.value) })}>{#each Array.from({ length: N }, (_, i) => i) as i (i)}<option value={i}>{i}</option>{/each}</select>
		</label>
		<span class="num">{t({ en: `${g.edges.length} friendships`, zh: `${g.edges.length} 段朋友关系` })}</span>
	</div>
	{#if !compact}
		<Club
			{g}
			focus={g.focus}
			onnode={(i) => set({ focus: i })}
			oncell={(i, j) => onchange?.(toggleEdge(g, i, j))}
			matrix={{ data: adjacency(g), title: { en: 'Click a member to focus on it, a square to add or remove a friendship.', zh: '点一个成员来关注它，点一个格子来加上或去掉一段朋友关系。' } }}
		/>
	{/if}
</div>

<style>
	.cap {
		display: grid;
		gap: 8px;
		min-width: 0;
		font-size: var(--fs-sm);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.btn {
		min-height: 32px;
		padding: 2px 10px;
		font-size: var(--fs-xs);
	}
	.btn[aria-pressed='true'] {
		border-color: var(--accent);
		background: var(--accent-soft);
	}
	select {
		margin-left: 6px;
		min-height: 30px;
	}
</style>
