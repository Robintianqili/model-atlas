<!--
  gcn Capture (local adapter 'graph'): the learner's whole input. Three starting graphs, the member in focus, the
  labelled members and, in the full copy, the editable drawing + adjacency table (click a square: add or remove a
  friendship). The compact copy beside the lesson keeps the buttons and the focus menu.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { adjacency, graphOf, KARATE, N, toggleEdge, type GraphRaw } from '../../graph.ts';
	import Graph from '../../parts/Graph.svelte';
	import { ISOLATED } from './adapter.ts';

	interface Props { raw: unknown; onchange: (raw: unknown) => void; compact?: boolean }
	let { raw, onchange, compact = false }: Props = $props();
	const g = $derived(graphOf(raw));
	const set = (p: Partial<GraphRaw>) => onchange({ ...g, ...p });
	const STARTS: [{ en: string; zh: string }, GraphRaw['edges']][] = [
		[{ en: 'Zachary’s club (1977)', zh: 'Zachary 的俱乐部（1977）' }, KARATE.edges],
		[{ en: 'Member 11 alone', zh: '成员 11 落单' }, ISOLATED.edges],
		[{ en: 'No friendships', zh: '没有朋友关系' }, []]
	];
	const same = (e: GraphRaw['edges']) => JSON.stringify(graphOf({ ...g, edges: e }).edges) === JSON.stringify(g.edges);
</script>

<div class="cap" class:compact>
	<div class="row" role="group" aria-label={t({ en: 'Friendships', zh: '朋友关系' })}>
		{#each STARTS as [label, edges], k (k)}<button type="button" class="btn" aria-pressed={same(edges)} onclick={() => set({ edges })}>{t(label)}</button>{/each}
	</div>
	<div class="row">
		<label>{t({ en: 'Member in focus', zh: '关注的成员' })}
			<select value={g.focus} onchange={(e) => set({ focus: Number(e.currentTarget.value) })}>{#each Array.from({ length: N }, (_, i) => i) as i (i)}<option value={i}>{i}</option>{/each}</select>
		</label>
		<span class="num">{t({ en: `${g.edges.length} friendships · labelled: ${g.labelled.join(', ') || '–'}`, zh: `${g.edges.length} 段朋友关系 · 已标注：${g.labelled.join('、') || '–'}` })}</span>
		{#if JSON.stringify(g.labelled) !== JSON.stringify(KARATE.labelled)}<button type="button" class="btn" onclick={() => set({ labelled: KARATE.labelled })}>{t({ en: 'Label 0 and 33 again', zh: '重新标注 0 和 33' })}</button>{/if}
	</div>
	{#if !compact}
		<Graph {g} focus={g.focus} onnode={(i) => set({ focus: i })} oncell={(i, j) => onchange(toggleEdge(g, i, j))}
			matrix={{ data: adjacency(g), title: { en: 'Click a member to focus on it, a square to add or remove a friendship.', zh: '点一个成员来关注它，点一个格子来加上或去掉一段朋友关系。' } }} />
	{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	select { margin-left: 6px; min-height: 30px; }
</style>
