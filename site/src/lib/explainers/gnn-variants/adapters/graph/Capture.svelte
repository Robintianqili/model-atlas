<!--
  gnn-variants Capture (local adapter 'graph'): three starting graphs, the member in focus and, in the full copy, the
  club drawing (click a member to follow it) and a friendship editor (add or remove the friendship between two members).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { graphOf, ISOLATED, KARATE, N, toggleEdge, type GraphRaw } from '../../graph.ts';
	import Club from '../../parts/Club.svelte';

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
	const ids = Array.from({ length: N }, (_, i) => i);
	let a = $state(16);
	let b = $state(33);
</script>

<div class="cap" class:compact>
	<div class="row" role="group" aria-label={t({ en: 'Friendships', zh: '朋友关系' })}>
		{#each STARTS as [label, edges], k (k)}<button type="button" class="btn" aria-pressed={same(edges)} onclick={() => set({ edges })}>{t(label)}</button>{/each}
	</div>
	<div class="row">
		<label>{t({ en: 'Member in focus', zh: '关注的成员' })}
			<select value={g.focus} onchange={(e) => set({ focus: Number(e.currentTarget.value) })}>{#each ids as i (i)}<option value={i}>{i}</option>{/each}</select>
		</label>
		<span class="num">{t({ en: `${g.edges.length} friendships`, zh: `${g.edges.length} 段朋友关系` })}</span>
	</div>
	{#if !compact}
		<Club {g} onnode={(i) => set({ focus: i })} />
		<div class="row">
			<span>{t({ en: 'Friendship between', zh: '在这两人之间' })}</span>
			<select bind:value={a} aria-label={t({ en: 'First member', zh: '第一个成员' })}>{#each ids as i (i)}<option value={i}>{i}</option>{/each}</select>
			<select bind:value={b} aria-label={t({ en: 'Second member', zh: '第二个成员' })}>{#each ids as i (i)}<option value={i}>{i}</option>{/each}</select>
			<button type="button" class="btn" disabled={a === b} onclick={() => onchange(toggleEdge(g, a, b))}>{t({ en: 'Add or remove', zh: '加上或去掉' })}</button>
		</div>
	{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	select { margin-left: 6px; min-height: 30px; }
</style>
