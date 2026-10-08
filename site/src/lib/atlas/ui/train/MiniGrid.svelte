<!--
  A small matrix of signed values as colored cells (diverging scale, blue < 0 < orange), with one cell outlined.
  For the kind-B views (a weight or gradient matrix of a toy network). Hover or focus a cell to read it.
-->
<script lang="ts">
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3 } from '../../core/format.ts';

	let { data, rows, cols, label, mark = null, maxCell = 26 }: { data: ArrayLike<number>; rows: number; cols: number; label: string; mark?: [number, number] | null; maxCell?: number } = $props();
	let W = $state(240);
	const cell = $derived(Math.max(6, Math.min(maxCell, Math.floor(W / Math.max(1, cols)))));
	const scale = $derived.by(() => {
		let m = 0;
		for (let i = 0; i < rows * cols; i++) m = Math.max(m, Math.abs(data[i] ?? 0));
		return makeScale('diverging', [-(m || 1), m || 1], themeState.resolved);
	});
	let hover = $state<[number, number] | null>(null);
	const shown = $derived(hover ?? mark);
</script>

<div class="mg" bind:clientWidth={W}>
	<svg width={cols * cell} height={rows * cell} viewBox="0 0 {cols * cell} {rows * cell}" role="img" aria-label={label} onpointerleave={() => (hover = null)}>
		{#each Array.from({ length: rows }, (_, r) => r) as r (r)}
			{#each Array.from({ length: cols }, (_, c) => c) as c (c)}
				<rect x={c * cell} y={r * cell} width={cell - 1} height={cell - 1} fill={scale.css(data[r * cols + c])} onpointerenter={() => (hover = [r, c])} role="presentation" />
			{/each}
		{/each}
		{#if mark}<rect class="mark" x={mark[1] * cell - 0.5} y={mark[0] * cell - 0.5} width={cell} height={cell} />{/if}
	</svg>
	{#if shown}<p class="small muted num">[{shown[0]}, {shown[1]}] = {fmt3(data[shown[0] * cols + shown[1]])}</p>{/if}
</div>

<style>
	.mg {
		display: grid;
		gap: 4px;
		min-width: 0;
		max-width: 100%;
	}
	svg {
		display: block;
		max-width: 100%;
	}
	.mark {
		fill: none;
		stroke: var(--ink);
		stroke-width: 2;
	}
	p {
		margin: 0;
	}
</style>
