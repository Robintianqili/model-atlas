<!--
  One tensor with one cell outlined, drawn by rank: [C,H,W] or [H,T,T] → the slice that holds the cell (a grid),
  [R,C] → a grid, [N] → a strip. −∞ cells are hatched (masked attention). Picking a cell calls onpick(index).
-->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import TensorStrip from '$lib/kit/TensorStrip.svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { domainOf } from './stage.ts';

	interface Props {
		data: ArrayLike<number>;
		shape: readonly number[];
		/** the cell to outline (multi-index), or null */
		cell?: readonly number[] | null;
		label: Text;
		/** name of a slice along axis 0 ("channel", "head") */
		sliceName?: { en: string; zh: string };
		kind?: 'diverging' | 'sequential';
		domain?: [number, number];
		onpick?: (idx: number[]) => void;
		maxCell?: number;
		/** pick the slice (axis 0) to show when there is no cell */
		slice?: number;
	}
	let { data, shape, cell = null, label, sliceName = { en: 'slice', zh: '切片' }, kind = 'diverging', domain, onpick, maxCell = 18, slice = 0 }: Props = $props();

	const rank = $derived(shape.length);
	const s0 = $derived(rank === 3 ? Math.max(0, Math.min(shape[0] - 1, cell?.[0] ?? slice)) : 0);
	const rows = $derived(rank === 3 ? shape[1] : rank === 2 ? shape[0] : 1);
	const cols = $derived(rank === 3 ? shape[2] : rank === 2 ? shape[1] : shape[0]);
	const sub = (a: ArrayLike<number>, from: number, to: number): ArrayLike<number> =>
		'subarray' in a ? (a as Float32Array).subarray(from, to) : Array.prototype.slice.call(a, from, to);
	const view = $derived(rank === 3 ? sub(data, s0 * rows * cols, (s0 + 1) * rows * cols) : data);
	const dom = $derived(domain ?? domainOf(kind, data));
	const rc = $derived(cell ? (rank === 3 ? [cell[1], cell[2]] : rank === 2 ? [cell[0], cell[1]] : [0, cell[0]]) : null);
	const anyInf = $derived.by(() => {
		for (let i = 0; i < view.length; i++) if (!Number.isFinite(view[i])) return true;
		return false;
	});
	function pickRC(r: number, c: number) {
		if (!onpick) return;
		onpick(rank === 3 ? [s0, r, c] : rank === 2 ? [r, c] : [c]);
	}
</script>

<figure class="tp">
	<figcaption>
		{t(label)}{#if rank === 3}<span>&nbsp;· {t(sliceName)} {s0}</span>{/if}
		<span class="num muted">[{shape.join('×')}]</span>
	</figcaption>
	{#if rank >= 2}
		<TensorGrid
			data={view}
			{rows}
			{cols}
			{kind}
			domain={dom}
			{maxCell}
			masked={anyInf ? (r, c) => !Number.isFinite(view[r * cols + c]) : undefined}
			highlights={rc ? [{ row: rc[0], col: rc[1], color: 'var(--accent)' }] : []}
			{label}
			bind:cursor={() => (rc ? { row: rc[0], col: rc[1] } : null), (c) => { if (c) pickRC(c.row, c.col); }}
			onselect={(c) => pickRC(c.row, c.col)}
		/>
	{:else}
		<TensorStrip {data} {kind} domain={dom} maxCells={32} shape={[...shape]} name={label} selected={rc ? rc[1] : null} onselect={(i) => pickRC(0, i)} />
	{/if}
</figure>

<style>
	.tp {
		margin: 0;
		display: grid;
		/* one shrinkable column: a long strip ([256] after a flatten) scrolls inside its own box instead of widening the
		   page (R0: 582 px of content at 390 px) */
		grid-template-columns: minmax(0, 1fr);
		gap: 4px;
		min-width: 0;
		max-width: 100%;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
</style>
