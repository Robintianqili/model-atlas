<!--
  AttentionMatrix: query rows × key columns, sequential purple, causal mask hatched.
  Query labels get a blue frame and key labels a red frame when highlighted (frames only,
  never cell fill — brief §4.7).

  <AttentionMatrix weights={att} queries={toks} causal bind:row={q} onhover={(c) => …} />

  `weights` is row-major [Tq × Tk]. Labels show when cells are ≥ 12 px; otherwise use the
  readout line (hover / keyboard) and a TokenChips row elsewhere on the page.
-->
<script lang="ts">
	import TensorGrid from './TensorGrid.svelte';
	import { visibleToken, fmt3 } from './format';
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';
	import type { CellValue, Domain, Highlight } from './types';

	interface Props {
		weights: ArrayLike<number>;
		queries: string[];
		keys?: string[];
		causal?: boolean;
		domain?: Domain;
		/** highlighted query row (bindable) */
		row?: number | null;
		/** highlighted key column (bindable) */
		col?: number | null;
		label?: Text;
		maxCell?: number;
		onhover?: (c: CellValue | null) => void;
	}
	let {
		weights,
		queries,
		keys,
		causal = false,
		domain = [0, 1],
		row = $bindable(null),
		col = $bindable(null),
		label,
		maxCell = 34,
		onhover
	}: Props = $props();

	const kLabels = $derived(keys ?? queries);
	const tq = $derived(queries.length);
	const tk = $derived(kLabels.length);
	let avail = $state(0);
	let wrap: HTMLElement | undefined = $state();
	const labelW = $derived(avail > 0 && avail < 480 ? 44 : 64);
	const cell = $derived(
		avail > 0 ? Math.max(4, Math.min(maxCell, Math.floor((avail - labelW - 8) / Math.max(1, tk)))) : 20
	);
	const showLabels = $derived(cell >= 12);
	const masked = $derived(causal ? (r: number, c: number) => c > r : undefined);
	let hover = $state<CellValue | null>(null);
	const hr = $derived(hover?.row ?? row);
	const hc = $derived(hover?.col ?? col);
	const highlights = $derived.by(() => {
		const h: Highlight[] = [];
		if (hr !== null && hr !== undefined) h.push({ row: hr, col: 0, h: 1, w: tk, style: 'dashed', color: 'var(--q)' });
		if (hc !== null && hc !== undefined) h.push({ row: 0, col: hc, h: tq, w: 1, style: 'dashed', color: 'var(--k)' });
		return h;
	});
	$effect(() => {
		if (!wrap) return;
		const ro = new ResizeObserver((e) => (avail = Math.floor(e[0].contentRect.width)));
		ro.observe(wrap);
		return () => ro.disconnect();
	});

	function onHover(c: CellValue | null) {
		hover = c;
		onhover?.(c);
	}
	function onSelect(c: CellValue) {
		row = c.row;
		col = c.col;
	}
</script>

<figure class="am" bind:this={wrap}>
	{#if label}<figcaption>{t(label)}</figcaption>{/if}
	<div class="axis-note">
		<span class="q">{ui('query')} ↓</span>
		<span class="k">{ui('key')} →</span>
	</div>
	<div class="grid" style="--cell: {cell}px; --lw: {labelW}px">
		<div class="corner"></div>
		<div class="cols" aria-hidden="true">
			{#if showLabels}
				{#each kLabels as k, j (j)}
					<span class="cl" class:on={hc === j}><span>{visibleToken(k)}</span></span>
				{/each}
			{/if}
		</div>
		<div class="rows" aria-hidden="true">
			{#if showLabels}
				{#each queries as q, i (i)}
					<span class="rl" class:on={hr === i}>{visibleToken(q)}</span>
				{/each}
			{/if}
		</div>
		<div class="m">
			<TensorGrid
				data={weights}
				rows={tq}
				cols={tk}
				kind="sequential"
				{domain}
				{cell}
				{masked}
				{highlights}
				readout={false}
				labels={false}
				label={label ?? ui('attentionLabel')}
				onhover={onHover}
				onselect={onSelect}
			/>
		</div>
	</div>
	<div class="readout num" aria-live="off">
		{#if hover}
			<span class="qq">{visibleToken(queries[hover.row])}</span> → <span class="kk">{visibleToken(kLabels[hover.col])}</span>:
			{masked?.(hover.row, hover.col) ? ui('masked') : fmt3(hover.value)}
		{:else}&nbsp;{/if}
	</div>
</figure>

<style>
	.am {
		margin: 0;
		min-width: 0;
	}
	figcaption {
		font-size: var(--fs-sm);
		margin-bottom: 4px;
	}
	.axis-note {
		display: flex;
		gap: 12px;
		font-size: var(--fs-xs);
		color: var(--muted);
		margin-bottom: 2px;
	}
	.axis-note .q {
		color: var(--q-text);
	}
	.axis-note .k {
		color: var(--k-text);
	}
	.grid {
		display: grid;
		grid-template-columns: var(--lw) auto;
		grid-template-rows: auto auto;
		max-width: 100%;
	}
	.cols {
		display: flex;
		align-items: flex-end;
		height: 3.2em;
		overflow: visible;
	}
	.cl {
		width: var(--cell);
		height: 100%;
		position: relative;
		border-bottom: 2px solid transparent;
	}
	.cl > span {
		position: absolute;
		left: 50%;
		bottom: 2px;
		transform-origin: left bottom;
		transform: rotate(-55deg);
		font-family: var(--font-mono);
		font-size: min(12px, calc(var(--cell) * 0.7));
		white-space: pre;
	}
	.cl.on {
		border-bottom-color: var(--k);
	}
	.rows {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		padding-right: 4px;
	}
	.rl {
		height: var(--cell);
		line-height: var(--cell);
		max-width: 100%;
		overflow: hidden;
		font-family: var(--font-mono);
		font-size: min(12px, calc(var(--cell) * 0.7));
		white-space: pre;
		padding-right: 3px;
		border-right: 2px solid transparent;
	}
	.rl.on {
		border-right-color: var(--q);
	}
	.m {
		min-width: 0;
	}
	.readout {
		min-height: 1.5em;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		margin-top: 4px;
	}
	.qq {
		border-bottom: 2px solid var(--q);
	}
	.kk {
		border-bottom: 2px solid var(--k);
	}
</style>
