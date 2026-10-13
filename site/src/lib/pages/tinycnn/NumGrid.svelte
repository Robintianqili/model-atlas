<!--
  NumGrid: a small grid (3x3 kernel, window, products; 2x2 pool window) with the value printed in
  each colored cell. `hover` (bindable) is the linked cell index: hovering a cell in one grid
  highlights the same position in the grids next to it.
-->
<script lang="ts">
	import { makeScale, type Domain, type ScaleKind } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtFull } from '$lib/kit/format';
	import { t, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		values: ArrayLike<number>;
		rows: number;
		cols: number;
		kind?: ScaleKind;
		domain: Domain;
		label: Text;
		/** linked hover index (bindable) */
		hover?: number | null;
		/** index to mark (e.g. the max of a pool window) */
		mark?: number | null;
		/** a role chip next to the label: 'W' for parameters */
		role?: 'param' | 'act';
		cell?: number;
	}
	let { values, rows, cols, kind = 'diverging', domain, label, hover = $bindable(null), mark = null, role = 'act', cell = 56 }: Props = $props();
	const scale = $derived(makeScale(kind, domain, themeState.resolved));
	const idx = $derived(Array.from({ length: rows * cols }, (_, i) => i));
</script>

<figure class="ng" class:param={role === 'param'}>
	<figcaption>{#if role === 'param'}<span class="w" aria-hidden="true">W</span>{/if}{t(label)}</figcaption>
	<div class="cells" style="grid-template-columns: repeat({cols}, {cell}px)" role="group" aria-label={t(label)}>
		{#each idx as i (i)}
			{@const v = values[i]}
			<span
				class="c num"
				class:on={hover === i}
				class:mark={mark === i}
				class:long={fmt3(v).length > 6}
				style="background: {scale.css(v)}; color: {scale.ink(v)}; height: {Math.round(cell * 0.62)}px"
				title={fmtFull(v)}
				role="presentation"
				onpointerenter={() => (hover = i)}
				onpointerleave={() => (hover = null)}>{fmt3(v)}</span
			>
		{/each}
	</div>
</figure>

<style>
	.ng {
		margin: 0;
		display: grid;
		gap: 3px;
		justify-items: start;
	}
	figcaption {
		font-size: var(--t-xs);
		color: var(--muted);
		display: flex;
		gap: 4px;
		align-items: center;
		white-space: nowrap;
	}
	.w {
		font-family: var(--font-mono);
		font-weight: 700;
		font-size: var(--t-xs);
		padding: 0 3px;
		border: 1.5px double var(--param);
		border-radius: 2px;
		color: var(--param);
		line-height: 1.2;
	}
	.cells {
		display: grid;
		gap: 1px;
		padding: 1px;
		background: var(--rule);
		border-radius: 3px;
	}
	.param .cells {
		padding: 2px;
		background: var(--param);
		outline: 1px solid var(--param);
		outline-offset: 1px;
	}
	.c {
		display: grid;
		place-items: center;
		font-size: var(--t-xs);
		letter-spacing: -0.02em;
		white-space: nowrap;
		overflow: hidden;
		cursor: crosshair;
	}
	.c.long {
		font-size: var(--t-xs);
		letter-spacing: -0.05em;
	}
	.c.on {
		outline: 2px solid var(--highlight);
		outline-offset: -2px;
	}
	.c.mark {
		box-shadow: inset 0 0 0 2px var(--highlight);
		font-weight: 700;
	}
</style>
