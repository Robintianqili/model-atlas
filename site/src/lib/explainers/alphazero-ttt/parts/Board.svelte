<!-- alphazero-ttt: one tic-tac-toe board. Marks X and O; on an empty cell an optional label (a number), a fill from 0 to 1
     (the sequential purple: priors, visit shares), a ring (the accent: the network's or the search's choice), a dashed ring
     (your guess) and a pale mark (the move about to be played). With onpick, empty cells are buttons. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { CELL } from '../rules.ts';

	type Cell = { text?: string; fill?: number; ring?: boolean; guess?: boolean; ghost?: string; dim?: boolean };
	interface Props { b: ArrayLike<number>; cells?: (i: number) => Cell; onpick?: (i: number) => void; win?: number[] | null; small?: boolean; label: string }
	let { b, cells = () => ({}), onpick, win = null, small = false, label }: Props = $props();
	const MARKS: Record<number, string> = { 1: 'X', [-1]: 'O' };
</script>

<div class="board" class:small role="group" aria-label={label}>
	{#each Array.from({ length: 9 }, (_, i) => i) as i (i)}
		{@const c = cells(i)}
		{@const free = b[i] === 0}
		{@const name = `${t(CELL[i])}: ${free ? t({ en: 'empty', zh: '空' }) : MARKS[b[i]]}${c.text ? ` (${c.text})` : ''}`}
		{#if onpick && free}
			<button type="button" class="cell" class:ring={c.ring} class:guess={c.guess} style:--f={c.fill ?? 0} aria-label={name} onclick={() => onpick(i)}>
				{#if c.text}<span class="num">{c.text}</span>{/if}
			</button>
		{:else}
			<div class="cell" class:ring={c.ring} class:guess={c.guess} class:win={win?.includes(i)} class:dim={c.dim} style:--f={c.fill ?? 0} role="img" aria-label={name}>
				{#if !free}<span class="mark">{MARKS[b[i]]}</span>{:else if c.ghost}<span class="mark ghost">{c.ghost}</span>{/if}
				{#if c.text}<span class="num">{c.text}</span>{/if}
			</div>
		{/if}
	{/each}
</div>

<style>
	.board { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); grid-template-rows: repeat(3, minmax(0, 1fr)); gap: 4px; width: min(100%, 264px); aspect-ratio: 1; }
	.board.small { width: 76px; gap: 2px; }
	.cell { position: relative; display: grid; place-items: center; min-width: 0; padding: 0; border: 1px solid var(--rule-strong); border-radius: var(--r1); font: inherit; color: var(--ink);
		background: color-mix(in oklab, var(--seq-hi) calc(var(--f) * 70%), var(--surface)); }
	button.cell { cursor: pointer; }
	button.cell:hover, button.cell:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
	.cell.ring { box-shadow: inset 0 0 0 3px var(--accent); }
	.cell.guess::after { content: ''; position: absolute; inset: 6px 12px 22px; border: 2px dashed var(--ink-2); border-radius: 50%; }
	.cell.win { background: var(--accent-soft); }
	.cell.dim { opacity: 0.45; }
	.mark { font-family: var(--font-display); font-size: 2.2em; font-weight: 700; line-height: 1; color: var(--ink); }
	.mark.ghost { opacity: 0.35; }
	/* the number sits on a small surface-colored chip, so it keeps 4.5:1 on any fill (a mid purple fell to about 3:1) */
	.num { position: absolute; bottom: 3px; left: 50%; transform: translateX(-50%); padding: 0 3px; border-radius: 3px; background: var(--surface); line-height: 1.35; white-space: nowrap; font-family: var(--font-mono); font-size: var(--fs-xs); color: var(--ink); }
	.small .mark { font-size: 1em; }
	.small .num { font-size: 9px; bottom: 1px; }
</style>
