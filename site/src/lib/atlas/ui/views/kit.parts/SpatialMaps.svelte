<!--
  Input map (one channel, the window the unit reads outlined) → output map (the unit's cell outlined), as in TinyCNN's
  ConvDetail. Hovering, tapping or arrow keys on the output map pick another cell of the same channel (onpick).
  With `slide`, a StepPlayer moves the cell over every position of the channel, row by row ("Slide the kernel").
-->
<script lang="ts">
	import { untrack } from 'svelte';
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import type { Box } from '../../../types/index.ts';

	interface Map {
		data: ArrayLike<number>;
		rows: number;
		cols: number;
		kind?: 'diverging' | 'sequential';
		domain: [number, number];
		label: Text;
		/** cells drawn hatched (e.g. −∞ masks, or cells not computed yet) */
		masked?: (r: number, c: number) => boolean;
	}
	interface Props {
		input: Map;
		output: Map;
		/** window in the input (inclusive cells) */
		box?: Box | null;
		/** the unit's cell in the output map */
		cell: { row: number; col: number };
		/** text on the arrow ("3×3", "max", "ReLU") */
		arrow?: string;
		onpick?: (row: number, col: number) => void;
		/** slide the cell over the whole map with a StepPlayer */
		slide?: { autoplay: boolean; interval?: number; onend?: () => void; title?: Text } | null;
		/** keep both maps side by side on phones (lesson stages) */
		pair?: boolean;
		maxCell?: number;
	}
	let { input, output, box = null, cell, arrow = '', onpick, slide = null, pair = true, maxCell }: Props = $props();
	/** small maps get bigger cells (a 26×26 map stays at 14 px, as ConvDetail) */
	const mc = (n: number) => maxCell ?? Math.max(8, Math.min(28, Math.floor(364 / Math.max(1, n))));

	let playing = $state(false);
	const count = $derived(output.rows * output.cols);
	const inHl = $derived(box ? [{ row: box.y0, col: box.x0, h: box.y1 - box.y0 + 1, w: box.x1 - box.x0 + 1, color: 'var(--accent)' }] : []);
	const outHl = $derived([{ row: cell.row, col: cell.col, color: 'var(--accent)' }]);
	const stepNow = $derived(cell.row * output.cols + cell.col);
	const outMasked = $derived.by(() => {
		const base = output.masked;
		if (!(slide && playing)) return base;
		const at = stepNow;
		const cols = output.cols;
		return (r: number, c: number) => r * cols + c > at || (base ? base(r, c) : false);
	});
	function setStep(v: number) {
		onpick?.(Math.floor(v / output.cols), v % output.cols);
	}
	// a stage that plays itself ('play: view') slides from the first position when its autoplay turns on (a lesson
	// stage turns it on once the stage has been in view: ctx.shown)
	let started = false;
	$effect(() => {
		if (slide?.autoplay && !started) {
			started = true;
			untrack(() => onpick?.(0, 0));
		}
	});
</script>

{#snippet maps()}
	<div class="maps" class:pair>
		<figure class="m">
			<figcaption>{t(input.label)} <span class="num muted">[{input.rows}×{input.cols}]</span></figcaption>
			<TensorGrid
				data={input.data}
				rows={input.rows}
				cols={input.cols}
				kind={input.kind ?? 'diverging'}
				domain={input.domain}
				masked={input.masked}
				maxCell={mc(Math.max(input.rows, input.cols))}
				interactive={false}
				readout={false}
				highlights={inHl}
				label={input.label}
			/>
		</figure>
		<div class="arrow" aria-hidden="true">
			{#if arrow}<span>{arrow}</span>{/if}
			<svg viewBox="0 0 40 16" width="40" height="16"><path d="M2 8h32m-6-5 6 5-6 5" fill="none" stroke="currentColor" stroke-width="1.8" /></svg>
		</div>
		<figure class="m">
			<figcaption>{t(output.label)} <span class="num muted">[{output.rows}×{output.cols}]</span></figcaption>
			<TensorGrid
				data={output.data}
				rows={output.rows}
				cols={output.cols}
				kind={output.kind ?? 'diverging'}
				domain={output.domain}
				masked={outMasked}
				maxCell={mc(Math.max(output.rows, output.cols))}
				highlights={outHl}
				label={output.label}
				bind:cursor={() => cell, (c) => { if (c) onpick?.(c.row, c.col); }}
				onhover={(c) => { if (c) onpick?.(c.row, c.col); }}
			/>
		</figure>
	</div>
{/snippet}

{#if slide && count > 1}
	<StepPlayer
		{count}
		bind:step={() => stepNow, setStep}
		bind:playing
		autoplay={slide.autoplay}
		interval={slide.interval ?? 60}
		onend={slide.onend}
		speeds={[0.5, 1, 3, 8]}
		label={slide.title ?? t({ en: 'Slide the window: press Play, or hover or tap the output map', zh: '滑动窗口：点“播放”，或在输出图上悬停、点按' })}
		counter={{ en: 'Position {i} of {n}', zh: '位置 {i} / {n}' }}
	>
		{#snippet children()}{@render maps()}{/snippet}
	</StepPlayer>
{:else}
	{@render maps()}
{/if}

<style>
	.maps {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 10px 14px;
		min-width: 0;
	}
	.m {
		margin: 0;
		display: grid;
		gap: 4px;
		min-width: 0;
		flex: 1 1 200px;
		max-width: 380px;
	}
	figcaption {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.arrow {
		display: grid;
		justify-items: center;
		color: var(--muted);
		font-size: var(--t-xs);
		font-family: var(--font-mono);
	}
	@media (max-width: 560px) {
		.arrow {
			flex-basis: 100%;
		}
		.arrow svg {
			transform: rotate(90deg);
			margin: 10px 0;
		}
		.maps.pair {
			display: grid;
			grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
			gap: 6px;
			align-items: end;
		}
		.maps.pair .m {
			max-width: none;
		}
		.maps.pair .arrow {
			align-self: center;
		}
		.maps.pair .arrow svg {
			transform: none;
			margin: 0;
			width: 22px;
		}
	}
</style>
