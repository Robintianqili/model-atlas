<!--
  One vector as a row of signed bars (orange up = positive, blue down = negative: the kit's diverging scale), with an
  optional mean line and ±spread band. Rows that share `max` can be compared by eye (LayerNorm: x, x̂, y).
  Hover or tap reads a value; click, tap or the arrow keys pick an index (onselect).
-->
<script lang="ts">
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtFull } from '../../../core/format.ts';
	import { t, ui, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		data: ArrayLike<number>;
		/** half of the y range; default max |v| */
		max?: number;
		selected?: number | null;
		band?: { mean: number; sd: number } | null;
		height?: number;
		label: Text;
		/** short name left of the bars ("x", "x̂") */
		name?: string;
		/** a sequential row (probabilities): bars grow up from 0 */
		kind?: 'diverging' | 'sequential';
		onselect?: (i: number) => void;
	}
	let { data, max, selected = null, band = null, height = 56, label, name, kind = 'diverging', onselect }: Props = $props();

	const n = $derived(data.length);
	const m = $derived.by(() => {
		if (max && max > 0) return max;
		let a = 0;
		for (let i = 0; i < n; i++) if (Number.isFinite(data[i])) a = Math.max(a, Math.abs(data[i]));
		return a || 1;
	});
	const seq = $derived(kind === 'sequential');
	const sc = $derived(makeScale(kind, seq ? [0, m] : [-m, m], themeState.resolved));
	const BW = 6;
	const W = $derived(Math.max(1, n * BW));
	const mid = $derived(seq ? height - 1 : height / 2);
	const yOf = (v: number) => (seq ? height - 1 - (Math.max(0, Math.min(m, v)) / m) * (height - 3) : mid - (Math.max(-m, Math.min(m, v)) / m) * (mid - 2));
	const bars = $derived(
		Array.from({ length: n }, (_, i) => {
			const v = Number.isFinite(data[i]) ? data[i] : 0;
			const y = yOf(v);
			return { i, x: i * BW + 0.75, y: Math.min(y, mid), h: Math.max(0.6, Math.abs(y - mid)), fill: sc.css(v) };
		})
	);
	let hover = $state<number | null>(null);
	let focused = $state(false);
	const active = $derived(hover ?? selected);
	function at(e: PointerEvent): number {
		const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		return Math.max(0, Math.min(n - 1, Math.floor(((e.clientX - r.left) / Math.max(1, r.width)) * n)));
	}
	function onkeydown(e: KeyboardEvent) {
		const c = selected ?? -1;
		let j: number;
		if (e.key === 'ArrowRight') j = Math.min(n - 1, c + 1);
		else if (e.key === 'ArrowLeft') j = Math.max(0, c - 1);
		else if (e.key === 'Home') j = 0;
		else if (e.key === 'End') j = n - 1;
		else return;
		e.preventDefault();
		hover = null;
		onselect?.(j);
	}
</script>

<div class="rb">
	{#if name}<span class="nm num">{name}</span>{/if}
	<div class="plot">
		<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions (composite widget: arrow keys pick an index) -->
		<svg
			viewBox="0 0 {W} {height}"
			preserveAspectRatio="none"
			style="height: {height}px; max-width: {Math.max(n * 10, 200)}px"
			tabindex={onselect ? 0 : -1}
			role="application"
			aria-roledescription={t({ en: 'vector', zh: '向量' })}
			aria-label="{t(label)} [{n}]"
			onpointermove={(e) => (hover = at(e))}
			onpointerleave={() => (hover = null)}
			onpointerdown={(e) => onselect?.(at(e))}
			onfocus={() => (focused = true)}
			onblur={() => (focused = false)}
			{onkeydown}
		>
			{#if band}
				{@const y1 = yOf(band.mean + band.sd)}
				{@const y2 = yOf(band.mean - band.sd)}
				<rect class="band" x="0" width={W} y={y1} height={Math.max(0.5, y2 - y1)} />
				<line class="mean" x1="0" x2={W} y1={yOf(band.mean)} y2={yOf(band.mean)} />
			{/if}
			<line class="axis" x1="0" x2={W} y1={mid} y2={mid} />
			{#each bars as b (b.i)}
				<rect x={b.x} width={BW - 1.5} y={b.y} height={b.h} fill={b.fill} />
			{/each}
			{#if active !== null && active < n}
				<rect class="sel" x={active * BW + 0.25} width={BW - 0.5} y="0.5" height={height - 1} />
			{/if}
		</svg>
		<div class="ro num" aria-hidden="true">
			{#if active !== null && active < n}[{active}] = {fmt3(data[active])}{:else}&nbsp;{/if}
		</div>
		<div class="visually-hidden" aria-live="polite">
			{#if focused && selected !== null && selected < n}{ui('itemAt', { i: selected, v: fmtFull(data[selected]) })}{/if}
		</div>
	</div>
</div>

<style>
	.rb {
		display: grid;
		grid-template-columns: 4.5em minmax(0, 1fr);
		gap: 8px;
		align-items: start;
		min-width: 0;
	}
	.nm {
		font-size: var(--fs-xs);
		color: var(--ink-2);
		padding-top: 4px;
		overflow-wrap: anywhere;
	}
	.plot {
		min-width: 0;
	}
	svg {
		display: block;
		width: 100%;
		touch-action: pan-y;
		cursor: crosshair;
		background: var(--surface-2);
		border-radius: 3px;
	}
	svg:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.axis {
		stroke: var(--rule-strong);
		stroke-width: 1;
		vector-effect: non-scaling-stroke;
	}
	.band {
		fill: var(--accent-soft);
	}
	.mean {
		stroke: var(--accent);
		stroke-width: 1;
		stroke-dasharray: 3 2;
		vector-effect: non-scaling-stroke;
	}
	.sel {
		fill: none;
		stroke: var(--highlight);
		stroke-width: 1.5;
		vector-effect: non-scaling-stroke;
	}
	.ro {
		font-size: var(--t-xs);
		color: var(--ink-2);
		min-height: 1.3em;
	}
	@media (max-width: 480px) {
		.rb {
			grid-template-columns: minmax(0, 1fr);
			gap: 2px;
		}
	}
</style>
