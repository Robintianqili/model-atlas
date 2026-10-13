<!--
  The Model Atlas mark (redesign spec §4.2): a 3 x 3 kernel in ink with one highlighter cell, the product in one glyph.
  It is the favicon (static/favicon.svg) and sits at 28-30 px in the bars. Decorative: the link around it carries the name.
    <Mark />  <Mark size={26} />
-->
<script lang="ts">
	let { size = 28 }: { size?: number } = $props();
	/** cell opacities, row by row; cell (1, 2) is the highlighter cell */
	const OP = [0.4, 0.85, 0.22, 0.65, 1, 1, 0.16, 0.5, 0.92];
</script>

<svg class="mark" width={size} height={size} viewBox="0 0 28 28" aria-hidden="true" focusable="false">
	<rect width="28" height="28" rx="7" class="bg" />
	{#each OP as o, i (i)}
		{@const r = Math.floor(i / 3)}
		{@const c = i % 3}
		<rect x={5 + c * 6.4} y={5 + r * 6.4} width="5" height="5" rx="1.2" class={r === 1 && c === 2 ? 'hot' : 'cell'} opacity={r === 1 && c === 2 ? 1 : o} />
	{/each}
</svg>

<style>
	.mark {
		display: block;
		flex: none;
	}
	.bg {
		fill: var(--ink);
	}
	.cell {
		fill: var(--paper);
	}
	.hot {
		fill: var(--mark);
	}
	@media (forced-colors: active) {
		.bg {
			fill: CanvasText;
		}
		.cell,
		.hot {
			fill: Canvas;
		}
	}
</style>
