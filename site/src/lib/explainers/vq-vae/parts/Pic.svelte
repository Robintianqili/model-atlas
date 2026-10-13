<!--
  vq-vae Pic (from vae-2d's Pic): one square picture on a canvas, side = √(data length) (28 for a drawing, 7 for a
  patch or a code's block). kind 'ink' (default): 0 = paper, 1 = full ink, in the theme's colours; 'heat': a
  non-negative map on the sequential (purple) scale from 0 to `max`; 'change': a signed map on the diverging scale
  (orange: more ink, blue: less) from −max to max. `frame` colours the border: 'real' for the learner's drawing (green),
  'generated' for a redrawing (magenta). `grid` draws the G × G patch borders; children are laid over the picture.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import { themeState } from '$lib/kit/theme.svelte';
	import { divergingRgb, sequentialRgb } from '$lib/kit/color';

	interface Props { data: ArrayLike<number> | null | undefined; size?: number; kind?: 'ink' | 'heat' | 'change'; max?: number; frame?: 'real' | 'generated' | 'none'; grid?: number; label: string; children?: Snippet }
	let { data, size = 112, kind = 'ink', max = 1, frame = 'none', grid = 0, label, children }: Props = $props();
	let cv = $state<HTMLCanvasElement>();
	const side = $derived(Math.round(Math.sqrt(data?.length ?? 784)));

	function rgb(css: string): number[] {
		const s = css.trim();
		if (s.startsWith('#')) return [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));
		return (s.match(/[\d.]+/g) ?? ['0', '0', '0']).slice(0, 3).map(Number);
	}
	$effect(() => {
		const theme = themeState.resolved;
		if (!cv || !data) return;
		const g = cv.getContext('2d');
		if (!g) return;
		const st = getComputedStyle(cv);
		const paper = rgb(st.getPropertyValue('--surface-2') || '#f4f4f4');
		const ink = rgb(st.getPropertyValue('--ink') || '#111111');
		const img = g.createImageData(side, side);
		for (let i = 0; i < side * side; i++) {
			const v = data[i];
			const c = kind === 'heat' ? sequentialRgb(Math.max(0, Math.min(1, v / (max || 1))), [0, 1], theme)
				: kind === 'change' ? divergingRgb(v, max || 1, theme)
				: paper.map((p, k) => p + (ink[k] - p) * Math.max(0, Math.min(1, v)));
			img.data.set([c[0], c[1], c[2], 255], i * 4);
		}
		g.putImageData(img, 0, 0);
	});
</script>

<span class="pic" role={children ? 'group' : 'img'} aria-label={label} style:width="{size}px" style:height="{size}px">
	<canvas bind:this={cv} width={side} height={side} class={frame} aria-hidden="true"></canvas>
	{#if grid > 1}<span class="grid" aria-hidden="true" style:background-size="{100 / grid}% {100 / grid}%"></span>{/if}
	{@render children?.()}
</span>

<style>
	.pic { position: relative; display: block; flex: none; }
	canvas { image-rendering: pixelated; border: 2px solid var(--rule); border-radius: 4px; display: block; width: 100%; height: 100%; box-sizing: border-box; }
	.real { border-color: var(--real); } .generated { border-color: var(--generated); }
	.grid { position: absolute; inset: 2px; pointer-events: none; background-image: linear-gradient(to right, var(--muted) 1px, transparent 1px), linear-gradient(to bottom, var(--muted) 1px, transparent 1px); opacity: 0.55; }
</style>
