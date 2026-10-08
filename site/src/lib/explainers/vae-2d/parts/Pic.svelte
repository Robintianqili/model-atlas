<!--
  vae-2d Pic: one 28×28 picture on a canvas. kind 'ink' (default): 0 = paper, 1 = full ink, in the theme's colours;
  'heat': a non-negative map on the sequential (purple) scale from 0 to `max`. `frame` colours the border: 'real' for the
  learner's drawing (green), 'generated' for a redrawing (magenta), as everywhere on the site.
-->
<script lang="ts">
	import { themeState } from '$lib/kit/theme.svelte';
	import { sequentialRgb } from '$lib/kit/color';

	interface Props { data: ArrayLike<number> | null | undefined; size?: number; kind?: 'ink' | 'heat'; max?: number; frame?: 'real' | 'generated' | 'none'; label: string }
	let { data, size = 112, kind = 'ink', max = 1, frame = 'none', label }: Props = $props();
	let cv = $state<HTMLCanvasElement>();

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
		const img = g.createImageData(28, 28);
		for (let i = 0; i < 784; i++) {
			const v = Math.max(0, Math.min(1, kind === 'heat' ? data[i] / (max || 1) : data[i]));
			const c = kind === 'heat' ? sequentialRgb(v, [0, 1], theme) : paper.map((p, k) => p + (ink[k] - p) * v);
			img.data.set([c[0], c[1], c[2], 255], i * 4);
		}
		g.putImageData(img, 0, 0);
	});
</script>

<span role="img" aria-label={label}><canvas bind:this={cv} width="28" height="28" class={frame} style:width="{size}px" style:height="{size}px" aria-hidden="true"></canvas></span>

<style>
	canvas { image-rendering: pixelated; border: 2px solid var(--rule); border-radius: 4px; display: block; }
	.real { border-color: var(--real); } .generated { border-color: var(--generated); }
</style>
