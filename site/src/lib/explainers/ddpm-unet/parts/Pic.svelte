<!--
  ddpm-unet Pic (a copy of vae-2d's, plus 'signed'): one square picture on a canvas. kind 'ink' (default): `lo` = paper,
  1 = full ink, in the theme's colours (lo 0 for a drawing, −1 for the DDPM's pictures; values outside are clamped);
  'signed': noise and other signed numbers on the site's diverging scale (blue negative, orange positive) up to ±max.
  `frame` colours the border: 'real' for the learner's drawing (green), 'generated' for a sample (magenta); `mark`
  outlines one pixel (a flat index).
-->
<script lang="ts">
	import { themeState } from '$lib/kit/theme.svelte';
	import { divergingRgb } from '$lib/kit/color';

	interface Props { data: ArrayLike<number> | null | undefined; size?: number; kind?: 'ink' | 'signed'; max?: number; frame?: 'real' | 'generated' | 'none'; label: string; mark?: number; lo?: number }
	let { data, size = 112, kind = 'ink', max = 3, frame = 'none', label, mark = -1, lo = -1 }: Props = $props();
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
			const v = Math.max(0, Math.min(1, (data[i] - lo) / (1 - lo)));
			const c = kind === 'signed' ? divergingRgb(data[i], max, theme) : paper.map((p, k) => p + (ink[k] - p) * v);
			img.data.set([c[0], c[1], c[2], 255], i * 4);
		}
		g.putImageData(img, 0, 0);
	});
</script>

<span role="img" aria-label={label} class="pic"><canvas bind:this={cv} width={side} height={side} class={frame} style:width="{size}px" style:height="{size}px" aria-hidden="true"></canvas>{#if mark >= 0}<span class="mark" style:left="{2 + ((mark % side) * size) / side}px" style:top="{2 + (Math.floor(mark / side) * size) / side}px" style:width="{size / side}px" style:height="{size / side}px"></span>{/if}</span>

<style>
	canvas { image-rendering: pixelated; border: 2px solid var(--rule); border-radius: 4px; display: block; }
	.real { border-color: var(--real); } .generated { border-color: var(--generated); }
	.pic { position: relative; display: block; }
	.mark { position: absolute; outline: 2px solid var(--generated); outline-offset: 1px; pointer-events: none; }
</style>
