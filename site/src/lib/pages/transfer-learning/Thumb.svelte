<!--
  A 28×28 model input drawn as ink on paper (theme-aware), pixel-exact (image-rendering: pixelated).
  This is exactly what the network receives (framed like Quick, Draw!), not the raw pad strokes.
-->
<script lang="ts">
	import { themeState } from '$lib/kit/theme.svelte';

	interface Props {
		data: ArrayLike<number> | null;
		/** CSS px */
		size?: number;
		alt: string;
	}
	let { data, size = 42, alt }: Props = $props();
	let canvas: HTMLCanvasElement | undefined = $state();

	function hex(v: string, fallback: [number, number, number]): [number, number, number] {
		const m = /^#([0-9a-f]{6})$/i.exec(v.trim());
		if (!m) return fallback;
		const n = parseInt(m[1], 16);
		return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
	}

	$effect(() => {
		void themeState.resolved;
		const d = data;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const cs = getComputedStyle(canvas);
		const paper = hex(cs.getPropertyValue('--surface'), [255, 253, 248]);
		const ink = hex(cs.getPropertyValue('--ink'), [29, 28, 24]);
		const img = ctx.createImageData(28, 28);
		for (let i = 0; i < 784; i++) {
			const v = d ? Math.max(0, Math.min(1, d[i])) : 0;
			img.data[i * 4] = paper[0] + (ink[0] - paper[0]) * v;
			img.data[i * 4 + 1] = paper[1] + (ink[1] - paper[1]) * v;
			img.data[i * 4 + 2] = paper[2] + (ink[2] - paper[2]) * v;
			img.data[i * 4 + 3] = 255;
		}
		ctx.putImageData(img, 0, 0);
	});
</script>

{#if alt}
	<span class="wrap" role="img" aria-label={alt}><canvas bind:this={canvas} width="28" height="28" style:width="{size}px" style:height="{size}px" aria-hidden="true"></canvas></span>
{:else}
	<canvas bind:this={canvas} width="28" height="28" style:width="{size}px" style:height="{size}px" aria-hidden="true"></canvas>
{/if}

<style>
	.wrap {
		display: inline-block;
	}
	canvas {
		display: block;
		image-rendering: pixelated;
		border-radius: 3px;
		background: var(--surface);
	}
</style>
