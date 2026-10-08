<!--
  MapCanvas: one feature map (or image) drawn at its native resolution on a tiny canvas and scaled
  up with crisp pixels. Non-interactive (a picture); interaction belongs to the parent button or to
  TensorGrid. Optional boxes (in cells) are drawn as CSS outlines on top, so they stay sharp.
-->
<script lang="ts">
	import { makeScale, type Domain, type ScaleKind } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';

	interface BoxCell {
		y0: number;
		x0: number;
		y1: number;
		x1: number;
		/** 'solid' (default), 'dashed' or 'max' (filled corner marker) */
		style?: 'solid' | 'dashed' | 'soft';
		color?: string;
	}
	interface Props {
		data: ArrayLike<number>;
		rows: number;
		cols: number;
		kind?: ScaleKind;
		domain: Domain;
		/** CSS width in px (height follows the aspect ratio unless `height` is set) */
		size: number;
		height?: number;
		boxes?: BoxCell[];
		/** draw ink-on-paper instead of a value scale (for "what the model sees" next to the pad) */
		paper?: boolean;
		alt?: string;
	}
	let { data, rows, cols, kind = 'diverging', domain, size, height, boxes = [], paper = false, alt }: Props = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	const scale = $derived(makeScale(kind, domain, themeState.resolved));

	$effect(() => {
		const sc = scale;
		const d = data;
		const dark = themeState.resolved === 'dark';
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const img = ctx.createImageData(cols, rows);
		const px = img.data;
		const n = rows * cols;
		if (paper) {
			// ink on paper in the page's own ink / surface colors (theme aware)
			const ink = dark ? [236, 232, 222] : [29, 28, 24];
			const pap = dark ? [26, 29, 33] : [255, 253, 248];
			for (let i = 0; i < n; i++) {
				const v = Math.max(0, Math.min(1, d[i] ?? 0));
				px[i * 4] = pap[0] + (ink[0] - pap[0]) * v;
				px[i * 4 + 1] = pap[1] + (ink[1] - pap[1]) * v;
				px[i * 4 + 2] = pap[2] + (ink[2] - pap[2]) * v;
				px[i * 4 + 3] = 255;
			}
		} else {
			for (let i = 0; i < n; i++) sc.write(d[i] ?? 0, px, i * 4);
		}
		ctx.putImageData(img, 0, 0);
	});
</script>

<span class="mc" style="width: {size}px; height: {height ?? (size * rows) / cols}px">
	<canvas bind:this={canvas} width={cols} height={rows} role={alt ? 'img' : undefined} aria-label={alt} aria-hidden={alt ? undefined : 'true'}></canvas>
	{#each boxes as b, i (i)}
		<span
			class="box {b.style ?? 'solid'}"
			style="left: {(b.x0 / cols) * 100}%; top: {(b.y0 / rows) * 100}%; width: {((b.x1 - b.x0 + 1) / cols) * 100}%; height: {((b.y1 - b.y0 + 1) / rows) * 100}%; {b.color ? `--bc: ${b.color}` : ''}"
		></span>
	{/each}
</span>

<style>
	.mc {
		position: relative;
		display: block;
		flex: 0 0 auto;
	}
	canvas {
		width: 100%;
		height: 100%;
		image-rendering: pixelated;
		border-radius: 2px;
	}
	.box {
		position: absolute;
		box-sizing: border-box;
		border: 2px solid var(--bc, var(--highlight));
		box-shadow:
			0 0 0 1px var(--highlight-halo),
			inset 0 0 0 1px var(--highlight-halo);
		pointer-events: none;
	}
	.box.dashed {
		border-style: dashed;
	}
	.box.soft {
		border-width: 1.5px;
		box-shadow: none;
		opacity: 0.85;
	}
</style>
