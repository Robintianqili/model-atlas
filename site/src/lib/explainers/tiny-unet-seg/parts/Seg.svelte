<!--
  tiny-unet-seg Seg: the 32×32 picture in grey (0 = paper, 1 = ink, in the theme's colours) with a class map over it
  (class k tinted with the site's class colour k − 1; background untinted) at opacity `alpha`, or with `probs` ([4, 32·32]
  shares) every class's colour mixed in by its share, so an unsure pixel shows a blend; optional marks: the
  selected pixel, wrong pixels (a slash), and windows (boxes, solid / dashed / dotted). With `onpick`, a click, a drag
  or the arrow keys pick a pixel; with `onmove` too, the arrow keys move the cursor `sel` and Enter or Space picks it.
-->
<script lang="ts">
	import { themeState } from '$lib/kit/theme.svelte';
	import { categorical, hexToRgb } from '$lib/kit/color';
	import type { Box } from '../../../atlas/types/index.ts';
	import { S } from '../scenes.ts';

	interface Props { img: ArrayLike<number>; cls?: ArrayLike<number> | null; probs?: ArrayLike<number> | null; alpha?: number; sel?: [number, number] | null; wrong?: ArrayLike<number> | null; boxes?: (Box | null)[]; size?: number; label: string; onpick?: (rc: [number, number], key?: boolean) => void; onmove?: (rc: [number, number]) => void }
	let { img, cls = null, probs = null, alpha = 0.55, sel = null, wrong = null, boxes = [], size = 256, label, onpick, onmove }: Props = $props();
	let cv = $state<HTMLCanvasElement>();
	const C = 8;
	const css = (el: Element, v: string, d: string) => getComputedStyle(el).getPropertyValue(v).trim() || d;
	const rgb = (s: string) => (s.startsWith('#') ? hexToRgb(s) : (s.match(/[\d.]+/g) ?? ['0', '0', '0']).slice(0, 3).map(Number));

	$effect(() => {
		const theme = themeState.resolved;
		const g = cv?.getContext('2d');
		if (!cv || !g) return;
		const paper = rgb(css(cv, '--surface-2', '#f4f4f4')), ink = rgb(css(cv, '--ink', '#111111'));
		const tint = [0, 1, 2].map((k) => hexToRgb(categorical(k, theme)));
		for (let i = 0; i < S * S; i++) {
			const v = Math.max(0, Math.min(1, img[i]));
			let c = paper.map((p, j) => p + (ink[j] - p) * v);
			const k = cls ? cls[i] : 0;
			if (probs) c = c.map((q, j) => (1 - alpha) * q + alpha * (probs[i] * q + [1, 2, 3].reduce((a, m) => a + probs[m * S * S + i] * tint[m - 1][j], 0)));
			else if (k > 0) c = c.map((q, j) => q + (tint[k - 1][j] - q) * alpha);
			g.fillStyle = `rgb(${c.map(Math.round).join(',')})`;
			g.fillRect((i % S) * C, Math.floor(i / S) * C, C, C);
			if (wrong?.[i]) {
				g.strokeStyle = css(cv, '--surface', '#fff');
				g.lineWidth = 1.5;
				g.beginPath(), g.moveTo((i % S) * C + 1, Math.floor(i / S) * C + C - 1), g.lineTo((i % S) * C + C - 1, Math.floor(i / S) * C + 1), g.stroke();
			}
		}
		g.strokeStyle = css(cv, '--ink', '#111');
		boxes.forEach((b, j) => {
			if (!b) return;
			g.lineWidth = 2;
			g.setLineDash(j === 0 ? [] : j === 1 ? [6, 3] : [2, 3]);
			g.strokeRect(b.x0 * C + 1 + j, b.y0 * C + 1 + j, (b.x1 - b.x0 + 1) * C - 2 - 2 * j, (b.y1 - b.y0 + 1) * C - 2 - 2 * j);
		});
		g.setLineDash([]);
		if (sel) {
			for (const [w, v] of [[4, '--surface'], [2, '--ink']] as const) (g.lineWidth = w), (g.strokeStyle = css(cv, v, '#888')), g.strokeRect(sel[1] * C - 1, sel[0] * C - 1, C + 2, C + 2);
		}
	});
	function at(e: PointerEvent): [number, number] {
		const r = cv!.getBoundingClientRect();
		const q = (v: number, n: number) => Math.max(0, Math.min(S - 1, Math.floor((v / n) * S)));
		return [q(e.clientY - r.top, r.height), q(e.clientX - r.left, r.width)];
	}
	const pick = (e: PointerEvent) => onpick && (e.buttons || e.type === 'pointerdown') && onpick(at(e));
	function key(e: KeyboardEvent) {
		const d: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
		if (!onpick || !sel) return;
		if (onmove && (e.key === 'Enter' || e.key === ' ')) return e.preventDefault(), onpick(sel, true);
		if (!d[e.key]) return;
		e.preventDefault();
		(onmove ?? onpick)([Math.max(0, Math.min(S - 1, sel[0] + d[e.key][0])), Math.max(0, Math.min(S - 1, sel[1] + d[e.key][1]))]);
	}
</script>

<canvas bind:this={cv} width={S * C} height={S * C} style:max-width="{size}px" class:pick={!!onpick} aria-label={label} role={onpick ? 'application' : 'img'} tabindex={onpick ? 0 : -1} onpointerdown={pick} onpointermove={pick} onkeydown={key}></canvas>

<style>
	canvas { width: 100%; aspect-ratio: 1; image-rendering: pixelated; border: 2px solid var(--rule); border-radius: 4px; display: block; }
	.pick { cursor: crosshair; }
	.pick:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>
