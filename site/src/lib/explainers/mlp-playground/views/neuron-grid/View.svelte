<!-- mlp-playground/neuron-grid (props: meta.ts). One canvas; nothing animates by itself: the picture follows the frame. -->
<script lang="ts">
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import type { I18n } from '../../../../atlas/types/core.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical, hexToRgb, makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
	import { dec, RES, SPAN } from '../../net.ts';
	import { untrack } from 'svelte';
	import meta, { describePic, netPic, type Choice } from './meta.ts';

	let { rt, source, at, props }: StageProps = $props();
	const show = $derived((props?.show as 'net' | 'layer' | 'neuron' | undefined) ?? 'net');
	const choices = $derived((props?.choices as Choice[] | undefined) ?? []);
	let picked = $state<string | null>(null);
	const src = $derived(picked ?? (typeof props?.source === 'string' ? (props.source as string) : source));

	// redraw while the shown source records frames
	let version = $state(0);
	$effect(() => {
		const tp = rt.sources[src]?.tape as { subscribe?: (fn: () => void) => () => void } | null | undefined;
		if (tp && typeof tp.subscribe === 'function') return tp.subscribe(() => version++);
	});
	// recompute only when the shown state changes (a new `at` object with the same cached state draws nothing new)
	const atState = $derived(at?.state);
	const pic = $derived.by(() => {
		const shown = src;
		void version;
		void atState;
		return untrack(() => netPic({ rt, source, at }, shown));
	});
	const description = $derived(t(describePic(pic)));

	let W = $state(360);
	let canvas = $state<HTMLCanvasElement | null>(null);
	const theme = $derived(themeState.resolved);
	const legend = $derived(makeScale('diverging', [-1, 1], theme));
	const P = RES * RES;

	/** a square at (x, y) of side s showing grid g of pic.grids (−1 / −2: the x / y input); k: the neuron in its layer */
	type Box = { x: number; y: number; s: number; g: number; k?: number };
	/** Where every square goes, and the canvas height. */
	const layout = $derived.by(() => {
		const w = Math.max(240, Math.min(W, 640));
		const boxes: Box[][] = [];
		if (!pic) return { w, h: 200, boxes, out: null as Box | null };
		const hidden = pic.sizes.slice(1, -1);
		const offs = hidden.map((_, l) => hidden.slice(0, l).reduce((a, b) => a + b, 0));
		const nOut = hidden.reduce((a, b) => a + b, 0);
		if (show === 'net') {
			const O = Math.round(Math.min(200, Math.max(104, w * 0.3)));
			const cols = hidden.length + 1;
			const step = (w - O - 8) / cols;
			const m = Math.max(2, ...hidden);
			const s = Math.round(Math.max(18, Math.min(52, step * 0.5, (300 - (m - 1) * 6) / m)));
			const h = Math.max(O, m * s + (m - 1) * 6) + 22;
			const col = (c: number, k: number, g0: number) =>
				Array.from({ length: k }, (_, i) => ({ x: Math.round(4 + c * step), y: Math.round(22 + (h - 22 - (k * s + (k - 1) * 6)) / 2 + i * (s + 6)), s, g: g0 + i }));
			boxes.push(col(0, 2, -2).map((b, i) => ({ ...b, g: i === 0 ? -1 : -2 })));
			hidden.forEach((k, l) => boxes.push(col(l + 1, k, offs[l])));
			return { w, h, boxes, out: { x: w - O - 4, y: 22 + (h - 22 - O) / 2, s: O, g: nOut } };
		}
		const l = Math.min(hidden.length - 1, Math.max(0, Number(props?.layer ?? 0) || 0));
		const unit = Math.min(hidden[l] - 1, Math.max(0, Number(props?.unit ?? 0) || 0));
		const ids = show === 'neuron' ? [offs[l] + unit] : Array.from({ length: hidden[l] }, (_, i) => offs[l] + i);
		const per = show === 'neuron' ? 1 : Math.max(1, Math.min(ids.length, Math.floor((w + 10) / 130)));
		const s = Math.round(Math.min(show === 'neuron' ? 320 : 180, (w - (per - 1) * 10) / per));
		const rows = Math.ceil(ids.length / per);
		boxes.push(ids.map((g, i) => ({ x: (i % per) * (s + 10), y: 18 + Math.floor(i / per) * (s + 28), s, g, k: g - offs[l] })));
		return { w, h: 18 + rows * (s + 28) - 10, boxes, out: null };
	});

	function heat(g: CanvasRenderingContext2D, b: Box, colors: { base: number[]; c0: number[]; c1: number[]; ink: string }) {
		if (!pic?.grids) return;
		const img = new ImageData(RES, RES);
		const out = b.g === pic.grids.shape[0] - 1;
		const sc = makeScale('diverging', [-1, 1], theme);
		for (let i = 0; i < P; i++) {
			let rgb: number[];
			if (b.g === -1) rgb = sc.rgb((((i % RES) + 0.5) / RES) * 2 - 1);
			else if (b.g === -2) rgb = sc.rgb(1 - ((Math.floor(i / RES) + 0.5) / RES) * 2);
			else {
				const v = pic.grids.data[b.g * P + i];
				if (out) {
					const u = Math.min(1, Math.abs(v - 0.5) * 2);
					const tgt = v >= 0.5 ? colors.c1 : colors.c0;
					const a = 0.12 + 0.48 * u;
					rgb = [0, 1, 2].map((k) => colors.base[k] + (tgt[k] - colors.base[k]) * a);
				} else rgb = sc.rgb(pic.act === 'relu' ? Math.min(1, v) : v);
			}
			img.data.set([rgb[0], rgb[1], rgb[2], 255], i * 4);
		}
		const off = Object.assign(document.createElement('canvas'), { width: RES, height: RES });
		off.getContext('2d')!.putImageData(img, 0, 0);
		g.imageSmoothingEnabled = true;
		g.drawImage(off, b.x, b.y, b.s, b.s);
		g.strokeStyle = colors.ink;
		g.lineWidth = out ? 1.5 : 1.2;
		if (b.g >= 0) zeroLine(g, b, out ? 0.5 : 0);
	}

	/** the contour where the square's value crosses `level` (marching squares on the RES grid) */
	function zeroLine(g: CanvasRenderingContext2D, b: Box, level: number) {
		const d = pic!.grids!.data;
		const v = (r: number, c: number) => d[b.g * P + r * RES + c] - level;
		const cw = b.s / RES;
		g.beginPath();
		for (let r = 0; r + 1 < RES; r++)
			for (let c = 0; c + 1 < RES; c++) {
				const [a, e, f, h] = [v(r, c), v(r, c + 1), v(r + 1, c + 1), v(r + 1, c)];
				const pts: number[][] = [];
				const x0 = b.x + (c + 0.5) * cw;
				const y0 = b.y + (r + 0.5) * cw;
				if (a * e < 0) pts.push([x0 + (a / (a - e)) * cw, y0]);
				if (e * f < 0) pts.push([x0 + cw, y0 + (e / (e - f)) * cw]);
				if (h * f < 0) pts.push([x0 + (h / (h - f)) * cw, y0 + cw]);
				if (a * h < 0) pts.push([x0, y0 + (a / (a - h)) * cw]);
				for (let k = 0; k + 1 < pts.length; k += 2) {
					g.moveTo(pts[k][0], pts[k][1]);
					g.lineTo(pts[k + 1][0], pts[k + 1][1]);
				}
			}
		g.stroke();
	}

	function dots(g: CanvasRenderingContext2D, b: Box, surface: string, accent: string) {
		if (!pic) return;
		const xy = pic.points.xy;
		const n = pic.points.n;
		const r = b.s > 150 ? (n > 400 ? 2 : 3) : 1.6;
		const at = (x: number, y: number) => [b.x + ((x + SPAN) / (2 * SPAN)) * b.s, b.y + ((SPAN - y) / (2 * SPAN)) * b.s];
		g.lineWidth = 0.8;
		g.strokeStyle = surface;
		for (let i = 0; i < n; i++) {
			const [px, py] = at(xy[2 * i], xy[2 * i + 1]);
			g.beginPath();
			g.arc(px, py, r, 0, Math.PI * 2);
			g.fillStyle = categorical(pic.points.label[i], theme);
			g.fill();
			if (b.s > 150) g.stroke();
		}
		const k = typeof props?.point === 'number' ? (props.point as number) : -1;
		if (k >= 0 && k < n) {
			const [px, py] = at(xy[2 * k], xy[2 * k + 1]);
			g.beginPath();
			g.arc(px, py, 8, 0, Math.PI * 2);
			g.lineWidth = 2.5;
			g.strokeStyle = accent;
			g.stroke();
		}
	}

	function draw() {
		const c = canvas;
		if (!c || !pic) return;
		const { w, h, boxes, out } = layout;
		const dpr = typeof devicePixelRatio === 'number' ? Math.min(3, devicePixelRatio) : 1;
		c.width = Math.round(w * dpr);
		c.height = Math.round(h * dpr);
		const g = c.getContext('2d');
		if (!g) return;
		g.setTransform(dpr, 0, 0, dpr, 0, 0);
		const css = getComputedStyle(c);
		const surface = css.getPropertyValue('--surface').trim() || (theme === 'dark' ? '#1a1d21' : '#fffdf8');
		const ink2 = css.getPropertyValue('--ink-2').trim() || '#555555';
		const accent = css.getPropertyValue('--accent').trim() || '#0b6b70';
		const colors = { base: surface.startsWith('#') ? hexToRgb(surface) : [255, 253, 248], c0: hexToRgb(categorical(0, theme)), c1: hexToRgb(categorical(1, theme)), ink: ink2 };
		g.clearRect(0, 0, w, h);
		g.font = '12px system-ui, sans-serif';
		g.fillStyle = ink2;
		g.textBaseline = 'top';
		if (show === 'net' && out) {
			// edges: the weight from each square to each square of the next column (to the output: w1 − w0, its pull
			// towards class 1), blue negative, orange positive, thicker for larger weights
			const sc = makeScale('diverging', [-1, 1], theme);
			const targets = [...boxes.slice(1), [out]];
			targets.forEach((col, l) => {
				const W8 = pic.layers[l].w;
				const I = W8.shape[1];
				const wts = col.map((_, j) => Array.from({ length: I }, (_, i) => (l === pic.layers.length - 1 ? W8.data[I + i] - W8.data[i] : W8.data[j * I + i])));
				const max = Math.max(1e-9, ...wts.flat().map(Math.abs));
				col.forEach((tb, j) =>
					boxes[l].forEach((fb, i) => {
						const v = wts[j][i];
						g.beginPath();
						g.moveTo(fb.x + fb.s, fb.y + fb.s / 2);
						g.lineTo(tb.x, tb.y + tb.s / 2);
						g.lineWidth = 0.4 + 3.2 * (Math.abs(v) / max);
						g.strokeStyle = sc.css(v >= 0 ? 0.85 : -0.85);
						g.globalAlpha = 0.25 + 0.6 * (Math.abs(v) / max);
						g.stroke();
						g.globalAlpha = 1;
					})
				);
			});
			const names: I18n[] = [{ en: 'x, y', zh: 'x、y' }, ...pic.sizes.slice(1, -1).map((_, l) => ({ en: `layer ${l + 1}`, zh: `第 ${l + 1} 层` }))];
			boxes.forEach((col, l) => g.fillText(t(names[l]), col[0].x, 2));
			g.fillText(t({ en: 'output', zh: '输出' }), out.x, 2);
		}
		for (const col of boxes)
			for (const b of col) {
				heat(g, b, colors);
				if (show !== 'net') {
					dots(g, b, surface, accent);
					g.fillStyle = ink2;
					g.fillText(t({ en: `neuron ${b.k}`, zh: `神经元 ${b.k}` }), b.x, b.y + b.s + 4);
				}
			}
		if (out) {
			heat(g, out, colors);
			dots(g, out, surface, accent);
		}
	}

	$effect(() => {
		void pic;
		void layout;
		void theme;
		draw();
	});
</script>

<figure class="ng" bind:clientWidth={W} style:min-height="{meta.minHeight}px">
	{#if choices.length}
		<div class="choices" role="group" aria-label={t({ en: 'Network size', zh: '网络大小' })}>
			{#each choices as c (c.source)}
				<button type="button" class="btn" aria-pressed={src === c.source} onclick={() => ((picked = c.source), void rt.ensure(c.source, 'last').catch(() => -1))}>{t(c.label)}</button>
			{/each}
		</div>
	{/if}
	{#if pic}
		<div class="pic" role="img" aria-label={description}><canvas bind:this={canvas} style:width="{layout.w}px" style:height="{layout.h}px" aria-hidden="true"></canvas></div>
		<figcaption class="small">
			<span class="key"><i style:background={legend.css(-1)}></i>−1 <i style:background={legend.css(1)}></i>+1</span>
			<span>{t({ en: 'step', zh: '第' })} <span class="num">{fmtInt(pic.step)}</span>{t({ en: '', zh: ' 步' })} · {pic.sizes.join(' → ')}</span>
			{#if Number.isFinite(pic.loss)}<span>{t({ en: 'loss', zh: '损失' })} <span class="num">{dec(pic.loss)}</span></span>{/if}
			{#if pic.ev}<span>{t({ en: 'right:', zh: '答对：' })} <span class="num">{fmtPct(pic.ev.accuracy)}</span> {t({ en: `of ${fmtInt(pic.points.n)} points`, zh: `（共 ${fmtInt(pic.points.n)} 个点）` })}</span>{/if}
		</figcaption>
	{:else}
		<p class="muted small">{t({ en: 'Training this network… (it starts when you pick it)', zh: '正在训练这个网络……（选中后开始）' })}</p>
	{/if}
</figure>

<style>
	.ng { margin: 0; display: grid; gap: 8px; min-width: 0; align-content: start; }
	canvas { display: block; max-width: 100%; }
	.choices, figcaption { display: flex; flex-wrap: wrap; gap: 6px 14px; }
	.btn[aria-pressed='true'] { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
	figcaption { color: var(--ink-2); }
	.key i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 2px; vertical-align: -1px; }
</style>
