<!--
  MiniHeat: a small, non-interactive heatmap of a [rows × cols] matrix stretched to a fixed box
  (cells need not be square). Used for thumbnails (overview cards, the layers × heads grid).
  Real values, the kit color scales; masked cells (causal mask) are drawn in the mask color.
  Drawn often (every keystroke, ~25 instances), so the loop reads plain locals and the CSS colors
  are read once per theme.
-->
<script module lang="ts">
	const cssCache = new Map<string, string>();
	let lastTheme = '';
	/** Theme token from :root, cached per resolved theme (reading computed styles forces a style recalc). */
	function themeVar(theme: string, name: string, fallback: string): string {
		if (theme !== lastTheme) {
			cssCache.clear();
			lastTheme = theme;
		}
		const key = theme + name;
		let v = cssCache.get(key);
		if (v === undefined) {
			v = getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
			cssCache.set(key, v);
		}
		return v;
	}
</script>

<script lang="ts">
	import { makeScale, domainOf, type Domain, type ScaleKind } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';

	interface Props {
		data: ArrayLike<number>;
		rows: number;
		cols: number;
		width: number;
		height: number;
		kind?: ScaleKind;
		domain?: Domain;
		/** causal: cells above the diagonal (col > row) are masked */
		causal?: boolean;
		/** outline one row (e.g. the selected token) */
		hlRow?: number | null;
		hlColor?: string;
		label: string;
	}
	let { data, rows, cols, width, height, kind = 'diverging', domain, causal = false, hlRow = null, hlColor = '--highlight', label }: Props = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	let bm: HTMLCanvasElement | null = null;

	function hexRgb(v: string, fb: [number, number, number]): [number, number, number] {
		const m = /^#([0-9a-f]{6})$/i.exec(v);
		if (!m) return fb;
		const h = m[1];
		return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
	}

	$effect(() => {
		const theme = themeState.resolved;
		const R = rows;
		const C = cols;
		const d = data;
		const W = width;
		const Hh = height;
		const cz = causal;
		const hl = hlRow;
		if (!canvas || R <= 0 || C <= 0) return;
		let dom = domain;
		if (!dom) {
			if (cz) {
				const fin: number[] = [];
				for (let r = 0; r < R; r++) for (let c = 0; c <= r; c++) fin.push(d[r * C + c]);
				dom = domainOf(kind, fin);
			} else dom = domainOf(kind, d);
		}
		const sc = makeScale(kind, dom, theme);
		if (!bm) bm = document.createElement('canvas');
		bm.width = C;
		bm.height = R;
		const b = bm.getContext('2d');
		if (!b) return;
		const img = b.createImageData(C, R);
		const px = img.data;
		const [mr, mg, mb] = hexRgb(themeVar(theme, '--surface-3', '#e6e0d1'), [230, 224, 209]);
		for (let r = 0; r < R; r++)
			for (let c = 0; c < C; c++) {
				const i = r * C + c;
				if (cz && c > r) {
					px[i * 4] = mr;
					px[i * 4 + 1] = mg;
					px[i * 4 + 2] = mb;
					px[i * 4 + 3] = 255;
				} else sc.write(d[i], px, i * 4);
			}
		b.putImageData(img, 0, 0);
		const dpr = window.devicePixelRatio || 1;
		const cw = Math.round(W * dpr);
		const ch = Math.round(Hh * dpr);
		if (canvas.width !== cw) canvas.width = cw;
		if (canvas.height !== ch) canvas.height = ch;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.imageSmoothingEnabled = false;
		ctx.clearRect(0, 0, W, Hh);
		ctx.drawImage(bm, 0, 0, W, Hh);
		if (hl !== null && hl >= 0 && hl < R) {
			const rh = Hh / R;
			ctx.lineWidth = 1.5;
			ctx.strokeStyle = themeVar(theme, hlColor, '#1d1c18');
			ctx.strokeRect(0.75, hl * rh + 0.75, W - 1.5, Math.max(1.5, rh - 1.5));
		}
	});
</script>

<span class="mh" role="img" aria-label={label}><canvas bind:this={canvas} style="width: {width}px; height: {height}px" aria-hidden="true"></canvas></span>

<style>
	.mh {
		display: block;
		line-height: 0;
	}
	canvas {
		display: block;
		border-radius: 2px;
		image-rendering: pixelated;
		max-width: 100%;
	}
</style>
