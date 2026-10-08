<!--
  TensorGrid: a matrix or one feature map drawn on canvas with crisp cells.

  <TensorGrid data={fmap} rows={26} cols={26} label="conv1 channel 3"
              kind="diverging" scope="layer" scopeDomains={{ layer: layerDom }}
              highlights={[{ row: r, col: c, h: 3, w: 3 }]}
              onhover={(c) => (hovered = c)} />

  - `data` is row-major (length rows*cols). Pass a NEW array (or bump `version`) to redraw.
  - Color domain: `domain` if given, else from `scope` ('unit' = this map; 'layer'/'global'
    use `scopeDomains`, computed by the page with domainOf()).
  - Cell size: `cell` (CSS px) or auto-fit to the container (between minCell and maxCell);
    below minCell the grid scrolls horizontally instead of dropping cells.
  - Grid lines appear when cells are ≥ 7 px; value labels when cells are ≥ 30 px (labels="auto").
  - Highlight colors may be CSS colors or "var(--token)" (resolved against the theme).
  - Mouse/touch: hover (tap on touch) reports the cell; click selects. Keyboard: Tab to focus,
    arrow keys move, Home/End jump, Enter selects. A live region announces the focused cell.
-->
<script module lang="ts">
	// CSS variables depend only on the colour scheme. getComputedStyle forces a style recalculation (expensive on
	// big pages, and it ran on every repaint of every grid), so each variable is read once per scheme and shared.
	const cssCache = new Map<string, string>();
	let darkMq: MediaQueryList | null = null;
	function schemeKey(): string {
		const a = document.documentElement.getAttribute('data-theme');
		if (a === 'light' || a === 'dark') return a;
		darkMq ??= matchMedia('(prefers-color-scheme: dark)');
		return darkMq.matches ? 'dark' : 'light';
	}
</script>

<script lang="ts">
	import { untrack } from 'svelte';
	import { keyScroll } from './keyscroll';
	import { makeScale, scopeDomain, type ColorScope, type Domain, type ScaleKind } from './color';
	import { themeState } from './theme.svelte';
	import { fmt3, fmtFull } from './format';
	import { ui, t, type Text } from '$lib/i18n/i18n.svelte';
	import type { Cell, CellValue, Highlight } from './types';

	interface Props {
		data: ArrayLike<number>;
		rows: number;
		cols: number;
		kind?: ScaleKind;
		domain?: Domain;
		scope?: ColorScope;
		scopeDomains?: { layer?: Domain; global?: Domain };
		cell?: number;
		maxCell?: number;
		minCell?: number;
		gridLines?: boolean;
		labels?: 'auto' | boolean;
		highlights?: Highlight[];
		/** cells for which this returns true are drawn hatched (e.g. a causal mask) */
		masked?: (row: number, col: number) => boolean;
		/** keyboard/selection cursor (bindable) */
		cursor?: Cell | null;
		/** false = a static picture (no focus, no hover), e.g. inside TileStack */
		interactive?: boolean;
		/** accessible name */
		label: Text;
		/** show the "row, col: value" line under the grid */
		readout?: boolean;
		/** bump to force a redraw when `data` was mutated in place */
		version?: number;
		onhover?: (c: CellValue | null) => void;
		onselect?: (c: CellValue) => void;
	}

	let {
		data,
		rows,
		cols,
		kind = 'diverging',
		domain,
		scope = 'unit',
		scopeDomains = {},
		cell,
		maxCell = 40,
		minCell = 2,
		gridLines = true,
		labels = 'auto',
		highlights = [],
		masked,
		cursor = $bindable(null),
		interactive = true,
		label,
		readout = true,
		version = 0,
		onhover,
		onselect
	}: Props = $props();

	let wrap: HTMLDivElement | undefined = $state();
	let canvas: HTMLCanvasElement | undefined = $state();
	let avail = $state(0);
	let hover = $state<Cell | null>(null);
	let focused = $state(false);
	let bitmapTick = $state(0);
	let bitmap: HTMLCanvasElement | null = null;

	const dom = $derived(domain ?? scopeDomain(kind, scope, data, scopeDomains));
	const scale = $derived(makeScale(kind, dom, themeState.resolved));
	const cellPx = $derived(
		cell ?? (avail > 0 ? Math.max(minCell, Math.min(maxCell, Math.floor(avail / cols))) : Math.min(maxCell, 12))
	);
	const cssW = $derived(cols * cellPx);
	const cssH = $derived(rows * cellPx);
	const showLabels = $derived(labels === true || (labels === 'auto' && cellPx >= 30));

	const valueAt = (c: Cell) => data[c.row * cols + c.col];
	const active = $derived(hover ?? (focused ? cursor : null));
	const readoutText = $derived(
		active
			? ui('cellAt', { r: active.row, c: active.col, v: masked?.(active.row, active.col) ? ui('masked') : fmt3(valueAt(active)) })
			: ''
	);
	const liveText = $derived(
		focused && cursor
			? ui('cellAt', {
					r: cursor.row,
					c: cursor.col,
					v: masked?.(cursor.row, cursor.col) ? ui('masked') : fmtFull(valueAt(cursor))
				})
			: ''
	);

	// measure available width
	$effect(() => {
		if (!wrap || cell) return;
		const ro = new ResizeObserver((entries) => {
			avail = Math.floor(entries[0].contentRect.width);
		});
		ro.observe(wrap);
		return () => ro.disconnect();
	});

	// build the 1-pixel-per-cell bitmap whenever data or the scale change
	$effect(() => {
		void version;
		const sc = scale;
		const n = rows * cols;
		if (!bitmap) bitmap = document.createElement('canvas');
		bitmap.width = cols;
		bitmap.height = rows;
		const bctx = bitmap.getContext('2d');
		if (!bctx || n === 0) return;
		const img = bctx.createImageData(cols, rows);
		const px = img.data;
		const mask = masked;
		// only read the mask color when there is a mask (getComputedStyle forces a style recalc)
		const [mr, mg, mb] = mask ? maskRgb() : [0, 0, 0];
		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				const i = r * cols + c;
				if (mask && mask(r, c)) {
					px[i * 4] = mr;
					px[i * 4 + 1] = mg;
					px[i * 4 + 2] = mb;
					px[i * 4 + 3] = 255;
				} else sc.write(data[i], px, i * 4);
			}
		}
		bctx.putImageData(img, 0, 0);
		// bump without reading (reading would make this effect depend on its own output)
		bitmapTick = untrack(() => bitmapTick) + 1;
	});

	function cssVar(name: string, fallback: string): string {
		if (!canvas) return fallback;
		const key = schemeKey() + name;
		let v = cssCache.get(key);
		if (v === undefined) {
			v = getComputedStyle(canvas).getPropertyValue(name).trim();
			if (v) cssCache.set(key, v);
		}
		return v || fallback;
	}
	function maskRgb(): [number, number, number] {
		void themeState.resolved;
		const v = cssVar('--surface-2', '#f0ece1');
		const m = /^#([0-9a-f]{6})$/i.exec(v);
		if (!m) return [230, 226, 214];
		const h = m[1];
		return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
	}

	// paint: bitmap scaled up with nearest-neighbor, then lines, labels, highlights, cursor
	$effect(() => {
		void bitmapTick;
		void themeState.resolved;
		const hl = highlights;
		const act = active;
		const cur = focused ? cursor : null;
		if (!canvas || !bitmap) return;
		const dpr = window.devicePixelRatio || 1;
		const W = cssW;
		const H = cssH;
		canvas.width = Math.max(1, Math.round(W * dpr));
		canvas.height = Math.max(1, Math.round(H * dpr));
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.imageSmoothingEnabled = false;
		ctx.clearRect(0, 0, W, H);
		ctx.drawImage(bitmap, 0, 0, W, H);
		const cp = cellPx;

		if (masked && cp >= 5) {
			ctx.save();
			ctx.strokeStyle = cssVar('--mask-hatch', '#c9c0aa');
			ctx.lineWidth = 1;
			for (let r = 0; r < rows; r++)
				for (let c = 0; c < cols; c++) {
					if (!masked(r, c)) continue;
					const x = c * cp;
					const y = r * cp;
					ctx.save();
					ctx.beginPath();
					ctx.rect(x, y, cp, cp);
					ctx.clip();
					ctx.beginPath();
					for (let k = -cp; k <= cp; k += 4) {
						ctx.moveTo(x + k, y + cp);
						ctx.lineTo(x + k + cp, y);
					}
					ctx.stroke();
					ctx.restore();
				}
			ctx.restore();
		}

		if (gridLines && cp >= 7) {
			ctx.save();
			ctx.strokeStyle = cssVar('--surface', '#fffdf8');
			ctx.globalAlpha = cp >= 14 ? 0.9 : 0.55;
			ctx.lineWidth = 1;
			ctx.beginPath();
			for (let c = 1; c < cols; c++) {
				ctx.moveTo(c * cp + 0.5, 0);
				ctx.lineTo(c * cp + 0.5, H);
			}
			for (let r = 1; r < rows; r++) {
				ctx.moveTo(0, r * cp + 0.5);
				ctx.lineTo(W, r * cp + 0.5);
			}
			ctx.stroke();
			ctx.restore();
		}

		if (showLabels) {
			ctx.save();
			const fs = Math.max(9, Math.min(13, Math.floor(cp / 3.4)));
			ctx.font = `500 ${fs}px ${cssVar('--font-mono', 'monospace')}`;
			ctx.textAlign = 'center';
			ctx.textBaseline = 'middle';
			for (let r = 0; r < rows; r++)
				for (let c = 0; c < cols; c++) {
					if (masked?.(r, c)) continue;
					const v = data[r * cols + c];
					ctx.fillStyle = scale.ink(v);
					ctx.fillText(fmt3(v), c * cp + cp / 2, r * cp + cp / 2 + 0.5, cp - 3);
				}
			ctx.restore();
		}

		const ink = cssVar('--highlight', '#1d1c18');
		const halo = cssVar('--highlight-halo', '#fffdf8');
		const rect = (h: Highlight, lw: number) => {
			const x = h.col * cp;
			const y = h.row * cp;
			const w = (h.w ?? 1) * cp;
			const hh = (h.h ?? 1) * cp;
			ctx.save();
			ctx.lineWidth = lw + 2;
			ctx.strokeStyle = halo;
			ctx.setLineDash([]);
			ctx.strokeRect(x + 1, y + 1, w - 2, hh - 2);
			ctx.lineWidth = lw;
			// canvas cannot read CSS variables: resolve "var(--name)" here
			ctx.strokeStyle = h.color?.startsWith('var(') ? cssVar(h.color.slice(4, -1).trim(), ink) : (h.color ?? ink);
			if (h.style === 'dashed') ctx.setLineDash([4, 3]);
			ctx.strokeRect(x + 1, y + 1, w - 2, hh - 2);
			ctx.restore();
		};
		for (const h of hl) rect(h, 2);
		if (act) rect({ row: act.row, col: act.col }, 2);
		if (cur && (!act || cur.row !== act.row || cur.col !== act.col)) rect({ ...cur, style: 'dashed' }, 2);
	});

	function cellFrom(e: PointerEvent | MouseEvent): Cell | null {
		if (!canvas) return null;
		const r = canvas.getBoundingClientRect();
		const col = Math.floor(((e.clientX - r.left) / r.width) * cols);
		const row = Math.floor(((e.clientY - r.top) / r.height) * rows);
		if (row < 0 || col < 0 || row >= rows || col >= cols) return null;
		return { row, col };
	}
	const withValue = (c: Cell): CellValue => ({ ...c, value: valueAt(c) });

	function onpointermove(e: PointerEvent) {
		if (!interactive) return;
		const c = cellFrom(e);
		if (c?.row === hover?.row && c?.col === hover?.col) return;
		hover = c;
		onhover?.(c ? withValue(c) : null);
	}
	function onpointerleave(e: PointerEvent) {
		if (!interactive || e.pointerType !== 'mouse') return;
		hover = null;
		onhover?.(cursor && focused ? withValue(cursor) : null);
	}
	function onclick(e: MouseEvent) {
		if (!interactive) return;
		const c = cellFrom(e);
		if (!c) return;
		cursor = c;
		hover = c;
		onhover?.(withValue(c));
		onselect?.(withValue(c));
	}
	function onkeydown(e: KeyboardEvent) {
		if (!interactive) return;
		const c = cursor ?? { row: 0, col: 0 };
		let { row, col } = c;
		switch (e.key) {
			case 'ArrowRight':
				col = Math.min(cols - 1, col + 1);
				break;
			case 'ArrowLeft':
				col = Math.max(0, col - 1);
				break;
			case 'ArrowDown':
				row = Math.min(rows - 1, row + 1);
				break;
			case 'ArrowUp':
				row = Math.max(0, row - 1);
				break;
			case 'Home':
				col = 0;
				if (e.ctrlKey) row = 0;
				break;
			case 'End':
				col = cols - 1;
				if (e.ctrlKey) row = rows - 1;
				break;
			case 'Enter':
			case ' ':
				e.preventDefault();
				onselect?.(withValue(c));
				return;
			default:
				return;
		}
		e.preventDefault();
		cursor = { row, col };
		hover = null;
		onhover?.(withValue(cursor));
	}
</script>

<div class="tg" bind:this={wrap}>
	<div class="scroll" style={cell ? '' : 'width: 100%'} use:keyScroll>
		<canvas
			bind:this={canvas}
			class:interactive
			width={cssW}
			height={cssH}
			style="width: {cssW}px; height: {cssH}px"
			tabindex={interactive ? 0 : -1}
			role={interactive ? 'application' : 'img'}
			aria-roledescription={interactive ? 'data grid' : undefined}
			aria-label="{t(label)} [{rows}×{cols}]. {interactive ? ui('gridKeysHelp') : ''}"
			{onpointermove}
			{onpointerleave}
			{onclick}
			{onkeydown}
			onfocus={() => (focused = true)}
			onblur={() => (focused = false)}
		></canvas>
	</div>
	{#if interactive && readout}
		<div class="readout num" aria-hidden="true">{readoutText || ' '}</div>
		<div class="visually-hidden" aria-live="polite">{liveText}</div>
	{/if}
</div>

<style>
	.tg {
		max-width: 100%;
		min-width: 0;
	}
	.scroll {
		max-width: 100%;
		overflow-x: auto;
		overscroll-behavior-x: contain;
	}
	canvas {
		display: block;
		border-radius: 2px;
		image-rendering: pixelated;
		touch-action: pan-y;
	}
	canvas.interactive {
		cursor: crosshair;
	}
	canvas:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 3px;
	}
	.readout {
		min-height: 1.5em;
		margin-top: 4px;
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
</style>
