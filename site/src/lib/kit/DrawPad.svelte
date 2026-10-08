<!--
  DrawPad: a square drawing pad for mouse, touch and pen (pointer events), with brush size,
  undo, clear and presets. Works on phones: a touch that starts on the pad draws (touch-action:
  none); scroll the page by touching outside it. Only the primary pointer draws.

  <DrawPad bind:this={pad} bind:strokes onchange={(s) => run(pad.getImageData(28))} presets={presets} />

  - `strokes` (bindable) are normalized 0..1 (see draw.ts), so presets render at any size.
  - `getImageData(px?)` returns a px×px ImageData (default: the pad's own resolution), always
    BLACK ink on WHITE paper regardless of theme. `getInk(px)` returns Float32 ink intensity in [0, 1].
  - `onchange(strokes)` fires when a stroke ends, on undo, clear and preset; `ondraw` fires
    while drawing (throttled to animation frames) for live updates.
  - Keyboard users: presets are buttons; Ctrl/Cmd+Z undoes when the pad has focus.
-->
<script lang="ts">
	import { paintStrokes, rasterize, toInk, type Stroke, type StrokePoint } from './draw';
	import { themeState } from './theme.svelte';
	import { ui, t, colon, type Text } from '$lib/i18n/i18n.svelte';

	interface Props {
		strokes?: Stroke[];
		/** brush diameter as a fraction of the pad side */
		brush?: number;
		/** largest CSS size of the pad; it shrinks to fit narrow screens */
		size?: number;
		/** internal resolution in px (export default) */
		resolution?: number;
		presets?: { name: Text; strokes: Stroke[] }[];
		showBrush?: boolean;
		label?: Text;
		onchange?: (strokes: Stroke[]) => void;
		ondraw?: (strokes: Stroke[]) => void;
	}
	let {
		strokes = $bindable([]),
		brush = $bindable(0.065),
		size = 280,
		resolution = 280,
		presets = [],
		showBrush = true,
		label,
		onchange,
		ondraw
	}: Props = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	let current: Stroke | null = null;
	let drawing = $state(false);
	let rafPending = false;
	let redrawTick = $state(0);
	// On-screen backing store follows devicePixelRatio so strokes stay sharp on phones and
	// retina screens. Exports (getImageData/getInk) are rasterized separately at `resolution`.
	let dpr = $state(1);
	const backing = $derived(Math.round(resolution * dpr));
	$effect(() => {
		dpr = Math.min(3, Math.max(1, window.devicePixelRatio || 1));
	});

	export function getImageData(px: number = resolution): ImageData {
		return rasterize(strokes, px);
	}
	export function getInk(px: number = resolution): Float32Array {
		return toInk(rasterize(strokes, px));
	}
	export function clear() {
		strokes = [];
		onchange?.(strokes);
	}
	export function undo() {
		if (!strokes.length) return;
		strokes = strokes.slice(0, -1);
		onchange?.(strokes);
	}

	function colors(): { ink: string; paper: string } {
		void themeState.resolved;
		if (!canvas) return { ink: '#1d1c18', paper: '#fffdf8' };
		const cs = getComputedStyle(canvas);
		return {
			ink: cs.getPropertyValue('--ink').trim() || '#1d1c18',
			paper: cs.getPropertyValue('--surface').trim() || '#fffdf8'
		};
	}

	function render() {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const { ink, paper } = colors();
		ctx.setTransform(1, 0, 0, 1, 0, 0);
		ctx.fillStyle = paper;
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		const all = current ? [...strokes, current] : strokes;
		paintStrokes(ctx, all, canvas.width, ink);
	}

	$effect(() => {
		void strokes;
		void redrawTick;
		void themeState.resolved;
		void backing;
		render();
	});

	function point(e: PointerEvent): StrokePoint {
		const r = canvas!.getBoundingClientRect();
		return {
			x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
			y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
			p: e.pointerType === 'pen' && e.pressure > 0 ? e.pressure : 0.5
		};
	}

	function down(e: PointerEvent) {
		if (e.button !== 0 && e.pointerType === 'mouse') return;
		// a second finger must not replace the stroke in progress
		if (!e.isPrimary || drawing) return;
		e.preventDefault();
		// preventDefault on pointerdown suppresses the focus that a click would give: focus
		// explicitly so Ctrl/Cmd+Z works right after drawing
		canvas!.focus({ preventScroll: true });
		try {
			canvas!.setPointerCapture(e.pointerId);
		} catch {
			/* pointer already gone (or a synthetic event): draw without capture */
		}
		current = { points: [point(e)], size: brush };
		drawing = true;
		redrawTick++;
	}
	function move(e: PointerEvent) {
		if (!drawing || !current || !e.isPrimary) return;
		e.preventDefault();
		const events = typeof e.getCoalescedEvents === 'function' ? e.getCoalescedEvents() : [e];
		for (const ev of events.length ? events : [e]) current.points.push(point(ev));
		if (!rafPending) {
			rafPending = true;
			requestAnimationFrame(() => {
				rafPending = false;
				render();
				if (current) ondraw?.([...strokes, current]);
			});
		}
	}
	function up(e: PointerEvent) {
		if (!drawing || !current || !e.isPrimary) return;
		drawing = false;
		const s = current;
		current = null;
		strokes = [...strokes, s];
		onchange?.(strokes);
	}
	function onkeydown(e: KeyboardEvent) {
		if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
			e.preventDefault();
			undo();
		}
	}
	function usePreset(p: { strokes: Stroke[] }) {
		strokes = p.strokes.map((s) => ({ size: s.size, points: s.points.map((q) => ({ ...q })) }));
		onchange?.(strokes);
	}
</script>

<div class="pad" style="--size: {size}px">
	<canvas
		bind:this={canvas}
		width={backing}
		height={backing}
		tabindex="0"
		aria-label="{label ? t(label) + '. ' : ''}{ui('drawPadLabel')} {strokes.length ? ui('strokes', { n: strokes.length }) : ui('emptyCanvas')}"
		onpointerdown={down}
		onpointermove={move}
		onpointerup={up}
		onpointercancel={up}
		onlostpointercapture={up}
		{onkeydown}
	></canvas>
	{#if strokes.length === 0 && !drawing}
		<span class="hint" aria-hidden="true">{ui('drawHere')}</span>
	{/if}
	<div class="tools">
		<button type="button" class="btn" onclick={undo} disabled={!strokes.length}>
			<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path d="M5 4 2 7l3 3M2.5 7H10a4 4 0 0 1 0 8H7" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" /></svg>
			{ui('undo')}
		</button>
		<button type="button" class="btn" onclick={clear} disabled={!strokes.length}>{ui('clear')}</button>
		{#if showBrush}
			<label class="brush">
				<span>{ui('brush')}</span>
				<input type="range" min="0.025" max="0.14" step="0.005" bind:value={brush} />
				<span class="dot" style="--d: {Math.round(brush * 100)}px" aria-hidden="true"></span>
			</label>
		{/if}
	</div>
	{#if presets.length}
		<div class="presets" role="group" aria-label={ui('presets')}>
			<span class="muted">{ui('presets')}{colon()}</span>
			{#each presets as p, i (i)}
				<button type="button" class="chip-btn" onclick={() => usePreset(p)}>{t(p.name)}</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.pad {
		position: relative;
		display: grid;
		gap: 8px;
		width: min(var(--size), 100%);
	}
	canvas {
		width: 100%;
		height: auto;
		aspect-ratio: 1;
		border: 1.5px solid var(--rule-strong);
		border-radius: var(--r2);
		background: var(--surface);
		touch-action: none;
		cursor: crosshair;
		box-shadow: inset 0 1px 3px rgb(0 0 0 / 0.06);
	}
	.hint {
		position: absolute;
		top: 0;
		left: 0;
		width: 100%;
		aspect-ratio: 1;
		display: grid;
		place-items: center;
		color: var(--muted);
		font-family: var(--font-display);
		font-size: var(--fs-lg);
		font-style: italic;
		pointer-events: none;
	}
	.tools {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px;
	}
	.brush {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.brush input {
		width: 90px;
	}
	.dot {
		width: max(4px, calc(var(--d) * 0.55));
		height: max(4px, calc(var(--d) * 0.55));
		border-radius: 50%;
		background: var(--ink);
	}
	.presets {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.chip-btn {
		min-height: 32px;
		padding: 0 10px;
		border-radius: var(--r-pill);
		border: 1px solid var(--rule-strong);
		background: var(--surface);
		font-size: var(--fs-sm);
	}
	.chip-btn:hover {
		background: var(--surface-2);
	}
</style>
