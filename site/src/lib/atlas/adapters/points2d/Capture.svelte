<!--
  Capture for the 'points2d' adapter: paint points of a class on a plane, erase them, or pick a preset.
  Props (shared Capture contract; W1-B2): raw (bindable PointsRaw), presets?, options? { classes (1 = unlabelled,
  default 2), presets?: string[] }, onchange?(raw), label?.
  Mouse/touch/pen: press or drag to paint (or erase). Keyboard: arrows move the cross (Shift: bigger steps), Enter or
  Space adds a point there (erases in erase mode), 1 / 2 pick the class, E toggles erase. The raw value only changes when
  a stroke ends, so a running trainer gets one new input per stroke.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { Text } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { I18n } from '../../types/core.ts';
	import { classCounts, MAX_POINTS, pointsOf, POINT_PRESETS, presetRaw, type PaintedPoint, type PointsRaw } from './adapter.ts';

	interface Props {
		raw?: PointsRaw;
		presets?: { id: string; label: I18n; raw: unknown }[];
		options?: Record<string, unknown>;
		onchange?: (raw: PointsRaw) => void;
		label?: Text;
	}
	let { raw = $bindable({ points: [] }), presets, options = {}, onchange, label }: Props = $props();

	const classes = $derived(Math.max(1, Math.floor(Number(options.classes ?? 2)) || 2));
	// an empty list (a page without a presets file passes []) falls back to the generated presets. K1: an unlabelled page
	// (classes 1) may use the labelled shapes too, drawn in one color (two moons for flow matching), after its own
	const list = $derived(
		presets?.length
			? presets
			: [...POINT_PRESETS.filter((p) => !p.labelled), ...POINT_PRESETS.filter((p) => p.labelled)]
					.filter((p) => (classes > 1 ? p.labelled : true))
					.filter((p) => !Array.isArray(options.presets) || (options.presets as string[]).includes(p.id))
					.map((p) => ({ id: p.id, label: p.label, raw: presetRaw(p.id) }))
	);
	/** the chip of the learner's input: the same preset (name, size, seed, noise) and nothing painted or erased (K1) */
	const same = (a: PointsRaw | undefined, b: PointsRaw) =>
		!!a?.preset && !!b.preset && a.preset.name === b.preset.name && (a.preset.n ?? null) === (b.preset.n ?? null) && (a.preset.seed ?? 1) === (b.preset.seed ?? 1) && (a.preset.noise ?? null) === (b.preset.noise ?? null) && !(a.points?.length || a.removed?.length);
	const B = 1.25; // the plane is [−B, B]²

	let cls = $state(0);
	let erase = $state(false);
	let draft = $state.raw<PointsRaw | null>(null);
	const shown = $derived(draft ?? raw);
	const pts = $derived(pointsOf(shown));
	const counts = $derived(classCounts(pts));
	const theme = $derived(themeState.resolved);

	let W = $state(300);
	const side = $derived(Math.max(200, Math.min(W, 320)));
	let canvas = $state<HTMLCanvasElement | null>(null);
	let cross = $state<{ x: number; y: number } | null>(null);
	let colors: { theme: string; surface: string; rule: string; ink: string; real: string } | null = null;

	const sx = (x: number) => ((x + B) / (2 * B)) * side;
	const sy = (y: number) => ((B - y) / (2 * B)) * side;
	const ux = (px: number) => (px / side) * 2 * B - B;
	const uy = (py: number) => B - (py / side) * 2 * B;

	function draw() {
		const c = canvas;
		if (!c) return;
		const dpr = typeof devicePixelRatio === 'number' ? Math.min(3, devicePixelRatio) : 1;
		c.width = Math.round(side * dpr);
		c.height = Math.round(side * dpr);
		const g = c.getContext('2d');
		if (!g) return;
		g.setTransform(dpr, 0, 0, dpr, 0, 0);
		if (!colors || colors.theme !== theme) {
			const css = getComputedStyle(c);
			colors = { theme, surface: css.getPropertyValue('--surface').trim() || '#fff', rule: css.getPropertyValue('--rule').trim() || '#ccc', ink: css.getPropertyValue('--ink').trim() || '#222', real: css.getPropertyValue('--real').trim() || '#1a8a4a' };
		}
		g.fillStyle = colors.surface;
		g.fillRect(0, 0, side, side);
		g.strokeStyle = colors.rule;
		g.beginPath();
		g.moveTo(sx(0), 0);
		g.lineTo(sx(0), side);
		g.moveTo(0, sy(0));
		g.lineTo(side, sy(0));
		g.stroke();
		const r = pts.n > 600 ? 2 : pts.n > 200 ? 3 : 4;
		for (let i = 0; i < pts.n; i++) {
			g.beginPath();
			g.arc(sx(pts.xy[2 * i]), sy(pts.xy[2 * i + 1]), r, 0, Math.PI * 2);
			g.fillStyle = classes > 1 ? categorical(pts.label[i], theme) : colors.real;
			g.fill();
		}
		if (cross) {
			g.strokeStyle = erase ? colors.ink : classes > 1 ? categorical(cls, theme) : colors.real;
			g.lineWidth = 2;
			g.beginPath();
			g.arc(sx(cross.x), sy(cross.y), erase ? 0.08 * (side / (2 * B)) : 7, 0, Math.PI * 2);
			g.stroke();
		}
	}
	$effect(() => {
		void pts;
		void side;
		void theme;
		void cross;
		void erase;
		draw();
	});

	function commit(next: PointsRaw) {
		draft = null;
		raw = next;
		onchange?.(next);
	}

	/** add one point (painted) or erase the points near (x, y) in a working copy */
	function apply(work: PointsRaw, x: number, y: number): PointsRaw {
		if (erase) {
			const R2 = 0.08 * 0.08;
			const keep = (work.points ?? []).filter((q) => (q.x - x) ** 2 + (q.y - y) ** 2 > R2);
			let removed = work.removed ?? [];
			if (work.preset) {
				const base = pointsOf({ preset: work.preset });
				const gone = new Set(removed);
				for (let i = 0; i < base.n; i++) if (!gone.has(i) && (base.xy[2 * i] - x) ** 2 + (base.xy[2 * i + 1] - y) ** 2 <= R2) gone.add(i);
				removed = [...gone].sort((a, b) => a - b);
			}
			return { ...work, points: keep, removed };
		}
		if (pointsOf(work).n >= MAX_POINTS) return work;
		const p: PaintedPoint = classes > 1 ? { x, y, c: cls } : { x, y };
		return { ...work, points: [...(work.points ?? []), p] };
	}

	let drawing = false;
	let last: { x: number; y: number } | null = null;
	function local(e: PointerEvent) {
		const r = (e.currentTarget as HTMLCanvasElement).getBoundingClientRect();
		return { x: Math.max(-B, Math.min(B, ux(((e.clientX - r.left) / r.width) * side))), y: Math.max(-B, Math.min(B, uy(((e.clientY - r.top) / r.height) * side))) };
	}
	function down(e: PointerEvent) {
		if (!e.isPrimary) return;
		(e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
		drawing = true;
		const q = local(e);
		last = q;
		draft = apply(raw, q.x, q.y);
	}
	function moveP(e: PointerEvent) {
		const q = local(e);
		cross = q;
		if (!drawing || !draft || !last) return;
		if (Math.hypot(q.x - last.x, q.y - last.y) < (erase ? 0.03 : 0.06)) return;
		last = q;
		draft = apply(draft, q.x, q.y);
	}
	function up() {
		if (!drawing) return;
		drawing = false;
		if (draft) commit(draft);
	}
	function key(e: KeyboardEvent) {
		const c = cross ?? { x: 0, y: 0 };
		const s = e.shiftKey ? 0.2 : 0.05;
		if (e.key === 'ArrowLeft') cross = { x: Math.max(-B, c.x - s), y: c.y };
		else if (e.key === 'ArrowRight') cross = { x: Math.min(B, c.x + s), y: c.y };
		else if (e.key === 'ArrowUp') cross = { x: c.x, y: Math.min(B, c.y + s) };
		else if (e.key === 'ArrowDown') cross = { x: c.x, y: Math.max(-B, c.y - s) };
		else if (e.key === 'Enter' || e.key === ' ') commit(apply(raw, c.x, c.y));
		else if (e.key === 'Delete' || e.key === 'Backspace') {
			const was = erase;
			erase = true;
			commit(apply(raw, c.x, c.y));
			erase = was;
		} else if (e.key === 'e' || e.key === 'E') erase = !erase;
		else if (/^[1-9]$/.test(e.key) && Number(e.key) <= classes) cls = Number(e.key) - 1;
		else return;
		e.preventDefault();
	}
	const summary = $derived(
		classes > 1
			? t({ en: `${pts.n} points: ${counts.map((n, k) => `${n} in class ${k}`).join(', ')}.`, zh: `${pts.n} 个点：${counts.map((n, k) => `第 ${k} 类 ${n} 个`).join('，')}。` })
			: t({ en: `${pts.n} points.`, zh: `${pts.n} 个点。` })
	);
</script>

<div class="cap" bind:clientWidth={W}>
	{#if list.length}
		<div class="chips" role="group" aria-label={t({ en: 'Point sets', zh: '点集' })}>
			{#each list as p (p.id)}
				<button type="button" class="chip" aria-pressed={same(raw, p.raw as PointsRaw)} onclick={() => commit(p.raw as PointsRaw)}>{t(p.label)}</button>
			{/each}
		</div>
	{/if}
	<canvas
		bind:this={canvas}
		style:width="{side}px"
		style:height="{side}px"
		tabindex="0"
		aria-label="{label ? t(label) + '. ' : ''}{t({ en: 'Point editor: press to add a point, drag to paint; arrow keys move the cross and Enter adds a point.', zh: '点编辑器：按下添加一个点，拖动连续画点；方向键移动十字，回车添加一个点。' })} {summary}"
		onpointerdown={down}
		onpointermove={moveP}
		onpointerup={up}
		onpointercancel={up}
		onpointerleave={() => (cross = drawing ? cross : null)}
		onkeydown={key}
		onblur={() => (cross = null)}
	></canvas>
	<div class="tools">
		{#if classes > 1}
			<div class="segmented" role="radiogroup" aria-label={t({ en: 'Class to paint', zh: '要画的类别' })}>
				{#each Array.from({ length: classes }, (_, k) => k) as k (k)}
					<button type="button" role="radio" aria-checked={cls === k && !erase} onclick={() => ((cls = k), (erase = false))}>
						<i style:background={categorical(k, theme)}></i>{t({ en: `class ${k}`, zh: `第 ${k} 类` })}
					</button>
				{/each}
			</div>
		{/if}
		<button type="button" class="btn ghost" aria-pressed={erase} onclick={() => (erase = !erase)}>{t({ en: 'Erase', zh: '橡皮擦' })}</button>
		<button type="button" class="btn ghost" onclick={() => commit({ points: [] })}>{t({ en: 'Clear', zh: '清空' })}</button>
	</div>
	<p class="small muted" aria-live="polite">{summary}</p>
</div>

<style>
	.cap {
		display: grid;
		gap: var(--s2);
		justify-items: start;
		min-width: 0;
	}
	.chips,
	.tools {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	canvas {
		display: block;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r1);
		touch-action: none;
		cursor: crosshair;
		max-width: 100%;
	}
	canvas:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.segmented button {
		display: inline-flex;
		gap: 6px;
		align-items: center;
	}
	.segmented i {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 50%;
	}
	p {
		margin: 0;
	}
	button.chip {
		background: var(--surface);
		color: var(--ink-2);
		cursor: pointer;
		min-height: 30px;
		font-size: var(--fs-sm);
	}
	button.chip[aria-pressed='true'] {
		border-color: var(--accent);
		background: var(--accent-soft);
		color: var(--ink);
	}
</style>
