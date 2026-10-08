<!--
  Capture for the 'series' adapter: draw a curve over time, drag observation points, or pick a preset.
  Props (shared Capture contract; W1-B2): raw (bindable SeriesRaw), presets?, options? { mode?: 'curve' | 'points'
  (default: 'points' when the series has observation points), yRange?: [lo, hi] }, onchange?(raw), label?.
  Curve mode: press and drag across the plot to redraw that stretch. Points mode: drag a point; press empty space to add
  one (up to 12). Keyboard: ←/→ pick a value (or a point), ↑/↓ change it (Shift: bigger steps), Delete removes a point.
  The raw value changes when a stroke ends.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { Text } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '../../core/format.ts';
	import type { I18n } from '../../types/core.ts';
	import { SERIES_PRESETS, sameSeries, seriesOf, seriesPresetRaw, type SeriesRaw } from './adapter.ts';

	interface Props {
		raw?: SeriesRaw;
		presets?: { id: string; label: I18n; raw: unknown }[];
		options?: Record<string, unknown>;
		onchange?: (raw: SeriesRaw) => void;
		label?: Text;
	}
	let { raw = $bindable({ y: [] }), presets, options = {}, onchange, label }: Props = $props();

	// an empty list (a page without a presets file passes []) falls back to the generated presets
	// K1: options.presets (an id list) picks among them, as the points2d Capture does
	const list = $derived(
		presets?.length
			? presets
			: SERIES_PRESETS.filter((p) => p.id !== 'flat')
					.filter((p) => !Array.isArray(options.presets) || (options.presets as string[]).includes(p.id))
					.map((p) => ({ id: p.id, label: p.label, raw: seriesPresetRaw(p.id) }))
	);
	let draft = $state.raw<SeriesRaw | null>(null);
	const s = $derived(seriesOf(draft ?? raw));
	const mode = $derived(options.mode === 'curve' || options.mode === 'points' ? (options.mode as 'curve' | 'points') : s.points.length ? 'points' : 'curve');
	const yr = $derived.by((): [number, number] => {
		const o = options.yRange as number[] | undefined;
		if (Array.isArray(o) && o.length === 2) return [o[0], o[1]];
		let lo = -1,
			hi = 1;
		for (const v of s.y) {
			lo = Math.min(lo, v);
			hi = Math.max(hi, v);
		}
		for (const p of s.points) {
			lo = Math.min(lo, p.y);
			hi = Math.max(hi, p.y);
		}
		const pad = 0.1 * (hi - lo);
		return [lo - pad, hi + pad];
	});
	const xr = $derived.by((): [number, number] => {
		const n = s.t.length;
		const a = n ? s.t[0] : 0;
		const b = n ? s.t[n - 1] : 1;
		return b > a ? [a, b] : [a, a + 1];
	});

	let W = $state(480);
	const H = 180;
	const m = { l: 8, r: 8, t: 8, b: 8 };
	const X = (x: number) => m.l + ((x - xr[0]) / (xr[1] - xr[0])) * (W - m.l - m.r);
	const Y = (y: number) => m.t + (1 - (y - yr[0]) / (yr[1] - yr[0])) * (H - m.t - m.b);
	const invX = (px: number) => xr[0] + ((px - m.l) / (W - m.l - m.r)) * (xr[1] - xr[0]);
	const invY = (py: number) => yr[0] + (1 - (py - m.t) / (H - m.t - m.b)) * (yr[1] - yr[0]);
	const path = $derived(Array.from(s.y, (v, i) => `${i ? 'L' : 'M'}${X(s.t[i]).toFixed(1)},${Y(v).toFixed(1)}`).join(''));

	let sel = $state(0);

	function commit(next: SeriesRaw) {
		draft = null;
		raw = next;
		onchange?.(next);
	}
	const base = (): SeriesRaw => ({ ...(raw ?? {}), y: Array.from(seriesOf(raw).y), t: Array.from(seriesOf(raw).t) });

	// drawing
	let drawing = false;
	let lastI = -1;
	let grab = -1;
	function pos(e: PointerEvent) {
		const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		return { x: ((e.clientX - r.left) / r.width) * W, y: ((e.clientY - r.top) / r.height) * H };
	}
	function nearestIndex(x: number): number {
		let best = 0,
			bd = Infinity;
		for (let i = 0; i < s.t.length; i++) {
			const d = Math.abs(s.t[i] - x);
			if (d < bd) {
				bd = d;
				best = i;
			}
		}
		return best;
	}
	function down(e: PointerEvent) {
		if (!e.isPrimary) return;
		(e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
		const q = pos(e);
		if (mode === 'points') {
			let best = -1,
				bd = 18 * 18;
			s.points.forEach((p, i) => {
				const d = (X(p.t) - q.x) ** 2 + (Y(p.y) - q.y) ** 2;
				if (d < bd) {
					bd = d;
					best = i;
				}
			});
			if (best < 0 && s.points.length < 12) {
				const pts = [...s.points, { t: invX(q.x), y: invY(q.y) }];
				draft = { ...(raw ?? {}), points: pts };
				best = [...pts].sort((a, b) => a.t - b.t).findIndex((p) => p.t === invX(q.x));
			}
			grab = best;
			sel = Math.max(0, best);
			drawing = best >= 0;
			return;
		}
		if (!s.t.length) return;
		drawing = true;
		const w = base();
		const i = nearestIndex(invX(q.x));
		w.y![i] = invY(q.y);
		lastI = i;
		sel = i;
		draft = w;
	}
	function move(e: PointerEvent) {
		if (!drawing) return;
		const q = pos(e);
		if (mode === 'points') {
			if (grab < 0) return;
			const pts = seriesOf(draft ?? raw).points.slice();
			pts[grab] = { t: Math.max(xr[0], Math.min(xr[1], invX(q.x))), y: invY(q.y) };
			draft = { ...(raw ?? {}), points: pts };
			return;
		}
		const w = { ...(draft ?? base()) };
		const y = (w.y ?? []).slice();
		const i = nearestIndex(invX(q.x));
		const v = invY(q.y);
		const a = lastI < 0 ? i : lastI;
		const from = Math.min(a, i),
			to = Math.max(a, i);
		const va = y[a] ?? v;
		for (let k = from; k <= to; k++) y[k] = to === from ? v : a <= i ? va + ((v - va) * (k - a)) / (i - a) : va + ((v - va) * (a - k)) / (a - i);
		lastI = i;
		sel = i;
		draft = { ...w, y };
	}
	function up() {
		if (!drawing) return;
		drawing = false;
		grab = -1;
		lastI = -1;
		if (draft) commit(draft);
	}
	function key(e: KeyboardEvent) {
		const n = mode === 'points' ? s.points.length : s.y.length;
		const step = (yr[1] - yr[0]) * (e.shiftKey ? 0.1 : 0.02);
		if (e.key === 'ArrowRight' && n) sel = Math.min(n - 1, sel + (e.shiftKey && mode === 'curve' ? 5 : 1));
		else if (e.key === 'ArrowLeft' && n) sel = Math.max(0, sel - (e.shiftKey && mode === 'curve' ? 5 : 1));
		else if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && n) {
			const d = e.key === 'ArrowUp' ? step : -step;
			if (mode === 'points') {
				const pts = s.points.slice();
				pts[sel] = { ...pts[sel], y: pts[sel].y + d };
				commit({ ...(raw ?? {}), points: pts });
			} else {
				const w = base();
				w.y![sel] += d;
				commit(w);
			}
		} else if ((e.key === 'Delete' || e.key === 'Backspace') && mode === 'points' && n) {
			commit({ ...(raw ?? {}), points: s.points.filter((_, i) => i !== sel) });
			sel = Math.max(0, sel - 1);
		} else return;
		e.preventDefault();
	}
	const readout = $derived.by(() => {
		if (mode === 'points') {
			const p = s.points[sel];
			return p ? t({ en: `Point ${sel + 1} of ${s.points.length}: t = ${fmt3(p.t)}, value ${fmt3(p.y)}`, zh: `第 ${sel + 1} / ${s.points.length} 个点：t = ${fmt3(p.t)}，值 ${fmt3(p.y)}` }) : t({ en: 'No points yet: press the plot to add one.', zh: '还没有点：在图上按一下就能添加。' });
		}
		if (!s.y.length) return t({ en: 'No values yet: pick a preset.', zh: '还没有值：选一个预设。' });
		const i = Math.min(sel, s.y.length - 1);
		return t({ en: `${s.y.length} values. Value ${i + 1}: t = ${fmt3(s.t[i])}, ${fmt3(s.y[i])}`, zh: `${s.y.length} 个值。第 ${i + 1} 个：t = ${fmt3(s.t[i])}，${fmt3(s.y[i])}` });
	});
</script>

<div class="cap" bind:clientWidth={W}>
	{#if list.length}
		<div class="chips" role="group" aria-label={t({ en: 'Series', zh: '序列' })}>
			{#each list as p (p.id)}
				<button type="button" class="chip" aria-pressed={sameSeries(raw, p.raw)} onclick={() => commit(p.raw as SeriesRaw)}>{t(p.label)}</button>
			{/each}
		</div>
	{/if}
	<svg
		width={W}
		height={H}
		viewBox="0 0 {W} {H}"
		tabindex="0"
		role="slider"
		aria-label={label ? t(label) : mode === 'points' ? t({ en: 'Observation points: drag a point, or press empty space to add one.', zh: '观测点：拖动一个点，或在空白处按下以添加。' }) : t({ en: 'Series: press and drag to redraw the curve.', zh: '序列：按下并拖动来重画曲线。' })}
		aria-valuemin={0}
		aria-valuemax={Math.max(0, (mode === 'points' ? s.points.length : s.y.length) - 1)}
		aria-valuenow={sel}
		aria-valuetext={readout}
		onpointerdown={down}
		onpointermove={move}
		onpointerup={up}
		onpointercancel={up}
		onkeydown={key}
	>
		<rect class="bg" x="0" y="0" width={W} height={H} rx="4" />
		{#if yr[0] < 0 && yr[1] > 0}<line class="zero" x1={m.l} x2={W - m.r} y1={Y(0)} y2={Y(0)} />{/if}
		<path class="line" d={path} />
		{#if mode === 'curve' && s.y.length}
			<circle class="sel" cx={X(s.t[Math.min(sel, s.y.length - 1)])} cy={Y(s.y[Math.min(sel, s.y.length - 1)])} r="4" />
		{/if}
		{#each s.points as p, i (i)}
			<circle class="pt" class:on={mode === 'points' && i === sel} cx={X(p.t)} cy={Y(p.y)} r={mode === 'points' && i === sel ? 7 : 5.5} />
		{/each}
	</svg>
	<p class="small muted" aria-live="polite">{readout}</p>
</div>

<style>
	.cap {
		display: grid;
		gap: var(--s2);
		min-width: 0;
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	svg {
		display: block;
		max-width: 100%;
		touch-action: none;
		cursor: crosshair;
		border-radius: var(--r1);
	}
	svg:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.bg {
		fill: var(--surface);
		stroke: var(--rule-strong);
	}
	.zero {
		stroke: var(--rule);
	}
	.line {
		fill: none;
		stroke: var(--ink-2);
		stroke-width: 2;
	}
	.sel {
		fill: var(--accent);
	}
	.pt {
		fill: var(--surface);
		stroke: var(--k);
		stroke-width: 2.2;
	}
	.pt.on {
		fill: var(--k);
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
