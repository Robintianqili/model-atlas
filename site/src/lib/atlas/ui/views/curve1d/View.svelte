<!--
  curve1d: curves over time (data, the network's curve, a true curve), observation points and shaded windows. SVG.
  Keyboard: focus the chart, ←/→ move the readout along the first curve (Shift: 10 at a time), Home/End.
-->
<script lang="ts">
	import type { StageProps } from '../../../types/view.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtNum } from '../../../core/format.ts';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { downsample, isLive, tapeOf } from '../../train/refs.ts';
	import meta, { curveData, interp } from './meta.ts';

	// props one by one (no rest/spread props: they would grow the Svelte runtime chunk every page preloads)
	let { rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props }: StageProps = $props();
	const p: StageProps = $derived({ rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props });

	let version = $state(0);
	$effect(() => {
		const tp = tapeOf(p.rt, p.source);
		if (isLive(tp)) return tp.subscribe(() => version++);
	});
	const d = $derived.by(() => {
		void version;
		return curveData(p);
	});
	const theme = $derived(themeState.resolved);

	let W = $state(560);
	const H = $derived(W < 420 ? 210 : 250);
	// K1: the left margin fits the longest y tick ("−0.00716" was clipped to ".00716") and the y label
	const yLabel = $derived(p.props?.yLabel ? t(p.props.yLabel as never) : '');
	const ml = $derived(Math.max(46, 12 + 7 * Math.max(...[d.y0, (d.y0 + d.y1) / 2, d.y1].map((v) => fmt3(v).length))) + (yLabel ? 16 : 0));
	const m = $derived({ l: ml, r: 12, t: 10, b: 34 });
	const xOf = (x: number) => m.l + ((x - d.x0) / (d.x1 - d.x0)) * (W - m.l - m.r);
	const yOf = (y: number) => m.t + (1 - (Math.min(d.y1, Math.max(d.y0, y)) - d.y0) / (d.y1 - d.y0)) * (H - m.t - m.b);

	const colorOf = (kind: string, k: number) =>
		kind === 'data' ? 'var(--ink-2)' : kind === 'model' ? 'var(--accent)' : kind === 'truth' ? 'var(--real)' : kind === 'error' ? 'var(--k)' : categorical(k, theme);
	const paths = $derived(
		d.lines.map((l, k) => {
			const s = downsample(l.x, l.y, Math.max(120, Math.floor(W * 1.5)));
			let path = '';
			let pen = false;
			s.x.forEach((x, i) => {
				path += `${pen ? 'L' : 'M'}${xOf(x).toFixed(1)},${yOf(s.y[i]).toFixed(1)}`;
				pen = true;
			});
			return { k, d: path, color: colorOf(l.kind, k), dash: l.kind === 'truth' ? '5 4' : '', label: l.label, width: l.kind === 'model' ? 2.4 : 1.6 };
		})
	);

	let probe = $state<number | null>(null);
	const pi = $derived(probe ?? d.highlight);
	const probeInfo = $derived.by(() => {
		const l = d.lines[0];
		if (pi === null || !l || pi >= l.x.length) return null;
		const x = l.x[pi];
		return { x, values: d.lines.map((q, k) => ({ label: q.label, v: k === 0 ? l.y[pi] : interp(q, x), color: paths[k]?.color })) };
	});
	function key(e: KeyboardEvent) {
		const n = d.lines[0]?.x.length ?? 0;
		if (!n) return;
		const cur = pi ?? 0;
		let next = cur;
		if (e.key === 'ArrowRight') next = Math.min(n - 1, cur + (e.shiftKey ? 10 : 1));
		else if (e.key === 'ArrowLeft') next = Math.max(0, cur - (e.shiftKey ? 10 : 1));
		else if (e.key === 'Home') next = 0;
		else if (e.key === 'End') next = n - 1;
		else if (e.key === 'Escape') {
			probe = null;
			return;
		} else return;
		e.preventDefault();
		probe = next;
	}
	function move(e: PointerEvent) {
		const l = d.lines[0];
		if (!l) return;
		const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		const x = d.x0 + (((e.clientX - r.left) / r.width) * W - m.l) / (W - m.l - m.r) * (d.x1 - d.x0);
		let best = 0,
			bd = Infinity;
		for (let i = 0; i < l.x.length; i++) {
			const dd = Math.abs(l.x[i] - x);
			if (dd < bd) {
				bd = dd;
				best = i;
			}
		}
		probe = best;
	}
	const description = $derived.by(() => {
		void version;
		return t(meta.describe(p));
	});
</script>

<figure class="cv" bind:clientWidth={W} style:min-height="{meta.minHeight}px">
	{#if d.lines.length > 1 || d.points.length || (p.props?.legend === true && d.lines.length)}
		<div class="legend small">
			{#each paths as q (q.k)}<span><i style:background={q.color}></i>{t(q.label)}</span>{/each}
			{#if d.points.length}<span><b class="pt"></b>{t({ en: 'observations', zh: '观测点' })}</span>{/if}
		</div>
	{/if}
	<svg
		width={W}
		height={H}
		viewBox="0 0 {W} {H}"
		role="slider"
		tabindex="0"
		aria-label={description}
		aria-valuemin={0}
		aria-valuemax={Math.max(0, (d.lines[0]?.x.length ?? 1) - 1)}
		aria-valuenow={pi ?? 0}
		aria-valuetext={probeInfo ? `t ${fmtNum(probeInfo.x)}: ${probeInfo.values.map((v) => `${t(v.label)} ${fmt3(v.v)}`).join(', ')}` : ''} onkeydown={key} onpointermove={move} onpointerleave={() => (probe = null)} onblur={() => (probe = null)}>
		{#each d.windows as w, i (i)}
			<rect class="win" x={xOf(w.from)} y={m.t} width={Math.max(1, xOf(w.to) - xOf(w.from))} height={H - m.t - m.b} />
			{#if w.label}<text class="wlbl" x={xOf(w.from) + 4} y={m.t + 12}>{t(w.label)}</text>{/if}
		{/each}
		{#each [d.y0, (d.y0 + d.y1) / 2, d.y1] as v, i (i)}
			<line class="grid" x1={m.l} x2={W - m.r} y1={yOf(v)} y2={yOf(v)} />
			<text class="tick" x={m.l - 6} y={yOf(v) + 4} text-anchor="end">{fmt3(v)}</text>
		{/each}
		{#if d.y0 < 0 && d.y1 > 0}<line class="zero" x1={m.l} x2={W - m.r} y1={yOf(0)} y2={yOf(0)} />{/if}
		<!-- K2: an integer x (training rounds) prints as itself: "40", was "40.0" -->
		<text class="tick" x={m.l} y={H - m.b + 16} text-anchor="start">{fmtNum(d.x0)}</text>
		<text class="tick" x={W - m.r} y={H - m.b + 16} text-anchor="end">{fmtNum(d.x1)}</text>
		<text class="axis" x={m.l + (W - m.l - m.r) / 2} y={H - 4} text-anchor="middle">{p.props?.xLabel ? t(p.props.xLabel as never) : t({ en: 'time', zh: '时间' })}</text>
		{#if yLabel}<text class="axis" x={12} y={m.t + (H - m.t - m.b) / 2} text-anchor="middle" transform="rotate(-90 12 {m.t + (H - m.t - m.b) / 2})">{yLabel}</text>{/if}
		{#each paths as q (q.k)}<path class="line" d={q.d} style:stroke={q.color} stroke-width={q.width} stroke-dasharray={q.dash} />{/each}
		{#each d.points as q, i (i)}<circle class="obs" cx={xOf(q.x)} cy={yOf(q.y)} r="4.5" />{/each}
		{#if probeInfo}
			<line class="probe" x1={xOf(probeInfo.x)} x2={xOf(probeInfo.x)} y1={m.t} y2={H - m.b} />
			{#each probeInfo.values as v, i (i)}{#if Number.isFinite(v.v)}<circle class="pdot" cx={xOf(probeInfo.x)} cy={yOf(v.v)} r="4" style:fill={v.color} />{/if}{/each}
		{/if}
	</svg>
	<figcaption class="small">
		{#if probeInfo}
			<span>t = <span class="num">{fmtNum(probeInfo.x)}</span></span>
			{#each probeInfo.values as v, i (i)}<span><span class="muted">{t(v.label)}</span> <span class="num">{fmt3(v.v)}</span></span>{/each}
		{:else if !d.lines.length}
			<span class="muted">{t({ en: 'No curve yet.', zh: '还没有曲线。' })}</span>
		{/if}
	</figcaption>
</figure>

<style>
	.cv {
		margin: 0;
		display: grid;
		gap: 6px;
		min-width: 0;
		align-content: start;
	}
	svg {
		display: block;
		max-width: 100%;
		touch-action: pan-y;
	}
	svg:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.grid {
		stroke: var(--rule);
	}
	.zero {
		stroke: var(--rule-strong);
	}
	.tick {
		fill: var(--muted);
		font: 11px var(--font-mono);
	}
	.axis {
		fill: var(--ink-2);
		font-size: 12px;
	}
	.line {
		fill: none;
		stroke-linejoin: round;
	}
	.win {
		fill: var(--accent-soft);
		opacity: 0.55;
	}
	.wlbl {
		fill: var(--ink-2);
		font-size: 11px;
	}
	.obs {
		fill: var(--surface);
		stroke: var(--k);
		stroke-width: 2;
	}
	.probe {
		stroke: var(--muted);
	}
	.pdot {
		stroke: var(--surface);
		stroke-width: 1.5;
	}
	.legend,
	figcaption {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		color: var(--ink-2);
	}
	.legend i {
		display: inline-block;
		width: 14px;
		height: 3px;
		border-radius: 2px;
		margin-right: 6px;
		vertical-align: middle;
	}
	.legend .pt {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 50%;
		border: 2px solid var(--k);
		margin-right: 6px;
		vertical-align: -1px;
	}
</style>
