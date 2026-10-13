<!--
  loss-curve: a metric (default the loss) over the recorded steps of a train or process source; the cursor's step is
  marked. Live sources redraw as frames arrive (subscribe, throttled to animation frames). Keyboard: focus the chart,
  ←/→ move the readout, Home/End jump. Divergence (non-finite values) is marked in words, not drawn.
  K2: steps print as counts ("step 50", was "step 50.0"); with several lines "lowest" names its line; props.refs draws
  dashed reference lines ([{ y, label }], e.g. a GAN's equilibrium losses 2 ln 2 and ln 2), kept inside the y range.
  K4: props.xLabel ({ en, zh }) names the x axis when it does not count training steps (episodes, rounds), and
  props.lowest: false leaves out the "lowest … at step" line (a metric where lower is not better).
-->
<script lang="ts">
	import type { StageProps } from '../../../types/view.ts';
	import type { I18n } from '../../../types/core.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtNum } from '../../../core/format.ts';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { downsample, isLive, tapeOf } from '../../train/refs.ts';
	import meta, { lossSeries, summarize } from './meta.ts';

	// props one by one (no rest/spread props: they would grow the Svelte runtime chunk every page preloads)
	let { rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props }: StageProps = $props();
	const p: StageProps = $derived({ rt, source, at, focus, sel, ctx, provenance, autoplay, onend, props });

	// bump on every frame batch of the source and of the compared sources
	let version = $state(0);
	$effect(() => {
		const names = [p.source, ...(((p.props?.compare as string[] | undefined) ?? []) as string[])];
		const offs: (() => void)[] = [];
		for (const n of names) {
			const tp = tapeOf(p.rt, n);
			if (isLive(tp)) offs.push(tp.subscribe(() => version++));
		}
		return () => offs.forEach((f) => f());
	});

	// the compared sources run as far as this one (K0: nothing ran them, so they were empty unless an earlier step had)
	const compareKey = $derived(((p.props?.compare as string[] | undefined) ?? []).join('|'));
	$effect(() => {
		const names = compareKey ? compareKey.split('|') : [];
		const rt = p.rt;
		const own = tapeOf(rt, p.source) as { done?: boolean; length: number } | null;
		const step = p.at?.frame?.step;
		const c = !own || own.done !== false || step === undefined ? 'last' : { step };
		for (const n of names) if (rt.sources[n] && typeof rt.ensure === 'function') void rt.ensure(n, c).catch(() => undefined);
	});

	const series = $derived.by(() => {
		void version;
		return lossSeries(p);
	});
	const sum = $derived(summarize(series[0], p.at.index));
	/** K4: the x axis's name (default "training step") and whether the readout names the lowest value */
	const xLabel = $derived(p.props?.xLabel && typeof (p.props.xLabel as I18n).en === 'string' ? (p.props.xLabel as I18n) : null);
	const showLowest = $derived(p.props?.lowest !== false);
	/** K2: dashed reference lines, props.refs: [{ y, label }] */
	const refs = $derived(
		(Array.isArray(p.props?.refs) ? (p.props.refs as { y: number; label?: I18n }[]) : []).filter((r) => r && typeof r.y === 'number' && Number.isFinite(r.y))
	);

	let W = $state(560);
	const H = $derived(W < 420 ? 200 : 240);

	const all = $derived.by(() => {
		let x0 = Infinity,
			x1 = -Infinity,
			y0 = Infinity,
			y1 = -Infinity;
		for (const s of series)
			for (let i = 0; i < s.y.length; i++) {
				const v = s.y[i];
				if (!Number.isFinite(v)) continue;
				x0 = Math.min(x0, s.x[i]);
				x1 = Math.max(x1, s.x[i]);
				y0 = Math.min(y0, v);
				y1 = Math.max(y1, v);
			}
		if (!Number.isFinite(x0)) return null;
		// K2: reference lines stay inside the drawn range
		for (const r of refs) {
			y0 = Math.min(y0, r.y);
			y1 = Math.max(y1, r.y);
		}
		if (x1 === x0) x1 = x0 + 1;
		// a log scale needs positive values: it starts at the smallest positive one (a loss of exactly 0 sits on the axis);
		// with none, the scale is linear (K0: log: true on a zero loss threw "Invalid array length")
		let pos0 = Infinity;
		for (const s of series) for (let i = 0; i < s.y.length; i++) if (Number.isFinite(s.y[i]) && s.y[i] > 0 && s.y[i] < pos0) pos0 = s.y[i];
		if (y1 === y0) {
			y1 = y0 + 1e-3 * Math.max(1, Math.abs(y0));
			y0 = y0 - 1e-3 * Math.max(1, Math.abs(y0));
		}
		const logOpt = p.props?.log ?? 'auto';
		const log = Number.isFinite(pos0) && pos0 < y1 && (logOpt === true || (logOpt === 'auto' && y0 > 0 && y1 / y0 > 30));
		return { x0, x1, y0: log ? pos0 : Math.min(0, y0), y1, log };
	});

	const xOf = (x: number) => (all ? m.l + ((x - all.x0) / (all.x1 - all.x0)) * (W - m.l - m.r) : 0);
	const yOf = (y: number) => {
		if (!all) return 0;
		const u = all.log ? (y > 0 ? (Math.log10(y) - Math.log10(all.y0)) / (Math.log10(all.y1) - Math.log10(all.y0)) : 0) : (y - all.y0) / (all.y1 - all.y0);
		return m.t + (1 - Math.min(1, Math.max(0, u))) * (H - m.t - m.b);
	};
	const theme = $derived(themeState.resolved);
	const smooth = $derived(typeof p.props?.smooth === 'number' && (p.props.smooth as number) > 1 ? Math.floor(p.props.smooth as number) : 0);
	const toPath = (xs: ArrayLike<number>, ys: ArrayLike<number>) => {
		const d = downsample(xs, ys, Math.max(100, Math.floor(W)));
		let path = '';
		d.x.forEach((x, i) => (path += `${i ? 'L' : 'M'}${xOf(x).toFixed(1)},${yOf(d.y[i]).toFixed(1)}`));
		return path;
	};
	const paths = $derived(
		series.map((s, k) => ({ key: s.key, d: toPath(s.x, s.y), color: k === 0 ? (smooth ? 'var(--muted)' : 'var(--accent)') : categorical(k, theme), label: s.label, thin: k === 0 && smooth > 0 }))
	);
	/** trailing mean over the last `smooth` finite values of the first line (the trend under a noisy batch loss) */
	const smoothPath = $derived.by(() => {
		const s = series[0];
		if (!smooth || !s) return '';
		const xs: number[] = [];
		const ys: number[] = [];
		const win: number[] = [];
		let sum = 0;
		for (let i = 0; i < s.y.length; i++) {
			const v = s.y[i];
			if (!Number.isFinite(v)) continue;
			win.push(v);
			sum += v;
			if (win.length > smooth) sum -= win.shift()!;
			xs.push(s.x[i]);
			ys.push(sum / win.length);
		}
		return toPath(xs, ys);
	});
	const yTicks = $derived.by(() => {
		if (!all) return [];
		if (all.log) {
			const out: number[] = [];
			for (let e = Math.floor(Math.log10(all.y0)); e <= Math.ceil(Math.log10(all.y1)); e++) {
				const v = 10 ** e;
				if (v >= all.y0 * 0.999 && v <= all.y1 * 1.001) out.push(v);
			}
			return out.length >= 2 ? out : [all.y0, all.y1];
		}
		return [all.y0, (all.y0 + all.y1) / 2, all.y1];
	});

	// the left margin fits the widest y label (log scales print 1.00e−4)
	const m = $derived({ l: Math.max(40, 12 + 7 * Math.max(0, ...yTicks.map((v) => fmt3(v).length))), r: 12, t: 10, b: 34 });

	// readout: the cursor's frame, or the one the learner moved to
	let probe = $state<number | null>(null);
	const probeIndex = $derived(probe ?? p.at.index);
	const probeAt = $derived.by(() => {
		const s = series[0];
		if (!s || probeIndex < 0 || probeIndex >= s.y.length) return null;
		return { step: s.x[probeIndex], values: series.map((q) => ({ label: q.label, v: probeIndex < q.y.length ? q.y[probeIndex] : NaN })) };
	});

	function key(e: KeyboardEvent) {
		const n = series[0]?.y.length ?? 0;
		if (!n) return;
		const cur = probe ?? Math.max(0, p.at.index);
		const big = Math.max(1, Math.round(n / 20));
		let next = cur;
		if (e.key === 'ArrowRight') next = Math.min(n - 1, cur + (e.shiftKey ? big : 1));
		else if (e.key === 'ArrowLeft') next = Math.max(0, cur - (e.shiftKey ? big : 1));
		else if (e.key === 'Home') next = 0;
		else if (e.key === 'End') next = n - 1;
		else return;
		e.preventDefault();
		probe = next;
	}
	function move(e: PointerEvent) {
		const s = series[0];
		if (!s || !s.x.length) return;
		const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		const x = ((e.clientX - r.left) / r.width) * W;
		let best = 0;
		let bd = Infinity;
		for (let i = 0; i < s.x.length; i += Math.max(1, Math.floor(s.x.length / 800))) {
			const d = Math.abs(xOf(s.x[i]) - x);
			if (d < bd) {
				bd = d;
				best = i;
			}
		}
		probe = best;
	}

	let ended = false;
	$effect(() => {
		void version;
		const tp = tapeOf(p.rt, p.source);
		if (!ended && p.autoplay && tp && (!isLive(tp) || tp.done)) {
			ended = true;
			p.onend();
		}
	});
	const description = $derived.by(() => {
		void version;
		return t(meta.describe(p));
	});
</script>

<figure class="lc" bind:clientWidth={W} style:min-height="{meta.minHeight}px">
	{#if paths.length > 1}
		<div class="legend small">
			{#each paths as q (q.key)}<span><i style:background={q.color}></i>{t(q.label)}</span>{/each}
		</div>
	{/if}
	{#if all}
		<svg
			width={W}
			height={H}
			viewBox="0 0 {W} {H}"
			role="slider"
			tabindex="0"
			aria-label={description}
			aria-valuemin={0}
			aria-valuemax={Math.max(0, (series[0]?.y.length ?? 1) - 1)}
			aria-valuenow={Math.max(0, probeIndex)}
			aria-valuetext={probeAt ? `${t({ en: 'step', zh: '第' })} ${fmtNum(probeAt.step)}: ${probeAt.values.map((v) => `${t(v.label)} ${fmt3(v.v)}`).join(', ')}` : ''}
			onkeydown={key}
			onpointermove={move}
			onpointerleave={() => (probe = null)}
			onblur={() => (probe = null)}
		>
			{#each yTicks as v, i (i)}
				<line class="grid" x1={m.l} x2={W - m.r} y1={yOf(v)} y2={yOf(v)} />
				<text class="tick" x={m.l - 6} y={yOf(v) + 4} text-anchor="end">{fmt3(v)}</text>
			{/each}
			<text class="tick" x={m.l} y={H - m.b + 16} text-anchor="start">{fmtNum(all.x0)}</text>
			<text class="tick" x={W - m.r} y={H - m.b + 16} text-anchor="end">{fmtNum(all.x1)}</text>
			{#each refs as r, i (i)}
				<line class="ref" x1={m.l} x2={W - m.r} y1={yOf(r.y)} y2={yOf(r.y)} />
				<text class="tick" x={W - m.r - 4} y={yOf(r.y) - 4} text-anchor="end">{r.label ? `${t(r.label)} ` : ''}{fmt3(r.y)}</text>
			{/each}
			<text class="axis" x={m.l + (W - m.l - m.r) / 2} y={H - 4} text-anchor="middle">
				{xLabel ? `${t(xLabel)}${all.log ? t({ en: ' (log scale)', zh: '（对数刻度）' }) : ''}` : t({ en: all.log ? 'training step (loss on a log scale)' : 'training step', zh: all.log ? '训练步数（损失为对数刻度）' : '训练步数' })}
			</text>
			{#each paths as q (q.key)}<path class="line" class:thin={q.thin} d={q.d} style:stroke={q.color} />{/each}
			{#if smoothPath}<path class="line" d={smoothPath} style:stroke="var(--accent)" />{/if}
			{#if sum && Number.isFinite(sum.atStep)}
				<line class="cursor" x1={xOf(sum.atStep)} x2={xOf(sum.atStep)} y1={m.t} y2={H - m.b} />
				{#if Number.isFinite(sum.atValue)}<circle class="dot" cx={xOf(sum.atStep)} cy={yOf(sum.atValue)} r="4.5" />{/if}
			{/if}
			{#if probe !== null && probeAt}
				<line class="probe" x1={xOf(probeAt.step)} x2={xOf(probeAt.step)} y1={m.t} y2={H - m.b} />
			{/if}
		</svg>
	{:else}
		<p class="empty muted">{t({ en: 'No steps recorded yet.', zh: '还没有记录任何步。' })}</p>
	{/if}
	<figcaption class="small">
		{#if probeAt}
			<span class="num">{t({ en: 'step', zh: '第' })} {fmtNum(probeAt.step)}{t({ en: '', zh: ' 步' })}</span>
			{#each probeAt.values as v, i (i)}<span><span class="lbl">{t(v.label)}</span> <span class="num">{Number.isFinite(v.v) ? fmt3(v.v) : '—'}</span></span>{/each}
		{/if}
		{#if smooth}<span class="muted">{t({ en: `thick line: average of the last ${smooth} steps`, zh: `粗线：最近 ${smooth} 步的平均` })}</span>{/if}
		{#if sum && sum.n > 0 && showLowest}
			<span class="muted">{t({ en: 'lowest', zh: '最低' })}{series.length > 1 ? ` ${t(series[0].label)}` : ''} <span class="num">{fmt3(sum.min)}</span> {t({ en: 'at step', zh: '在第' })} <span class="num">{fmtNum(sum.minStep)}</span>{t({ en: '', zh: ' 步' })}</span>
		{/if}
		{#if sum && sum.diverged >= 0}
			<strong class="warn">{t({ en: `Diverged at step ${sum.diverged}: the loss is no longer a finite number.`, zh: `在第 ${sum.diverged} 步发散：损失不再是有限的数。` })}</strong>
		{/if}
	</figcaption>
</figure>

<style>
	.lc {
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
		border-radius: var(--r1);
	}
	svg:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.grid {
		stroke: var(--rule);
		stroke-width: 1;
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
		stroke-width: 2;
		stroke-linejoin: round;
	}
	.line.thin {
		stroke-width: 1;
		opacity: 0.7;
	}
	.cursor {
		stroke: var(--ink-2);
		stroke-dasharray: 3 3;
	}
	.probe {
		stroke: var(--muted);
	}
	.ref {
		stroke: var(--muted);
		stroke-dasharray: 6 4;
	}
	.dot {
		fill: var(--accent);
		stroke: var(--surface);
		stroke-width: 2;
	}
	.legend {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
	}
	.legend i {
		display: inline-block;
		width: 14px;
		height: 3px;
		border-radius: 2px;
		margin-right: 6px;
		vertical-align: middle;
	}
	figcaption {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 14px;
		color: var(--ink-2);
	}
	.lbl {
		color: var(--muted);
	}
	.warn {
		color: var(--k-text);
	}
	.empty {
		padding: var(--s4) 0;
	}
</style>
