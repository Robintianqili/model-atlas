<!-- pinn/residual-strip (props: meta.ts). SVG; nothing animates by itself: the picture follows the frame. -->
<script lang="ts">
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { t } from '$lib/i18n/i18n.svelte';
	import { gradientCss, makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { untrack } from 'svelte';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { C, K, N, SPLIT, T, TRUTH, pointsOf } from '../../osc.ts';
	import { describeFit, fitPic, physOf, type Choice } from './meta.ts';

	let { rt, source, at, props }: StageProps = $props();
	const choices = $derived((props?.choices as Choice[] | undefined) ?? []);
	let picked = $state<string | null>(null);
	const src = $derived(picked ?? source);
	let version = $state(0);
	$effect(() => {
		const tp = rt.sources[src]?.tape as { subscribe?: (fn: () => void) => () => void } | null | undefined;
		if (tp && typeof tp.subscribe === 'function') return tp.subscribe(() => version++);
	});
	const atState = $derived(at?.state);
	const pic = $derived.by(() => {
		const shown = src;
		void version;
		void atState;
		return untrack(() => fitPic({ rt, source, at }, shown));
	});
	const pts = $derived(pointsOf(rt.input));
	const show = (k: string) => props?.[k] !== false;

	let W = $state(560);
	const H = $derived(W < 420 ? 200 : 240);
	const m = { l: 48, r: 10, t: 8, b: 34 };
	const range = $derived.by(() => {
		let lo = -1,
			hi = 1.1;
		for (const v of [...(pic?.u ?? []), ...pts.u]) if (Number.isFinite(v)) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
		return [Math.max(-3, lo - 0.1), Math.min(3, hi + 0.1)];
	});
	const X = (x: number) => m.l + x * (W - m.l - m.r);
	const Y = (y: number) => m.t + (1 - (Math.min(range[1], Math.max(range[0], y)) - range[0]) / (range[1] - range[0])) * (H - m.t - m.b);
	const line = (ys: ArrayLike<number>) => Array.from({ length: N }, (_, i) => `${i ? 'L' : 'M'}${X(T[i]).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join('');
	const theme = $derived(themeState.resolved);
	const scale = $derived(makeScale('diverging', [-1, 1], theme));
	let probe = $state<number | null>(null);
	function move(e: PointerEvent) {
		const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
		probe = Math.max(0, Math.min(N - 1, Math.round((((e.clientX - r.left) / r.width) * W - m.l) / (W - m.l - m.r) * (N - 1))));
	}
	const description = $derived(t(describeFit(pic, props ?? {})));
	// the band between the network's curve and the true curve after t = 0.4 (prop gap): its mean height is the caption's gap
	const band = $derived.by(() => {
		if (!props?.gap || !pic?.u) return '';
		const u = pic.u;
		const ix = Array.from({ length: N }, (_, i) => i).filter((i) => T[i] >= SPLIT);
		return ix.map((i, j) => `${j ? 'L' : 'M'}${X(T[i]).toFixed(1)},${Y(u[i]).toFixed(1)}`).join('') + ix.reverse().map((i) => `L${X(T[i]).toFixed(1)},${Y(TRUTH[i]).toFixed(1)}`).join('') + 'Z';
	});
	/** the run's own law, with its numbers: (u″ + 3.6·u′ + 144·u) / 144 */
	const law = $derived.by(() => {
		const c = String(+(pic?.c ?? C).toPrecision(6));
		const k = String(+K.toPrecision(6));
		return `(u″ + ${c}·u′ + ${k}·u) / ${k}`;
	});
</script>

<figure class="rs" bind:clientWidth={W}>
	{#if choices.length}
		<div class="choices" role="group" aria-label={t({ en: 'Weight of the law', zh: '定律的权重' })}>
			{#each choices as c (c.source)}
				<button type="button" class="btn" aria-pressed={src === c.source} onclick={() => ((picked = c.source), void rt.ensure(c.source, 'last').catch(() => -1))}>{t(c.label)}</button>
			{/each}
		</div>
	{/if}
	<div class="legend small">
		{#if show('truth')}<span><i class="dash"></i>{t({ en: 'true curve (never shown to the networks)', zh: '真实曲线（网络从没见过）' })}</span>{/if}
		{#if show('net') && pic?.u}<span><i class="net"></i>{t(pic.pinn ? { en: 'PINN', zh: 'PINN' } : { en: 'plain network', zh: '普通网络' })}{pic.lam !== undefined && pic.pinn ? ` (λ = ${pic.lam}${pic.c !== undefined && Math.abs(pic.c - C) > 1e-9 ? `, c = ${pic.c}` : ''})` : ''}</span>{/if}
		{#if show('points')}<span><b class="pt"></b>{t({ en: 'your points', zh: '你的数据点' })}</span>{/if}
		{#if band}<span><i class="gapk"></i>{t({ en: `gap to the true curve after t = ${SPLIT}`, zh: `t = ${SPLIT} 之后与真实曲线的差距` })}</span>{/if}
	</div>
	<svg width={W} height={H} viewBox="0 0 {W} {H}" role="img" aria-label={description} onpointermove={move} onpointerleave={() => (probe = null)}>
		{#if show('split')}
			<rect class="win" x={X(SPLIT)} y={m.t} width={Math.max(0, X(1) - X(SPLIT))} height={H - m.t - m.b} />
			<text class="tick" x={X(SPLIT) + 4} y={m.t + 12}>{t({ en: `after t = ${SPLIT}`, zh: `t = ${SPLIT} 之后` })}</text>
		{/if}
		{#each [range[0], 0, range[1]] as v (v)}<text class="tick" x={m.l - 5} y={Y(v) + 4} text-anchor="end">{fmt3(v)}</text>{/each}
		<line class="zero" x1={m.l} x2={W - m.r} y1={Y(0)} y2={Y(0)} />
		{#each [0, 0.2, 0.4, 0.6, 0.8, 1] as x (x)}<text class="tick" x={X(x)} y={H - 19} text-anchor="middle">{x}</text>{/each}
		<text class="axis" x={X(0.5)} y={H - 3} text-anchor="middle">{t({ en: 'time t', zh: '时间 t' })}</text>
		<text class="axis" transform="translate(11 {(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">{t({ en: 'position u', zh: '位置 u' })}</text>
		{#if band}<path class="gapband" d={band} />{/if}
		{#if show('truth')}<path class="truth" d={line(TRUTH)} />{/if}
		{#if show('net') && pic?.u}<path class="net" d={line(pic.u)} />{/if}
		{#each (show('points') ? pts.t : []) as tt, i (i)}<circle class="obs" cx={X(tt)} cy={Y(pts.u[i])} r="4.5" />{/each}
		{#if probe !== null}<line class="probe" x1={X(T[probe])} x2={X(T[probe])} y1={m.t} y2={H - m.b} />{/if}
	</svg>
	{#if show('strip') && pic?.r}
		<div class="strip" style:margin-left="{m.l}px" style:margin-right="{m.r}px" aria-hidden="true">
			{#each pic.r as v, i (i)}<span style:background={scale.css(v)} class:on={probe === i}></span>{/each}
		</div>
		<p class="small cap">
			<span class="grad" style:background={gradientCss('diverging', theme)}></span>
			{t({ en: `residual ${law}: the background color = obeys the law; blue −1, orange +1`, zh: `残差 ${law}：和背景同色 = 遵守定律；蓝色 −1，橙色 +1` })}
		</p>
	{/if}
	<figcaption class="small num">
		{#if probe !== null}
			t = {fmt3(T[probe])} · {t({ en: 'true', zh: '真实' })} {fmt3(TRUTH[probe])}{#if pic?.u} · {t({ en: 'network', zh: '网络' })} {fmt3(pic.u[probe])}{/if}{#if pic?.r && show('strip')} · {t({ en: 'residual', zh: '残差' })} {fmt3(pic.r[probe])}{/if}
		{:else if pic && Number.isFinite(pic.err) && show('net')}
			{@const dl = Number.isFinite(pic.data)}
			{@const ph = show('strip') && pic.r ? physOf(pic.r) : NaN}
			{t({ en: `step ${pic.step}${dl ? ` · data loss ${fmt3(pic.data)}` : ''}${Number.isFinite(ph) ? ` · physics loss ${fmt3(ph)}` : ''} · mean gap to the true curve after t = ${SPLIT}: ${fmt3(pic.err)}`, zh: `第 ${pic.step} 步${dl ? ` · 数据损失 ${fmt3(pic.data)}` : ''}${Number.isFinite(ph) ? ` · 物理损失 ${fmt3(ph)}` : ''} · t = ${SPLIT} 之后与真实曲线的平均差距：${fmt3(pic.err)}` })}
		{/if}
	</figcaption>
</figure>

<style>
	.rs { margin: 0; display: grid; gap: 6px; min-width: 0; grid-template-columns: minmax(0, 1fr); }
	svg { display: block; max-width: 100%; touch-action: pan-y; }
	.win { fill: var(--accent-soft); opacity: 0.35; }
	.tick { fill: var(--muted); font: 11px var(--font-mono); }
	.axis { fill: var(--ink-2); font-size: 11px; }
	.gapband { fill: var(--k); opacity: 0.22; }
	.zero { stroke: var(--rule); }
	.truth { fill: none; stroke: var(--real); stroke-width: 1.8; stroke-dasharray: 5 4; }
	.net { fill: none; stroke: var(--accent); stroke-width: 2.6; }
	.obs { fill: var(--surface); stroke: var(--k); stroke-width: 2.2; }
	.probe { stroke: var(--muted); }
	.strip { display: flex; height: 22px; border: 1px solid var(--rule); border-radius: 3px; overflow: hidden; }
	.strip span { flex: 1 1 0; min-width: 0; }
	.strip span.on { outline: 2px solid var(--ink); }
	.cap { margin: 0; display: flex; gap: 8px; align-items: center; color: var(--ink-2); }
	.grad { flex: 0 0 60px; height: 10px; border-radius: 2px; }
	.legend, .choices, figcaption { display: flex; flex-wrap: wrap; gap: 4px 14px; color: var(--ink-2); }
	.choices { gap: 6px; }
	.legend i { display: inline-block; width: 16px; height: 0; margin-right: 6px; vertical-align: middle; border-top: 2.6px solid var(--accent); }
	.legend i.dash { border-top: 2px dashed var(--real); }
	.legend i.gapk { height: 10px; border: 0; background: var(--k); opacity: 0.35; }
	.legend .pt { display: inline-block; width: 9px; height: 9px; border-radius: 50%; border: 2px solid var(--k); margin-right: 6px; vertical-align: -1px; }
	.btn { min-height: 34px; }
	.small { font-size: var(--fs-sm); }
</style>
