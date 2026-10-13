<!-- fno: one chart over an index (a point of the ring, or a mode). lines: curves over the 64 points ('sol' green: the
     solver, 'pred' magenta: the network, 'c0'..'c3' the channels' categorical colors, 'faint' a ghost); bars: one number
     per index from 0 (bars at or above `cut` are drawn dashed: dropped); dots: a second series on the bars; mark: a
     dashed vertical line at the left edge of an index; hot: one bar drawn dark; x0: the label of index 0. Axes are labelled; y runs over [lo, hi] (default: the data and 0); log: a log10 y axis (ticks at powers of 10), bars rise from its bottom. -->
<script lang="ts">
	import { t, type Text } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '../../../atlas/core/format.ts';

	type Line = { y: ArrayLike<number>; cls: string; name: Text };
	interface Props { lines?: Line[]; bars?: { v: number[]; name: Text; cut?: number }; dots?: { v: number[]; name: Text }; mark?: number | null; hot?: number; x0?: number; log?: boolean; xLabel: Text; yLabel: Text; lo?: number; hi?: number; h?: number; label: Text }
	let { lines = [], bars, dots, mark = null, hot = -1, x0 = 0, log = false, xLabel, yLabel, lo, hi, h = 210, label }: Props = $props();
	let W = $state(480);
	let el = $state<HTMLDivElement>();
	$effect(() => {
		if (!el) return;
		const ro = new ResizeObserver((e) => { const w = Math.max(200, Math.round(e[0].contentRect.width)); if (w !== W) W = w; }); // prettier-ignore: at least 200 px wide, so no size is negative while the stage lays out
		ro.observe(el);
		return () => ro.disconnect();
	});
	const n = $derived(bars ? bars.v.length : (lines[0]?.y.length ?? 64));
	const tf = (v: number) => (log ? Math.log10(Math.max(v, 1e-9)) : v);
	const all = $derived([...lines.flatMap((l) => Array.from(l.y)), ...(bars?.v ?? []), ...(dots?.v ?? []), ...(log ? [] : [0])].filter(Number.isFinite).map(tf));
	const y0 = $derived(lo !== undefined ? tf(lo) : Math.min(...all));
	const y1 = $derived(Math.max(hi !== undefined ? tf(hi) : Math.max(...all), y0 + 1e-6));
	const ML = 54, MR = 8, MT = 8, MB = 34;
	const X = (i: number) => ML + ((bars ? i + 0.5 : i) / (bars ? n : n - 1)) * (W - ML - MR);
	const Yt = (v: number) => MT + ((y1 - Math.max(y0, Math.min(y1, v))) / (y1 - y0)) * (h - MT - MB);
	const Y = (v: number) => Yt(tf(v));
	const path = (y: ArrayLike<number>) => Array.from(y, (v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('');
	const yt = $derived(log ? Array.from({ length: Math.floor(y1) - Math.ceil(y0) + 1 }, (_, k) => 10 ** (Math.ceil(y0) + k)) : [y0, (y0 + y1) / 2, y1]);
	const xt = $derived(bars ? [0, Math.round(n / 4), Math.round(n / 2), Math.round((3 * n) / 4), n - 1] : [0, 16, 32, 48, 63]);
	const bw = $derived(Math.max(1, ((W - ML - MR) / n) * 0.7));
</script>

<div class="chart" bind:this={el}>
	<svg width={W} height={h} viewBox="0 0 {W} {h}" role="img" aria-label={t(label)}>
		<rect x={ML} y={MT} width={W - ML - MR} height={h - MT - MB} class="bg" />
		{#each yt as v, k (k)}<line x1={ML} x2={W - MR} y1={log ? Y(v) : Yt(v)} y2={log ? Y(v) : Yt(v)} class="grid" /><text x={ML - 4} y={(log ? Y(v) : Yt(v)) + 3.5} class="ax end">{log ? String(+v.toPrecision(1)) : fmt3(v)}</text>{/each}
		{#if !log && y0 < 0 && y1 > 0}<line x1={ML} x2={W - MR} y1={Y(0)} y2={Y(0)} class="grid zero" />{/if}
		{#each xt as i (i)}<text x={X(i)} y={h - MB + 13} class="ax mid">{i + x0}</text>{/each}
		<text x={(ML + W - MR) / 2} y={h - 4} class="ax mid">{t(xLabel)}</text>
		<text x={10} y={(MT + h - MB) / 2} class="ax mid" transform="rotate(-90 10 {(MT + h - MB) / 2})">{t(yLabel)}</text>
		{#if bars}{#each bars.v as v, i (i)}<rect x={X(i) - bw / 2} y={Math.min(Y(v), Yt(log ? y0 : 0))} width={bw} height={Math.abs(Yt(log ? y0 : 0) - Y(v))} class="bar" class:drop={bars.cut !== undefined && i >= bars.cut} class:hot={i === hot} />{/each}{/if}
		{#if bars && hot >= 0}<circle cx={X(hot)} cy={Y(bars.v[hot])} r="5" class="ring" />{/if}
		{#if dots}{#each dots.v as v, i (i)}<circle cx={X(i)} cy={Y(v)} r="3" class="dot" />{/each}{/if}
		{#each lines as l, k (k)}<path d={path(l.y)} class="ln {l.cls}" />{/each}
		{#if mark !== null}<line x1={X(mark) - (bars ? (W - ML - MR) / n / 2 : 0)} x2={X(mark) - (bars ? (W - ML - MR) / n / 2 : 0)} y1={MT} y2={h - MB} class="mark" />{/if}
	</svg>
	<p class="legend">
		{#each lines as l, k (k)}<span><i class="k {l.cls}"></i>{t(l.name)}</span>{/each}
		{#if bars}<span><i class="k sq"></i>{t(bars.name)}</span>{#if bars.cut !== undefined && bars.cut < n}<span><i class="k sq drop"></i>{t({ en: 'dropped', zh: '丢掉的' })}</span>{/if}{/if}
		{#if dots}<span><i class="k dt"></i>{t(dots.name)}</span>{/if}
	</p>
</div>

<style>
	.chart { min-width: 0; }
	svg { display: block; max-width: 100%; }
	.bg { fill: var(--surface); stroke: var(--rule-strong); }
	.grid { stroke: var(--rule); stroke-width: 0.6; }
	.grid.zero { stroke: var(--rule-strong); }
	.ax { font-size: 10px; fill: var(--ink-2); }
	.ax.mid { text-anchor: middle; }
	.ax.end { text-anchor: end; }
	.ln { fill: none; stroke-width: 2; stroke-linejoin: round; }
	.sol { stroke: var(--real); stroke-width: 3.2; }
	.pred { stroke: var(--generated); stroke-dasharray: 5 3; }
	.u0 { stroke: var(--ink); }
	.faint { stroke: var(--ink-2); stroke-width: 1.2; stroke-dasharray: 2 3; }
	.c0 { stroke: var(--cat-0); } .c1 { stroke: var(--cat-1); } .c2 { stroke: var(--cat-2); } .c3 { stroke: var(--cat-3); }
	.bar { fill: var(--cat-0); }
	.bar.hot { fill: var(--ink); }
	.bar.drop { fill: none; stroke: var(--muted); stroke-dasharray: 2 2; }
	.ring { fill: none; stroke: var(--ink); stroke-width: 1.8; }
	.dot { fill: var(--generated); stroke: var(--surface); stroke-width: 1; }
	.mark { stroke: var(--ink); stroke-width: 1.4; stroke-dasharray: 4 3; }
	.legend { display: flex; flex-wrap: wrap; gap: 2px 14px; margin: 4px 0 0; color: var(--ink-2); font-size: var(--fs-xs); }
	.k { display: inline-block; width: 18px; height: 0; margin-right: 5px; vertical-align: middle; border-top: 2.5px solid; }
	.k.sol { border-color: var(--real); } .k.pred { border-top-style: dashed; border-color: var(--generated); } .k.u0 { border-color: var(--ink); } .k.faint { border-top-style: dotted; border-color: var(--ink-2); }
	.k.c0 { border-color: var(--cat-0); } .k.c1 { border-color: var(--cat-1); } .k.c2 { border-color: var(--cat-2); } .k.c3 { border-color: var(--cat-3); }
	.k.sq { width: 10px; height: 10px; border: 0; background: var(--cat-0); }
	.k.sq.drop { background: none; border: 1.5px dashed var(--muted); }
	.k.dt { width: 8px; height: 8px; border: 0; border-radius: 50%; background: var(--generated); }
</style>
