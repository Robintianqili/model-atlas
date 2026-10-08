<!-- ecg-autoencoder: beats drawn over their 140 points (height up, point across, the input box's range). Lines: 'real'
     green (a beat as it came in), 'gen' magenta (the rebuild), 'faint' (a crowd of training beats). gap shades the error
     between the first two lines; regions marks where the P, QRS, ST and T parts usually are; mark rings one point;
     strip draws numbers per point as bars under the plot (the errors: orange where the beat is above its rebuild). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '../../../atlas/core/format.ts';
	import { REGIONS } from '../ecg.ts';

	interface Props { lines: { y: ArrayLike<number>; cls: 'real' | 'gen' | 'faint' }[]; gap?: boolean; regions?: boolean; mark?: number | null; label: string; h?: number; strip?: number[] }
	let { lines, gap = false, regions = false, mark = null, label, h = 220, strip }: Props = $props();
	let W = $state(480);
	let el = $state<HTMLDivElement>();
	$effect(() => {
		if (!el) return;
		const ro = new ResizeObserver((e) => {
			const w = Math.round(e[0].contentRect.width);
			if (w > 0 && w !== W) W = w;
		});
		ro.observe(el);
		return () => ro.disconnect();
	});
	const Y0 = -0.6, Y1 = 1.4, ML = 40, MR = 6, MT = 14, MB = 18;
	const X = (i: number) => ML + (i / 139) * (W - ML - MR);
	const Y = (v: number) => MT + ((Y1 - Math.max(Y0, Math.min(Y1, v))) / (Y1 - Y0)) * (h - MT - MB);
	const path = (y: ArrayLike<number>) => Array.from(y, (v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('');
	const SH = 64;
	const top = $derived(h + 8);
	const sMax = $derived(strip ? Math.max(0.05, ...strip.map(Math.abs)) : 1);
	const sY = (v: number) => top + SH / 2 - (v / sMax) * (SH / 2 - 3);
	const shade = $derived(gap && lines.length > 1 ? path(lines[0].y) + Array.from(lines[1].y, (v, i) => [i, v] as const).reverse().map(([i, v]) => `L${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('') + 'Z' : '');
</script>

<div class="plot" bind:this={el}>
	<svg width={W} height={strip ? top + SH : h} viewBox="0 0 {W} {strip ? top + SH : h}" role="img" aria-label={label}>
		<rect x={ML} y={MT} width={W - ML - MR} height={h - MT - MB} class="bg" />
		{#if regions}{#each REGIONS as r, k (r.l)}<rect x={X(r.from)} y={MT} width={X(r.to) - X(r.from)} height={h - MT - MB} class="reg" class:alt={k % 2 === 1} /><text x={(X(r.from) + X(r.to)) / 2} y={MT - 3} class="ax mid">{r.l}</text>{/each}{/if}
		{#each [-0.5, 0, 0.5, 1] as v (v)}<line x1={ML} x2={W - MR} y1={Y(v)} y2={Y(v)} class="grid" class:zero={v === 0} /><text x={ML - 4} y={Y(v) + 3.5} class="ax end">{String(v).replace('-', '−')}</text>{/each}
		{#each [0, 35, 70, 105, 139] as i (i)}<text x={X(i)} y={h - 5} class="ax" class:mid={i > 0 && i < 139} class:end={i === 139}>{i}</text>{/each}
		{#if shade}<path d={shade} class="gap" />{/if}
		{#each lines as l, k (k)}<path d={path(l.y)} class="ln {l.cls}" />{/each}
		{#if mark !== null && lines[0]}<circle cx={X(mark)} cy={Y(lines[0].y[mark])} r="6" class="mark" />{/if}
		{#if strip}
			<rect x={ML} y={top} width={W - ML - MR} height={SH} class="bg" /><line x1={ML} x2={W - MR} y1={sY(0)} y2={sY(0)} class="grid zero" />
			<text x={ML - 4} y={top + 9} class="ax end">{fmt3(sMax)}</text><text x={ML - 4} y={top + SH - 2} class="ax end">−{fmt3(sMax)}</text>
			{#each strip as v, i (i)}<line x1={X(i)} x2={X(i)} y1={sY(0)} y2={sY(v)} class="bar" class:pos={v > 0} class:hot={i === mark} />{/each}
		{/if}
	</svg>
	{#if lines.some((l) => l.cls !== 'faint')}
		<p class="legend">
			{#if lines.some((l) => l.cls === 'real')}<span><i class="k-real"></i>{t({ en: 'the beat that came in', zh: '输入的心跳' })}</span>{/if}
			{#if lines.some((l) => l.cls === 'gen')}<span><i class="k-gen"></i>{t({ en: 'the rebuilt beat', zh: '重建的心跳' })}</span>{/if}
			{#if shade}<span><i class="k-gap"></i>{t({ en: 'the error between them', zh: '两者之间的误差' })}</span>{/if}
			{#if strip}<span><i class="k-pos"></i>{t({ en: 'error: beat above its rebuild', zh: '误差：心跳高于重建' })}</span><span><i class="k-neg"></i>{t({ en: 'beat below its rebuild', zh: '心跳低于重建' })}</span>{/if}
		</p>
	{/if}
</div>

<style>
	.plot { min-width: 0; }
	svg { display: block; max-width: 100%; }
	.bg { fill: var(--surface); stroke: var(--rule-strong); }
	.grid { stroke: var(--rule); stroke-width: 0.6; }
	.grid.zero { stroke: var(--rule-strong); }
	.reg { fill: var(--surface-2); }
	.reg.alt { fill: var(--surface-3); opacity: 0.7; }
	.ax { font-size: 10px; fill: var(--ink-2); }
	.ax.mid { text-anchor: middle; }
	.ax.end { text-anchor: end; }
	.ln { fill: none; stroke-width: 1.8; stroke-linejoin: round; }
	.real { stroke: var(--real); }
	.gen { stroke: var(--generated); stroke-width: 2.2; }
	.faint { stroke: var(--real); stroke-width: 1; opacity: 0.35; }
	.gap { fill: var(--generated); opacity: 0.18; }
	.bar { stroke: var(--div-neg); stroke-width: 1.6; }
	.bar.pos { stroke: var(--div-pos); }
	.bar.hot { stroke: var(--ink); stroke-width: 2.4; }
	.mark { fill: none; stroke: var(--ink); stroke-width: 1.8; }
	.legend { display: flex; flex-wrap: wrap; gap: 2px 14px; margin: 4px 0 0; color: var(--ink-2); font-size: var(--fs-xs); }
	.legend i { display: inline-block; width: 18px; height: 0; margin-right: 5px; vertical-align: middle; border-top: 2.5px solid; }
	.k-real { border-color: var(--real) !important; }
	.k-gen { border-color: var(--generated) !important; }
	.k-pos { border-color: var(--div-pos) !important; }
	.k-neg { border-color: var(--div-neg) !important; }
	.k-gap { height: 9px !important; border: 0 !important; background: var(--generated); opacity: 0.3; }
</style>
