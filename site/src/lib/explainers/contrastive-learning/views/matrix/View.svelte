<!-- contrastive-learning 'matrix': a batch's table of pairs (props.mode, see data.ts). Adapted from TinyCLIP's 'match' view
     (thumbnails and strips with the kit TensorGrid; the table is an ARIA table whose rows are subgrids). The τ curve is a
     small SVG with a log x axis. -->
<script lang="ts">
	import TensorGrid from '$lib/kit/TensorGrid.svelte';
	import { makeScale } from '$lib/kit/color';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtFull, fmtInt, fmtPct, fmtSetting } from '../../../../atlas/core/format.ts';
	import { TAU_RANGE } from '../../adapters/batch/adapter.ts';
	import { viewData, type Show } from './data.ts';
	import { describeMatrix } from './meta.ts';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'cos'));
	let cols = $state(false);
	const toggled = $derived<Show | undefined>(mode === 'P' && cols ? 'Q' : undefined);
	const d = $derived(viewData(P, toggled));
	const L = (en: string, zh: string) => t({ en, zh });
	$effect(() => void (P.autoplay && P.onend()));

	/** one colour domain per kind of table: cosines −1..1 (so the training frames compare), shares 0..1, gradients ±max */
	const scale = $derived.by(() => {
		if (d.kind !== 'table') return null;
		if (d.domain === 'share') return makeScale('sequential', [0, 1], themeState.resolved);
		const m = d.domain === 'unit' ? 1 : Math.max(1e-9, ...(d.M ?? []).map(Math.abs));
		return makeScale('diverging', [-m, m], themeState.resolved);
	});
	const cell = (v: number) => (d.kind === 'table' && d.domain === 'share' ? (v >= 0.9995 ? '>99.9%' : fmtPct(v)) : fmt3(v, 2));
	const isMark = (i: number, j: number) => d.kind === 'table' && d.marks.some(([a, b]) => a === i && b === j);
	const idx = (n: number) => Array.from({ length: n }, (_, k) => k);

	// the τ curve
	const W = 440, H = 230, pad = { l: 46, r: 12, t: 14, b: 40 };
	const lx = (v: number) => Math.log10(v);
	const [x0, x1] = TAU_RANGE.map(lx);
	const sx = (tau: number) => pad.l + ((lx(tau) - x0) / (x1 - x0)) * (W - pad.l - pad.r);
	const yMax = $derived(d.kind === 'curve' ? Math.max(d.chance * 1.15, ...d.loss.filter(Number.isFinite)) : 1);
	const sy = (v: number) => pad.t + (1 - v / yMax) * (H - pad.t - pad.b);
	const yTicks = $derived.by(() => {
		const step = yMax > 20 ? 10 : yMax > 8 ? 2 : yMax > 3 ? 1 : 0.5;
		return Array.from({ length: Math.floor(yMax / step) + 1 }, (_, k) => k * step);
	});
	const xTicks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2];
	const line = $derived(d.kind === 'curve' ? d.taus.map((x, k) => `${sx(x).toFixed(1)},${sy(d.loss[k]).toFixed(1)}`).join(' ') : '');
</script>

{#snippet thumb(data: ArrayLike<number>, px: number, label: string)}
	<TensorGrid {data} rows={28} cols={28} kind="sequential" domain={[0, 1]} cell={px} interactive={false} gridLines={false} labels={false} label={{ en: label, zh: label }} />
{/snippet}
{#snippet strip(data: number[], label: string, domain: [number, number])}
	<div class="strip"><span class="sl">{label}</span><TensorGrid {data} rows={1} cols={data.length} kind="diverging" {domain} maxCell={16} minCell={4} labels={false} readout={false} label={{ en: label, zh: label }} /></div>
{/snippet}

<div class="mt">
	{#if d.kind === 'pair'}
		{@const m = Math.max(1e-9, ...d.a.map(Math.abs), ...d.b.map(Math.abs))}
		{@render strip(d.a, L('drawing 0’s vector a (length 1)', '画 0 的向量 a（长度 1）'), [-m, m])}
		{@render strip(d.b, L('caption 0’s vector b (length 1)', '描述 0 的向量 b（长度 1）'), [-m, m])}
		{@render strip(d.prods, L('products a × b, number by number', '逐个数的乘积 a × b'), [-Math.max(1e-9, ...d.prods.map(Math.abs)), Math.max(1e-9, ...d.prods.map(Math.abs))])}
		<p class="num sum">Σ a·b = cos = <strong>{fmt3(d.cos)}</strong></p>
		<p class="muted">{L('Orange is positive, blue negative. Where both vectors have the same sign the product is positive and raises the cosine.', '橙色为正，蓝色为负。两个向量同号的地方乘积为正，会让余弦变大。')}</p>
	{:else if d.kind === 'curve'}
		<svg viewBox="0 0 {W} {H}" class="curve" role="img" aria-label={t(describeMatrix(P))}>
			{#each yTicks as y (y)}
				<line x1={pad.l} x2={W - pad.r} y1={sy(y)} y2={sy(y)} class="grid" />
				<text x={pad.l - 6} y={sy(y) + 4} text-anchor="end" class="tick">{fmtSetting(y)}</text>
			{/each}
			{#each xTicks as x (x)}
				<line x1={sx(x)} x2={sx(x)} y1={pad.t} y2={H - pad.b} class="grid" />
				<text x={sx(x)} y={H - pad.b + 14} text-anchor="middle" class="tick">{x}</text>
			{/each}
			<line x1={pad.l} x2={W - pad.r} y1={sy(d.chance)} y2={sy(d.chance)} class="ref" />
			<text x={W - pad.r - 4} y={sy(d.chance) - 5} text-anchor="end" class="tick">{L(`blind guess: log ${Math.round(Math.exp(d.chance))} = ${fmt3(d.chance)}`, `瞎猜：log ${Math.round(Math.exp(d.chance))} = ${fmt3(d.chance)}`)}</text>
			<polyline points={line} class="loss" />
			<circle cx={sx(d.tau)} cy={sy(d.now)} r="6" class="now" />
			<text x={(pad.l + W - pad.r) / 2} y={H - 4} text-anchor="middle" class="axis">{L('temperature τ (log scale)', '温度 τ（对数刻度）')}</text>
			<text x="12" y={(pad.t + H - pad.b) / 2} text-anchor="middle" class="axis" transform="rotate(-90 12 {(pad.t + H - pad.b) / 2})">{L('loss (nats)', '损失（奈特）')}</text>
		</svg>
		<p class="num readout">
			{L(`τ = ${fmtSetting(d.tau)}: loss ${fmt3(d.now)}`, `τ = ${fmtSetting(d.tau)}：损失 ${fmt3(d.now)}`)} · {d.best.atEdge ? L('it keeps falling as τ shrinks', 'τ 越小，损失越低') : L(`lowest ${fmt3(d.best.loss)} at τ = ${fmt3(d.best.tau)}`, `最低 ${fmt3(d.best.loss)}（τ = ${fmt3(d.best.tau)}）`)}
		</p>
	{:else if d.kind === 'shift'}
		<div class="views">
			<figure>{@render thumb(d.images[0], 3, L('drawing 0', '画 0'))}<figcaption>{L('drawing 0', '画 0')}</figcaption></figure>
			<figure>{@render thumb(d.images[1], 3, L('a distorted view', '一个变形视图'))}<figcaption>{L('a distorted view', '一个变形视图')}<br /><span class="num">cos {fmt3(d.jitter)}</span></figcaption></figure>
			<figure>{@render thumb(d.images[2], 3, L('moved across', '移到另一边'))}<figcaption>{L('moved across', '移到另一边')}<br /><span class="num">cos {fmt3(d.shift)}</span></figcaption></figure>
		</div>
		<p class="muted">{L('Cosines with drawing 0, through TinyCLIP’s image tower.', '与画 0 的余弦，由 TinyCLIP 的图像塔算出。')}</p>
	{:else if d.kind === 'table' && scale}
		{#if mode === 'P'}
			<div class="tog" role="group" aria-label={L('Softmax direction', 'softmax 的方向')}>
				<button type="button" class="chip" aria-pressed={!cols} onclick={() => (cols = false)}>{L('along rows', '沿行')}</button>
				<button type="button" class="chip" aria-pressed={cols} onclick={() => (cols = true)}>{L('along columns', '沿列')}</button>
			</div>
		{/if}
		<p class="muted">{t(d.what)}</p>
		{#if mode === 'progress' && P.at.frame?.metrics}
			{@const m = P.at.frame.metrics}
			<p class="num readout">{L(`update ${fmtInt(m.update)} of ${fmtInt(m.last)} · τ = ${fmt3(m.tau)} · loss ${fmt3(m.loss)} · own caption first in ${m.right} of ${m.N} rows`, `第 ${fmtInt(m.update)} 次更新（共 ${fmtInt(m.last)} 次）· τ = ${fmt3(m.tau)} · 损失 ${fmt3(m.loss)} · ${m.N} 行中有 ${m.right} 行把自己的描述排第一`)}</p>
		{/if}
		{#if d.M}
			{@const n = d.rows.length}
			<div class="box" use:keyScroll>
				<div class="mx" style:--n={n} role="table" aria-label={t(d.what)}>
					<div class="r" role="row"><span role="columnheader"><span class="sr">{L('row', '行')}</span></span>{#each idx(n) as j (j)}<span class="hd num" role="columnheader">{j}</span>{/each}</div>
					{#each idx(n) as i (i)}
						<div class="r" role="row">
							<span class="rh" role="rowheader"><span class="ri num">{i}</span>{@render thumb(d.rows[i], 1, d.captions ? (d.captions[i] ?? '') : L(`view ${i}`, `视图 ${i}`))}</span>
							{#each idx(n) as j (j)}
								{@const v = d.M[i * n + j]}
								{#if d.blank && i === j}
									<span class="c self" role="cell">—</span>
								{:else}
									<span class="c num" class:dg={isMark(i, j)} role="cell" style:background={scale.css(v)} style:color={scale.ink(v)} title={fmtFull(v)}>{cell(v)}</span>
								{/if}
							{/each}
						</div>
					{/each}
				</div>
			</div>
			{#if d.captions}
				<ol class="legend" start="0">{#each d.captions as c, j (j)}<li>“{c}”</li>{/each}</ol>
			{:else}
				<p class="muted">{L('Views 2i and 2i + 1 are the two views of drawing i (outlined: each view’s partner).', '视图 2i 和 2i + 1 是画 i 的两个视图（带框的是每个视图的搭档）。')}</p>
			{/if}
		{/if}
	{/if}
</div>

<style>
	.mt { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.strip { display: grid; gap: 2px; min-width: 0; }
	.sl { font-size: var(--fs-xs); }
	.sum, .muted, .readout { margin: 0; }
	.tog { display: flex; gap: 6px; }
	.chip { cursor: pointer; font-size: var(--fs-xs); padding: 2px 8px; }
	.chip[aria-pressed='true'] { background: var(--accent-soft); border-color: var(--accent); }
	.box { overflow-x: auto; max-width: 100%; }
	.mx { display: grid; grid-template-columns: 46px repeat(var(--n), 40px); gap: 2px; align-items: center; justify-content: start; }
	.r { display: grid; grid-template-columns: subgrid; grid-column: 1 / -1; align-items: center; }
	.hd { text-align: center; font-size: var(--fs-xs); }
	.rh { display: flex; align-items: center; gap: 3px; justify-content: flex-end; }
	.ri { font-size: var(--fs-xs); }
	.c { display: grid; place-items: center; height: 30px; font-size: 10px; border-radius: 3px; overflow: hidden; }
	.c.dg { outline: 2px solid var(--ink); outline-offset: -2px; }
	.c.self { color: var(--muted); border: 1px dashed var(--rule); }
	.legend { margin: 0; padding-left: 2.2em; font-size: var(--fs-xs); }
	.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
	.curve { width: 100%; max-width: 520px; height: auto; }
	.grid { stroke: var(--rule); stroke-width: 1; }
	.ref { stroke: var(--muted); stroke-dasharray: 5 4; stroke-width: 1.5; }
	.loss { fill: none; stroke: var(--accent); stroke-width: 2.5; }
	.now { fill: var(--highlight); stroke: var(--ink); stroke-width: 1.5; }
	.tick { font-size: 12px; fill: var(--ink-2); }
	.axis { font-size: 13px; fill: var(--ink); }
	.views { display: flex; flex-wrap: wrap; gap: 14px; }
	.views figure { margin: 0; display: grid; gap: 4px; justify-items: center; font-size: var(--fs-xs); text-align: center; }
</style>
