<!-- matrix-factorization 'matrix': the rating matrix (movies as rows, people as columns, your column first and wide).
     A rating's cell is colored blue (−2) to orange (+2) with its number; an empty cell stays blank, or with props.fill
     shows the prediction of the step's weights in a paler color (numbers in your column only). The box scrolls sideways
     on a narrow screen. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { KINDS, MOVIES, N_OTHERS, YOU, kindColor, known, predict, ranked, toRaw, type RatingsRaw } from '../../world.ts';
	import { factorsAt } from '../../picture.ts';
	import RateList from '../../parts/RateList.svelte';
	import { GRID } from './meta.ts';

	let P: StageProps = $props();
	const TW = 150, YW = 46, CW = 16, RH = 18, HD = 30;
	const mine = $derived(toRaw(P.rt.input).ratings);
	const f = $derived(P.props.fill ? factorsAt(P.at) : null);
	const top = $derived(f && P.props.rank ? ranked(f, P.rt.input).slice(0, 3).map((x) => x.i) : []);
	const color = (v: number, pale = false) => `color-mix(in srgb, var(${v >= 0 ? '--div-pos' : '--div-neg'}) ${Math.round(Math.min(1, Math.abs(v) / 2) * (pale ? 45 : 85))}%, var(--surface))`;
	const sign = (v: number) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0');
	const filled = $derived(known(P.rt.input).n);
	const cells = (N_OTHERS + 1) * MOVIES.length;
	const setInput = (r: RatingsRaw) => (P.setInput ? P.setInput(r) : P.rt.setInput(r));
</script>

<div class="mx">
	<div class="box" use:keyScroll>
		<svg viewBox="0 0 {TW + YW + N_OTHERS * CW + 2} {HD + MOVIES.length * RH + 2}" role="img" aria-label={t({ en: 'The rating matrix', zh: '评分矩阵' })}>
			<text x={TW + YW / 2} y={HD - 8} class="hd you">{t({ en: 'you', zh: '你' })}</text>
			<text x={TW + YW + (N_OTHERS * CW) / 2} y={HD - 8} class="hd">{t({ en: `${N_OTHERS} other people`, zh: `其他 ${N_OTHERS} 个人` })}</text>
			{#each MOVIES as m, i (i)}
				{@const y = HD + i * RH}
				{#if P.props.focus === i}<rect x="0" {y} width={TW + YW} height={RH} class="focus" />{/if}
				<circle cx="6" cy={y + RH / 2} r="4" style:fill={kindColor(m.kind)}><title>{t(KINDS[m.kind])}</title></circle>
				<text x="14" y={y + RH / 2 + 4} class="ttl">{t(m.title)}</text>
				{#if top.includes(i)}<text x={TW - 4} y={y + RH / 2 + 4} class="rank">{top.indexOf(i) + 1}</text>{/if}
				{@const v = mine[i]}
				{@const p = v === null && f ? predict(f, YOU, i) : null}
				<rect x={TW + 1} y={y + 1} width={YW - 2} height={RH - 2} class="cell" style:fill={v !== null ? color(v) : p !== null ? color(p, true) : null} />
				{#if v !== null}<text x={TW + YW / 2} y={y + RH / 2 + 4} class="v b">{sign(v)}</text>
				{:else if p !== null}<text x={TW + YW / 2} y={y + RH / 2 + 4} class="v i">{fmt3(p)}</text>{/if}
				{#each GRID[i] as r, u (u)}
					{@const x = TW + YW + u * CW}
					{@const q = r === null && f ? predict(f, u, i) : null}
					<rect x={x + 0.5} y={y + 0.5} width={CW - 1} height={RH - 1} class="cell" style:fill={r !== null ? color(r) : q !== null ? color(q, true) : null} />
					{#if r !== null}<text x={x + CW / 2} y={y + RH / 2 + 3.5} class="v s">{sign(r)}</text>{/if}
				{/each}
			{/each}
			<line x1={TW + YW + 0.5} y1={HD - 4} x2={TW + YW + 0.5} y2={HD + MOVIES.length * RH} class="sep" />
		</svg>
	</div>
	<p class="note muted">
		{t({ en: `Colors: blue −2 (hated it) to orange +2 (loved it). Blank: not rated (${filled} of ${cells} cells are rated).`, zh: `颜色：蓝色 −2（讨厌）到橙色 +2（超爱）。空白：没评过（${cells} 个格子里有 ${filled} 个有评分）。` })}
		{#if f}{t({ en: ' Pale cells: the model’s predictions (in your column with their numbers).', zh: '浅色格子：模型的预测（你那一列标出了数值）。' })}{/if}
		{#if top.length}{t({ en: ' 1, 2, 3: your top recommendations.', zh: '1、2、3：给你的前三个推荐。' })}{/if}
	</p>
	<p class="note muted">{t({ en: 'Dot before a title: the movie’s kind, for you only (training never sees it):', zh: '片名前的圆点：电影的类型，只是给你看的（训练从来看不到）：' })}{#each KINDS as k, i (i)}<span class="k"><i style:background={kindColor(i)}></i>{t(k)}</span>{/each}</p>
	{#if P.props.edit}<RateList raw={P.rt.input} onchange={setInput} />{/if}
</div>

<style>
	.mx { display: grid; gap: 8px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	.box { overflow-x: auto; }
	svg { display: block; width: 100%; min-width: 520px; max-width: 760px; height: auto; }
	.cell { fill: var(--surface-2); stroke: none; }
	.ttl { font: 11px var(--font-body); fill: var(--ink); }
	.hd { font: 10px var(--font-body); fill: var(--ink-2); text-anchor: middle; } .hd.you { font-weight: 700; fill: var(--accent-strong); }
	.v { font-family: var(--font-mono); text-anchor: middle; fill: var(--ink); } .v.b { font-size: 11px; font-weight: 700; } .v.i { font-size: 10px; font-style: italic; } .v.s { font-size: 8.5px; }
	.rank { font: 700 11px var(--font-mono); fill: var(--accent-strong); text-anchor: end; }
	.focus { fill: var(--accent-soft); } .sep { stroke: var(--rule-strong); }
	.note { margin: 0; font-size: var(--fs-xs); }
	.k { margin-left: 10px; white-space: nowrap; } .k i { display: inline-block; width: 9px; height: 9px; margin-right: 4px; border-radius: 50%; }
</style>
