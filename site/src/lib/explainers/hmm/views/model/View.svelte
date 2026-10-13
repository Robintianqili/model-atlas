<!-- hmm 'model' (meta.ts). story: the two weather states as circles with the transition probabilities on the arrows,
     each state's emission probabilities as bars (length = probability, one shared 0-to-1 scale), and her days with the
     weather unknown. sample: the made-up days so far and the current day's two draws: a uniform number u on a 0-to-1
     line cut into the row's intervals (the weather row, then that weather's activity row). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical, hexToRgb, inkOn } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { ACTS, I, obsOf, percentOf, STATES } from '../../hmm.ts';
	import { drawn } from './meta.ts';

	let P: StageProps = $props();
	const uid = $props.id();
	const show = $derived(String(P.props.show ?? 'story'));
	const cat = $derived((void themeState.resolved, [0, 1].map((k) => categorical(k))));
	const ink = $derived(cat.map((c) => inkOn(hexToRgb(c))));
	const tb = $derived((void P.at, percentOf(P.rt.input)));
	const obs = $derived((void P.at, obsOf(P.rt.input)));
	const q = (v: number) => String(v / 100);
	const f = $derived(P.at.frame);
	const m = $derived(f?.metrics ?? {});
	const D = $derived(drawn(P));
	// the current day's two lines: the row each draw used, its intervals and the draw
	const lines = $derived.by(() => {
		if (show !== 'sample' || f?.phase !== 'day') return [];
		const wRow = m.t ? tb.A[m.prev] : tb.pi, cut = (row: number[]) => row.reduce<number[]>((a, v) => [...a, (a[a.length - 1] ?? 0) + v / 100], []);
		return [
			{ title: m.t ? I(`Weather of day ${m.t}: the transition row of ${STATES[m.prev].en} (day ${m.t - 1})`, `第 ${m.t} 天的天气：用${STATES[m.prev].zh}（第 ${m.t - 1} 天）那一行转移概率`) : I('Weather of day 0: the start row', '第 0 天的天气：用初始那一行'), cuts: cut(wRow), names: STATES, u: m.u, pick: m.s, sym: 'u' },
			{ title: I(`Her activity on day ${m.t}: the emission row of ${STATES[m.s].en}`, `第 ${m.t} 天她的活动：用${STATES[m.s].zh}那一行发射概率`), cuts: cut(tb.B[m.s]), names: ACTS, u: m.v, pick: m.act, sym: 'v' }
		];
	});
	const NW = 300, BX = 10;
</script>

<div class="mv">
	{#if show === 'story'}
		<svg viewBox="0 0 360 230" class="dia" role="img" aria-label={t({ en: 'The two weather states, their transition probabilities and emission bars', zh: '两种天气状态、它们的转移概率和发射概率条' })}>
			<defs><marker id="mh-{uid}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="mk" /></marker></defs>
			<path d="M70,36 C50,-4 130,-4 110,36" class="ar" marker-end="url(#mh-{uid})" /><text x="90" y="12" class="lb">{q(tb.A[0][0])}</text>
			<path d="M250,36 C230,-4 310,-4 290,36" class="ar" marker-end="url(#mh-{uid})" /><text x="270" y="12" class="lb">{q(tb.A[1][1])}</text>
			<path d="M124,48 Q180,28 236,48" class="ar" marker-end="url(#mh-{uid})" /><text x="180" y="32" class="lb">{q(tb.A[0][1])}</text>
			<path d="M236,74 Q180,94 124,74" class="ar" marker-end="url(#mh-{uid})" /><text x="180" y="98" class="lb">{q(tb.A[1][0])}</text>
			{#each STATES as s, j (j)}
				<circle cx={90 + 180 * j} cy="61" r="30" style:fill={cat[j]} />
				<text x={90 + 180 * j} y="65" class="sn" style:fill={ink[j]}>{t(s)}</text>
				{#each ACTS as a, k (k)}
					<text x={50 + 180 * j} y={136 + k * 24} class="an">{t(a)}</text>
					<rect x={56 + 180 * j} y={126 + k * 24} width={tb.B[j][k] * 1.0} height="14" class="bar" style:fill={cat[j]} />
					<text x={60 + 180 * j + tb.B[j][k]} y={137 + k * 24} class="bv">{q(tb.B[j][k])}</text>
				{/each}
			{/each}
			<text x="180" y="222" class="cap">{t({ en: 'Bars: what she does on that weather (length = probability, 0 to 1)', zh: '横条：这种天气下她做什么（长度 = 概率，0 到 1）' })}</text>
		</svg>
		<div class="scroll" use:keyScroll role="region" aria-label={t({ en: 'Her days', zh: '她的日子' })}>
			<ol class="days">
				{#each obs as o, d (d)}<li><small>{t({ en: `day ${d}`, zh: `第 ${d} 天` })}</small><span class="act">{t(ACTS[o])}</span><span class="q" aria-label={t({ en: 'weather unknown', zh: '天气未知' })}>?</span></li>{/each}
			</ol>
		</div>
	{:else}
		<div class="scroll" use:keyScroll role="region" aria-label={t({ en: 'The made-up days', zh: '编出来的日子' })}>
			<table class="sm">
				<thead><tr><th scope="col"><span class="vh">{t({ en: 'row', zh: '行' })}</span></th>{#each D.w as _, d (d)}<th scope="col" class:now={f?.phase === 'day' && m.t === d}>{d}</th>{/each}</tr></thead>
				<tbody>
					<tr><th scope="row">{t({ en: 'Weather (hidden)', zh: '天气（隐藏）' })}</th>{#each D.w as s, d (d)}<td>{#if s >= 0}<span class="chip" style:background={cat[s]} style:color={ink[s]}>{t(STATES[s])}</span>{/if}</td>{/each}</tr>
					<tr><th scope="row">{t({ en: 'Activity (seen)', zh: '活动（看得到）' })}</th>{#each D.a as a, d (d)}<td>{#if a >= 0}<span class="ac">{t(ACTS[a])}</span>{/if}</td>{/each}</tr>
				</tbody>
			</table>
		</div>
		{#each lines as ln (ln.sym)}
			<figure class="line">
				<figcaption>{t(ln.title)}: {ln.sym} = {fmt3(ln.u)} → <b>{t(ln.names[ln.pick])}</b></figcaption>
				<svg viewBox="0 0 {NW + 2 * BX} 44" role="img" aria-label={t({ en: `${ln.sym} = ${fmt3(ln.u)} lands in the interval of ${ln.names[ln.pick].en}`, zh: `${ln.sym} = ${fmt3(ln.u)} 落在“${ln.names[ln.pick].zh}”的区间里` })}>
					{#each ln.cuts as c, k (k)}
						{@const a = k ? ln.cuts[k - 1] : 0}
						<rect x={BX + a * NW} y="8" width={(c - a) * NW} height="16" class="seg" class:pick={k === ln.pick} />
						<text x={BX + ((a + c) / 2) * NW} y="20" class="sl">{t(ln.names[k])}</text>
						<text x={BX + c * NW} y="38" class="tk">{fmt3(c)}</text>
					{/each}
					<text x={BX} y="38" class="tk">0</text>
					<line x1={BX + ln.u * NW} x2={BX + ln.u * NW} y1="2" y2="28" class="u" />
				</svg>
			</figure>
		{/each}
		{#if f?.phase === 'done'}<p class="sum">{t({ en: `${m.n} made-up days: the weather stayed the same ${m.stays} times and changed ${m.switches} times; the longest spell lasted ${m.longest} days.`, zh: `编出了 ${m.n} 天：天气 ${m.stays} 次保持不变，${m.switches} 次发生变化；最长的一段持续了 ${m.longest} 天。` })}</p>{/if}
	{/if}
</div>

<style>
	.mv { display: grid; gap: 10px; min-width: 0; font-size: var(--fs-sm); }
	.dia { width: 100%; max-width: 420px; height: auto; }
	.ar { fill: none; stroke: var(--ink-2); stroke-width: 1.5; }
	.mk { fill: var(--ink-2); }
	text { font-size: 11px; fill: var(--ink); text-anchor: middle; }
	.lb { font-weight: 600; font-variant-numeric: tabular-nums; }
	.sn { font-weight: 700; font-size: 12px; }
	.an { text-anchor: end; }
	.bv { text-anchor: start; font-variant-numeric: tabular-nums; }
	.cap { fill: var(--muted); font-size: 10px; }
	.scroll { overflow-x: auto; max-width: 100%; }
	.scroll:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
	.days { list-style: none; margin: 0; padding: 0; display: flex; gap: 4px; }
	.days li { display: grid; justify-items: center; min-width: 50px; padding: 4px; border: 1px solid var(--rule-strong); border-radius: 6px; }
	.days small { color: var(--muted); font-size: var(--fs-xs); }
	.act { font-weight: 600; }
	.q { font-weight: 700; color: var(--muted); }
	.sm { border-collapse: separate; border-spacing: 2px; font-size: var(--fs-xs); }
	.sm th { font-weight: 600; text-align: center; padding: 0 2px; }
	.sm th[scope='row'] { text-align: right; white-space: nowrap; }
	.sm th.now { outline: 2px solid var(--accent); border-radius: 4px; }
	.sm td { min-width: 44px; text-align: center; height: 24px; }
	.chip { display: inline-block; padding: 1px 6px; border-radius: 10px; font-weight: 600; }
	.vh { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
	.line { margin: 0; }
	.line figcaption { font-size: var(--fs-xs); }
	.line svg { width: 100%; max-width: 480px; height: auto; display: block; }
	.seg { fill: var(--surface-2); stroke: var(--rule-strong); }
	.seg.pick { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 2; }
	.sl { font-size: 10px; }
	.tk { font-size: 9px; fill: var(--muted); }
	.u { stroke: var(--accent); stroke-width: 2.5; }
	.sum { margin: 0; }
</style>
