<!-- transe 'space': the entity points flattened by PCA (see meta.ts), coloured by type, with relation arrows and a
     question. A step with play: 'view' replays the snapshots up to its frame (autoplay; Replay), then rests on that
     frame, which the narration describes (reduced motion: no animation). -->
<script lang="ts">
	import { untrack } from 'svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
	import { ENT, REL, TYPES } from '../../kg.ts';
	import { spacePic } from './meta.ts';

	let P: StageProps = $props();
	let anim = $state(-1);
	let timer: ReturnType<typeof setInterval> | undefined;
	function replay() {
		const tp = P.at.tape;
		if (!tp) return;
		const snaps = Array.from({ length: P.at.index + 1 }, (_, i) => i).filter((i) => ['init', 'snapshot'].includes(tp.frame(i).phase));
		clearInterval(timer);
		let k = 0;
		anim = snaps[0] ?? -1;
		timer = setInterval(() => {
			if ((k += 2) < snaps.length) anim = snaps[k];
			else (clearInterval(timer), (anim = -1), P.onend());
		}, 60);
	}
	$effect(() => {
		if (P.autoplay && P.props.animate) untrack(replay);
		return () => clearInterval(timer);
	});
	const shown = $derived(anim >= 0 && P.at.tape ? { ...P, at: { index: anim, frame: P.at.tape.frame(anim), state: P.at.tape.state(anim), tape: P.at.tape } } : P);
	const m = $derived(spacePic(shown));
	const theme = $derived(themeState.resolved);
	const W = 400, H = 300, PAD = 26;
	const box = $derived.by(() => {
		const pts = [...(m?.final ?? [[0, 0]])];
		const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
		const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
		return { x0, y0, s: Math.min((W - 2 * PAD) / Math.max(1e-9, x1 - x0), (H - 2 * PAD) / Math.max(1e-9, y1 - y0)) };
	});
	const X = (v: number) => Math.max(4, Math.min(W - 4, PAD + (v - box.x0) * box.s));
	const Y = (v: number) => Math.max(4, Math.min(H - 4, H - PAD - (v - box.y0) * box.s));
	const hot = $derived(m ? new Set([...m.ring, ...(m.q ? [m.q[0], ...m.near.slice(0, 3).map((c) => c.x)] : []), ...m.arrows.flatMap((a) => [a.h, a.t])]) : new Set<number>());
	const relColor = (k: number) => categorical(4 + k, theme);
</script>

<div class="sp" style:min-height="380px">
	{#if !m}
		<p class="muted">{t({ en: 'Training has not made any entity points yet…', zh: '训练还没有产生实体点…' })}</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: 'Entities on a map (PCA) with relation arrows', zh: '实体图（PCA）和关系箭头' })}>
			<defs>{#each REL as _, k (k)}<marker id="transe-sp-a{k}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill={relColor(k)} /></marker>{/each}</defs>
			{#each m.arrows as a (a.h + '-' + a.t)}<line x1={X(m.xy[a.h][0])} y1={Y(m.xy[a.h][1])} x2={X(a.to[0])} y2={Y(a.to[1])} stroke={relColor(m.rel)} stroke-width="1.4" marker-end="url(#transe-sp-a{m.rel})" />{/each}
			{#if m.q && m.land}
				{#each m.near.slice(0, 3) as c (c.x)}<line x1={X(m.land[0])} y1={Y(m.land[1])} x2={X(m.xy[c.x][0])} y2={Y(m.xy[c.x][1])} class="nn" />{/each}
				<line x1={X(m.xy[m.q[0]][0])} y1={Y(m.xy[m.q[0]][1])} x2={X(m.land[0])} y2={Y(m.land[1])} stroke={relColor(m.q[1])} stroke-width="2.4" marker-end="url(#transe-sp-a{m.q[1]})" />
				<text x={X(m.land[0])} y={Y(m.land[1]) + 4} class="land">✕</text>
			{/if}
			{#each m.xy as [a, b], i (i)}
				<g transform="translate({X(a)},{Y(b)})" class:hot={hot.has(i)}>
					{#if m.ring.includes(i) || m.q?.[0] === i}<circle r="7.5" class="ring" />{/if}
					<circle r={hot.has(i) ? 4.5 : 3.5} style:fill={categorical(ENT[i].type, theme)} class="pt" />
					<text x="6" y="3">{t(ENT[i].name)}</text>
				</g>
			{/each}
		</svg>
		<p class="legend small">{#each TYPES as ty, k (k)}<span><i style:background={categorical(k, theme)}></i>{t(ty)}</span>{/each}{#if m.rel >= 0}<span><b style:background={relColor(m.rel)}></b>{t(REL[m.rel])}</span>{/if}</p>
		<p class="small muted">
			{t({ en: `step ${fmtInt(m.step)}`, zh: `第 ${fmtInt(m.step)} 步` })}{#if Number.isFinite(m.loss)}{' · '}{t({ en: 'average loss', zh: '平均损失' })} {fmt3(m.loss)}{/if}{' · '}{t({ en: `${m.nFacts} facts`, zh: `${m.nFacts} 个事实` })} · {t({ en: `the map keeps ${fmtPct(m.pca.share)} of the spread of 8 numbers per entity`, zh: `这张图保留了每个实体 8 个数的分布中的 ${fmtPct(m.pca.share)}` })}
			{#if P.props.animate}<button type="button" class="btn ghost" onclick={replay}>{t({ en: 'Replay the training', zh: '重放训练过程' })}</button>{/if}
		</p>
		{#if m.q}
			<table class="num-table small">
				<caption>{t({ en: `${t(ENT[m.q[0]].name)} + “${t(REL[m.q[1]])}” → ? (✕ on the map)${m.kind >= 0 ? `: the nearest of type “${TYPES[m.kind].en}”` : ''}`, zh: `${t(ENT[m.q[0]].name)} + “${t(REL[m.q[1]])}” → ?（图上的 ✕）${m.kind >= 0 ? `：类型为“${TYPES[m.kind].zh}”的实体中最近的` : ''}` })}</caption>
				<thead><tr><th>{t({ en: 'rank', zh: '名次' })}</th><th>{t({ en: 'entity', zh: '实体' })}</th><th>{t({ en: 'distance', zh: '距离' })}</th><th>{t({ en: 'a fact says so', zh: '有事实这么说' })}</th></tr></thead>
				<tbody>{#each m.near as c (c.x)}<tr><td class="num">{c.rank}</td><td>{t(ENT[c.x].name)}{#if m.kind < 0}{' '}<span class="muted">({t(TYPES[ENT[c.x].type])})</span>{/if}</td><td class="num">{fmt3(c.d)}</td><td>{m.truth.includes(c.x) ? t({ en: 'yes', zh: '是' }) : ''}</td></tr>{/each}</tbody>
			</table>
		{/if}
	{/if}
</div>

<style>
	.sp { display: grid; gap: 4px; min-width: 0; align-content: start; } svg { width: 100%; height: auto; max-height: 460px; display: block; } .small { font-size: var(--fs-sm); } p { margin: 0; }
	.pt { stroke: var(--ink-2); stroke-width: 0.6; } text { font: 500 8px var(--font-body); fill: var(--ink-2); pointer-events: none; } .hot text { font-weight: 700; font-size: 9.5px; fill: var(--ink); }
	.land { font-size: 13px; font-weight: 700; fill: var(--ink); text-anchor: middle; } .ring { fill: none; stroke: var(--ink); stroke-width: 1.3; stroke-dasharray: 3 2; } .nn { stroke: var(--ink-2); stroke-width: 1; stroke-dasharray: 3 3; }
	.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; } .legend i { display: inline-block; width: 9px; height: 9px; margin-right: 5px; border-radius: 50%; } .legend b { display: inline-block; width: 14px; height: 4px; margin-right: 5px; vertical-align: middle; }
	table { border-collapse: collapse; } caption { text-align: left; font-weight: 600; } th, td { padding: 1px 8px; text-align: left; } th { font-weight: 600; } .btn { min-height: 28px; padding: 0 8px; font-size: var(--fs-xs); }
</style>
