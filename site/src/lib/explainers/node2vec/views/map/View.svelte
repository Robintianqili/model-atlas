<!-- node2vec 'map': the member vectors flattened by PCA (see meta.ts), each dot a member coloured by its side (blue Mr.
     Hi, orange the Officer). A step with play: 'view' replays the snapshots up to its frame (autoplay; Replay), then rests
     on that frame, which the narration describes (reduced motion: no animation). -->
<script lang="ts">
	import { untrack } from 'svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
	import { FACTIONS } from '../../walks.ts';
	import { mapPic } from './meta.ts';

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
	const m = $derived(mapPic(shown));
	/** the member in focus is drawn last, on top */
	const order = $derived(m ? [...m.xy.keys()].sort((a, b) => +(a === m.focus) - +(b === m.focus)) : []);
	const W = 400, H = 300, PAD = 18;
	const box = $derived.by(() => {
		const end = m?.final ?? [[0, 0]];
		const xs = end.map((p) => p[0]), ys = end.map((p) => p[1]);
		const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
		return { x0, y0, sx: (W - 2 * PAD) / Math.max(1e-9, x1 - x0), sy: (H - 2 * PAD) / Math.max(1e-9, y1 - y0) };
	});
	const X = (v: number) => Math.max(6, Math.min(W - 6, PAD + (v - box.x0) * box.sx));
	const Y = (v: number) => Math.max(6, Math.min(H - 6, H - PAD - (v - box.y0) * box.sy));
	const fac = (i: number) => `color-mix(in srgb, var(${FACTIONS[i] ? '--div-pos' : '--div-neg'}) 60%, var(--surface))`;
	const side = (f: number) => t(f ? { en: 'Officer', zh: '干事' } : { en: 'Mr. Hi', zh: 'Mr. Hi' });
</script>

<div class="map" style:min-height="360px">
	{#if !m}
		<p class="muted">{t({ en: 'Training has not made any member vectors yet…', zh: '训练还没有产生成员向量…' })}</p>
	{:else}
		<svg viewBox="0 0 {W} {H}" role="img" aria-label={t({ en: 'Member vectors on a map (PCA)', zh: '成员向量图（PCA）' })}>
			{#if m.showCut && Number.isFinite(m.cut.x)}<line x1={X(m.cut.x)} x2={X(m.cut.x)} y1="4" y2={H - 4} class="cut" />{/if}
			{#each m.near.slice(0, 3) as x (x.j)}<line x1={X(m.xy[m.focus][0])} y1={Y(m.xy[m.focus][1])} x2={X(m.xy[x.j][0])} y2={Y(m.xy[x.j][1])} class="nn" />{/each}
			{#each order as i (i)}
				{@const [a, b] = m.xy[i]}
				{@const wrong = m.leaders && m.st.wrong.includes(i)}
				<g transform="translate({X(a)},{Y(b)})" class:f={i === m.focus}>
					{#if m.group.includes(i) || (m.leaders && (i === 0 || i === 33))}<circle r="11" class="ring" />{/if}
					<circle r={i === m.focus ? 9.5 : 8} style:fill={fac(i)} class="pt" />
					<text y="3.5">{i}</text>
					{#if wrong}<text y="-11" class="x">✕</text>{/if}
				</g>
			{/each}
		</svg>
		<p class="small muted">
			{t({ en: `step ${fmtInt(m.step)}`, zh: `第 ${fmtInt(m.step)} 步` })}{#if Number.isFinite(m.loss)}{' · '}{t({ en: 'average loss', zh: '平均损失' })} {fmt3(m.loss)}{/if}{' · '}{t({ en: `p = ${m.g.p}, q = ${m.g.q}`, zh: `p = ${m.g.p}，q = ${m.g.q}` })} · {t({ en: `the map keeps ${fmtPct(m.pca.share)} of the spread`, zh: `这张图保留了 ${fmtPct(m.pca.share)} 的分布` })}{' · '}{t({ en: `nearest member on the same side: ${m.st.same} of 34`, zh: `最近的成员同一方：34 人中 ${m.st.same} 人` })}
			{#if P.props.animate}<button type="button" class="btn ghost" onclick={replay}>{t({ en: 'Replay the training', zh: '重放训练过程' })}</button>{/if}
		</p>
		{#if m.showCut}<p class="small">{t({ en: `The dashed line: one vertical cut puts ${m.cut.ok} of 34 members on their own side.`, zh: `虚线：竖着切这一刀，34 个成员中有 ${m.cut.ok} 个落在自己一方。` })}</p>{/if}
		{#if Number.isFinite(m.far)}<p class="small">{t({ en: `Walks end ${fmt3(m.far)} hops from their start on average; each member’s 3 nearest members are ${fmt3(m.st.hop)} hops away.`, zh: `游走结束时平均离起点 ${fmt3(m.far)} 跳；每个成员最近的 3 个成员平均相隔 ${fmt3(m.st.hop)} 跳。` })}</p>{/if}
		{#if m.twin}<p class="small">{t({ en: `Ringed: the members with the same friends, mean cosine similarity ${fmt3(m.twin.g)} (two random members: ${fmt3(m.twin.all)}).`, zh: `加圈的：朋友完全相同的成员，平均余弦相似度 ${fmt3(m.twin.g)}（随便两个成员：${fmt3(m.twin.all)}）。` })}</p>{/if}
		{#if m.leaders}<p class="small">{t({ en: `✕ sits closer to the other side’s leader: ${m.st.wrong.length ? m.st.wrong.join(', ') : 'nobody'} (${m.st.right} of 32 right)`, zh: `✕ 离另一方的领头人更近：${m.st.wrong.length ? m.st.wrong.join('、') : '没有人'}（32 人中 ${m.st.right} 人正确）` })}</p>{/if}
		{#if m.list && m.focus >= 0}
			<table class="num-table small">
				<thead><tr><th>{t({ en: `nearest to ${m.focus}`, zh: `离 ${m.focus} 最近` })}</th><th>{t({ en: 'cosine', zh: '余弦相似度' })}</th><th>{t({ en: 'hops', zh: '跳数' })}</th><th>{t({ en: 'side', zh: '一方' })}</th></tr></thead>
				<tbody>{#each m.near as x (x.j)}<tr><td class="num">{x.j}</td><td class="num">{fmt3(x.cos)}</td><td class="num">{m.st.H[m.focus][x.j] < 0 ? '–' : m.st.H[m.focus][x.j]}</td><td>{side(FACTIONS[x.j])}</td></tr>{/each}</tbody>
			</table>
		{/if}
	{/if}
</div>

<style>
	.map { display: grid; gap: 4px; min-width: 0; align-content: start; } svg { width: 100%; height: auto; max-height: 420px; display: block; } .small { font-size: var(--fs-sm); } p { margin: 0; }
	.pt { stroke: var(--ink-2); stroke-width: 0.8; } .f .pt { stroke: var(--ink); stroke-width: 2.4; } text { font: 600 9px var(--font-mono); text-anchor: middle; fill: var(--ink); pointer-events: none; }
	.x { font-size: 11px; fill: var(--ink); } .ring { fill: none; stroke: var(--ink); stroke-width: 1.2; stroke-dasharray: 3 2; } .nn { stroke: var(--accent); stroke-width: 1.6; stroke-dasharray: 4 3; } .cut { stroke: var(--ink-2); stroke-width: 1.2; stroke-dasharray: 6 4; }
	table { border-collapse: collapse; } th, td { padding: 1px 8px; text-align: left; } th { font-weight: 600; } .btn { min-height: 28px; padding: 0 8px; font-size: var(--fs-xs); }
</style>
