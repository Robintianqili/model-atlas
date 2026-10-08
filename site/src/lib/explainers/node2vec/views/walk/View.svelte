<!-- node2vec 'walk': the club and its walks (see meta.ts). Members are coloured by the side they joined (blue Mr. Hi,
     orange the Officer); the walk's edges are outlined, its newest move thick. In 'step', v is filled, t half-filled,
     every friend of v marked with its weight, the others dimmed, and a table gives each friend's chance and the draw. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import Club from '../../../../atlas/adapters/graph/Club.svelte';
	import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
	import { FACTION_NAMES } from '../../../../atlas/data/karate.ts';
	import { FACTIONS, WIN } from '../../walks.ts';
	import { kindText, walkPic } from './meta.ts';

	let P: StageProps = $props();
	const s = $derived(walkPic(P));
	const fac = (i: number, pct = 55) => `color-mix(in srgb, var(${FACTIONS[i] ? '--div-pos' : '--div-neg'}) ${pct}%, var(--surface))`;
	const link = (a: number, b: number, x: number, y: number) => (a === x && b === y) || (a === y && b === x);
	const onPath = (a: number, b: number) => s.path.some((x, i) => i > 0 && link(a, b, s.path[i - 1], x));
	const isNow = (a: number, b: number) => s.path.length > 1 && link(a, b, s.path[s.path.length - 2], s.path[s.path.length - 1]);
	const cand = (i: number) => s.step?.x.indexOf(i) ?? -1;
	const fill = (i: number) =>
		s.step ? (i === s.step.v ? 'var(--accent)' : i === s.step.t ? 'color-mix(in srgb, var(--accent) 40%, var(--surface))' : cand(i) >= 0 ? fac(i) : undefined) : s.mode === 'club' || s.path.includes(i) ? fac(i) : undefined;
	const edge = (a: number, b: number) => {
		if (s.step) return (a === s.step.v && cand(b) >= 0) || (b === s.step.v && cand(a) >= 0) || onPath(a, b) ? 'on' : '';
		return s.mode === 'walks' && isNow(a, b) ? 'now' : onPath(a, b) ? 'on' : '';
	};
</script>

{#snippet walkRow(ids: number[], hi: number)}<span class="walk">{#each ids as id, k (k)}<span class="m" class:c={k === hi} class:w={hi >= 0 && k !== hi && Math.abs(k - hi) <= WIN} style:background={fac(id, 30)}>{id}</span>{/each}</span>{/snippet}

<div class="wv" style:min-height="360px">
	<Club
		g={s.g}
		focus={s.mode === 'club' ? s.g.focus : -1}
		{fill}
		ring={(i) => (s.step ? i === s.step.x[s.step.xi] : s.mode === 'walks' && i === s.path[0])}
		mark={(i) => (cand(i) >= 0 ? `×${s.step!.a[cand(i)]}` : '')}
		{edge}
		dim={(i) => !!s.step && cand(i) < 0 && i !== s.step.v && i !== s.step.t}
		label={{ en: 'The club with the walk', zh: '俱乐部和游走' }}
	/>
	{#if s.step}
		{@const st = s.step}
		<div class="scroll" use:keyScroll>
			<table class="num-table small">
				<thead><tr><th>{t({ en: `friend x of ${st.v}`, zh: `${st.v} 的朋友 x` })}</th><th>{t({ en: 'kind', zh: '类别' })}</th><th>{t({ en: 'weight α', zh: '权重 α' })}</th><th>{t({ en: 'running sum', zh: '累计' })}</th><th>{t({ en: 'chance', zh: '概率' })}</th></tr></thead>
				<tbody>
					{#each st.x as x, i (x)}
						<tr class:pick={i === st.xi}><td class="num">{x}</td><td>{t(kindText(st.t, st.kind[i]))}</td><td class="num">{st.a[i]}</td><td class="num">{s.cum[i]}</td><td class="num"><span class="bar" style:width="{Math.round(st.prob[i] * 100)}px"></span> {fmtPct(st.prob[i])}</td></tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="small num">{t({ en: 'draw', zh: '抽样' })} u = {fmt3(st.u)} · u × {st.total} = {fmt3(st.ux)} → {t({ en: 'the first running sum above it: member', zh: '第一个超过它的累计：成员' })} {st.x[st.xi]}</p>
	{:else if s.mode === 'walks' && s.path.length}
		<p class="small">{t({ en: `The ${s.same.length} walks from member ${s.path[0]}, one per round (a walk is a sentence of member ids):`, zh: `从成员 ${s.path[0]} 出发的 ${s.same.length} 条游走，每轮一条（一条游走就是由成员编号组成的句子）：` })}</p>
		<p class="small">{t({ en: `With p = ${s.g.p} and q = ${s.g.q}, a walk ends on average ${fmt3(s.W.far)} hops from its start and meets ${fmt3(s.W.distinct)} different members.`, zh: `p = ${s.g.p}、q = ${s.g.q} 时，一条游走结束时平均离起点 ${fmt3(s.W.far)} 跳，遇到 ${fmt3(s.W.distinct)} 个不同的成员。` })}</p>
		<div class="walks">{#each s.same as x (x.k)}<div class:cur={x.k === s.w}>{@render walkRow(x.k === s.w ? s.path : x.ids, -1)}</div>{/each}</div>
	{:else if s.mode === 'pairs' && s.path.length}
		<div class="walks"><div class="cur">{@render walkRow(s.path, WIN)}</div></div>
		<p class="small list"><strong>{t({ en: `Pairs of member ${s.c} here:`, zh: `成员 ${s.c} 在这里的训练对：` })}</strong>{#each s.path as o, k (k)}{#if k !== WIN && Math.abs(k - WIN) <= WIN}<span class="num">({s.c}, {o})</span>{/if}{/each}</p>
		<p class="small">{t({ en: `All the walks give ${fmtInt(s.W.pairs.length)} (center, context) pairs.`, zh: `全部游走一共给出 ${fmtInt(s.W.pairs.length)} 个（中心，上下文）训练对。` })}</p>
		<p class="small muted list"><span>{t({ en: `Over all walks, member ${s.c}’s most frequent other context members:`, zh: `在全部游走中，成员 ${s.c} 最常见的其他上下文成员：` })}</span>{#each s.top as x (x.j)}<span>{t({ en: `${x.j} (${x.n} times)`, zh: `${x.j}（${x.n} 次）` })}</span>{/each}</p>
	{:else if s.mode === 'club'}
		<p class="legend small">{#each FACTION_NAMES as f, c (c)}<span><i style:background={fac(c ? 33 : 0)}></i>{t({ en: `${f.en}’s side (${FACTIONS.filter((x) => x === c).length})`, zh: `${f.zh}一方（${FACTIONS.filter((x) => x === c).length} 人）` })}</span>{/each}</p>
	{/if}
</div>

<style>
	.wv { display: grid; gap: 8px; min-width: 0; align-content: start; } p { margin: 0; } .small { font-size: var(--fs-sm); } .scroll { overflow-x: auto; }
	table { border-collapse: collapse; } th, td { padding: 1px 6px; text-align: left; white-space: nowrap; } th { font-weight: 600; } tr.pick td { background: var(--accent-soft); font-weight: 700; }
	.bar { display: inline-block; height: 10px; background: var(--seq-mid); border-radius: 2px; vertical-align: middle; }
	/* the walk being built: a solid accent bar; the others a dashed rule (no opacity on text: AK604 contrast) */
	.walks { display: grid; gap: 3px; } .walks > div { padding-left: 5px; border-left: 3px dashed var(--rule); } .walks > .cur { border-left: 3px solid var(--accent); } .walk { display: inline-flex; flex-wrap: wrap; gap: 2px; }
	.m { min-width: 22px; padding: 0 3px; text-align: center; border: 1px solid transparent; border-radius: var(--r1); font: 600 var(--fs-xs) var(--font-mono); }
	.m.c { border-color: var(--ink); outline: 2px solid var(--accent); } .m.w { border-color: var(--ink); border-style: dashed; }
	.list { display: flex; flex-wrap: wrap; gap: 2px 10px; } .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; } .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
	@media (max-width: 560px) { th, td { padding: 1px 3px; } th { white-space: normal; } }
</style>
