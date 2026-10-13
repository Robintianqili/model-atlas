<!-- behavior-cloning 'board' (meta.ts lists the props): the field, the demonstrations, the dataset, the policy's arrows,
     the episodes of a roll tape and the forward pass at the probed cell; 'edit' adds the demonstration recorder. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { ARROWS, DIR_NAMES, featuresOf, GOAL, OPEN, PIT, step, toRaw } from '../../world.ts';
	import DemoPad from '../../adapters/demos/DemoPad.svelte';
	import { boardOf } from './meta.ts';

	let P: StageProps = $props();
	const show = $derived(new Set(String(P.props.show ?? '').split(',')));
	const isRoll = $derived(P.source === 'roll' || P.source === 'droll');
	onMount(() => {
		const src = P.props.policy === 'dagger' ? 'dagger' : 'train';
		if (!P.rt.sources[src]?.tape) void P.rt.ensure(src, 'last');
	});
	const b = $derived(boardOf(P.rt, P.props, isRoll ? { tape: P.at.tape, index: P.at.index } : undefined));
	const W = $derived(b.W);
	const visits = $derived.by(() => {
		const v = W.cell.map(() => 0);
		if (show.has('roll')) for (const e of b.eps) for (const s of e.cells.slice(1)) v[s]++;
		return v;
	});
	const heat = $derived(makeScale('sequential', [0, Math.max(1, ...visits)], themeState.resolved));
	const sig = $derived(makeScale('diverging', [-1, 1], themeState.resolved));
	const ctr = (s: number) => [(s % W.w) * 100 + 50, Math.floor(s / W.w) * 100 + 50];
	const ARROW = 'M0,-30 L12,-12 L4,-12 L4,20 L-4,20 L-4,-12 L-12,-12 Z';
	const demoPaths = $derived(b.raw.demos.map((dm) => dm.moves.reduce((c, a) => [...c, step(W, c[c.length - 1], a)], [dm.start])));
	const cur = $derived(isRoll && P.at.frame?.phase === 'move' ? P.at.frame.metrics!.s2 : -1);
	const outcomes = $derived({ goal: b.eps.filter((e) => e.outcome === 1).length, pit: b.eps.filter((e) => e.outcome === -1).length, out: b.eps.filter((e) => e.outcome === 0).length });
	const setProbe = (a: number) => P.setInput?.({ ...toRaw(P.rt.input), probe: step(W, b.raw.probe, a) });
	const OUT = { 1: 'goal', '-1': 'pit', 0: 'out' } as Record<string, string>;
</script>

<div class="bd">
	<svg viewBox="-4 -4 {W.w * 100 + 8} {W.h * 100 + 8}" style:max-width="{W.w * 64}px" role="img" aria-label={t({ en: 'The windy field', zh: '刮风的空地' })}>
		{#each W.cell as c, s (s)}
			{@const x = (s % W.w) * 100}
			{@const y = Math.floor(s / W.w) * 100}
			<rect {x} {y} width="100" height="100" class="c c{c}" style:fill={c === OPEN && show.has('roll') && visits[s] ? heat.css(visits[s]) : null} />
			{#if c === GOAL}<text x={x + 50} y={y + 64} class="lbl g">G</text>{:else if c === PIT}<text x={x + 50} y={y + 64} class="lbl p">×</text>{/if}
			{#if s === W.start}<text x={x + 8} y={y + 26} class="s">S</text>{/if}
			{#if show.has('data') && b.dataCells.has(s)}<rect x={x + 7} y={y + 7} width="86" height="86" class="data" class:added={b.added.has(s)} />{/if}
			{#if show.has('policy') && b.moves && b.moves[s] >= 0}<path d={ARROW} transform="translate({x + 50},{y + 50}) rotate({b.moves[s] * 90}) scale(0.8)" class="arr" class:bad={show.has('wrong') && b.wrong.includes(s)} />{/if}
			{#if show.has('probe') && s === b.raw.probe}<rect x={x + 3} y={y + 3} width="94" height="94" class="probe" />{/if}
		{/each}
		{#if show.has('demos')}
			{#each demoPaths as cells, i (i)}<polyline points={cells.map((s) => ctr(s).join(',')).join(' ')} class="demo" />{/each}
		{/if}
		{#if show.has('roll')}
			{#each b.eps as e, i (i)}<polyline points={e.cells.map((s) => { const [x, y] = ctr(s); return `${x + ((i % 5) - 2) * 6},${y + ((i % 3) - 1) * 6}`; }).join(' ')} class="ep {OUT[String(e.outcome)] ?? 'live'}" />{/each}
			{#if cur >= 0}<circle cx={ctr(cur)[0]} cy={ctr(cur)[1]} r="16" class="agent" />{/if}
		{/if}
	</svg>

	{#if show.has('roll') && b.eps.length}
		<p class="num small">{t({ en: `Episodes shown: ${b.eps.length} · goal ${outcomes.goal} · pit ${outcomes.pit} · out of moves ${outcomes.out}`, zh: `显示的回合：${b.eps.length} · 到终点 ${outcomes.goal} · 掉进陷阱 ${outcomes.pit} · 步数用完 ${outcomes.out}` })}</p>
		{#if isRoll && P.at.frame?.phase === 'move'}{@const m = P.at.frame.metrics!}<p class="num small">{t({ en: `Episode ${m.e + 1}, move ${m.t + 1}: the policy chose ${DIR_NAMES[m.a].en}${m.gust ? ', but a gust blew it down' : ''}; this cell ${m.seen ? 'is' : 'is not'} in the dataset.`, zh: `第 ${m.e + 1} 个回合第 ${m.t + 1} 步：策略选了${DIR_NAMES[m.a].zh}${m.gust ? '，但一阵风把它吹了下去' : ''}；这个格子${m.seen ? '在' : '不在'}数据集里。` })}</p>{/if}
	{/if}

	{#if show.has('probe') && b.probe}
		{@const f = featuresOf(W, b.raw.probe)}
		<div class="fwd" role="group" aria-label={t({ en: 'The forward pass at the outlined cell', zh: '外框格子上的前向传播' })}>
			<p class="num small">{t({ en: 'input', zh: '输入' })} x = ({fmt3(f[0])}, {fmt3(f[1])})</p>
			<div class="strip" aria-label={t({ en: `hidden layer: ${b.probe.h.length} numbers between −1 and 1`, zh: `隐藏层：${b.probe.h.length} 个介于 −1 和 1 之间的数` })} role="img">
				{#each b.probe.h as v, j (j)}<span style:background={sig.css(v)} title={fmt3(v)}></span>{/each}
			</div>
			<div class="bars">
				{#each b.probe.p as p, k (k)}
					<span class="nm">{ARROWS[k]} {t(DIR_NAMES[k])}</span>
					<span class="bar"><span style:width="{p * 100}%" style:background={makeScale('sequential', [0, 1], themeState.resolved).css(p)}></span></span>
					<span class="num">z {fmt3(b.probe.z[k])} · p {fmt3(p)}</span>
				{/each}
			</div>
			<div class="row" role="group" aria-label={t({ en: 'Move the outlined cell', zh: '移动外框格子' })}>
				{#each ARROWS as ar, a (a)}<button type="button" class="btn" disabled={!P.setInput} onclick={() => setProbe(a)} aria-label={t({ en: `Probe the cell ${DIR_NAMES[a].en}`, zh: `探查${DIR_NAMES[a].zh}边的格子` })}>{ar}</button>{/each}
			</div>
		</div>
	{/if}

	{#if show.has('edit')}<DemoPad raw={b.raw} onedit={(r) => P.setInput?.(r)} />{/if}

	<ul class="legend small muted">
		{#if show.has('demos')}<li><span class="key demo"></span>{t({ en: 'your demonstrations', zh: '你的示范' })}</li>{/if}
		{#if show.has('data')}<li><span class="key data"></span>{t({ en: 'a cell in the dataset', zh: '数据集里的格子' })}{#if b.added.size}; <span class="key data added"></span>{t({ en: 'added by DAgger', zh: '由 DAgger 加入' })}{/if}</li>{/if}
		{#if show.has('policy')}<li>{t({ en: 'arrow: the policy’s move (the highest score)', zh: '箭头：策略的动作（分数最高的那个）' })}{#if show.has('wrong')}; <span class="key bad"></span>{t({ en: 'an arrow that does not lead closer to the goal', zh: '不能让智能体更接近终点的箭头' })}{/if}</li>{/if}
		{#if show.has('roll')}<li>{t({ en: 'episodes:', zh: '回合：' })} <span class="key line goal"></span>{t({ en: 'reached the goal', zh: '到达终点' })}, <span class="key line pit"></span>{t({ en: 'fell in a pit', zh: '掉进陷阱' })}, <span class="key line out"></span>{t({ en: 'ran out of moves', zh: '步数用完' })}; {t({ en: 'stronger cell color: visited more often', zh: '格子颜色越浓：被到访越多' })}</li>{/if}
		{#if show.has('probe')}<li>{t({ en: `outlined: the probed cell; strip: its ${b.probe?.h.length ?? 0} hidden numbers (orange above 0, blue below)`, zh: `外框：探查的格子；色条：它的 ${b.probe?.h.length ?? 0} 个隐藏层数值（橙色大于 0，蓝色小于 0）` })}</li>{/if}
	</ul>
</div>

<style>
	.bd { display: grid; gap: 8px; grid-template-columns: minmax(0, 1fr); min-width: 0; justify-items: center; }
	svg { width: 100%; height: auto; display: block; }
	.c { stroke: var(--rule); stroke-width: 2; fill: var(--surface); } .c1 { fill: var(--ink-2); } .c2 { fill: var(--live-soft); }
	.c3 { fill: color-mix(in srgb, var(--div-neg) 25%, var(--surface)); }
	.lbl { font: 700 34px var(--font-mono); text-anchor: middle; } .g { fill: var(--live); } .p { fill: var(--div-neg); }
	.s { font: 700 22px var(--font-mono); fill: var(--accent-strong); }
	.data { fill: none; stroke: var(--accent); stroke-width: 5; } .data.added { stroke-dasharray: 12 8; stroke: var(--pre, var(--accent)); }
	.probe { fill: none; stroke: var(--ink); stroke-width: 6; }
	.arr { fill: var(--ink); } .arr.bad { fill: var(--div-neg); }
	.demo { fill: none; stroke: var(--accent); stroke-width: 10; stroke-linejoin: round; stroke-linecap: round; opacity: 0.45; }
	.ep { fill: none; stroke-width: 4; stroke-linejoin: round; opacity: 0.6; } .ep.goal { stroke: var(--live); } .ep.pit { stroke: var(--div-neg); }
	.ep.out { stroke: var(--ink-2); stroke-dasharray: 10 8; } .ep.live { stroke: var(--accent); }
	.agent { fill: var(--accent); stroke: var(--surface); stroke-width: 5; }
	.small { margin: 0; font-size: var(--fs-xs); }
	.fwd { display: grid; gap: 6px; width: 100%; max-width: 30rem; }
	.strip { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr); gap: 1px; height: 18px; border: 1px solid var(--rule); }
	.bars { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 2px 8px; align-items: center; font-size: var(--fs-xs); }
	.bar { height: 10px; background: var(--surface-2); border: 1px solid var(--rule); } .bar span { display: block; height: 100%; }
	.row { display: flex; gap: 4px; flex-wrap: wrap; } .btn { min-height: 32px; min-width: 36px; padding: 2px 8px; font-size: var(--fs-xs); }
	.legend { margin: 0; padding: 0; list-style: none; display: grid; gap: 2px; width: 100%; }
	.key { display: inline-block; width: 1.4em; height: 0.8em; margin-right: 4px; vertical-align: middle; }
	.key.demo { background: color-mix(in srgb, var(--accent) 45%, transparent); } .key.data { border: 2px solid var(--accent); }
	.key.data.added { border-style: dashed; } .key.bad { background: var(--div-neg); height: 0.6em; }
	.key.line { height: 0; border-top: 3px solid; } .key.line.goal { border-color: var(--live); } .key.line.pit { border-color: var(--div-neg); } .key.line.out { border-color: var(--ink-2); border-top-style: dashed; }
</style>
