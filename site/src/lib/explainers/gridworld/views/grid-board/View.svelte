<!-- gridworld 'grid-board'. props.show (comma list): tri (Q triangles), heat (values), arrows, act (the choice, u against ε),
     move, update (with props.formula 'target' | 'update'), path (this episode), answer (rings where value iteration
     disagrees), edit (maze tools). props.pair: two vi sources side by side. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { Frame, StageProps } from '../../../../atlas/types/index.ts';
	import { alignTex, texValues } from '../../../../atlas/ui/views/kit.parts/tex.ts';
	import { fmt3, minus } from '../../../../atlas/core/format.ts';
	import { agreement, arrowOf, DIR_NAMES, GOAL, goalDistance, OPEN, PIT, toRaw, valueIteration } from '../../world.ts';
	import { edit } from '../../bus.ts';
	import Editor from '../../adapters/gridworld/Editor.svelte';
	import { boardAt, valueOf, type Board } from './meta.ts';

	let P: StageProps = $props();
	const show = $derived(new Set(String(P.props.show ?? '').split(',')));
	const m = $derived(P.at.frame?.metrics ?? {});
	const ph = $derived(P.at.frame?.phase);
	const pair = $derived((P.props.pair as string[] | undefined) ?? []);
	onMount(() => pair.forEach((s) => P.rt.sources[s]?.tape || void P.rt.ensure(s, 'last')));
	const boards = $derived(pair.length ? pair.map((s) => boardAt(P.rt.sources[s]?.tape, (P.rt.sources[s]?.tape?.length ?? 0) - 1)) : [boardAt(P.at.tape, P.at.index)]);
	/** the cells of this episode so far (its act frames up to this one; an end frame closes episode `step`) */
	const path = $derived.by(() => {
		const tp = P.at.tape, f = P.at.frame;
		if (!show.has('path') || !tp || !f) return [];
		const ep = f.phase === 'end' ? f.step : f.sub, out: number[] = [];
		for (let j = P.at.index - (f.phase === 'end' ? 1 : 0); j > 0 && tp.frame(j).sub === ep; j--) if (tp.frame(j).phase === 'act') out.unshift(tp.frame(j).metrics!.s2);
		return [tp.frame(0).metrics!.start, ...out];
	});
	const b0 = $derived(boards[0]);
	const wrong = $derived(new Set(show.has('answer') && b0?.Q ? agreement(b0, b0.Q, valueIteration(b0, b0.gamma), b0.gamma).wrong : []));
	const formula = $derived.by(() => {
		const f = P.at.frame;
		const phase = f ? (P.at.tape?.phases?.[f.phase] as { formula?: { tex: string; values: (x: Frame) => Record<string, number> } } | undefined) : undefined;
		if (!P.props.formula || !f || !phase?.formula) return null;
		const parts = phase.formula.tex.split(';\\quad');
		return { tex: alignTex(P.props.formula === 'target' ? parts[0] : parts[parts.length - 1]), values: texValues(phase.formula.values(f)) };
	});
	const fill = (v: number) => (v === 0 ? 'var(--surface)' : `color-mix(in srgb, var(${v > 0 ? '--div-pos' : '--div-neg'}) ${Math.round(Math.min(1, Math.sqrt(Math.abs(v))) * 80)}%, var(--surface))`);
	const corners = (x: number, y: number) => [[x, y], [x + 100, y], [x + 100, y + 100], [x, y + 100]];
	const tri = (x: number, y: number, a: number) => [corners(x, y)[a], corners(x, y)[(a + 1) % 4], [x + 50, y + 50]].join(' ');
	const ctr = (b: Board, s: number) => [(s % b.w) * 100 + 50, Math.floor(s / b.w) * 100 + 50];
	const ARROW = 'M0,-34 L13,-14 L4,-14 L4,22 L-4,22 L-4,-14 L-13,-14 Z';
	const agentAt = $derived(ph === 'act' ? (show.has('move') ? m.s2 : m.s) : ph === 'update' ? m.s : boards[0]?.pos);
	/** a cell's value: 2 digits, never in e-notation (a first-year reads 0.0020, not 2.0e−3); tiny values as <0.001 */
	const cellNum = (v: number) => (v !== 0 && Math.abs(v) < 0.001 ? (v > 0 ? '<0.001' : `>${minus('-0.001')}`) : fmt3(v, 2));
	const dec = (v: number) => (v === 0 || Math.abs(v) >= 1e-6 ? minus(String(Number(v.toPrecision(3)))) : fmt3(v));
	const actor = $derived(show.has('act') || show.has('move') || show.has('path') || show.has('update'));
</script>

{#snippet board(b: Board)}
	<svg viewBox="-4 -4 {b.w * 100 + 8} {b.h * 100 + 8}" style:max-width="{b.w * 76}px" role="img" aria-label={t({ en: 'The maze', zh: '迷宫' })}>
		{#each b.cell as c, s (s)}
			{@const x = (s % b.w) * 100}
			{@const y = Math.floor(s / b.w) * 100}
			<rect {x} {y} width="100" height="100" class="c c{c}" style:fill={c === OPEN && show.has('heat') ? fill(valueOf(b, s)) : null} />
			{#if c === OPEN && show.has('tri') && b.Q}
				{#each [0, 1, 2, 3] as a (a)}<polygon points={tri(x, y, a)} style:fill={fill(b.Q[s * 4 + a])} class="tri" class:hot={show.has('update') && ph === 'update' && m.s === s && m.a === a} />{/each}
			{/if}
			{#if c === GOAL}<text x={x + 50} y={y + 62} class="lbl g">+1</text>{:else if c === PIT}<text x={x + 50} y={y + 62} class="lbl p">−1</text>{/if}
			{#if s === b.start}<text x={x + 8} y={y + 26} class="s">S</text>{/if}
			{#if c === OPEN && show.has('heat')}<text x={x + 50} y={y + 90} class="v">{cellNum(valueOf(b, s))}</text>{/if}
			{#if c === OPEN && show.has('arrows') && b.Q && arrowOf(b.Q, s) >= 0}<path d={ARROW} transform="translate({x + 50},{y + 46}) rotate({arrowOf(b.Q, s) * 90}) scale(0.9)" class="arr" class:bad={wrong.has(s)} />{/if}
			{#if wrong.has(s)}<circle cx={x + 50} cy={y + 50} r="44" class="ring" />{/if}
		{/each}
		{#if path.length > 1}<polyline points={path.map((s) => ctr(b, s).join(',')).join(' ')} class="path" />{/if}
		{#if show.has('update') && ph === 'update'}<rect x={(m.s2 % b.w) * 100 + 6} y={Math.floor(m.s2 / b.w) * 100 + 6} width="88" height="88" class="next" />{/if}
		{#if agentAt !== undefined && actor}
			{#if show.has('move') && ph === 'act'}<circle cx={ctr(b, m.s)[0]} cy={ctr(b, m.s)[1]} r="14" class="ghost" />{/if}
			{#if show.has('act') && ph === 'act'}<path d={ARROW} transform="translate({ctr(b, m.s).join(',')}) rotate({m.a * 90}) translate(0,-30)" class="arr go" class:explore={m.explore === 1} />{/if}
			<circle cx={ctr(b, agentAt)[0]} cy={ctr(b, agentAt)[1]} r="17" class="agent" />
		{/if}
	</svg>
{/snippet}

<div class="gb">
	<div class="boards" class:two={boards.length > 1}>
		{#each boards as b, i (i)}
			{#if b}
				<figure>
					{@render board(b)}
					{#if pair.length}{@const d = goalDistance(b)}<figcaption class="num">γ = {+b.gamma.toFixed(4)} · {t({ en: 'the start is worth', zh: '起点的价值' })} {#if d > 1}{+b.gamma.toFixed(4)}<sup>{d - 1}</sup>{' = '}{/if}{dec(valueOf(b, b.start))}</figcaption>{/if}
				</figure>
			{:else}<p class="muted">{t({ en: 'Computing…', zh: '计算中…' })}</p>{/if}
		{/each}
	</div>
	{#if show.has('act') && ph === 'act'}
		<div class="gauge" aria-hidden="true"><span class="eps" style:width="{m.eps * 100}%"></span><span class="u" style:left="{m.u * 100}%"></span></div>
		<p class="num small">u = {fmt3(m.u)} {m.u < m.eps ? '<' : '≥'} ε = {+m.eps.toFixed(4)} → {t(m.explore ? { en: 'explore: a random move', zh: '探索：随机走一步' } : m.ties > 1 ? { en: `exploit: ${m.ties} moves share the best Q, one picked at random`, zh: `利用：有 ${m.ties} 个动作的 Q 并列最大，随机挑一个` } : { en: 'exploit: the move with the best Q', zh: '利用：选 Q 最大的动作' })}</p>
		<p class="num small">Q(s, ·): {[0, 1, 2, 3].map((a) => `${t(DIR_NAMES[a])} ${fmt3(m[`q${a}`])}`).join(' · ')}</p>
	{/if}
	{#if show.has('move') && ph === 'act'}<p class="num small">s = ({m.x}, {m.y}) → s′ = ({m.x2}, {m.y2}) · r = {m.r}</p>{/if}
	{#if formula}<FormulaCard tex={formula.tex} values={formula.values} provenance={P.provenance} />{/if}
	{#if show.has('tri')}<p class="muted small">{t({ en: 'Each cell: four triangles, one per move (up, right, down, left); orange = positive Q, blue = negative, no color = 0.', zh: '每个格子：四个三角形，各对应一个动作（上、右、下、左）；橙色表示 Q 为正，蓝色为负，无色为 0。' })}</p>{/if}
	{#if show.has('arrows')}<p class="muted small">{t({ en: 'Arrow: the move with the highest Q; no arrow where two or more moves tie.', zh: '箭头：Q 最大的动作；有两个或更多动作并列时不画箭头。' })}</p>{/if}
	{#if show.has('answer') && wrong.size}<p class="muted small">{t({ en: 'Rings: arrows that value iteration says are not a best move.', zh: '圆圈：价值迭代认为不是最佳动作的箭头。' })}</p>{/if}
	{#if show.has('edit')}<Editor raw={toRaw(P.rt.input)} onedit={(r) => edit(() => r) || P.rt.setInput(r)} />{/if}
</div>

<style>
	.gb { display: grid; gap: 10px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	.boards { display: grid; gap: 12px; grid-template-columns: minmax(0, 1fr); }
	.boards.two { grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); }
	figure { margin: 0; display: grid; gap: 4px; justify-items: center; } svg { width: 100%; height: auto; display: block; }
	.c { stroke: var(--rule); stroke-width: 2; fill: var(--surface); } .c1 { fill: var(--ink-2); } .c2 { fill: var(--live-soft); }
	.c3 { fill: color-mix(in srgb, var(--div-neg) 25%, var(--surface)); } .tri { stroke: var(--rule); stroke-width: 1; }
	.tri.hot { stroke: var(--ink); stroke-width: 6; } .lbl { font: 700 30px var(--font-mono); text-anchor: middle; } .g { fill: var(--live); }
	.p { fill: var(--div-neg); } .s { font: 700 22px var(--font-mono); fill: var(--accent-strong); }
	.v { font: 20px var(--font-mono); text-anchor: middle; fill: var(--ink); } .arr { fill: var(--ink); opacity: 0.8; } .arr.bad { fill: var(--div-neg); }
	.arr.go { fill: var(--accent); opacity: 1; } .arr.go.explore { fill: var(--pre); }
	.ring { fill: none; stroke: var(--div-neg); stroke-width: 4; stroke-dasharray: 8 6; }
	.path { fill: none; stroke: var(--accent); stroke-width: 7; stroke-linejoin: round; opacity: 0.55; }
	.next { fill: none; stroke: var(--ink); stroke-width: 4; stroke-dasharray: 10 8; }
	.agent { fill: var(--accent); stroke: var(--surface); stroke-width: 5; }
	.ghost { fill: none; stroke: var(--accent); stroke-width: 4; stroke-dasharray: 6 5; }
	.gauge { position: relative; height: 14px; border-radius: 7px; background: var(--surface-2); border: 1px solid var(--rule); overflow: hidden; }
	.gauge .eps { position: absolute; inset: 0 auto 0 0; background: color-mix(in srgb, var(--pre) 45%, transparent); }
	.gauge .u { position: absolute; top: 0; bottom: 0; width: 3px; margin-left: -1px; background: var(--ink); }
	.small { margin: 0; font-size: var(--fs-xs); } figcaption { font-size: var(--fs-xs); }
</style>
