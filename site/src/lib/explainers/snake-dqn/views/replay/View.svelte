<!-- snake-dqn 'replay': the replay buffer of the `replay` process. One chip per slot (the move's number, the action, the
     reward); the slot written by this frame is ringed; after the draw, the drawn slots are filled and the batch table
     gives each drawn move's Q(s, a), the target y and the loss (y − Q)². -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import Board from '../../parts/Board.svelte';
	import { ACTION, L } from '../../rules.ts';
	import { m, num, stateOf } from '../../picture.ts';

	let P: StageProps = $props();
	const s = $derived(stateOf(P.at));
	const ph = $derived(P.at.frame?.phase ?? '');
	const buf = $derived(s.buf ?? []);
	const cap = $derived((P.at.tape?.opt as { cap?: number } | undefined)?.cap ?? buf.length);
	const drawn = $derived(ph === 'store' ? [] : (s.pick ?? []));
	const times = (i: number) => drawn.filter((d) => d === i).length;
	const rows = $derived(ph === 'learn' ? (s.rows ?? []) : []);
	const ARR = ['↑', '↶', '↷'];
	const sg = (r: number) => (r > 0 ? '+1' : r < 0 ? '−1' : '0');
	const say = (o: { en: string; zh: string }) => t(o);
</script>

<div class="rv">
	<div class="top">
		{#if s.game}<Board g={s.game} small label={say(L('The game the agent is playing', '智能体正在玩的这一局'))} />{/if}
		<div class="side">
			<p class="lab">{say(L(`The buffer: ${buf.length} of ${cap} slots filled`, `缓冲区：${cap} 个格位里存了 ${buf.length} 个`))}</p>
			<ol class="slots num" aria-label={say(L('Slots of the buffer', '缓冲区的格位'))}>
				{#each buf as x, i (i)}
					<li class:now={ph === 'store' && i === m(P.at, 'slot')} class:drawn={times(i) > 0} class:pos={x.r > 0} class:neg={x.r < 0}>
						<span>#{x.t + 1}</span> {ARR[x.a]} {sg(x.r)}{#if times(i) > 1}<sup>×{times(i)}</sup>{/if}
					</li>
				{/each}
			</ol>
			<p class="small">{say(L('Each chip: the move’s number in play, its action (↑ straight on, ↶ left, ↷ right) and its reward. Ringed: just stored. Filled: drawn for the batch.', '每个小块：这一步在对局里的编号、它的动作（↑ 直走，↶ 左转，↷ 右转）和奖励。带圈的：刚存进去的。填色的：被抽进这一批的。'))}</p>
		</div>
	</div>
	{#if rows.length}
		<div class="tw" use:keyScroll>
			<table class="num">
				<thead><tr><th>{say(L('move', '步'))}</th><th>a</th><th>r</th><th>Q(s, a)</th><th>max Q<sub>target</sub>(s′)</th><th>y</th><th>(y − Q)²</th></tr></thead>
				<tbody>
					{#each rows as x, i (i)}<tr><td>#{x.t + 1}</td><td><span aria-hidden="true">{ARR[x.a]}</span><span class="sr">{t(ACTION[x.a])}</span></td><td>{sg(x.r)}</td><td>{fmt3(x.q)}</td><td class:end={!!x.done}>{x.done ? say(L('— (game over)', '—（终局）')) : fmt3(x.tmax)}</td><td>{fmt3(x.y)}</td><td>{num(x.loss)}</td></tr>{/each}
				</tbody>
			</table>
		</div>
		<p class="small">{say(L(`Mean loss of the batch: ${fmt3(m(P.at, 'mean'))}. One gradient step lowers this mean, for all ${rows.length} moves at once.`, `这一批的平均损失：${fmt3(m(P.at, 'mean'))}。一步梯度下降会一次性降低这 ${rows.length} 步的平均损失。`))}</p>
	{/if}
</div>

<style>
	.rv { display: grid; gap: 10px; min-width: 0; }
	.top { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; min-width: 0; }
	.side { flex: 1 1 220px; display: grid; gap: 6px; min-width: 0; }
	.lab { margin: 0; font-size: var(--fs-sm); font-weight: 600; }
	.slots { margin: 0; padding: 0; list-style: none; display: grid; grid-template-columns: repeat(auto-fill, minmax(76px, 1fr)); gap: 4px; }
	.slots li { padding: 2px 4px; border: 1px solid var(--rule-strong); border-radius: var(--r1); font-size: var(--fs-xs); white-space: nowrap; }
	.slots li span { color: var(--ink-2); }
	.slots li.pos { border-color: var(--div-pos); }
	.slots li.neg { border-color: var(--div-neg); }
	.slots li.now { box-shadow: 0 0 0 2px var(--accent); }
	.slots li.drawn { background: var(--accent-soft); font-weight: 700; }
	.small { margin: 0; font-size: var(--fs-xs); color: var(--ink-2); }
	.tw { overflow-x: auto; max-width: 100%; }
	table { border-collapse: collapse; font-size: var(--fs-xs); }
	th, td { padding: 2px 4px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }
	th { white-space: normal; vertical-align: bottom; }
	td.end { white-space: normal; }
	.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
</style>
