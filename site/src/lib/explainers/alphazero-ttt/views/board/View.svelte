<!-- alphazero-ttt 'board': the learner's position, with what one step of the lesson adds to it (props.mode):
     input   the nine numbers the network reads, and the mask        prior   tap a guess first, then the priors
     value   the value head on a scale from −1 to +1, and perfect play visits  the search's visit counts beside the priors
     pick    the move the search plays -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import Board from '../../parts/Board.svelte';
	import { rawOf } from '../../adapters/ttt-board/adapter.ts';
	import { ARROW, CELL, L, isOver, legal, markOf, solve, toMove, winLine } from '../../rules.ts';
	import { netOf, posOf, rootOf, stateOf } from '../../picture.ts';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'input'));
	const raw = $derived(rawOf(P.rt.input));
	const b = $derived(posOf(P.rt));
	const net = $derived(netOf(P.rt));
	const root = $derived(rootOf(stateOf(P.at)));
	const me = $derived(toMove(b));
	const guess = $derived(raw.guess ?? null);
	const shown = $derived(mode !== 'prior' || guess !== null);
	const setGuess = (g: number | null) => (P.setInput ?? P.rt.setInput)({ ...raw, guess: g });
	const sgn = (v: number) => (v === -Infinity ? '−∞' : v > 0 ? '+1' : v < 0 ? '−1' : '0');
	const cells = $derived((i: number) => {
		if (mode === 'input') return { text: sgn(net?.board[i] ?? 0) };
		if (mode === 'prior') return !net || !shown || b[i] ? { guess: i === guess } : { text: fmt3(net.p[i]), fill: net.p[i], ring: i === net.top, guess: i === guess };
		if ((mode === 'visits' || mode === 'pick') && root && !b[i]) return { text: String(root.n[i]), fill: root.n[i] / Math.max(1, root.sum), ring: i === root.most, ghost: mode === 'pick' && i === root.most ? markOf(me) : undefined };
		return {};
	});
	const x = (v: number) => 20 + ((v + 1) / 2) * 300;
	const rows = $derived(root ? legal(b).sort((a, c) => root.n[c] - root.n[a] || a - c) : []);
	const say = (o: { en: string; zh: string }) => t(o);
</script>

<div class="bv">
	<div class="top">
		<Board {b} {cells} win={winLine(b)} onpick={mode === 'prior' && guess === null && !isOver(b) ? setGuess : undefined} label={say(L(`The position, ${markOf(me)} to move`, `局面，轮到 ${markOf(me)}`))} />
		<div class="side">
			{#if mode === 'input' && net}
				{#each [{ name: L('board', '棋盘'), xs: net.board }, { name: L('mask', '掩码'), xs: net.mask }] as row (row.name.en)}
					<p class="lab">{t(row.name)}</p>
					<div class="strip num">{#each row.xs as v, i (i)}<span class:pos={v > 0} class:neg={v < 0}>{sgn(v)}</span>{/each}</div>
				{/each}
				<p class="small">{say(L(`Read from ${markOf(me)}’s side: +1 = ${markOf(me)}, −1 = ${markOf(-me)}, 0 = empty. Cells go row by row from the top left.`, `从 ${markOf(me)} 这一方来读：+1 = ${markOf(me)}，−1 = ${markOf(-me)}，0 = 空格。格子从左上角开始一行一行地排。`))}</p>
			{:else if mode === 'prior' && net}
				{#if guess === null}
					<p class="ask">{say(L('Tap the empty cell you think the network likes most. Its priors appear after your guess.', '点一个你觉得网络最喜欢的空格。猜完之后，它的先验才会显示出来。'))}</p>
				{:else}
					<p>{say(L('Your guess', '你的猜测'))}: <b>{t(CELL[guess])}</b> <span class="num">{fmt3(net.p[guess])}</span></p>
					<p>{say(L('The network’s favourite', '网络最喜欢的'))}: <b>{t(CELL[net.top])}</b> <span class="num">{fmt3(net.p[net.top])}</span></p>
					<button type="button" class="btn again" onclick={() => setGuess(null)}>{say(L('Guess again', '再猜一次'))}</button>
				{/if}
			{:else if mode === 'value' && net}
				<svg viewBox="0 0 340 70" role="img" aria-label={say(L('The value on a scale from −1 to +1', '−1 到 +1 刻度上的价值'))}>
					<line x1="20" y1="30" x2="320" y2="30" class="axis" />
					{#each [{ v: -1, s: L('−1 loss', '−1 输') }, { v: 0, s: L('0 draw', '0 平') }, { v: 1, s: L('+1 win', '+1 赢') }] as k (k.v)}<line x1={x(k.v)} y1="24" x2={x(k.v)} y2="36" class="axis" /><text x={x(k.v)} y="54" class="tick">{t(k.s)}</text>{/each}
					<circle cx={x(solve(b))} cy="30" r="9" class="truth" /><circle cx={x(net.v)} cy="30" r="6" class="v" />
					<text x={x(net.v)} y="14" class="tick strong">v = {fmt3(net.v)}</text>
				</svg>
				<p class="small"><span class="dot v"></span>{say(L('the value head’s guess', '价值头的猜测'))} <span class="dot truth"></span>{say(L('perfect play from here', '从这里起双方都下得完美的结果'))}</p>
			{:else if (mode === 'visits' || mode === 'pick') && root}
				<table class="num">
					<thead><tr><th>{say(L('move', '走法'))}</th><th>{say(L('prior', '先验'))}</th><th>{say(L('visits', '访问'))}</th></tr></thead>
					<tbody>{#each rows as c (c)}<tr class:best={c === root.most}><td>{ARROW[c]} <span class="nm">{t(CELL[c])}</span></td><td><span class="cv"><span class="val">{fmt3(root.p[c])}</span><span class="track"><i class="bar p" style:width="{root.p[c] * 100}%"></i></span></span></td><td><span class="cv"><span class="val">{root.n[c]}</span><span class="track"><i class="bar n" style:width="{(root.n[c] / Math.max(1, root.sum)) * 100}%"></i></span></span></td></tr>{/each}</tbody>
				</table>
			{/if}
		</div>
	</div>
</div>

<style>
	.bv { display: grid; gap: 10px; min-width: 0; }
	.top { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
	.side { flex: 1 1 240px; display: grid; gap: 6px; min-width: 0; align-content: start; }
	.side p { margin: 0; }
	.lab { font-size: var(--fs-xs); color: var(--muted); }
	.strip { display: grid; grid-template-columns: repeat(9, minmax(0, 1fr)); gap: 2px; max-width: 320px; }
	.strip span { padding: 4px 0; text-align: center; font-size: var(--fs-xs); border: 1px solid var(--rule); border-radius: 3px; background: var(--surface); }
	.strip .pos { background: color-mix(in oklab, var(--div-pos) 35%, var(--surface)); }
	.strip .neg { background: color-mix(in oklab, var(--div-neg) 35%, var(--surface)); }
	.small { font-size: var(--fs-xs); color: var(--ink-2); }
	.ask { padding: 8px; border-left: 3px solid var(--accent); background: var(--accent-soft); }
	svg { width: 100%; max-width: 340px; }
	.axis { stroke: var(--ink-2); stroke-width: 1.5; }
	.tick { font-size: 11px; fill: var(--ink-2); text-anchor: middle; } .tick.strong { fill: var(--ink); font-weight: 600; }
	.v { fill: var(--accent); } .truth { fill: none; stroke: var(--ink); stroke-width: 2; stroke-dasharray: 3 2; }
	.dot { display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 8px; border-radius: 50%; vertical-align: middle; }
	.dot.v { background: var(--accent); } .dot.truth { border: 2px dashed var(--ink); }
	table { width: 100%; border-collapse: collapse; font-size: var(--fs-xs); }
	th { text-align: left; font-weight: 600; color: var(--muted); }
	td { padding: 2px 4px; }
	tr.best td { font-weight: 700; }
	.cv { display: flex; align-items: center; gap: 6px; }
	.val { flex: 0 0 3.4em; }
	.track { flex: 1 1 auto; height: 10px; min-width: 0; }
	.bar { display: block; height: 100%; min-width: 1px; border-radius: 2px; }
	.bar.p { background: color-mix(in oklab, var(--seq-mid) 40%, transparent); } .bar.n { background: color-mix(in oklab, var(--seq-hi) 55%, transparent); }
	td:nth-child(2), td:nth-child(3) { width: 34%; }
	.again { justify-self: start; }
	@media (max-width: 480px) { .nm { display: none; } }
</style>
