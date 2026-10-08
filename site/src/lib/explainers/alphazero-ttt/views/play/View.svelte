<!-- alphazero-ttt 'play': a game against the search (process `game`). Your moves go into the input (raw.game), so the
     game is re-played from the start with the search's replies; New game and "The search starts" reset it. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import Board from '../../parts/Board.svelte';
	import { rawOf } from '../../adapters/ttt-board/adapter.ts';
	import { CELL, L, markOf, toMove, winLine } from '../../rules.ts';
	import { gameOf } from '../../picture.ts';

	let P: StageProps = $props();
	const raw = $derived(rawOf(P.rt.input));
	const g = $derived(gameOf(P.rt));
	const set = (game: number[], aiFirst = raw.aiFirst) => (P.setInput ?? P.rt.setInput)({ ...raw, game, aiFirst });
	const yours = $derived(!!g && !g.over && toMove(g.b) === g.you);
	const status = $derived.by(() => {
		if (!g) return L('Loading the game…', '正在载入棋局……');
		const me = markOf(g.you);
		if (g.over) return g.m.result > 0 ? L('You won! Can you do it again?', '你赢了！还能再赢一次吗？') : g.m.result < 0 ? L('The search won. Try another line.', '搜索赢了。换一种下法试试。') : L('A draw. With perfect play from both sides, every game of tic-tac-toe ends this way.', '平局。双方都下得完美的话，井字棋每一盘都是这个结果。');
		return yours ? L(`Your move: you are ${me}. Click an empty cell.`, `轮到你：你执 ${me}。点一个空格。`) : L('The search is thinking…', '搜索正在思考……');
	});
</script>

<div class="pv">
	{#if g}
		<Board b={g.b} win={winLine(g.b)} onpick={yours ? (i) => set([...raw.game!, i]) : undefined} cells={(i) => ({ ring: g.last?.cell === i })} label={t(L('Your game against the search', '你和搜索的这盘棋'))} />
	{/if}
	<div class="side">
		<p class="st" role="status">{t(status)}</p>
		{#if g?.last}<p class="small">{t(L(`Its last move: ${CELL[g.last.cell].en}, after ${g.last.n} of its ${g.last.sims} simulations went there.`, `它上一步下在${CELL[g.last.cell].zh}：${g.last.sims} 次模拟里有 ${g.last.n} 次走了这一步。`))}</p>{/if}
		<div class="row">
			<button type="button" class="btn" onclick={() => set([])}>{t(L('New game', '新开一盘'))}</button>
			<button type="button" class="btn ghost" aria-pressed={raw.aiFirst} onclick={() => set([], !raw.aiFirst)}>{t(raw.aiFirst ? L('You start (play X)', '你先走（执 X）') : L('The search starts (you play O)', '让搜索先走（你执 O）'))}</button>
		</div>
	</div>
</div>

<style>
	.pv { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; min-width: 0; }
	.side { flex: 1 1 220px; display: grid; gap: 8px; min-width: 0; }
	.side p { margin: 0; }
	.st { font-weight: 600; }
	.small { font-size: var(--fs-xs); color: var(--ink-2); }
	.row { display: flex; flex-wrap: wrap; gap: 6px; }
</style>
