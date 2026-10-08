<!-- alphazero-ttt Capture (local adapter 'ttt-board'): set up the position the lesson looks at. Click an empty cell to
     play the next mark (X and O take turns), undo, clear, or pick a position. A new position clears the step-2 guess. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/index.ts';
	import Board from '../../parts/Board.svelte';
	import { L, boardOf, isOver, markOf, toMove, winLine } from '../../rules.ts';
	import { overNote, rawOf, type TttRaw } from './adapter.ts';

	let { raw, presets = [], onchange, compact = false }: CaptureProps<TttRaw> = $props();
	const r = $derived(rawOf(raw));
	const b = $derived(boardOf(r.moves));
	const set = (moves: number[]) => onchange?.({ ...r, moves, guess: null });
	const same = (m: number[]) => m.join() === r.moves.join();
</script>

<div class="cap" class:compact>
	{#if presets.length}
		<div class="row" role="group" aria-label={t(L('Positions', '局面'))}>
			{#each presets as p (p.id)}<button type="button" class="btn" aria-pressed={same(rawOf(p.raw).moves)} onclick={() => set(rawOf(p.raw).moves)}>{t(p.label)}</button>{/each}
		</div>
	{/if}
	<div class="main">
		<Board {b} win={winLine(b)} onpick={isOver(b) ? undefined : (i) => set([...r.moves, i])} label={t(L('The position: click an empty cell to play the next mark', '局面：点一个空格，下一步棋'))} />
		<div class="side">
			<p class="small" role="status">{t(isOver(b) ? overNote(r.moves) : L(`${markOf(toMove(b))} to move. Click an empty cell to play the next mark: X and O take turns.`, `轮到 ${markOf(toMove(b))} 走。点一个空格下一步棋：X 和 O 轮流走。`))}</p>
			<div class="row">
				<button type="button" class="btn" disabled={!r.moves.length} onclick={() => set(r.moves.slice(0, -1))}>{t(L('Undo', '撤回一步'))}</button>
				<button type="button" class="btn ghost" disabled={!r.moves.length} onclick={() => set([])}>{t(L('Empty board', '清空棋盘'))}</button>
			</div>
		</div>
	</div>
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
	.main { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
	.side { flex: 1 1 160px; display: grid; gap: 8px; min-width: 0; }
	.small { margin: 0; font-size: var(--fs-xs); }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.compact :global(.board) { width: min(100%, 180px); }
</style>
