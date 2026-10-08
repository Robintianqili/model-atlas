<!-- snake-dqn Capture (local adapter 'snake-board'): set up the board the lesson looks at by steering the snake from the
     start (turn left, straight on, turn right), undo, start again, or pick a position. Every move keeps the food seed. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/index.ts';
	import Board from '../../parts/Board.svelte';
	import { ACTION, ENDS, L } from '../../rules.ts';
	import { crashNote, posOf, rawOf, type SnakeRaw } from './adapter.ts';

	let { raw, presets = [], onchange, compact = false }: CaptureProps<SnakeRaw> = $props();
	const r = $derived(rawOf(raw));
	const g = $derived(posOf(raw));
	const set = (moves: number[], seed = r.seed) => onchange?.({ seed, moves, draw: 0, game: r.game });
	const same = (p: unknown) => { const o = rawOf(p); return o.seed === r.seed && o.moves.join() === r.moves.join(); };
</script>

<div class="cap" class:compact>
	{#if presets.length}
		<div class="row" role="group" aria-label={t(L('Positions', '局面'))}>
			{#each presets as p (p.id)}<button type="button" class="btn" aria-pressed={same(p.raw)} onclick={() => set(rawOf(p.raw).moves, rawOf(p.raw).seed)}>{t(p.label)}</button>{/each}
		</div>
	{/if}
	<div class="main">
		<Board {g} small={compact} label={t(L(`The board: length ${g.snake.length}, ${g.score} eaten`, `棋盘：长度 ${g.snake.length}，吃了 ${g.score} 个`))} />
		<div class="side">
			<p class="small" role="status">{t(g.over ? crashNote(g.over) : L(`Steer the snake from the start (food seed ${r.seed}); ${r.moves.length} moves so far, ${g.score} eaten.`, `从开局开始操纵这条蛇（食物种子 ${r.seed}）；已经走了 ${r.moves.length} 步，吃了 ${g.score} 个。`))}</p>
			<div class="row">
				{#each [1, 0, 2] as a (a)}<button type="button" class="btn" disabled={!!g.over} onclick={() => set([...r.moves, a])}>{['↑', '↶', '↷'][a]} {t(ACTION[a])}</button>{/each}
			</div>
			<div class="row">
				<button type="button" class="btn" disabled={!r.moves.length} onclick={() => set(r.moves.slice(0, -1))}>{t(L('Undo', '撤回一步'))}</button>
				<button type="button" class="btn ghost" disabled={!r.moves.length} onclick={() => set([])}>{t(L('Back to the start', '回到开局'))}</button>
			</div>
			{#if g.over}<p class="small">{t(ENDS[g.over])}</p>{/if}
		</div>
	</div>
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
	.main { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }
	.side { flex: 1 1 180px; display: grid; gap: 8px; min-width: 0; }
	.small { margin: 0; font-size: var(--fs-xs); }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
</style>
