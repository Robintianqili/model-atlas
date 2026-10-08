<!-- snake-dqn 'watch': the trained agent's game (process `agent`, at the frame the scrubber shows; arrows: the Q-values it
     sees there) next to the learner's own game on the same food seed. The learner's moves go into the input (raw.game),
     so the task can see them; New game clears them. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import Board from '../../parts/Board.svelte';
	import { rawOf } from '../../adapters/snake-board/adapter.ts';
	import { ACTION, ENDS, L } from '../../rules.ts';
	import { m, stateOf, yourGame } from '../../picture.ts';

	let P: StageProps = $props();
	const g = $derived(stateOf(P.at).game);
	const q = $derived(g && !g.over ? [m(P.at, 'q0'), m(P.at, 'q1'), m(P.at, 'q2')] : null);
	const raw = $derived(rawOf(P.rt.input));
	const you = $derived(yourGame(P.rt.input));
	const end = $derived.by(() => {
		const tp = P.at.tape, f = tp && tp.length ? tp.frame(tp.length - 1) : null;
		return f?.phase === 'end' ? f.metrics ?? null : null;
	});
	const set = (game: number[]) => (P.setInput ?? P.rt.setInput)({ ...raw, game });
	const why = (code: number) => ENDS[['', 'wall', 'self', 'full', 'hungry', 'cap'][code]] ?? L('it reached the move limit', '到了步数上限');
	const say = (o: { en: string; zh: string }) => t(o);
</script>

<div class="wv">
	<div class="half">
		<p class="lab">{say(L('The agent', '智能体'))}</p>
		{#if g}<Board {g} {q} pick={q ? q.indexOf(Math.max(...q)) : -1} label={say(L(`The agent’s game: ${g.score} eaten after ${g.moves} moves`, `智能体的这盘：走了 ${g.moves} 步，吃了 ${g.score} 个`))} />{/if}
		<p class="small num">{say(L(`Move ${g?.moves ?? 0}: ${g?.score ?? 0} eaten, length ${g?.snake.length ?? 0}.`, `第 ${g?.moves ?? 0} 步：吃了 ${g?.score ?? 0} 个，长度 ${g?.snake.length ?? 0}。`))}</p>
		{#if end}<p class="small">{say(L(`Its game ends after ${end.moves} moves with ${end.score} eaten: ${why(end.why).en}.`, `它的这盘在第 ${end.moves} 步结束，吃了 ${end.score} 个：${why(end.why).zh}。`))}</p>{/if}
	</div>
	<div class="half">
		<p class="lab">{say(L('Your game, same food seed', '你的这盘，食物种子相同'))}</p>
		<Board g={you.g} label={say(L(`Your game: ${you.g.score} eaten after ${you.g.moves} moves`, `你的这盘：走了 ${you.g.moves} 步，吃了 ${you.g.score} 个`))} />
		<p class="small num" role="status">{say(you.done ? L(`Game over after ${you.g.moves} moves: ${you.g.score} eaten (${(ENDS[you.g.over || 'hungry']).en}).`, `第 ${you.g.moves} 步结束：吃了 ${you.g.score} 个（${(ENDS[you.g.over || 'hungry']).zh}）。`) : L(`Move ${you.g.moves}: ${you.g.score} eaten. Steer with the buttons.`, `第 ${you.g.moves} 步：吃了 ${you.g.score} 个。用按钮操纵。`))}</p>
		<div class="row">
			{#each [1, 0, 2] as a (a)}<button type="button" class="btn" disabled={you.done} onclick={() => set([...raw.game.slice(0, you.used), a])}>{['↑', '↶', '↷'][a]} {t(ACTION[a])}</button>{/each}
			<button type="button" class="btn ghost" disabled={!raw.game.length} onclick={() => set([])}>{say(L('New game', '新开一盘'))}</button>
		</div>
	</div>
</div>

<style>
	.wv { display: flex; flex-wrap: wrap; gap: 18px; min-width: 0; }
	.half { flex: 1 1 240px; display: grid; gap: 6px; align-content: start; min-width: 0; }
	.lab { margin: 0; font-weight: 600; font-size: var(--fs-sm); }
	.small { margin: 0; font-size: var(--fs-xs); color: var(--ink-2); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
</style>
