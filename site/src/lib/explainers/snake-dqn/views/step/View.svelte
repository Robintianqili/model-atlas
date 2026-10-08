<!-- snake-dqn 'step': the learner's board and what one lesson step adds (props.mode):
     input   the three planes of numbers the network reads          q       the online network's Q-values as arrows
     act     the ε-greedy draw u against ε (dqn frame 'act')        reward  the board before and after the move
     target  the target network's Q-values on s′ and y = r + γ·max  loss    δ = y − Q(s, a) and the loss δ² -->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtSetting } from '../../../../atlas/core/format.ts';
	import Board from '../../parts/Board.svelte';
	import { posOf, rawOf } from '../../adapters/snake-board/adapter.ts';
	import { ACTION, CELLS, L, encode } from '../../rules.ts';
	import { best, m, num, qOf, stateOf } from '../../picture.ts';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'input'));
	const s = $derived(stateOf(P.at));
	const g = $derived(s.game ?? posOf(P.rt.input));
	const f = (k: string) => m(P.at, k);
	const q = $derived(mode === 'input' || mode === 'q' ? qOf(P.rt) : [f('q0'), f('q1'), f('q2')]);
	const x = $derived(encode(g));
	const PL = [L('head', '蛇头'), L('body', '身体'), L('food', '食物')];
	const say = (o: { en: string; zh: string }) => t(o);
	const sgn = (v: number) => (v > 0 ? `+${v}` : String(v).replace('-', '−'));
	const raw = $derived(rawOf(P.rt.input));
	const redraw = () => (P.setInput ?? P.rt.setInput)({ ...raw, draw: raw.draw + 1 });
</script>

<div class="sv">
	{#if mode === 'input'}
		<div class="row">
			<Board {g} label={say(L('Your board', '你的棋盘'))} />
			<div class="planes">
				{#each PL as name, k (k)}
					<div>
						<p class="lab">{t(name)}</p>
						<div class="grid num" role="img" aria-label={t(name)}>{#each Array.from({ length: CELLS }, (_, i) => x[k * CELLS + i]) as v, i (i)}<span class:on={v > 0}>{v ? +v.toFixed(2) : '·'}</span>{/each}</div>
					</div>
				{/each}
			</div>
		</div>
		<p class="small">{say(L('Every cell of every plane is one number the network reads: 3 × 81 = 243 in all (· is 0).', '每一层的每个格子都是网络读到的一个数：一共 3 × 81 = 243 个（· 表示 0）。'))}</p>
	{:else if mode === 'q' || mode === 'act'}
		<div class="row">
			<Board {g} {q} pick={mode === 'act' ? f('a') : best(q)} label={say(L('Your board with a Q-value for each move', '你的棋盘，每一步标着 Q 值'))} />
			<div class="side">
				{#if q}<ul class="qs num">{#each q as v, a (a)}<li class:top={a === best(q)}>{t(ACTION[a])}: <b>{fmt3(v)}</b></li>{/each}</ul>{/if}
				{#if mode === 'act'}
					<svg viewBox="0 0 300 64" class="line" role="img" aria-label={say(L('The draw u on the line from 0 to 1, against ε', '0 到 1 的线上抽到的 u，与 ε 比较'))}>
						<rect x="10" y="24" width={280 * f('eps')} height="14" class="rand" /><rect x={10 + 280 * f('eps')} y="24" width={280 * (1 - f('eps'))} height="14" class="greedy" />
						<line x1={10 + 280 * f('u')} y1="18" x2={10 + 280 * f('u')} y2="44" class="u" />
						<text x="10" y="58">0</text><text x={10 + 280 * f('eps')} y="58">ε = {fmtSetting(f('eps'))}</text><text x="290" y="58" text-anchor="end">1</text>
						<text x={Math.min(250, Math.max(40, 10 + 280 * f('u')))} y="13" text-anchor="middle" class="ut">u = {fmt3(f('u'))}</text>
					</svg>
					<p class="small">{say(f('explore') ? L(`u < ε: explore. A second draw picks a move at random: ${ACTION[f('a')].en}.`, `u < ε：探索。再抽一次，随机选一步：${ACTION[f('a')].zh}。`) : L(`u ≥ ε: take the move with the highest Q, ${ACTION[f('a')].en}.`, `u ≥ ε：走 Q 最高的那一步，${ACTION[f('a')].zh}。`))}</p>
					<button type="button" class="btn" onclick={redraw}>{say(L('Draw u again', '再抽一次 u'))}</button>
				{/if}
			</div>
		</div>
	{:else if mode === 'reward' && s.next}
		<div class="row">
			<figure><Board {g} label={say(L('The board s before the move', '走之前的棋盘 s'))} small /><figcaption>s</figcaption></figure>
			<span class="to" aria-hidden="true">→ {t(ACTION[f('a')])} →</span>
			<figure><Board g={s.next} label={say(L('The board s′ after the move', '走之后的棋盘 s′'))} small /><figcaption>s′</figcaption></figure>
			<p class="chip num" class:pos={f('r') > 0} class:neg={f('r') < 0}>r = {sgn(f('r'))}</p>
		</div>
	{:else if (mode === 'target' || mode === 'loss') && s.next}
		<div class="col">
			{#if mode === 'target'}<Board g={s.next} q={f('terminal') ? null : [f('t0'), f('t1'), f('t2')]} pick={f('tbest')} label={say(L('The board s′ with the target network’s Q-values', '棋盘 s′，标着目标网络的 Q 值'))} />{/if}
			<div class="side">
				{#if mode === 'target'}
					<FormulaCard tex={f('terminal') ? 'y = r = {{r}}' : "\\begin{aligned} y &= r + \\gamma \\max_{a'} Q_{\\text{target}}(s', a') \\\\ &= {{r}} + {{gamma}} \\times {{tmax}} = {{y}} \\end{aligned}"} values={{ r: sgn(f('r')), gamma: fmtSetting(f('gamma')), tmax: f('tmax'), y: f('y') }} />
					<p class="small">{say(f('terminal') ? L('The game ended on this move: there is no next board to look ahead from.', '这一步之后游戏结束了：没有下一块棋盘可以往后看。') : L(`For comparison, the online network says max Q(s′, ·) = ${fmt3(f('omax'))}.`, `作为对照，在线网络给出的 max Q(s′, ·) = ${fmt3(f('omax'))}。`))}</p>
				{:else}
					<!-- two lines, as the target's formula: on one line δ's value was cut off at 390 px -->
					<FormulaCard tex={'\\begin{aligned} \\delta &= y - Q(s, a) \\\\ &= {{y}} - {{qa}} = {{delta}} \\end{aligned}'} values={{ y: f('y'), qa: f('qa'), delta: num(f('delta')) }} />
					<FormulaCard tex={'\\text{loss} = \\delta^2 = {{loss}}'} values={{ loss: num(f('loss')) }} />
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.sv { display: grid; gap: 10px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; min-width: 0; }
	.col { display: grid; gap: 10px; min-width: 0; }
	figure { margin: 0; display: grid; justify-items: center; gap: 2px; }
	figcaption { font-family: var(--font-mono); font-size: var(--fs-sm); }
	.side { flex: 1 1 220px; display: grid; gap: 8px; min-width: 0; }
	.planes { display: flex; flex-wrap: wrap; gap: 10px; min-width: 0; }
	.lab { margin: 0 0 2px; font-size: var(--fs-xs); color: var(--ink-2); }
	.grid { display: grid; grid-template-columns: repeat(9, 1fr); width: 243px; border: 1px solid var(--rule); font-size: 9px; line-height: 19px; text-align: center; }
	.grid span { color: var(--ink-2); }
	.grid .on { color: var(--ink); font-weight: 700; background: var(--accent-soft); }
	.small { margin: 0; font-size: var(--fs-xs); color: var(--ink-2); }
	.qs { margin: 0; padding: 0; list-style: none; display: grid; gap: 2px; }
	.qs .top b { color: var(--accent-strong); }
	.line { width: min(100%, 320px); }
	.line text { font-size: 11px; fill: var(--ink-2); }
	.line .ut { fill: var(--ink); font-weight: 700; }
	.rand { fill: var(--cat-3); opacity: 0.5; }
	.greedy { fill: var(--accent-soft); }
	.u { stroke: var(--ink); stroke-width: 2; }
	.to { font-size: var(--fs-sm); color: var(--ink-2); }
	.chip { margin: 0; padding: 2px 10px; border-radius: var(--r-pill); border: 1px solid var(--rule-strong); font-weight: 700; }
	.chip.pos { background: color-mix(in oklab, var(--div-pos) 25%, var(--surface)); }
	.chip.neg { background: color-mix(in oklab, var(--div-neg) 25%, var(--surface)); }
	.btn { justify-self: start; min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
</style>
