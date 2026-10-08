<!--
  gridworld Capture (local adapter 'gridworld'): the learner's whole input. The maze (presets, editor), the Q-learning
  settings γ, ε, α and the number of episodes, and their own slot-machine pulls. The compact copy beside the lesson
  keeps the presets, settings and pulls; the stage edits the maze and pulls through bus.ts, which lands here.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n } from '$lib/i18n/types';
	import { ARM_NAMES, MAX_PULLS, goalDistance, toRaw, worldOf, type GameRaw } from '../../world.ts';
	import { onEdit } from '../../bus.ts';
	import { UNREACHABLE } from './adapter.ts';
	import Editor from './Editor.svelte';

	interface Props { raw: unknown; presets?: { id: string; label: I18n; raw: unknown }[]; onchange: (raw: unknown, o?: { live?: boolean }) => void; compact?: boolean }
	let { raw, presets = [], onchange, compact = false }: Props = $props();
	const g = $derived(toRaw(raw));
	const set = (p: Partial<GameRaw>, live = false) => onchange({ ...g, ...p }, { live });
	onMount(() => onEdit((e) => onchange(e(toRaw(raw)))));
	const SLIDERS: [keyof GameRaw, string | I18n, number, number, number, I18n][] = [
		['gamma', 'γ', 0, 0.99, 0.01, { en: 'how much later rewards count', zh: '以后的奖励算多少' }],
		['epsilon', 'ε', 0, 1, 0.01, { en: 'how often to explore', zh: '多常去探索' }],
		['alpha', 'α', 0.05, 1, 0.05, { en: 'how far each update moves', zh: '每次更新挪多远' }],
		['episodes', { en: 'episodes', zh: '回合数' }, 1, 200, 1, { en: 'tries from start to finish to learn from', zh: '用来学习的回合（从起点到结束的尝试）数' }]
	];
</script>

<div class="cap" class:compact>
	<div class="row" role="group" aria-label={t({ en: 'Mazes', zh: '迷宫' })}>
		{#each presets as p (p.id)}<button type="button" class="btn" aria-pressed={JSON.stringify(toRaw(p.raw).rows) === JSON.stringify(g.rows)} onclick={() => set({ rows: toRaw(p.raw).rows })}>{t(p.label)}</button>{/each}
	</div>
	{#if !compact}<Editor raw={g} onedit={(r) => onchange(r)} />{/if}
	{#if goalDistance(worldOf(g)) < 0}<p class="warn" role="status">{t(UNREACHABLE)}</p>{/if}
	{#each SLIDERS as [k, sym, lo, hi, st, help] (k)}
		<label class="sl"><span class="num">{t(sym)} = {g[k]}</span><input type="range" min={lo} max={hi} step={st} value={g[k] as number} oninput={(e) => set({ [k]: Number(e.currentTarget.value) }, true)} onchange={(e) => set({ [k]: Number(e.currentTarget.value) })} /><span class="muted">{t(help)}</span></label>
	{/each}
	<div class="row" role="group" aria-label={t({ en: 'Your own pulls of the slot machines', zh: '你自己拉老虎机' })}>
		<span>{t({ en: `Your pulls: ${g.pulls.length}`, zh: `你拉了 ${g.pulls.length} 次` })}</span>
		{#each ARM_NAMES as a, k (a)}<button type="button" class="btn" disabled={g.pulls.length >= MAX_PULLS} onclick={() => set({ pulls: [...g.pulls, k] })}>{a}</button>{/each}
		<button type="button" class="btn ghost" disabled={!g.pulls.length} onclick={() => set({ pulls: [] })}>{t({ en: 'Reset', zh: '重来' })}</button>
	</div>
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); } .row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.sl { display: grid; grid-template-columns: 9em minmax(0, 1fr); gap: 0 8px; align-items: center; }
	.sl .muted { grid-column: 2; font-size: var(--fs-xs); } .compact .sl .muted { display: none; }
	.warn { margin: 0; padding: 6px 8px; border-left: 3px solid var(--div-neg); font-size: var(--fs-xs); }
</style>
