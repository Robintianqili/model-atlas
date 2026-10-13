<!--
  behavior-cloning Capture (local adapter 'demos'): the learner's whole input. The presets, their demonstrations
  (recorded on the DemoPad, listed with a delete button each) and the gust chance. The compact copy beside the lesson
  keeps the presets, the list and the slider.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n } from '$lib/i18n/types';
	import { toRaw, worldOf, type BcRaw } from '../../world.ts';
	import DemoPad from './DemoPad.svelte';

	interface Props { raw: unknown; presets?: { id: string; label: I18n; raw: unknown }[]; onchange: (raw: unknown, o?: { live?: boolean }) => void; compact?: boolean }
	let { raw, presets = [], onchange, compact = false }: Props = $props();
	const g = $derived(toRaw(raw));
	const W = $derived(worldOf(g.map));
	const set = (p: Partial<BcRaw>, live = false) => onchange(toRaw({ ...g, ...p }), { live });
	const same = (r: unknown) => JSON.stringify(toRaw(r)) === JSON.stringify(g);
</script>

<div class="cap" class:compact>
	<div class="row" role="group" aria-label={t({ en: 'Ready-made inputs', zh: '现成的输入' })}>
		{#each presets as p (p.id)}<button type="button" class="btn" aria-pressed={same(p.raw)} onclick={() => onchange(p.raw)}>{t(p.label)}</button>{/each}
	</div>
	{#if !compact}<DemoPad raw={g} onedit={(r) => onchange(r)} />{/if}
	<ul class="demos">
		{#each g.demos as dm, i (i)}
			<li>
				<span>{t({ en: `Demonstration ${i + 1}: from column ${dm.start % W.w}, row ${Math.floor(dm.start / W.w)}, ${dm.moves.length} moves`, zh: `示范 ${i + 1}：从第 ${dm.start % W.w} 列第 ${Math.floor(dm.start / W.w)} 行出发，${dm.moves.length} 步` })}</span>
				<button type="button" class="btn ghost" onclick={() => set({ demos: g.demos.filter((_, j) => j !== i) })} aria-label={t({ en: `Delete demonstration ${i + 1}`, zh: `删除示范 ${i + 1}` })}>×</button>
			</li>
		{:else}
			<li class="muted">{t({ en: 'No demonstrations yet.', zh: '还没有示范。' })}</li>
		{/each}
	</ul>
	<label class="sl">
		<span class="num">{t({ en: 'gust chance', zh: '阵风概率' })} = {g.wind}</span>
		<input type="range" min="0" max="0.5" step="0.05" value={g.wind} oninput={(e) => set({ wind: Number(e.currentTarget.value) }, true)} onchange={(e) => set({ wind: Number(e.currentTarget.value) })} />
		<span class="muted">{t({ en: 'how often a gust blows the agent one cell down instead of its move', zh: '阵风把智能体往下吹一格、顶替它这一步的概率' })}</span>
	</label>
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.demos { margin: 0; padding: 0; list-style: none; display: grid; gap: 2px; font-size: var(--fs-xs); }
	.demos li { display: flex; gap: 6px; align-items: center; justify-content: space-between; }
	.sl { display: grid; grid-template-columns: 9em minmax(0, 1fr); gap: 0 8px; align-items: center; }
	.sl .muted { grid-column: 2; font-size: var(--fs-xs); }
	.compact .sl .muted { display: none; }
</style>
