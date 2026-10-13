<!--
  hmm Capture (local adapter 'days'): the friend's activities, one chip per day. Press a chip to change that day's
  activity (walk → shop → clean), add a day with the three buttons, remove the last one, or pick a preset. The model's
  tables sit in a <details> below (sliders; each row keeps summing to 1). The tables are kept when the days change.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/index.ts';
	import Tables from '../../views/model/Tables.svelte';
	import adapter from './adapter.ts';
	import { ACTS, M, MAXT, obsOf, rawOf, type HmmRaw } from '../../hmm.ts';

	let { raw, presets = [], onchange, compact = false }: CaptureProps<HmmRaw> = $props();
	const obs = $derived(obsOf(raw));
	const list = $derived(presets.length ? presets : (adapter.presets?.({}) ?? []));
	const set = (o: number[]) => onchange?.({ ...rawOf(raw), obs: o });
	const same = (a: HmmRaw) => JSON.stringify(a) === JSON.stringify(rawOf(raw));
</script>

<div class="cap">
	{#if !compact}
		<div class="row" role="group" aria-label={t({ en: 'Example days', zh: '示例日子' })}>
			{#each list as p (p.id)}<button type="button" class="btn" aria-pressed={same(p.raw)} onclick={() => onchange?.(p.raw)}>{t(p.label)}</button>{/each}
		</div>
	{/if}
	<ol class="days" aria-label={t({ en: 'Her activity on each day (press to change it)', zh: '她每天的活动（点一下可以更换）' })}>
		{#each obs as o, d (d)}
			<li><button type="button" class="chip a{o}" aria-label={t({ en: `Day ${d}: ${ACTS[o].en}. Press for the next activity.`, zh: `第 ${d} 天：${ACTS[o].zh}。点一下换成下一种活动。` })} onclick={() => set(obs.map((x, i) => (i === d ? (x + 1) % M : x)))}><small>{d}</small>{t(ACTS[o])}</button></li>
		{/each}
	</ol>
	<div class="row">
		{#each ACTS as a, k (k)}<button type="button" class="btn" disabled={obs.length >= MAXT} onclick={() => set([...obs, k])}>+ {t(a)}</button>{/each}
		<button type="button" class="btn" disabled={!obs.length} onclick={() => set(obs.slice(0, -1))}>{t({ en: 'Remove the last day', zh: '删掉最后一天' })}</button>
	</div>
	{#if !compact}
		<p class="note">{t({ en: `Days count from 0, at most ${MAXT}. Her weather is hidden: the page has to infer it.`, zh: `天数从 0 数起，最多 ${MAXT} 天。她那里的天气是隐藏的：页面要把它推断出来。` })}</p>
		<details>
			<summary>{t({ en: 'The model’s tables (edit them here or in the task step)', zh: '模型的表格（可以在这里改，也可以在任务那一步改）' })}</summary>
			<Tables {raw} set={(r) => onchange?.(r)} />
		</details>
	{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.days { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 4px; }
	.chip { min-height: 36px; min-width: 52px; padding: 2px 8px; display: grid; justify-items: center; line-height: 1.1; font-size: var(--fs-xs); border: 1px solid var(--rule-strong); border-radius: 6px; background: var(--surface); }
	.chip small { color: var(--muted); }
	.chip.a2 { border-style: dashed; }
	.chip.a1 { border-style: double; border-width: 3px; }
	.note { margin: 0; font-size: var(--fs-xs); color: var(--muted); }
	details summary { cursor: pointer; font-size: var(--fs-xs); }
	details[open] summary { margin-bottom: 6px; }
</style>
