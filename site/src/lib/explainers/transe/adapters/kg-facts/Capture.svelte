<!--
  transe Capture (local adapter 'kg-facts'): the starting sets, a fact to add (head, relation, tail), the question
  (head, relation, ?) and, outside the compact copy beside the lesson, every fact grouped by relation with a button to
  remove it (the facts not in the start set are marked).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/index.ts';
	import { ENT, hasFact, inputOf, PT_FACT, REL, START_FACTS, TYPES, type Fact, type KgRaw } from '../../kg.ts';
	import { PRESETS } from './adapter.ts';

	let { raw, onchange, compact = false }: CaptureProps<KgRaw> = $props();
	const g = $derived(inputOf(raw));
	const same = (r: KgRaw) => JSON.stringify(inputOf(r)) === JSON.stringify(g);
	let nf: Fact = $state([...PT_FACT]);
	const dup = $derived(hasFact(g.facts, nf) || nf[0] === nf[2]);
	const name = (i: number) => t(ENT[i].name);
	const fact = (f: Fact) => `${name(f[0])} · ${t(REL[f[1]])} · ${name(f[2])}`;
	const byType = TYPES.map((label, k) => ({ label, ids: ENT.flatMap((e, i) => (e.type === k ? [i] : [])) }));
	const set = (p: Partial<KgRaw>) => onchange?.({ ...g, ...p });
</script>

{#snippet entity(value: number, label: { en: string; zh: string }, pick: (i: number) => void)}
	<select {value} aria-label={t(label)} onchange={(e) => pick(Number(e.currentTarget.value))}>
		{#each byType as grp (grp.label.en)}<optgroup label={t(grp.label)}>{#each grp.ids as i (i)}<option value={i}>{name(i)}</option>{/each}</optgroup>{/each}
	</select>
{/snippet}
{#snippet relation(value: number, label: { en: string; zh: string }, pick: (r: number) => void)}
	<select {value} aria-label={t(label)} onchange={(e) => pick(Number(e.currentTarget.value))}>{#each REL as r, k (k)}<option value={k}>{t(r)}</option>{/each}</select>
{/snippet}

<div class="cap">
	<div class="row" role="group" aria-label={t({ en: 'Starting sets of facts', zh: '起始的事实集' })}>
		{#each PRESETS as p (p.id)}<button type="button" class="btn" aria-pressed={same(p.raw)} onclick={() => onchange?.(p.raw)}>{t(p.label)}</button>{/each}
	</div>
	<div class="row" role="group" aria-label={t({ en: 'Add a fact', zh: '加一个事实' })}>
		<span>{t({ en: 'Add a fact:', zh: '加一个事实：' })}</span>
		{@render entity(nf[0], { en: 'Head of the new fact', zh: '新事实的头实体' }, (i) => (nf[0] = i))}
		{@render relation(nf[1], { en: 'Relation of the new fact', zh: '新事实的关系' }, (r) => (nf[1] = r))}
		{@render entity(nf[2], { en: 'Tail of the new fact', zh: '新事实的尾实体' }, (i) => (nf[2] = i))}
		<button type="button" class="btn" disabled={dup} onclick={() => set({ facts: [...g.facts, [nf[0], nf[1], nf[2]]] })}>{t({ en: 'Add', zh: '加上' })}</button>
	</div>
	<div class="row" role="group" aria-label={t({ en: 'The question', zh: '问题' })}>
		<span>{t({ en: 'Ask:', zh: '提问：' })}</span>
		{@render entity(g.query[0], { en: 'Head of the question', zh: '问题的头实体' }, (i) => set({ query: [i, g.query[1]] }))}
		{@render relation(g.query[1], { en: 'Relation of the question', zh: '问题的关系' }, (r) => set({ query: [g.query[0], r] }))}
		<span>→ ?</span>
	</div>
	{#if compact}
		<p class="num">{t({ en: `${g.facts.length} facts; the full list, to remove some, is below the lesson.`, zh: `${g.facts.length} 个事实；完整的列表（可以删掉事实）在课程下方。` })}</p>
	{:else}
		{#each REL as r, k (k)}
			{@const own = g.facts.filter((f) => f[1] === k)}
			<div class="facts" role="group" aria-label={t(r)}>
				<span class="rel">{t(r)} ({own.length})</span>
				{#each own as f (f.join())}<button type="button" class="chip" class:new={!hasFact(START_FACTS, f)} title={t({ en: 'Remove this fact', zh: '删掉这个事实' })} onclick={() => set({ facts: g.facts.filter((x) => x !== f) })}>{fact(f)}{#if !hasFact(START_FACTS, f)} <b>{t({ en: 'added', zh: '新加' })}</b>{/if} ×</button>{/each}
			</div>
		{/each}
	{/if}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row, .facts { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; } p { margin: 0; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); } .btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	select { min-height: 30px; max-width: 100%; } .rel { font-weight: 600; margin-right: 4px; }
	.chip { min-height: 28px; padding: 1px 8px; font-size: var(--fs-xs); border: 1px solid var(--border); border-radius: var(--r1); background: var(--surface); color: var(--ink); }
	.chip.new { border: 1px dashed var(--accent); } .chip b { font-weight: 600; }
</style>
