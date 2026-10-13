<!-- transe 'rank': link prediction one candidate at a time (see meta.ts); bar lengths are distances, to scale. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { ENT, TYPES } from '../../kg.ts';
	import { rankPic } from './meta.ts';

	let P: StageProps = $props();
	const m = $derived(rankPic(P));
	const theme = $derived(themeState.resolved);
	const name = (x: number) => t(ENT[x].name);
</script>

<div class="rk" style:min-height="300px">
	{#if !m}
		<p class="muted">{t({ en: 'Waiting for the trained points…', zh: '等待训练好的点…' })}</p>
	{:else}
		<p class="small num">{#if m.cur}{t({ en: `Candidate ${m.n} of 34: ${name(m.cur.x)}, distance ${fmt3(m.cur.d)}`, zh: `第 ${m.n} 个候选（共 34 个）：${name(m.cur.x)}，距离 ${fmt3(m.cur.d)}` })}{:else}{t({ en: 'All 34 candidates scored: the ranking', zh: '34 个候选都打完分了：排名' })}{/if}</p>
		<p class="small muted">{t({ en: m.done ? 'The 6 nearest entities (distance from h + r, shorter is better):' : 'Nearest so far (distance from h + r, shorter is better):', zh: m.done ? '最近的 6 个实体（离 h + r 的距离，越短越好）：' : '目前最近的（离 h + r 的距离，越短越好）：' })}</p>
		<ol class="bars">
			{#each m.best as c, k (c.x)}
				<li class:cur={m.cur?.x === c.x}>
					<span class="nm">{k + 1}. {name(c.x)} <span class="muted">({t(TYPES[ENT[c.x].type])})</span>{#if c.truth}{' · '}<b>{t({ en: 'a fact says so', zh: '有事实这么说' })}</b>{/if}</span>
					<span class="bar"><i style:width="{(c.d / m.max) * 100}%" style:background={categorical(ENT[c.x].type, theme)}></i></span>
					<span class="num">{fmt3(c.d)}</span>
				</li>
			{/each}
		</ol>
	{/if}
</div>

<style>
	.rk { display: grid; gap: 6px; min-width: 0; align-content: start; } p { margin: 0; } .small { font-size: var(--fs-sm); }
	.bars { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; font-size: var(--fs-sm); }
	.bars li { display: grid; grid-template-columns: minmax(0, 13em) minmax(0, 1fr) 3.5em; gap: 8px; align-items: center; padding: 1px 4px; }
	.bars li.cur { outline: 2px solid var(--highlight); } .nm { overflow-wrap: anywhere; } .bar { height: 12px; background: var(--surface-2); } .bar i { display: block; height: 100%; }
	.num { text-align: right; }
</style>
