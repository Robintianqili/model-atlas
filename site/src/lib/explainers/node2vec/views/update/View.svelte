<!-- node2vec 'update': one skip-gram step on one pair of members (see meta.ts). The worked number's row is outlined. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
	import { worked } from '../../vectors.ts';

	let P: StageProps = $props();
	const w = $derived(worked(P.at));
	const par = (v: number) => (v < 0 ? `(${fmt3(v)})` : fmt3(v));
</script>

<div class="up" style:min-height="340px">
	{#if !w}
		<p class="muted">{t({ en: 'Waiting for the first training step…', zh: '等待第一步训练…' })}</p>
	{:else}
		{@const vo = w.wout[w.o]}
		<p class="small"><span class="tag c">{t({ en: 'center', zh: '中心成员' })} {w.c}</span> <span class="tag o">{t({ en: 'context', zh: '上下文成员' })} {w.o}</span></p>
		<p class="small num">{t({ en: 'score', zh: '分数' })} u = h · v<sub>{w.o}</sub> = {fmt3(w.u[w.o])} · {t({ en: 'share', zh: '份额' })} p = {fmtPct(w.p[w.o])} ({t({ en: 'even split', zh: '平均分' })} 1/34 = {fmtPct(1 / 34)}) · {t({ en: 'loss', zh: '损失' })} −ln p = {fmt3(w.loss, 4)}</p>
		<div class="scroll" use:keyScroll>
			<table class="num-table small">
				<thead><tr><th>k</th><th>h<sub>k</sub> ({w.c})</th><th>v<sub>{w.o},k</sub></th><th>h<sub>k</sub> × v<sub>{w.o},k</sub></th><th>{t({ en: 'gradient g', zh: '梯度 g' })}</th><th>{t({ en: 'new h', zh: '新的 h' })}<sub>k</sub></th></tr></thead>
				<tbody>
					{#each w.h as x, k (k)}
						<tr class:hi={k === w.k}><td class="muted">{k}</td><td class="num">{fmt3(x)}</td><td class="num">{fmt3(vo[k])}</td><td class="num">{fmt3(x * vo[k])}</td><td class="num">{fmt3(w.gh[k])}</td><td class="num">{fmt3(w.h1[k])}</td></tr>
					{/each}
				</tbody>
			</table>
		</div>
		<p class="small num">{t({ en: `row k = ${w.k}:`, zh: `k = ${w.k} 那一行：` })} {fmt3(w.old)} − {w.lr} × {par(w.g)} = {fmt3(w.new)} · {t({ en: `after the step: score of ${w.o}`, zh: `这一步之后：${w.o} 的分数` })} {fmt3(w.u[w.o])} → {fmt3(w.u1)} · {t({ en: 'loss', zh: '损失' })} {fmt3(w.loss, 4)} → {fmt3(w.loss1, 4)}</p>
	{/if}
</div>

<style>
	.up { display: grid; gap: 6px; min-width: 0; align-content: start; } p { margin: 0; } .small { font-size: var(--fs-sm); } .scroll { overflow-x: auto; }
	.tag { padding: 0 6px; border-radius: var(--r1); font-weight: 600; } .tag.c { background: var(--accent-soft); } .tag.o { outline: 2px solid var(--highlight); }
	table { border-collapse: collapse; } th, td { padding: 1px 8px; text-align: right; white-space: nowrap; } th { font-weight: 600; } tr.hi td { outline: 2px solid var(--highlight); outline-offset: -2px; background: var(--accent-soft); }
	@media (max-width: 560px) { .up .scroll table { font-size: 10.5px; } th, td { padding: 1px 2px; } th { white-space: normal; } }
</style>
