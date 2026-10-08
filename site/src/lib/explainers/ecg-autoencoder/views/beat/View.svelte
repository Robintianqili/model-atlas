<!-- ecg-autoencoder 'beat' (meta.ts lists the modes). Green: the beat that came in; magenta: its rebuild. -->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import Plot from '../../parts/Plot.svelte';
	import { regionOf } from '../../ecg.ts';
	import { presetKind } from '../../adapters/series/adapter.ts';
	import { KIND_LABEL } from '../../beats.ts';
	import { beatData } from './meta.ts';

	let P: StageProps = $props();
	const d = $derived(beatData(P));
	const r = $derived(d.rec);
	const kind = $derived(presetKind(P.rt.input));
	const facts = $derived({ normal: P.rt.num('fact:medianNormal'), odd: P.rt.num('fact:medianOdd') });
	const label = $derived(t({ en: 'The beat over its 140 points', zh: '心跳的 140 个点' }));
	// the score next to the test set's medians and the threshold (bars from 0)
	const marks = $derived(r ? [
		{ k: 'normal', v: facts.normal, l: t({ en: 'median normal test beat', zh: '正常测试心跳的中位数' }) },
		{ k: 'thr', v: d.thr, l: t({ en: 'threshold', zh: '阈值' }) },
		{ k: 'odd', v: facts.odd, l: t({ en: 'median odd test beat', zh: '异常测试心跳的中位数' }) },
		{ k: 'you', v: r.score, l: t({ en: 'your beat', zh: '你的心跳' }) }
	] : []);
	const top = $derived(Math.max(...marks.map((m) => m.v)));
</script>

<div class="beat">
	{#if d.mode === 'odd'}
		<div class="two">
			<figure>
				<figcaption>{t({ en: `${d.gallery.length} of the normal training beats`, zh: `${d.gallery.length} 个正常的训练心跳` })}</figcaption>
				<Plot lines={d.gallery.map((y) => ({ y, cls: 'faint' as const }))} {label} h={200} />
			</figure>
			<figure class="answer">
				<figcaption>{t({ en: 'An odd beat: raised ST segment', zh: '一个异常心跳：ST 段抬高' })}</figcaption>
				{#if d.odd}
					<Plot lines={[{ y: d.odd.x, cls: 'real' }, { y: d.odd.xr, cls: 'gen' }]} gap regions {label} h={200} />
					<p class="cap">{t({ en: `Score ${fmt3(d.odd.score)}. The same beat without the raise scores ${fmt3(d.odd.twin.score)}.`, zh: `分数 ${fmt3(d.odd.score)}。同一个心跳去掉抬高后，分数是 ${fmt3(d.odd.twin.score)}。` })}</p>
				{/if}
			</figure>
		</div>
	{:else if !r}
		<p class="note">{t({ en: 'No beat yet: pick one or draw one in the input box.', zh: '还没有心跳：在输入框里选一个或画一个。' })}</p>
	{:else}
		<Plot
			lines={d.mode === 'input' ? [{ y: r.x, cls: 'real' }] : [{ y: r.x, cls: 'real' }, { y: r.xr, cls: 'gen' }]}
			gap={d.mode !== 'input' && d.mode !== 'rebuild'}
			regions={d.mode === 'input' || d.mode === 'error'}
			mark={d.mode === 'error' || d.mode === 'draw' ? r.worst : d.mode === 'input' ? d.max : null}
			strip={d.mode === 'error' || d.mode === 'draw' ? r.e : undefined}
			h={d.mode === 'score' ? 170 : 220}
			{label} />
		<p class="cap">
			{#if d.mode === 'input'}{t({ en: `${kind ? KIND_LABEL[kind].en.replace(/^./, (c) => c.toUpperCase()) + ' preset' : 'Your own drawing'}. Tallest point: ${fmt3(r.x[d.max])} at point ${d.max} (counting from 0).`, zh: `${kind ? '预设：' + KIND_LABEL[kind].zh : '你自己画的'}。最高点：第 ${d.max} 个点（从 0 数起），${fmt3(r.x[d.max])}。` })}
			{:else if d.mode === 'rebuild'}{t({ en: `The code: ${r.code.map(fmt3).join(', ')}.`, zh: `编码：${r.code.map(fmt3).join('，')}。` })}
			{:else}{t({ en: `Largest error ${fmt3(r.e[r.worst])} at point ${r.worst} (${regionOf(r.worst).en}). Score ${fmt3(r.score)}.`, zh: `最大误差 ${fmt3(r.e[r.worst])}，在第 ${r.worst} 个点（${regionOf(r.worst).zh}）。分数 ${fmt3(r.score)}。` })}{/if}
		</p>
		{#if d.mode === 'draw'}
			<p class="verdict" class:odd={r.score >= d.thr}>{r.score >= d.thr ? t({ en: `Flagged as odd: ${fmt3(r.score)} ≥ the threshold ${fmt3(d.thr)}`, zh: `标为异常：${fmt3(r.score)} ≥ 阈值 ${fmt3(d.thr)}` }) : t({ en: `Passes as normal: ${fmt3(r.score)} < the threshold ${fmt3(d.thr)}`, zh: `当作正常：${fmt3(r.score)} < 阈值 ${fmt3(d.thr)}` })}</p>
		{/if}
		{#if d.mode === 'score'}
			<!-- two lines, so the live numbers fit a phone; FormulaCard scrolls its formula (keyScroll) where it still does not fit -->
			<FormulaCard tex={'\\begin{aligned}\\text{score} &= \\sqrt{\\tfrac{1}{140}\\,(e_0^2 + \\dots + e_{139}^2)} \\\\ &= \\sqrt{\\tfrac{ {{ss}} }{140}} = {{score}}\\end{aligned}'} values={{ ss: r.ss, score: r.score }} where={[{ tex: 'e_i', text: { en: 'your beat minus its rebuild at point i', zh: '第 i 个点上你的心跳减去重建' } }]} />
			<div class="bars" role="group" aria-label={t({ en: 'Scores compared', zh: '分数对比' })}>
				{#each marks as m (m.k)}<div class="row"><span>{m.l}</span><i class="b-{m.k}" style:width="{(m.v / top) * 100}%"></i><b>{fmt3(m.v)}</b></div>{/each}
			</div>
		{/if}
	{/if}
</div>

<style>
	.beat { display: grid; gap: var(--s2); grid-template-columns: minmax(0, 1fr); }
	.two { display: grid; gap: var(--s3); grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
	figure { margin: 0; min-width: 0; }
	figcaption { font-size: var(--fs-sm); color: var(--ink-2); margin-bottom: 2px; }
	.cap, .note { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); font-variant-numeric: tabular-nums; }
	.verdict { margin: 0; font-weight: 600; color: var(--live); }
	.verdict.odd { color: var(--generated-text); }
	.bars { display: grid; gap: 4px; }
	.row { display: grid; grid-template-columns: minmax(7em, 13em) minmax(0, 1fr) 4em; gap: 6px; align-items: center; font-size: var(--fs-sm); }
	.row i { display: block; height: 12px; border-radius: 3px; min-width: 2px; }
	.row b { text-align: right; font-variant-numeric: tabular-nums; }
	.b-normal { background: var(--cat-0); }
	.b-odd { background: var(--cat-1); }
	.b-thr { background: var(--ink-2); }
	.b-you { background: var(--real); }
	.answer :global(.gap) { opacity: 0.26; }
	/* the predict gate leaves the stage readable: hide the answer (the rebuild) until the learner has answered */
	:global([inert]) .answer :global(.gen), :global([inert]) .answer :global(.gap), :global([inert]) .answer :global(.legend), :global([inert]) .answer .cap { visibility: hidden; }
</style>
