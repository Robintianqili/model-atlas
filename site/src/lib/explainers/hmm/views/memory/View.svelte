<!-- hmm 'memory' (meta.ts). limits: the made-up stretch as a strip of days (Sunny / Rainy colors, named in the legend),
     then the share of rainy spells of each length 1..8, as they happened and as the counted Markov chain implies (bars on
     one 0-to-1 scale). rnn: two cards, the HMM's memory against the char-RNN's (its numbers from facts.json), and a
     link to the char-RNN page. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { t } from '$lib/i18n/i18n.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
	import { FREE, STATES } from '../../hmm.ts';
	import { factOf, spellsOf } from './meta.ts';

	let P: StageProps = $props();
	const show = $derived(String(P.props.show ?? 'limits'));
	const cat = $derived((void themeState.resolved, [0, 1, 2, 3].map((k) => categorical(k))));
	const S = $derived(spellsOf(P));
	const m = $derived(P.at.frame?.metrics ?? {});
	const CW = 34, BH = 110, X0 = 34, TOP = 8;
	const W = $derived(X0 + S.model.length * CW + 6);
	const Y = (v: number) => TOP + BH * (1 - v);
	const fact = (k: string) => factOf(P, k);
</script>

<div class="mem">
	{#if show === 'limits'}
		<figure>
			<figcaption>{t({ en: `The made-up stretch: ${S.year.length} days, every rainy spell exactly 3 days`, zh: `编出来的一段日子：${S.year.length} 天，每段连雨天都正好 3 天` })}</figcaption>
			<div class="strip" role="img" aria-label={t({ en: 'One square per day, colored by its weather', zh: '每天一个方块，颜色表示当天的天气' })}>
				{#each S.year as s, d (d)}<span style:background={cat[s]}></span>{/each}
			</div>
			<p class="legend">{#each STATES as st, j (j)}<span><i style:background={cat[j]}></i>{t(st)}</span>{/each}</p>
		</figure>
		<figure>
			<figcaption>{t({ en: 'Share of rainy spells lasting 1, 2, 3, … days', zh: '持续 1、2、3……天的连雨天所占的比例' })}</figcaption>
			<svg width={W} height={TOP + BH + 30} viewBox="0 0 {W} {TOP + BH + 30}" role="img" aria-label={t({ en: 'Bars: spell lengths as they happened and as the chain implies', zh: '柱：实际的连雨天长度和马尔可夫链给出的长度' })}>
				{#each [0, 0.5, 1] as v (v)}<line x1={X0} x2={W - 4} y1={Y(v)} y2={Y(v)} class="grid" /><text x={X0 - 4} y={Y(v) + 3} class="ax end">{v}</text>{/each}
				{#each S.model as p, k (k)}
					<rect x={X0 + k * CW + 3} y={Y(S.truth[k] ?? 0)} width="13" height={BH - (Y(S.truth[k] ?? 0) - TOP)} style:fill={cat[2]} />
					<rect x={X0 + k * CW + 17} y={Y(p)} width="13" height={BH - (Y(p) - TOP)} style:fill={cat[3]} />
					<text x={X0 + k * CW + CW / 2} y={TOP + BH + 13} class="ax">{k + 1}</text>
				{/each}
				<text x={X0 + (S.model.length * CW) / 2} y={TOP + BH + 27} class="ax">{t({ en: 'spell length (days)', zh: '连雨天长度（天）' })}</text>
			</svg>
			<p class="legend"><span><i style:background={cat[2]}></i>{t({ en: 'as it happened', zh: '实际' })}</span><span><i style:background={cat[3]}></i>{t({ en: `the counted chain (rain stays rain with ${fmt3(m.aRR)})`, zh: `数出来的马尔可夫链（雨天保持雨天的概率 ${fmt3(m.aRR)}）` })}</span></p>
		</figure>
	{:else}
		<div class="cards">
			<section class="card">
				<h4>{t({ en: 'This hidden Markov model', zh: '这个隐马尔可夫模型' })}</h4>
				<ul>
					<li>{t({ en: 'Memory from one day to the next: one of 2 states', zh: '从一天带到下一天的记忆：2 种状态之一' })}</li>
					<li>{t({ en: `Its numbers: ${FREE} free probabilities in three tables, set by you`, zh: `它的数：三张表里 ${FREE} 个自由的概率，由你设定` })}</li>
					<li>{t({ en: 'Exact answers: the probability of the data (forward) and the best hidden path (Viterbi)', zh: '能精确算出：数据的概率（前向算法）和最好的隐藏路径（维特比算法）' })}</li>
				</ul>
			</section>
			<section class="card">
				<h4>{t({ en: 'The char-RNN on this site', zh: '本站的 char-RNN' })}</h4>
				<ul>
					<li>{t({ en: `Memory from one character to the next: ${fmtInt(fact('charRnnHidden'))} numbers in its hidden state h, and as many in its cell state c`, zh: `从一个字符带到下一个字符的记忆：隐藏状态 h 里 ${fmtInt(fact('charRnnHidden'))} 个数，细胞状态 c 里也有同样多` })}</li>
					<li>{t({ en: `Its numbers: ${fmtInt(fact('charRnnParams'))} parameters, learned by gradient descent`, zh: `它的数：${fmtInt(fact('charRnnParams'))} 个参数，用梯度下降学出来` })}</li>
					<li>{t({ en: `Unit ${fact('counterUnit')} (counting from 0) learned to count the letters of the current word (correlation ${fmt3(fact('counterR'))})`, zh: `单元 ${fact('counterUnit')}（从 0 数起）学会了数当前单词已有几个字母（相关系数 ${fmt3(fact('counterR'))}）` })}</li>
				</ul>
				<a href="{base}/m/char-rnn/">{t({ en: 'Open the char-RNN page →', zh: '打开 char-RNN 页面 →' })}</a>
			</section>
		</div>
	{/if}
</div>

<style>
	.mem { display: grid; gap: 12px; min-width: 0; font-size: var(--fs-sm); }
	figure { margin: 0; display: grid; gap: 4px; min-width: 0; }
	figcaption { font-size: var(--fs-xs); color: var(--muted); }
	.strip { display: flex; flex-wrap: wrap; gap: 1px; max-width: 100%; }
	.strip span { width: 8px; height: 14px; }
	.legend { margin: 0; display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: var(--fs-xs); }
	.legend i { display: inline-block; width: 12px; height: 12px; margin-right: 4px; vertical-align: -2px; border-radius: 2px; }
	svg { display: block; max-width: 100%; height: auto; }
	text { font-size: 11px; fill: var(--ink); text-anchor: middle; }
	.ax { fill: var(--muted); }
	.end { text-anchor: end; }
	.grid { stroke: var(--rule); }
	.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
	.card { border: 1px solid var(--rule-strong); border-radius: 8px; padding: 8px 12px; }
	.card h4 { margin: 0 0 4px; font-size: var(--fs-sm); }
	.card ul { margin: 0 0 6px; padding-left: 18px; display: grid; gap: 4px; }
</style>
